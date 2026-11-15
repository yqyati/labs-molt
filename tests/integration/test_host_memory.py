"""Measure real Molt buffer/offload lifetimes without loading a checkpoint.

Run directly in the training image. Each phase has a fresh process so allocator
high-water marks from another phase cannot masquerade as retained samples.
"""
import argparse
import ctypes
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def memory():
    values = {}
    for path in ('/proc/self/status', '/proc/self/smaps_rollup'):
        for line in Path(path).read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'VmLck', 'VmPin', 'Pss', 'Pss_Anon', 'Private_Dirty'):
                values[key + '_mib'] = round(int(value.split()[0]) / 1024, 2)
    return values


def record(stage, **fields):
    row = dict(stage=stage, pid=os.getpid(), **fields, **memory())
    print(json.dumps(row), flush=True)
    return row


def cpu_bytes(value):
    import torch
    # Count unique backing storage, including DTensor's actual local shard.
    pending, seen = [value], {}
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
        elif isinstance(item, torch.Tensor):
            item = getattr(item, '_local_tensor', item)
            if item.device.type == 'cpu':
                storage = item.untyped_storage()
                seen[storage.data_ptr()] = storage.nbytes()
    return sum(seen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all', 'buffer', 'routing', 'optimizer'), default='all')
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--tokens', type=int, default=131072)
    parser.add_argument('--optimizer-numel', type=int, default=2_000_000)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.iterations < 6 or args.samples < 1 or args.tokens < 2 or args.optimizer_numel < 1:
        parser.error('need iterations >= 6, samples >= 1, tokens >= 2, optimizer-numel >= 1')
    if args.phase == 'all':
        output = args.output or Path('/tmp') / f'molt-host-memory-{os.getpid()}.jsonl'
        failed = []
        with output.open('x') as log:
            for phase in ('buffer', 'routing', 'optimizer'):
                command = [sys.executable, str(Path(__file__).resolve()), '--phase', phase,
                           '--iterations', str(args.iterations), '--samples', str(args.samples),
                           '--tokens', str(args.tokens), '--optimizer-numel', str(args.optimizer_numel),
                           '--device', args.device]
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                try:
                    for line in child.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    if child.wait():
                        failed.append(phase)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.wait()
        print(json.dumps(dict(report=str(output), failed_phases=failed,
                              scope='component test; not an eight-rank full-model leak verdict')))
        return bool(failed)

    import torch
    torch.set_num_threads(2)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda':
        torch.cuda.init()
    gc.collect()
    record('baseline', phase=args.phase, device=device, torch_version=torch.__version__)
    post_clear = []
    if args.phase in ('buffer', 'routing'):
        from torch.utils.data import DataLoader
        from molt.trainer.algorithm.experience import Experience
        from molt.trainer.algorithm.replay_buffer import NaiveReplayBuffer
        buffer = NaiveReplayBuffer(sample_batch_size=1, cpu_offload=True)
        for iteration in range(args.iterations):
            refs = []
            for _ in range(args.samples):
                sample = Experience(
                    sequences=torch.ones((1, args.tokens), dtype=torch.long),
                    attention_mask=torch.ones((1, args.tokens), dtype=torch.long),
                    action_mask=torch.ones((1, args.tokens - 1), dtype=torch.bool),
                    action_log_probs=torch.zeros((1, args.tokens - 1)),
                    rollout_log_probs=torch.zeros((1, args.tokens - 1)),
                    advantages=torch.ones((1, args.tokens - 1)),
                    returns=torch.ones((1, args.tokens - 1)),
                    total_length=torch.tensor([args.tokens]),
                    routed_experts=(torch.ones((1, 40, 8, args.tokens), dtype=torch.int16)
                                    if args.phase == 'routing' else None),
                )
                buffer.append(sample)
                del sample
            for item in buffer.items:
                refs.extend(weakref.ref(t) for t in vars(item).values() if isinstance(t, torch.Tensor))
            del item
            backing = cpu_bytes([vars(item) for item in buffer.items])
            record('buffer_filled', iteration=iteration, samples=len(buffer.items), cpu_tensor_bytes=backing)
            loader = DataLoader(buffer, batch_size=1, pin_memory=True, collate_fn=buffer.collate_fn)
            window = list(loader)
            window_bytes = cpu_bytes([vars(item) for item in window])
            pinned = sum(item.sequences.is_pinned() for item in window)
            record('window_ready', iteration=iteration, buffer_bytes=backing,
                   window_bytes=window_bytes, pinned_samples=pinned)
            for item in window:
                item.to_device(torch.device(device))
                item.to_device(torch.device('cpu'))
            del item
            if device == 'cuda':
                torch.cuda.synchronize()
            record('after_device_roundtrip', iteration=iteration)
            del window, loader
            buffer.clear()
            record('cleared_before_gc', iteration=iteration, buffer_items=len(buffer.items))
            gc.collect()
            alive = sum(ref() is not None for ref in refs)
            post_clear.append(record('cleared_after_gc', iteration=iteration, tracked_tensors_alive=alive))
            if alive:
                raise AssertionError(f'{alive} tracked buffer tensor wrappers remain alive')
    else:
        # Import the real implementation without loading the FSDP strategy package.
        path = REPO / 'molt/trainer/fsdp/optimizer_offload.py'
        spec = importlib.util.spec_from_file_location('offloader_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        params = [torch.nn.Parameter(torch.ones(args.optimizer_numel, device=device))]
        optimizer = torch.optim.AdamW(params, lr=1e-3, foreach=False)
        offloader = module.CpuOptimizerOffloader()
        initial_pointers = None
        for iteration in range(args.iterations):
            optimizer.zero_grad(set_to_none=True)
            params[0].grad = torch.ones_like(params[0])
            offloader.step(optimizer, params)
            if device == 'cuda':
                torch.cuda.synchronize()
            pointers = tuple(t.data_ptr() for d in (offloader._cpu_master, offloader._cpu_grad)
                             for t in d.values())
            if initial_pointers is None:
                initial_pointers = pointers
            if pointers != initial_pointers or len(offloader._cpu_master) != len(params):
                raise AssertionError('persistent optimizer buffers grew or changed storage')
            record('optimizer_step', iteration=iteration,
                   master_bytes=cpu_bytes(offloader._cpu_master),
                   grad_bytes=cpu_bytes(offloader._cpu_grad), state_bytes=cpu_bytes(optimizer.state),
                   parameter_value=float(params[0][0].detach().cpu()))
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            post_clear.append(record('optimizer_steady', iteration=iteration))
        del params, optimizer, offloader
        gc.collect()
        record('optimizer_released')
    record('before_trim')
    libc = ctypes.CDLL(None)
    trim = getattr(libc, 'malloc_trim', None)
    if trim is not None:
        trim(0)
    record('after_trim', malloc_trim_available=trim is not None)
    tail = post_clear[3:]
    growth = round(tail[-1]['VmRSS_mib'] - tail[0]['VmRSS_mib'], 2)
    record('phase_summary', phase=args.phase, lifecycle_checks='PASS',
           post_warmup_rss_growth_mib=growth,
           interpretation='inspect the full time series; RSS growth alone does not establish a leak',
           full_model_or_distributed_tested=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
