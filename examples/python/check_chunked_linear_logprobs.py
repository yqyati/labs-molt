"""GPU-only numerical/FSDP smoke check; run on a cluster, never the dev host.

    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
        examples/python/check_chunked_linear_logprobs.py

This checks a tiny LM head and its FSDP lifecycle, not full-model 64k capacity.
"""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Cluster CUDA worker required; do not run on the development host")
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    torch.backends.cuda.matmul.allow_tf32 = False
    module_path = Path(__file__).resolve().parents[2] / "molt/models/experimental_linear_logprobs.py"
    spec = importlib.util.spec_from_file_location("linear_logprobs_probe", module_path)
    experimental = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experimental)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for temperature in (1.0, 0.7):
                torch.manual_seed(42)
                head = torch.nn.Linear(32, 97, bias=False, device="cuda", dtype=dtype)
                weight = head.weight.detach().clone().requires_grad_()
                fully_shard(head, mesh=mesh)
                model = SimpleNamespace(lm_head=head, config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=97)))
                for iteration in range(2):
                    # Cross chunk boundaries, vary targets by CP rank, and pass arbitrary upstream gradients.
                    torch.manual_seed(100 + dist.get_rank() + iteration)
                    hidden = torch.randn(1, 521, 32, device="cuda", dtype=dtype, requires_grad=True)
                    reference_hidden = hidden.detach().clone().requires_grad_()
                    targets = torch.randint(97, (1, 521), device="cuda")
                    upstream = torch.randn(1, 521, device="cuda") * 0.01
                    logits = F.linear(reference_hidden, weight).float() / temperature
                    expected = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
                    expected_entropy = logits.logsumexp(-1) - (logits.softmax(-1) * logits).sum(-1)
                    (expected * upstream).sum().backward()
                    dist.all_reduce(weight.grad)
                    weight.grad.div_(dist.get_world_size())
                    previous_forward = head.__dict__.get("forward")
                    with experimental.chunked_lm_head(model, targets, temperature, True):
                        actual = head(hidden)
                    assert head.__dict__.get("forward") is previous_forward
                    torch.testing.assert_close(actual[..., 0], expected, atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(actual[..., 1].detach(), expected_entropy, atol=2e-5, rtol=2e-5)
                    (actual[..., 0] * upstream).sum().backward()
                    tolerance = 1e-3 if dtype == torch.bfloat16 else 2e-6
                    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=tolerance, rtol=0.03)
                    torch.testing.assert_close(head.weight.grad.full_tensor(), weight.grad, atol=tolerance, rtol=0.03)
                    head.zero_grad(set_to_none=True)
                    weight.grad = None
                    if dist.get_rank() == 0:
                        print(f"PASS FSDP/chunk boundaries/logprobs/entropy/gradients: {dtype}, T={temperature}, iter={iteration}", flush=True)
                with experimental.chunked_lm_head(model, targets, temperature, False), torch.no_grad():
                    no_grad_output = head(hidden.detach())
                assert not no_grad_output.requires_grad
                torch.testing.assert_close(no_grad_output[..., 0], expected.detach(), atol=2e-5, rtol=2e-5)
                # Verify restoration when the model forward raises.
                try:
                    with experimental.chunked_lm_head(model, targets, temperature, False):
                        raise RuntimeError("intentional context restoration probe")
                except RuntimeError:
                    pass
                assert head.__dict__.get("forward") is previous_forward
        if dist.get_rank() == 0:
            print("PASS tiny GPU checks; real NeMo CP alignment and 64k RL remain to be validated", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
