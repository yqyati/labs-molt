"""Opt-in, process-local instrumentation; never changes installed Molt/vLLM files."""
import hashlib
import json
import os


if os.environ.get('MOLT_TEST_WEIGHT_VALUE_AUDIT') == '1':
    from molt.trainer.vllm.vllm_worker_wrap import WorkerWrap

    _original_reset = WorkerWrap.reset_weight_update_check
    _original_report = WorkerWrap.weight_update_missing

    def _fingerprints(worker):
        import torch

        result = {}
        for name, param in worker.model_runner.model.named_parameters():
            if 'visual' in name or not name.endswith('.weight'):
                continue
            if not any(part in name for part in ('qkv_proj', 'q_proj', 'v_proj')):
                continue
            # Hash complete tensors, not sparse samples that can miss a small LoRA update.
            raw = param.detach().cpu().contiguous().view(torch.uint8).numpy()
            result[name] = hashlib.sha256(raw.tobytes()).hexdigest()
            if len(result) == 3:
                break
        if not result:
            raise RuntimeError('weight audit found no language q/v/qkv projection weights')
        return result

    def _audited_reset(self):
        _original_reset(self)
        self._test_refit_index = getattr(self, '_test_refit_index', 0) + 1
        self._test_before = _fingerprints(self)

    def _audited_report(self):
        before = self._test_before
        after = _fingerprints(self)
        if before.keys() != after.keys():
            raise RuntimeError('weight audit parameter set changed during refit')
        changed = [name for name in before if before[name] != after[name]]
        print('[test_weight_values] ' + json.dumps({
            'event': 'refit', 'refit_index': self._test_refit_index,
            'matched_parameters': len(before), 'changed_parameters': len(changed),
            'before_sha256': before, 'after_sha256': after,
        }), flush=True)
        return _original_report(self)

    WorkerWrap.reset_weight_update_check = _audited_reset
    WorkerWrap.weight_update_missing = _audited_report
