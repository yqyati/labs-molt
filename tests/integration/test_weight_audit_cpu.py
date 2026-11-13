"""CPU checks for value instrumentation and fail-closed result aggregation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


class WeightAuditTest(unittest.TestCase):
    def test_hook_detects_single_element_change_and_unchanged_refit(self):
        program = '''
import torch
from types import SimpleNamespace
from molt.trainer.vllm.vllm_worker_wrap import WorkerWrap
assert WorkerWrap.reset_weight_update_check.__module__ == 'sitecustomize'
model = torch.nn.Module()
model.qkv_proj = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16)
worker = WorkerWrap()
worker.model_runner = SimpleNamespace(model=model)
worker.reset_weight_update_check()
worker.weight_update_missing()
worker.reset_weight_update_check()
with torch.no_grad():
    model.qkv_proj.weight[17, 29] += 1
worker.weight_update_missing()
'''
        env = dict(os.environ, MOLT_TEST_WEIGHT_VALUE_AUDIT='1',
                   PYTHONPATH=os.pathsep.join((str(HERE), str(REPO))))
        run = subprocess.run([sys.executable, '-c', program], env=env,
                             text=True, capture_output=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        events = [json.loads(line.split('] ', 1)[1]) for line in run.stdout.splitlines()
                  if line.startswith('[test_weight_values]')]
        self.assertEqual([e['changed_parameters'] for e in events], [0, 1])
        self.assertEqual([e['refit_index'] for e in events], [1, 2])

    def test_supervisor_requires_changed_weights_at_verified_boundary(self):
        for weight_boundary, expected in ((None, 2), (4, 2), (2, 0)):
            with self.subTest(weight_boundary=weight_boundary), tempfile.TemporaryDirectory() as directory:
                lines = ['Train epoch [1/1]']
                if weight_boundary is not None:
                    lines.append('[test_weight_values] ' + json.dumps(dict(
                        event='refit', refit_index=weight_boundary, changed_parameters=1)))
                for boundary in (1, 2, 3):
                    lines.append('[partial_routing] ' + json.dumps(dict(
                        event='paused', boundary=boundary, requests_with_tokens=64, snapshots=64)))
                    lines.append(f"Global step {boundary}: {{'actor_grad_norm': 0.1, 'timing/broadcast': 1.0}}")
                    lines.append('[partial_routing] ' + json.dumps(dict(
                        event='finished', boundary=boundary, continued=True, prefix_preserved=True)))
                if weight_boundary is not None:
                    lines.append('[test_weight_values] ' + json.dumps(dict(
                        event='refit', refit_index=weight_boundary, changed_parameters=1)))
                report = Path(directory) / 'audit.json'
                run = subprocess.run([sys.executable, str(HERE / 'watch_partial_routing.py'),
                                      str(report), '--', sys.executable, '-c',
                                      'print(' + repr('\n'.join(lines)) + ', flush=True)'],
                                     text=True, capture_output=True, timeout=30)
                self.assertEqual(run.returncode, expected, run.stdout + run.stderr)
                result = json.loads(report.read_text())
                self.assertEqual(result['status'] == 'PASS', expected == 0)
                if expected == 0:
                    self.assertEqual(result['verified_weight_and_routing_boundaries'], [2])


if __name__ == '__main__':
    unittest.main()
