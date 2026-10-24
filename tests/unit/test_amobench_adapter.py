"""Check AMO data conversion and extraction without loading GPU dependencies."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO.parent / 'harness/science-agent-sandbox'
spec = importlib.util.spec_from_file_location('amo_answer_extraction', HARNESS / 'science_agent/eval/answer_extraction.py')
extraction = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = extraction
spec.loader.exec_module(extraction)


class AmoAdapterTests(unittest.TestCase):
    def test_preserves_all_references_and_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'source.jsonl', Path(tmp) / 'output.jsonl'
            source.write_text(json.dumps(dict(id='amo1', question='Compute x.', answer=['0', '2'])) + '\n')
            subprocess.run([sys.executable, str(REPO / 'examples/python/utils/prepare_amobench.py'),
                            '--source', str(source), '--output', str(output), '--expected-rows', '1'],
                           check=True, capture_output=True)
            row = json.loads(output.read_text())
            self.assertEqual(json.loads(row['science_item']['answer']), ['0', '2'])
            self.assertEqual(row['prompt'], [{'role': 'user', 'content': 'Compute x.'}])
            self.assertEqual(row['datasource'], 'amobench')

    def test_extraction_keeps_multiple_nested_boxes_and_ignores_thinking(self):
        tree = ast.parse((REPO / 'examples/python/agents/science_legacy_react_sandbox.py').read_text())
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Name) and n.test.id == 'amo_protocol')
        for raw, expected in [
            (r'<think>\boxed{99}</think>Final: \boxed{\frac{1}{2}}, \boxed{3}<|im_end|>',
             '\\boxed{\\frac{1}{2}}\n\\boxed{3}'),
            ('<think>discard</think>6<|endoftext|>', '6'),
            ('', ''),
        ]:
            with self.subTest(raw=raw):
                scope = dict(amo_protocol=True, raw_prediction=raw, find_boxed_spans=extraction.find_boxed_spans)
                exec(compile(ast.Module(body=[block], type_ignores=[]), '<amo extraction>', 'exec'), scope)
                self.assertEqual(scope['prediction'], expected)
                self.assertTrue(scope['next_protocol'])


if __name__ == '__main__':
    unittest.main()
