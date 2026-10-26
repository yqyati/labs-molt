"""Validate FSO reference isolation and multi-part answers without GPU imports."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]


class FsoAdapterTests(unittest.TestCase):
    def test_conversion_keeps_reference_out_of_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.jsonl", Path(tmp) / "output.jsonl"
            source.write_text(json.dumps(dict(id="fso1", question="Find both quantities.",
                                              answer=["secret-reference"], subject="physics")) + "\n")
            subprocess.run([sys.executable, str(REPO / "examples/python/utils/prepare_frontierscience_olympiad.py"),
                            "--source", str(source), "--output", str(output), "--expected-rows", "1"],
                           check=True, capture_output=True)
            row = json.loads(output.read_text())
            self.assertEqual(row["science_item"]["answer"], "secret-reference")
            self.assertEqual(row["science_item"]["subject"], "physics")
            self.assertEqual(row["datasource"], "frontierscience_olympiad")
            self.assertNotIn("secret-reference", row["prompt"][0]["content"])
            self.assertIn("each requested answer", row["prompt"][0]["content"])

    def test_final_response_keeps_units_labels_and_multiple_boxes(self):
        tree = ast.parse((REPO / "examples/python/agents/science_legacy_react_sandbox.py").read_text())
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Name) and n.test.id == "fso_protocol"
                     and any(isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prediction"
                             for t in s.targets) for s in n.body))
        expected = r"(a) \boxed{2} m; (b) \boxed{3} s"
        scope = dict(fso_protocol=True, raw_prediction="<think>discard \\boxed{99}</think>" + expected + "<|im_end|>")
        exec(compile(ast.Module(body=[block], type_ignores=[]), "<fso extraction>", "exec"), scope)
        self.assertEqual(scope["prediction"], expected)
        self.assertTrue(scope["next_protocol"])

    def test_judge_uses_olympiad_equivalence(self):
        tree = ast.parse((REPO / "examples/python/agents/science_legacy_react_sandbox.py").read_text())
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Name) and n.test.id == "fso_protocol"
                     and any(isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "judge_user_prompt"
                             for t in s.targets) for s in n.body))
        scope = dict(fso_protocol=True, reference="gold", prediction="candidate", ctx=type("Context", (), {"prompt": "question"})())
        exec(compile(ast.Module(body=[block], type_ignores=[]), "<fso judge>", "exec"), scope)
        prompt = scope["judge_user_prompt"]
        for text in ("unit conversions", "every part", "gold", "candidate", '"correct": true/false'):
            self.assertIn(text, prompt)


if __name__ == "__main__":
    unittest.main()
