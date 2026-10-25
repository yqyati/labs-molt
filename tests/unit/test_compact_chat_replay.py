"""Check compacted message replay without importing GPU or Ray dependencies."""

import ast
import asyncio
import copy
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[2] / "molt/agents/_chat_server.py"
function = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_messages_to_chat")
namespace = {"os": os, "re": re, "json": json, "deepcopy": copy.deepcopy,
             "_content_to_text_and_images": lambda value: (value, [])}
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                         function], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)


class CompactReplayTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"AGENT_CONTEXT_COMPACTION": "1", "AGENT_OBSERVATION_MASKING": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.messages = [
            {"role": "system", "content": "Solve the task."},
            {"role": "user", "content": "Original problem"},
            {"role": "user", "content": "[Context summary] Earlier work"},
        ]
        for i in range(3):
            self.messages.extend([
                {"role": "assistant", "content": f"<think>Compute {i}</think>\n\n<|im_end|>",
                 "reasoning_content": "", "tool_calls": [{"id": f"call{i}", "type": "function",
                 "function": {"name": "python", "arguments": json.dumps({"code": f"print({i})"})}}]},
                {"role": "tool", "tool_call_id": f"call{i}", "name": "python", "content": str(i)},
            ])

    def render_rows(self):
        return namespace["_messages_to_chat"](SimpleNamespace(expand_image_placeholder=False), self.messages)[0]

    def test_nonmask_compaction_keeps_calls_results_and_original_messages(self):
        original = copy.deepcopy(self.messages)
        rows = self.render_rows()
        for i in range(3):
            assistant, tool = rows[3 + 2*i:5 + 2*i]
            call = assistant["tool_calls"][0]
            self.assertEqual(call["id"], tool["tool_call_id"])
            self.assertEqual(call["function"]["arguments"], {"code": f"print({i})"})
            self.assertEqual(assistant["reasoning_content"], f"Compute {i}")
            self.assertEqual(assistant["content"], "")
        self.assertEqual(self.messages, original)

    def test_trailing_framing_removed_without_removing_visible_content(self):
        self.messages[3]["content"] = "<think>Work</think>\nVisible result\n<|im_end|><|endoftext|>\n"
        row = self.render_rows()[3]
        self.assertEqual(row["reasoning_content"], "Work")
        self.assertEqual(row["content"], "Visible result")

    def test_existing_separate_reasoning_preserved(self):
        self.messages[3].update(content="Visible result<|im_end|>", reasoning_content="Separate thought")
        row = self.render_rows()[3]
        self.assertEqual(row["reasoning_content"], "Separate thought")
        self.assertEqual(row["content"], "Visible result")

    def test_disabled_preserves_legacy_rendering(self):
        with patch.dict(os.environ, {"AGENT_CONTEXT_COMPACTION": "0"}):
            rows = self.render_rows()
        self.assertEqual(rows, [{"role": m["role"], "content": m["content"]} for m in self.messages])

    @unittest.skipUnless(os.environ.get("MOLT_TEST_CHAT_TEMPLATE"), "Set path to deployed Qwen template")
    def test_actual_qwen_template_renders_three_calls_without_duplicate_endings(self):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def fail(message):
            raise ValueError(message)

        env = ImmutableSandboxedEnvironment()
        env.globals["raise_exception"] = fail
        template = env.from_string(Path(os.environ["MOLT_TEST_CHAT_TEMPLATE"]).read_text())
        prompt = template.render(messages=self.render_rows(), add_generation_prompt=True)
        self.assertEqual(prompt.count("<tool_call>"), 3)
        self.assertEqual(prompt.count("<tool_response>"), 3)
        self.assertEqual(prompt.count("<|im_end|>"), len(self.messages))
        self.assertEqual(prompt.count("<think>"), 4)  # three retained turns + next generation
        self.assertIn("<parameter=code>\nprint(2)", prompt)
        self.assertNotIn("<|im_end|><|im_end|>", prompt)

        # Before compaction, the server appends only this suffix to the original token trace.
        suffixes = []
        for enabled in ("0", "1"):
            with patch.dict(os.environ, {"AGENT_CONTEXT_COMPACTION": enabled}):
                rows = self.render_rows()
            full = template.render(messages=rows, add_generation_prompt=True)
            prefix = template.render(messages=rows[:-1], add_generation_prompt=False)
            self.assertTrue(full.startswith(prefix))
            suffixes.append(full[len(prefix):])
        self.assertEqual(suffixes[0], suffixes[1])


class TextOnlyImageTokenTests(unittest.TestCase):
    def run_turn(self, images):
        nodes = ast.parse(SOURCE.read_text()).body
        selected = [n for n in nodes if getattr(n, "name", "") in {"_run_turn", "_build_sampling_params"}]
        prompts, calls = [], []
        sampling = SimpleNamespace(max_tokens=100, logprobs=1, min_tokens=0, bad_words=["existing"])
        processor = SimpleNamespace(image_token="<|image_pad|>", apply_chat_template=lambda *a, **kw: "quote <|image_pad|>")

        async def generate(ids, sp, **kwargs):
            calls.append(sp)
            output = SimpleNamespace(token_ids=[7], text="ok", finish_reason="stop", logprobs=[])
            return SimpleNamespace(outputs=[output]), 0

        def tokenize(proc, text, imgs):
            prompts.append(text)
            return [1, 2], {"pixel_values": "pixels"} if imgs else None, imgs

        async def inline_executor(executor, function, *args):
            return function(*args)

        session = SimpleNamespace(last_messages=None, last_response=None, trajectories=[], sampling_params=sampling,
                                  prompt="Q", label="A", images=images, session_id="session")
        state = SimpleNamespace(processor=processor, default_sampling=sampling, max_length=1000,
                                transport=SimpleNamespace(generate=generate))
        ns = dict(asyncio=SimpleNamespace(get_running_loop=lambda: SimpleNamespace(run_in_executor=inline_executor)),
                  deepcopy=copy.deepcopy, _TOKENIZE_EXECUTOR=None,
                  _messages_to_chat=lambda *a: ([], images), _tokenize_observation=tokenize,
                  estimate_vllm_input_expansion_delta=lambda *a: 0,
                  Trajectory=lambda **kwargs: SimpleNamespace(**kwargs, truncated=False, append_action=Mock(), absorb_routing=Mock()),
                  _extract_generation_logprobs=lambda *a: [])
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
        asyncio.run(ns["_run_turn"](state, session, {"messages": [{"role": "user", "content": "Q"}]}))
        self.assertEqual(sampling.bad_words, ["existing"])
        return prompts, calls

    def test_text_only_prompt_quotes_marker_and_blocks_generated_image_token(self):
        prompts, calls = self.run_turn([])
        self.assertEqual(prompts, ["quote < |image_pad|>"])
        self.assertEqual(calls[0].bad_words, ["existing", "<|image_pad|>"])
        router = SOURCE.parents[1] / "trainer/rollout/router.py"
        function = next(n for n in ast.parse(router.read_text()).body if getattr(n, "name", "") == "_inference_sampling_params")
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(router), "exec"), ns)
        calls[0].temperature = 0.6
        calls[0].top_p = 1.0
        self.assertEqual(ns["_inference_sampling_params"](calls[0])["bad_words"], calls[0].bad_words)

    def test_real_image_prompt_and_generation_settings_remain_unchanged(self):
        prompts, calls = self.run_turn(["image"])
        self.assertEqual(prompts, ["quote <|image_pad|>"])
        self.assertEqual(calls[0].bad_words, ["existing"])


if __name__ == "__main__":
    unittest.main()
