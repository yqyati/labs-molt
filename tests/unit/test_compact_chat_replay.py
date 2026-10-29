"""Check compacted message replay without importing GPU or Ray dependencies."""

import ast
import asyncio
import copy
from dataclasses import dataclass, field
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

    def test_plain_messages_render_identically_without_compaction(self):
        self.messages = self.messages[:2]
        with patch.dict(os.environ, {"AGENT_CONTEXT_COMPACTION": "0"}):
            rows = self.render_rows()
        self.assertEqual(rows, [{"role": m["role"], "content": m["content"]} for m in self.messages])

    def test_finalization_keeps_separate_reasoning_and_calls_without_compaction_or_masking(self):
        self.messages = self.messages[:2] + [
            {"role": "assistant", "content": "", "reasoning_content": "Check the calculation",
             "tool_calls": [{"id": "call0", "type": "function", "function": {
                 "name": "python", "arguments": '{"code":"print(6*7)"}'}}]},
            {"role": "tool", "tool_call_id": "call0", "name": "python", "content": "42"},
            {"role": "user", "content": "Do not call more tools; give the final answer."},
        ]
        original = copy.deepcopy(self.messages)
        with patch.dict(os.environ, {"AGENT_CONTEXT_COMPACTION": "0", "AGENT_OBSERVATION_MASKING": "0"}):
            rows = self.render_rows()
        self.assertEqual(rows[2]["reasoning_content"], "Check the calculation")
        self.assertEqual(rows[2]["tool_calls"][0]["function"]["arguments"], {"code": "print(6*7)"})
        self.assertEqual(rows[2]["tool_calls"][0]["id"], rows[3]["tool_call_id"])
        self.assertEqual(rows[3]["name"], "python")
        self.assertEqual(rows[-1], original[-1])
        self.assertEqual(self.messages, original)

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


class AppendFinalizationTests(unittest.TestCase):
    def setUp(self):
        base_source = SOURCE.with_name("base.py")
        trajectory = next(n for n in ast.parse(base_source.read_text()).body
                          if isinstance(n, ast.ClassDef) and n.name == "Trajectory")
        ns = {"dataclass": dataclass, "field": field}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[trajectory], type_ignores=[])),
                     str(base_source), "exec"), ns)
        self.old_tokens = [11, 12, 13, 14]
        self.traj = ns["Trajectory"]("Q", "A", None, "", self.old_tokens.copy(),
                                      action_ranges=[(2, 4)], rollout_log_probs=[0, 0, -.1, -.1])
        self.calls, self.renders = [], []

        def render(chat, **kwargs):
            self.renders.append(copy.deepcopy(chat))
            # Deliberately rewrite the old assistant if given a full history.
            if any(m["role"] == "assistant" for m in chat):
                return "REWRITTEN"
            return "".join("USER:" + m["content"] for m in chat) + (
                " ASSISTANT:" if kwargs.get("add_generation_prompt") else "")

        async def generate(ids, sp, **kwargs):
            self.calls.append((list(ids), sp.max_tokens))
            return SimpleNamespace(outputs=[SimpleNamespace(token_ids=[91, 92], text="answer",
                       finish_reason="stop", logprobs=None, routed_experts=None)]), 0

        async def inline_executor(executor, function, *args):
            return function(*args)

        def tokenize_feedback(processor, text, images, traj, max_length):
            self.delta = [ord(c) for c in text]
            return self.delta

        self.processor = SimpleNamespace(eos_token="<END>", apply_chat_template=render)
        sampling = SimpleNamespace(max_tokens=100, logprobs=1, min_tokens=0)
        self.state = SimpleNamespace(processor=self.processor, default_sampling=sampling, max_length=1000,
                                     transport=SimpleNamespace(generate=generate))
        old_messages = [{"role": "user", "content": "Q"}]
        self.session = SimpleNamespace(last_messages=old_messages, last_response=("old reasoning<END>", "stop"),
                                       trajectories=[self.traj], sampling_params=sampling, session_id="sid")
        self.body = {"messages": old_messages + [
            {"role": "assistant", "content": "old reasoning"},
            {"role": "user", "content": "FINAL"}], "molt_append_finalization": True}
        selected = [n for n in ast.parse(SOURCE.read_text()).body
                    if getattr(n, "name", "") in {"_run_turn", "_build_sampling_params"}]
        ns = dict(asyncio=SimpleNamespace(get_running_loop=lambda: SimpleNamespace(run_in_executor=inline_executor)),
                  deepcopy=copy.deepcopy, _TOKENIZE_EXECUTOR=None, HTTPException=RuntimeError,
                  _messages_to_chat=lambda state, messages: (messages, []),
                  _content_to_text_and_images=lambda content: (content, []),
                  _tokenize_feedback=tokenize_feedback,
                  _tokenize_observation=Mock(side_effect=AssertionError("History must not be rebuilt")),
                  _extract_generation_logprobs=lambda *args: [-.2, -.3])
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 *selected], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
        self.run_turn = ns["_run_turn"]
        self.forward_globals = ns

    def test_appends_only_final_prompt_and_keeps_action_and_logprob_alignment(self):
        result = asyncio.run(self.run_turn(self.state, self.session, self.body))
        self.assertEqual(result, ("answer", "stop"))
        anchor = {"role": "user", "content": ""}
        self.assertEqual(self.renders, [[anchor], [anchor, self.body["messages"][-1]]])
        self.assertEqual(self.calls[0][0], self.old_tokens + self.delta)
        self.assertEqual(len(self.session.trajectories), 1)
        self.assertEqual(self.traj.observation_tokens, self.old_tokens + self.delta + [91, 92])
        self.assertEqual(self.traj.action_ranges, [(2, 4), (4 + len(self.delta), 6 + len(self.delta))])
        self.assertEqual(self.traj.rollout_log_probs, [0, 0, -.1, -.1] + [0] * len(self.delta) + [-.2, -.3])
        asyncio.run(self.run_turn(self.state, self.session, self.body))
        self.assertEqual(len(self.calls), 1)  # Retry replays the result without appending twice.

    def test_closes_incomplete_assistant_before_appending(self):
        self.session.last_response = ("unfinished reasoning", "length")
        asyncio.run(self.run_turn(self.state, self.session, self.body))
        self.assertEqual("".join(map(chr, self.delta)), "<END>\nUSER:FINAL ASSISTANT:")

    def test_pending_tool_result_is_appended_before_final_instruction(self):
        self.body["messages"].insert(-1, {"role": "tool", "content": "RESULT"})
        asyncio.run(self.run_turn(self.state, self.session, self.body))
        self.assertEqual("".join(map(chr, self.delta)), "\nUSER:RESULTUSER:FINAL ASSISTANT:")
        self.assertEqual(self.calls[0][0], self.old_tokens + self.delta)

    def test_generation_is_clamped_to_remaining_context(self):
        delta_length = len("\nUSER:FINAL ASSISTANT:")
        self.state.max_length = len(self.old_tokens) + delta_length + 2
        asyncio.run(self.run_turn(self.state, self.session, self.body))
        self.assertEqual(self.calls[0][1], 2)

    def test_full_context_returns_length_without_mutating_history(self):
        self.state.max_length = len(self.old_tokens) + len("\nUSER:FINAL ASSISTANT:")
        self.assertEqual(asyncio.run(self.run_turn(self.state, self.session, self.body)), ("", "length"))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.traj.observation_tokens, self.old_tokens)
        self.assertTrue(self.traj.truncated)

    def test_rewritten_history_still_starts_a_new_segment(self):
        self.body["messages"][0] = {"role": "user", "content": "Compacted context"}
        self.session.prompt, self.session.label, self.session.images = "Q", "A", None
        self.forward_globals.update(
            _tokenize_observation=lambda *args: ([21, 22], None, []),
            estimate_vllm_input_expansion_delta=lambda *args: 0,
            Trajectory=type(self.traj),
        )
        asyncio.run(self.run_turn(self.state, self.session, self.body))
        self.assertEqual(len(self.session.trajectories), 2)
        self.assertEqual(self.traj.observation_tokens, self.old_tokens)
        self.assertEqual(self.calls[0][0], [21, 22])

    @unittest.skipUnless(os.environ.get("MOLT_TEST_CHAT_TEMPLATE"), "Set path to deployed Qwen template")
    def test_actual_qwen_template_appends_only_new_user_turn(self):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        template = ImmutableSandboxedEnvironment().from_string(
            Path(os.environ["MOLT_TEST_CHAT_TEMPLATE"]).read_text())
        self.processor.eos_token = "<|im_end|>"
        self.processor.apply_chat_template = lambda chat, **kw: template.render(messages=chat, **kw)
        self.session.last_response = ("<think>Keep this reasoning</think><|im_end|>", "stop")
        asyncio.run(self.run_turn(self.state, self.session, self.body))
        expected_delta = "\n<|im_start|>user\nFINAL<|im_end|>\n<|im_start|>assistant\n<think>\n"
        self.assertEqual(self.calls[0][0], self.old_tokens + list(map(ord, expected_delta)))
        self.assertEqual(len(self.session.trajectories), 1)

        self.setUp()
        self.processor.eos_token = "<|im_end|>"
        self.processor.apply_chat_template = lambda chat, **kw: template.render(messages=chat, **kw)
        self.body["messages"].insert(-1, {"role": "tool", "content": "RESULT"})
        asyncio.run(self.run_turn(self.state, self.session, self.body))
        delta = "".join(map(chr, self.delta))
        self.assertIn("<|im_start|>user\n<tool_response>\nRESULT\n</tool_response><|im_end|>", delta)
        self.assertIn("<|im_start|>user\nFINAL", delta)
        self.assertNotIn("old reasoning", delta)


if __name__ == "__main__":
    unittest.main()
