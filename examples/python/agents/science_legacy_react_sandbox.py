"""Isolated Sandbox evaluation adapter with a Sandbox-routed API judge.

The legacy agent owns the complete loop (model -> tools -> tool messages ->
model), while Molt owns the policy endpoint and trajectory capture.  This
module deliberately keeps the science-agent implementation out of Molt's
trainer: it only bridges ChatContext to the existing ``_run`` contract.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import sys
from pathlib import Path

import torch

from molt.agents import ChatAgent, ChatAgentRunner, ChatContext, Result
from molt.utils.logging_utils import init_logger

logger = init_logger(__name__)

SCIENCE_AGENT_ROOT = Path(
    os.environ.get(
        "SCIENCE_AGENT_ROOT",
        "/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/harness/science-agent",
    )
).resolve()
INFERENCE_ROOT = SCIENCE_AGENT_ROOT / "inference"
for path in (SCIENCE_AGENT_ROOT, INFERENCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from science_agent.eval.answer_extraction import find_boxed_spans, select_hle_candidate

from react_agent_openai_api_tools import MultiTurnReactAgent  # noqa: E402
from sandbox_gateway import AgentSandboxRun, SandboxGatewayError, SandboxStateError, sandbox_run_scope, shared_sandbox_run  # noqa: E402


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)

def _extract_hle_candidate(text: str) -> str:
    # Legacy ReAct has already extracted the final <answer> block. HLE's
    # boxed-answer selector expects raw model output; applying it to a plain
    # answer such as "5003" would replace a valid prediction with "".
    text = text.strip()
    spans = find_boxed_spans(text)
    if not spans:
        return text
    legacy = spans[-1].canonical if spans else ""
    return select_hle_candidate(text, legacy).value


def _token_f1(prediction: str, reference: str) -> float:
    """Temporary local reward used when the remote science judge is disabled."""
    pred = set(re.findall(r"\w+", _text(prediction).lower()))
    ref = set(re.findall(r"\w+", _text(reference).lower()))
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class ScienceLegacyReactAgent(ChatAgent):
    @sandbox_run_scope
    async def run(self, ctx: ChatContext) -> Result:
        # The legacy harness reads these values when constructing its OpenAI
        # client. Bind per instance so concurrent rollouts do not race through
        # process-global OPENAI_API_BASE/API_KEY.
        sample = ctx.label if isinstance(ctx.label, dict) else {}
        reference = _text(sample.get("answer", "") if sample else ctx.label)
        model_path = os.environ.get("SCIENCE_AGENT_MODEL_PATH", ctx.model_name)
        llm_cfg = {
            "model": model_path,
            "served_model_name": ctx.model_name,
            "model_type": "qwen_dashscope",
            "generate_cfg": {
                "max_input_tokens": int(os.environ.get("SCIENCE_MAX_INPUT_TOKENS", "320000")),
                "max_retries": int(os.environ.get("SCIENCE_MAX_RETRIES", "3")),
                "temperature": float(getattr(ctx.sampling_params, "temperature", 1.0)),
                "top_p": float(getattr(ctx.sampling_params, "top_p", 0.95)),
                "presence_penalty": 0.0,
            },
        }
        requested_tools = [
            item.strip()
            for item in os.environ.get("SCIENCE_AGENT_TOOLS", "search,visit,python,google_scholar").split(",")
            if item.strip()
        ]
        agent = MultiTurnReactAgent(llm=llm_cfg, function_list=requested_tools)
        agent._get_api_base = lambda: ctx.base_url
        agent._get_api_key = lambda: ctx.api_key
        task = {
            "item": {
                "id": os.environ.get("SCIENCE_ITEM_ID", "molt-science"),
                "question": ctx.prompt,
                "answer": reference,
                "messages": list(ctx.messages),
                "dataset_base_dir": os.environ.get("SCIENCE_DATASET_BASE_DIR", str(SCIENCE_AGENT_ROOT)),
                "source": sample.get("source", "Nemotron-RL-Science-v1"),
            },
            "planning_port": None,
            "tool_artifact_dir": os.environ.get("SCIENCE_TOOL_ARTIFACT_DIR", ""),
        }
        if sample.get("id") or sample.get("question_id"):
            task["item"]["id"] = str(sample.get("id") or sample["question_id"])
        task["item"].update({k: v for k, v in sample.items() if k not in task["item"]})
        agent_task = asyncio.create_task(asyncio.to_thread(agent._run, task, ctx.model_name))
        try:
            result = await asyncio.shield(agent_task)
        except asyncio.CancelledError:
            # The tool thread must finish before the outer scope deletes its Sandbox.
            try:
                await agent_task
            finally:
                raise asyncio.CancelledError
        raw_prediction = _text(result.get("prediction") if isinstance(result, dict) else result)
        amo_protocol = os.environ.get("SCIENCE_AGENT_EVAL_BENCHMARK", "") == "amobench"
        next_protocol = os.environ.get("SCIENCE_AGENT_NEXT_HLE_EVAL_PROTOCOL", "0").lower() in {"1", "true", "yes"}
        prediction = _extract_hle_candidate(raw_prediction) if next_protocol else raw_prediction
        if amo_protocol:
            prediction = raw_prediction.rsplit("</think>", 1)[-1].strip()
            for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
                prediction = prediction.replace(token, "")
            spans = find_boxed_spans(prediction)
            if spans:
                prediction = "\n".join(span.canonical for span in spans)
            next_protocol = True
        if prediction.strip().casefold() in {"your answer", "your final answer"}:
            prediction = ""
        prediction_missing = isinstance(result, dict) and "prediction" not in result
        prediction_empty = not prediction.strip()
        truncated = bool(result.get("truncated")) if isinstance(result, dict) else False
        generation_truncated = bool(result.get("generation_truncated", truncated)) if isinstance(result, dict) else False
        budget_exhausted = bool(result.get("trajectory_budget_exhausted")) if isinstance(result, dict) else False
        truncated = truncated or generation_truncated or budget_exhausted
        termination = result.get("termination", "") if isinstance(result, dict) else ""
        if budget_exhausted and prediction_empty:
            termination = "trajectory budget exhausted without valid final answer"
        truncation_info = {
            "science_generation_truncated": torch.tensor(float(generation_truncated), dtype=torch.float32),
            "science_trajectory_budget_exhausted": torch.tensor(float(budget_exhausted), dtype=torch.float32),
            "science_final_answer_valid": torch.tensor(float(not prediction_empty), dtype=torch.float32),
        }
        if prediction_empty:
            # An empty candidate is unanswerable. Sending the reference to the
            # judge with an empty candidate can produce false positives.
            return Result(
                truncated=truncated,
                reward=torch.tensor(0.0, dtype=torch.float32),
                score=torch.tensor(0.0, dtype=torch.float32),
                info={
                    **truncation_info,
                    "science_judge_score": torch.tensor(0.0, dtype=torch.float32),
                    "science_judge_valid": torch.tensor(0.0, dtype=torch.float32),
                    "science_judge_skipped_empty": torch.tensor(1.0, dtype=torch.float32),
                    "science_prediction_chars": torch.tensor(0.0, dtype=torch.float32),
                    "science_prediction_missing": torch.tensor(float(prediction_missing), dtype=torch.float32),
                    "science_prediction_empty": torch.tensor(1.0, dtype=torch.float32),
                    "science_tool_calls": torch.tensor(float(sum(
                        len(message.get("tool_calls") or [])
                        for message in (result.get("messages", []) if isinstance(result, dict) else [])
                        if isinstance(message, dict)
                    )), dtype=torch.float32),
                    "science_termination": termination,
                    "science_prediction": "",
                    "science_messages": json.dumps(result.get("messages", []) if isinstance(result, dict) else [], ensure_ascii=False, default=str),
                },
            )
        reward_mode = os.environ.get("SCIENCE_AGENT_REWARD_MODE", "judge").strip().lower()
        if reward_mode == "f1":
            f1 = _token_f1(prediction, reference)
            logger.warning(
                "Science reward mode=f1; remote judge bypassed prediction_chars=%d reference_chars=%d f1=%.6f",
                len(prediction), len(reference), f1,
            )
            return Result(
                truncated=truncated,
                reward=torch.tensor(f1, dtype=torch.float32),
                score=torch.tensor(f1, dtype=torch.float32),
                info={
                    **truncation_info,
                    "science_reward_mode": "f1",
                    "science_f1": torch.tensor(f1, dtype=torch.float32),
                    "science_judge_score": torch.tensor(f1, dtype=torch.float32),
                    "science_judge_valid": torch.tensor(0.0, dtype=torch.float32),
                    "science_judge_model": "disabled:f1",
                    "science_judge_error": "",
                    "science_judge_response": "",
                    "science_judge_finish_reason": "",
                    "science_judge_content_chars": torch.tensor(0.0, dtype=torch.float32),
                    "science_judge_reason": "local token-set F1",
                    "science_prediction_chars": torch.tensor(float(len(prediction)), dtype=torch.float32),
                    "science_prediction_missing": torch.tensor(float(prediction_missing), dtype=torch.float32),
                    "science_prediction_empty": torch.tensor(float(prediction_empty), dtype=torch.float32),
                    "science_tool_calls": torch.tensor(float(sum(
                        len(message.get("tool_calls") or [])
                        for message in (result.get("messages", []) if isinstance(result, dict) else [])
                        if isinstance(message, dict)
                    )), dtype=torch.float32),
                    "science_termination": termination,
                    "science_prediction": prediction,
                    "science_messages": json.dumps(result.get("messages", []) if isinstance(result, dict) else [], ensure_ascii=False, default=str),
                },
            )
        judge_url = os.environ["SCIENCE_AGENT_JUDGE_BASE_URL"].rstrip("/") + "/chat/completions"
        judge_model = os.environ["SCIENCE_AGENT_JUDGE_MODEL"]
        judge_result = None
        judge_error = ""
        judge_response_summary = ""
        judge_finish_reason = ""
        judge_content_chars = 0
        if amo_protocol:
            judge_system_prompt = "You are a strict grader for mathematical competition answers."
            judge_user_prompt = (
                "You are grading an attempted answer to a mathematical competition problem.\n"
                "The reference answer may be a list of acceptable answers. Mark correct only when the attempted "
                "final answer is mathematically equivalent to a reference answer. Ignore harmless formatting "
                "such as \\boxed{...}, surrounding prose, or equivalent LaTeX formatting. "
                "Leading zeros are harmless but a different value is incorrect.\n"
                f"Problem:\n{ctx.prompt}\n\nReference answer:\n{reference}\n\n"
                f"Attempted extracted answer:\n{prediction}\n\n"
                'Return only valid JSON: {"correct": true/false, "reason": "brief explanation"}.'
            )
        elif next_protocol:
            judge_system_prompt = ("You are a strict grader for problems across all academic disciplines. Given a problem and a candidate solution, decide whether the candidate answer is correct. Be strict: the final answer must match the ground truth (allow equivalent forms).")
            judge_user_prompt = ("You will output ONLY valid JSON with this schema:\n{\"correct\": true/false, \"reason\": \"one-sentence reason\"}\n\n" f"Problem:\n{ctx.prompt}\n\nCandidate answer (model output):\n{prediction}\n\nGround truth answer:\n{reference}\n")
        else:
            judge_system_prompt = (
                "You are a grader for problems across all academic disciplines. "
                "Judge whether the candidate answer is substantially correct and answers the question, "
                "using the reference answer as the grading reference. "
                "Accept equivalent wording, reasoning, and mathematical expressions. "
                "Make your grading decision without unnecessary deliberation, "
                "and return only the requested JSON with a brief reason."
            )
            judge_user_prompt = (
                'You will output ONLY valid JSON with this schema:\n'
                '{"score": 1 or 0, "correct": true or false, "reason": "one-sentence reason"}\n\n'
                "Use score 1 and correct true only when the answer is substantially correct.\n"
                "Otherwise, use score 0 and correct false.\n\n"
                f"Problem:\n{ctx.prompt}\n\n"
                f"Candidate answer (model output):\n{prediction}\n\n"
                f"Ground truth answer:\n{reference}\n"
            )
        # First try the task-owned Sandbox; replacements only judge the saved answer.
        judge_run = shared_sandbox_run()
        retry_delays = (1, 5, 10, 30, 60, 180, 180, 180)
        judge_attempts = 0
        judge_thinking_fallbacks = 0
        for sandbox_attempt in range(3):
            if sandbox_attempt:
                logger.warning("Retrying saved answer in replacement judge Sandbox %d/2", sandbox_attempt)
                judge_run = AgentSandboxRun()
            judge_connection_failures = 0
            try:
                for attempt in range(len(retry_delays) + 1):
                    judge_input = {
                        "model": judge_model,
                        "messages": [{"role": "system", "content": judge_system_prompt}, {"role": "user", "content": judge_user_prompt}],
                        "temperature": 0.0,
                        "max_tokens": int(os.environ.get("SCIENCE_AGENT_JUDGE_MAX_TOKENS", "6000" if next_protocol else "1800")),
                        "response_format": {"type": "json_object"},
                    }
                    judge_thinking = os.environ.get("SCIENCE_AGENT_JUDGE_THINKING", "").strip().lower()
                    if judge_thinking not in {"enabled", "disabled"}:
                        judge_thinking = ""
                    try:
                        thinking_modes = ("enabled", "disabled") if judge_thinking == "enabled" else (judge_thinking,)
                        for thinking_mode in thinking_modes:
                            if thinking_mode:
                                judge_input["thinking"] = {"type": thinking_mode}
                            judge_attempts += 1
                            request_task = asyncio.create_task(asyncio.to_thread(
                                judge_run.run_http,
                                {
                                    "method": "POST", "url": judge_url, "json": judge_input,
                                    "headers": {
                                        "Content-Type": "application/json",
                                        "Authorization": "Bearer " + (os.environ.get("SCIENCE_AGENT_JUDGE_API_KEY") or os.environ["API_KEY"]),
                                    },
                                },
                                timeout=float(os.environ.get("SCIENCE_AGENT_JUDGE_TIMEOUT", "180")),
                            ))
                            try:
                                response = await asyncio.shield(request_task)
                            except asyncio.CancelledError:
                                # to_thread keeps running after cancellation. Wait before
                                # deletion so it cannot create a Sandbox after cleanup.
                                try:
                                    await request_task
                                finally:
                                    raise asyncio.CancelledError
                            judge_connection_failures = 0
                            raw_response = response["text"]
                            judge_response_summary = raw_response[:500]
                            if response["status_code"] >= 400:
                                raise RuntimeError(f"Sandbox judge HTTP {response['status_code']}: {judge_response_summary}")
                            try:
                                judge_finish_reason = ""
                                judge_content_chars = 0
                                body = json.loads(raw_response)
                                choice = body["choices"][0]
                                judge_finish_reason = _text(choice.get("finish_reason", ""))
                                content = choice["message"]["content"]
                                if isinstance(content, list):
                                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                                content = _text(content).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                                judge_content_chars = len(content)
                                parsed = json.loads(content)
                                if not isinstance(parsed, dict) or not isinstance(parsed.get("correct"), bool):
                                    raise ValueError(f"invalid judge result: {parsed!r}")
                                if next_protocol:
                                    correct = parsed["correct"]
                                    judge_result = {"correct": correct, "score": 1.0 if correct else 0.0, "reason": str(parsed.get("reason", ""))}
                                else:
                                    score = float(parsed["score"])
                                    if score not in (0.0, 1.0) or bool(parsed.get("correct")) != bool(score):
                                        raise ValueError(f"invalid judge result: {parsed!r}")
                                    judge_result = parsed
                            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                                if thinking_mode != "enabled":
                                    raise
                                judge_thinking_fallbacks += 1
                                logger.warning(
                                    "Science judge returned no valid grade with thinking enabled; "
                                    "retrying with thinking disabled error_type=%s finish_reason=%s content_chars=%d",
                                    type(exc).__name__, judge_finish_reason, judge_content_chars,
                                )
                                continue
                            break
                        break
                    except Exception as exc:  # noqa: BLE001
                        judge_error = f"{type(exc).__name__}: {exc}"
                        connection_failed = (
                            isinstance(exc, SandboxGatewayError)
                            and "proxy HTTP 502" in str(exc)
                            and "Sandbox upstream connection failed" in str(exc)
                        )
                        judge_connection_failures = judge_connection_failures + 1 if connection_failed else 0
                        # Grading saved text needs no recovered Python state or model acknowledgement.
                        replace_sandbox = isinstance(exc, SandboxStateError) or judge_connection_failures >= 2
                        logger.warning(
                            "Science judge failure sandbox=%d/3 attempt=%d/%d url=%s model=%s prediction_chars=%d "
                            "missing_prediction=%s empty_prediction=%s error=%s response=%s next_retry_s=%s",
                            sandbox_attempt + 1,
                            attempt + 1,
                            len(retry_delays) + 1,
                            judge_url,
                            judge_model,
                            len(prediction),
                            prediction_missing,
                            prediction_empty,
                            judge_error,
                            judge_response_summary[:500],
                            retry_delays[attempt] if not replace_sandbox and attempt < len(retry_delays) else None,
                        )
                        if replace_sandbox:
                            logger.warning("Judge Sandbox unusable; ending retries in this instance error_type=%s", type(exc).__name__)
                            break
                        if attempt < len(retry_delays):
                            await asyncio.sleep(retry_delays[attempt])
            finally:
                if sandbox_attempt:
                    try:
                        await asyncio.to_thread(judge_run.close)
                    except Exception:
                        logger.exception("Replacement judge Sandbox cleanup failed")
            if judge_result is not None:
                break
        # -1 means ungraded: training rejects it; evaluation counts it as zero.
        reward = float(judge_result["score"]) if judge_result is not None else -1.0
        tool_calls = sum(
            len(message.get("tool_calls") or [])
            for message in (result.get("messages", []) if isinstance(result, dict) else [])
            if isinstance(message, dict)
        )
        return Result(
            truncated=truncated,
            reward=torch.tensor(reward, dtype=torch.float32),
            score=torch.tensor(reward, dtype=torch.float32),
            info={
                **truncation_info,
                "science_judge_score": torch.tensor(reward, dtype=torch.float32),
                "science_judge_valid": torch.tensor(float(judge_result is not None), dtype=torch.float32),
                "science_judge_model": judge_model,
                "science_judge_backend": "gateway_sandbox",
                "science_judge_attempts": torch.tensor(float(judge_attempts), dtype=torch.float32),
                "science_judge_thinking_fallbacks": torch.tensor(float(judge_thinking_fallbacks), dtype=torch.float32),
                "science_judge_sandbox_restarts": torch.tensor(float(sandbox_attempt), dtype=torch.float32),
                "science_judge_error": judge_error,
                "science_judge_response": judge_response_summary,
                "science_judge_finish_reason": judge_finish_reason,
                "science_judge_content_chars": torch.tensor(float(judge_content_chars), dtype=torch.float32),
                "science_judge_reason": str((judge_result or {}).get("reason", "")),
                "science_prediction_chars": torch.tensor(float(len(prediction)), dtype=torch.float32),
                "science_prediction_missing": torch.tensor(float(prediction_missing), dtype=torch.float32),
                "science_prediction_empty": torch.tensor(float(prediction_empty), dtype=torch.float32),
                "science_tool_calls": torch.tensor(float(tool_calls), dtype=torch.float32),
                "science_termination": termination,
                "science_prediction": prediction,
                "science_messages": json.dumps(result.get("messages", []) if isinstance(result, dict) else [], ensure_ascii=False, default=str),
            },
        )


class AgentRunner(ChatAgentRunner):
    def __init__(self):
        super().__init__(ScienceLegacyReactAgent)
