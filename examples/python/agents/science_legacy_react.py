"""Molt ChatAgent adapter for the science-agent Legacy ReAct harness.

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
import random
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

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
                "source": "Nemotron-RL-Science-v1",
            },
            "planning_port": None,
            "tool_artifact_dir": os.environ.get("SCIENCE_TOOL_ARTIFACT_DIR", ""),
        }
        if sample.get("id") or sample.get("question_id"):
            task["item"]["id"] = str(sample.get("id") or sample["question_id"])
        task["item"].update({k: v for k, v in sample.items() if k not in task["item"]})
        result = await asyncio.to_thread(agent._run, task, ctx.model_name)
        raw_prediction = _text(result.get("prediction") if isinstance(result, dict) else result)
        next_protocol = os.environ.get("SCIENCE_AGENT_NEXT_HLE_EVAL_PROTOCOL", "0").lower() in {"1", "true", "yes"}
        prediction = _extract_hle_candidate(raw_prediction) if next_protocol else raw_prediction
        prediction_missing = isinstance(result, dict) and "prediction" not in result
        prediction_empty = not prediction.strip()
        truncated = bool(result.get("truncated")) if isinstance(result, dict) else False
        if next_protocol and prediction_empty and not truncated:
            # An empty HLE candidate is unanswerable. Sending the reference to
            # the judge with an empty candidate can produce false positives.
            return Result(
                reward=torch.tensor(0.0, dtype=torch.float32),
                score=torch.tensor(0.0, dtype=torch.float32),
                info={
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
                    "science_termination": result.get("termination", "") if isinstance(result, dict) else "",
                    "science_prediction": "",
                    "science_messages": json.dumps(result.get("messages", []) if isinstance(result, dict) else [], ensure_ascii=False, default=str),
                },
            )
        if truncated:
            tool_calls = sum(
                len(message.get("tool_calls") or [])
                for message in (result.get("messages", []) if isinstance(result, dict) else [])
                if isinstance(message, dict)
            )
            # Debug-only reward probe: truncated trajectories used to receive
            # an unconditional zero, making an entire batch have zero
            # advantage. Give them a random score in [0.3, 0.7] so we can
            # distinguish the reward-collapse path from the distributed
            # training/collective path. Keep reward and score identical.
            truncated_reward = random.uniform(0.3, 0.7)
            logger.warning(
                "Science truncated reward debug: reward=%.6f prediction_chars=%d tool_calls=%d",
                truncated_reward,
                len(prediction),
                tool_calls,
            )
            return Result(
                reward=torch.tensor(truncated_reward, dtype=torch.float32),
                score=torch.tensor(truncated_reward, dtype=torch.float32),
                truncated=True,
                info={
                    "science_judge_score": torch.tensor(truncated_reward, dtype=torch.float32),
                    "science_judge_valid": torch.tensor(0.0, dtype=torch.float32),
                    "science_judge_skipped_truncated": torch.tensor(1.0, dtype=torch.float32),
                    "science_prediction_chars": torch.tensor(float(len(prediction)), dtype=torch.float32),
                    "science_prediction_missing": torch.tensor(float(prediction_missing), dtype=torch.float32),
                    "science_prediction_empty": torch.tensor(float(prediction_empty), dtype=torch.float32),
                    "science_tool_calls": torch.tensor(float(tool_calls), dtype=torch.float32),
                    "science_truncated": torch.tensor(1.0, dtype=torch.float32),
                    "science_termination": result.get("termination", "truncated response"),
                    "science_prediction": prediction,
                    "science_messages": json.dumps(result.get("messages", []), ensure_ascii=False, default=str),
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
                reward=torch.tensor(f1, dtype=torch.float32),
                score=torch.tensor(f1, dtype=torch.float32),
                info={
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
                    "science_termination": result.get("termination", "") if isinstance(result, dict) else "",
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
        if next_protocol:
            judge_system_prompt = ("You are a strict grader for physics problems. Given a problem and a candidate solution, decide whether the candidate answer is correct. Be strict: the final answer must match the ground truth (allow equivalent forms).")
            judge_user_prompt = ("You will output ONLY valid JSON with this schema:\n{\"correct\": true/false, \"reason\": \"one-sentence reason\"}\n\n" f"Problem:\n{ctx.prompt}\n\nCandidate answer (model output):\n{prediction}\n\nGround truth answer:\n{reference}\n")
        else:
            judge_system_prompt = ("Judge whether the candidate answer is scientifically correct and answers the question. Accept equivalent wording and reasoning. Return only JSON: {\"score\": 1 or 0, \"correct\": true or false, \"reason\": \"short explanation\"}. Use score 1 only when the answer is substantially correct.")
            judge_user_prompt = json.dumps({"question": ctx.prompt, "reference_answer": reference, "candidate_answer": prediction}, ensure_ascii=False)
        for attempt in range(3):
            judge_input = {
                "model": judge_model,
                "messages": [{"role": "system", "content": judge_system_prompt}, {"role": "user", "content": judge_user_prompt}],
                "temperature": 0.0,
                "max_tokens": int(os.environ.get("SCIENCE_AGENT_JUDGE_MAX_TOKENS", "6000" if next_protocol else "1800")),
                "response_format": {"type": "json_object"},
            }
            request = Request(
                judge_url,
                data=json.dumps(judge_input, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                opener = build_opener(ProxyHandler({}))
                response = await asyncio.to_thread(
                    lambda: opener.open(request, timeout=float(os.environ.get("SCIENCE_AGENT_JUDGE_TIMEOUT", "180")))
                )
                raw_response = response.read()
                judge_response_summary = raw_response[:500].decode("utf-8", errors="replace")
                body = json.loads(raw_response)
                choice = body["choices"][0]
                judge_finish_reason = _text(choice.get("finish_reason", ""))
                content = choice["message"]["content"]
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                content = _text(content).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                judge_content_chars = len(content)
                parsed = json.loads(content)
                if next_protocol:
                    correct = parsed["correct"]
                    if not isinstance(correct, bool):
                        raise ValueError(f"invalid judge result: {parsed!r}")
                    judge_result = {"correct": correct, "score": 1.0 if correct else 0.0, "reason": str(parsed.get("reason", ""))}
                else:
                    score = float(parsed["score"])
                    if score not in (0.0, 1.0) or bool(parsed.get("correct")) != bool(score):
                        raise ValueError(f"invalid judge result: {parsed!r}")
                    judge_result = parsed
                break
            except URLError as exc:
                judge_error = f"URLError: {exc}"
                logger.warning(
                    "Science judge network failure attempt=%d/3 url=%s model=%s prediction_chars=%d "
                    "missing_prediction=%s empty_prediction=%s error=%s",
                    attempt + 1, judge_url, judge_model, len(prediction), prediction_missing,
                    prediction_empty, judge_error,
                )
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
            except Exception as exc:  # noqa: BLE001
                judge_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Science judge failure attempt=%d/3 url=%s model=%s prediction_chars=%d "
                    "missing_prediction=%s empty_prediction=%s error=%s response=%s",
                    attempt + 1,
                    judge_url,
                    judge_model,
                    len(prediction),
                    prediction_missing,
                    prediction_empty,
                    judge_error,
                    judge_response_summary[:500],
                )
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        reward = float(judge_result["score"]) if judge_result is not None else -10.0
        filter_score = reward if judge_result is not None else -1e9
        tool_calls = sum(
            len(message.get("tool_calls") or [])
            for message in (result.get("messages", []) if isinstance(result, dict) else [])
            if isinstance(message, dict)
        )
        return Result(
            reward=torch.tensor(reward, dtype=torch.float32),
            score=torch.tensor(filter_score, dtype=torch.float32),
            info={
                "science_judge_score": torch.tensor(reward, dtype=torch.float32),
                "science_judge_valid": torch.tensor(float(judge_result is not None), dtype=torch.float32),
                "science_judge_model": judge_model,
                "science_judge_error": judge_error,
                "science_judge_response": judge_response_summary,
                "science_judge_finish_reason": judge_finish_reason,
                "science_judge_content_chars": torch.tensor(float(judge_content_chars), dtype=torch.float32),
                "science_judge_reason": str((judge_result or {}).get("reason", "")),
                "science_prediction_chars": torch.tensor(float(len(prediction)), dtype=torch.float32),
                "science_prediction_missing": torch.tensor(float(prediction_missing), dtype=torch.float32),
                "science_prediction_empty": torch.tensor(float(prediction_empty), dtype=torch.float32),
                "science_tool_calls": torch.tensor(float(tool_calls), dtype=torch.float32),
                "science_termination": result.get("termination", "") if isinstance(result, dict) else "",
                "science_prediction": prediction,
                "science_messages": json.dumps(result.get("messages", []) if isinstance(result, dict) else [], ensure_ascii=False, default=str),
            },
        )


class AgentRunner(ChatAgentRunner):
    def __init__(self):
        super().__init__(ScienceLegacyReactAgent)
