#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""GPU integration check: vLLM abort outputs retain captured MoE routing.

Run inside the molt-cu13-runtime image with MODEL_PATH set to a local MoE
checkpoint. The request is aborted only after vLLM has generated tokens and
captured routing for them.
"""

import asyncio
import json
import os
import sys
import time
from uuid import uuid4



async def _wait_for_captured_routing(engine, request_id, min_tokens, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        internal_ids = engine.output_processor.external_req_ids.get(request_id, ())
        states = [engine.output_processor.request_states[i] for i in internal_ids
                  if i in engine.output_processor.request_states]
        for state in states:
            token_count = len(state.detokenizer.output_token_ids)
            if token_count >= min_tokens and state.routed_experts_chunks:
                return token_count, sum(len(chunk) for chunk in state.routed_experts_chunks)
        await asyncio.sleep(0.05)
    raise TimeoutError(
        f"No captured routing after {timeout_s}s; verify MODEL_PATH is an MoE model "
        "and enable_return_routed_experts is supported by this vLLM build."
    )


async def _run():
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    model_path = os.environ.get(
        "MODEL_PATH",
        "/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/"
        "science_model/tool_models/hf_models/"
        "p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941",
    )

    args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=int(os.environ.get("TP", "1")),
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "2048")),
        max_num_seqs=1,
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.8")),
        enable_prefix_caching=False,
        enable_return_routed_experts=True,
        enforce_eager=True,
    )
    engine = AsyncLLM.from_engine_args(args)
    request_id = f"abort-routing-{uuid4().hex}"
    prompt = os.environ.get(
        "PROMPT",
        "Explain why the sky appears blue. Give a detailed answer in numbered steps.",
    )
    min_tokens = int(os.environ.get("MIN_GENERATED_TOKENS", "8"))
    timeout_s = float(os.environ.get("CAPTURE_TIMEOUT_S", "180"))

    try:
        generation = asyncio.create_task(
            _collect_final_output(
                engine,
                prompt,
                SamplingParams(max_tokens=512, temperature=0.8, logprobs=1, ignore_eos=True),
                request_id,
            )
        )
        observed_tokens, observed_route_rows = await _wait_for_captured_routing(
            engine, request_id, min_tokens, timeout_s
        )
        await engine.abort(request_id)
        final = await asyncio.wait_for(generation, timeout=timeout_s)

        if not final.outputs:
            raise AssertionError("Abort returned no completion output.")
        completion = final.outputs[0]
        token_ids = list(completion.token_ids)
        logprobs = completion.logprobs
        routed = completion.routed_experts
        prompt_ids = list(final.prompt_token_ids or [])
        if str(completion.finish_reason).lower() not in {"abort", "finishreason.abort"}:
            raise AssertionError(f"Expected abort finish reason, got {completion.finish_reason!r}.")
        if len(token_ids) < min_tokens:
            raise AssertionError(f"Abort output retained only {len(token_ids)} tokens.")
        if logprobs is None or len(logprobs) != len(token_ids):
            raise AssertionError(
                f"Logprob/token misalignment: {None if logprobs is None else len(logprobs)} "
                f"logprob rows for {len(token_ids)} tokens."
            )
        if routed is None:
            raise AssertionError("Abort output dropped all captured routed_experts.")

        expected_route_rows = len(prompt_ids) + len(token_ids) - 1
        if routed.shape[0] != expected_route_rows:
            raise AssertionError(
                f"Routing/token misalignment: got {routed.shape[0]} routing rows; expected "
                f"prompt({len(prompt_ids)}) + generated({len(token_ids)}) - final_unforwarded(1) "
                f"= {expected_route_rows}. Before abort the request had observed "
                f"{observed_route_rows} routing rows."
            )

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "vllm_version": vllm.__version__,
                    "request_id": request_id,
                    "finish_reason": str(completion.finish_reason),
                    "prompt_tokens": len(prompt_ids),
                    "generated_tokens_before_abort": observed_tokens,
                    "generated_tokens_returned": len(token_ids),
                    "logprob_rows": len(logprobs),
                    "routing_shape": list(routed.shape),
                    "routing_rows_observed_before_abort": observed_route_rows,
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        if "generation" in locals() and not generation.done():
            await engine.abort(request_id)
            await asyncio.gather(generation, return_exceptions=True)
        engine.shutdown()


async def _collect_final_output(engine, prompt, sampling_params, request_id):
    final = None
    async for output in engine.generate(prompt, sampling_params, request_id):
        final = output
    if final is None:
        raise RuntimeError("vLLM generate returned no final RequestOutput.")
    return final


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
