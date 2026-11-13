#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run directly on GPU: PREFIX_CACHE=0|1 TP=1 python this_file.py.

Exercises keep/reset/resume with unchanged weights, not weight broadcast correctness.
Runs one and three pauses, with chunked prefill and the same prompt for cache reuse.
"""

import asyncio
from copy import deepcopy
import json
import os

import numpy as np

from test_vllm_abort_routing import _collect_final_output, _wait_for_captured_routing


async def main():
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    prefix_cache = bool(int(os.environ.get("PREFIX_CACHE", "0")))
    timeout = float(os.environ.get("CAPTURE_TIMEOUT_S", "180"))
    model = os.environ.get(
        "MODEL_PATH",
        "/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/"
        "science_model/tool_models/hf_models/"
        "p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941",
    )
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=model, tensor_parallel_size=int(os.environ.get("TP", "1")),
        max_model_len=2048, max_num_seqs=1, max_num_batched_tokens=512,
        enable_chunked_prefill=True, enable_prefix_caching=prefix_cache,
        enable_return_routed_experts=True, enforce_eager=True,
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.8")),
    ))
    prompt = "Explain the physics of blue skies in detail. " * 80
    generation = None
    try:
        for pauses in (1, 3):
            request_id = f"keep-routing-{pauses}"
            generation = asyncio.create_task(_collect_final_output(
                engine, prompt,
                SamplingParams(max_tokens=256, temperature=0.8, logprobs=1, ignore_eos=True),
                request_id,
            ))
            snapshots = []
            threshold = 8
            for _ in range(pauses):
                await _wait_for_captured_routing(engine, request_id, threshold, timeout)
                await asyncio.wait_for(engine.pause_generation(mode="keep"), timeout)
                internal_ids = engine.output_processor.external_req_ids.get(request_id, ())
                states = [engine.output_processor.request_states[i] for i in internal_ids
                          if i in engine.output_processor.request_states]
                assert len(states) == 1, "Request finished before pause; test did not exercise keep."
                state = states[0]
                tokens = list(state.detokenizer.output_token_ids)
                routing = np.concatenate(state.routed_experts_chunks).copy()
                logprobs = deepcopy(state.logprobs_processor.logprobs)
                assert len(logprobs) == len(tokens)
                assert len(routing) == len(state.prompt_token_ids) + len(tokens) - 1
                if snapshots:
                    old_tokens, old_logprobs, old_routing = snapshots[-1]
                    assert tokens[:len(old_tokens)] == old_tokens
                    assert logprobs[:len(old_logprobs)] == old_logprobs
                    np.testing.assert_array_equal(routing[:len(old_routing)], old_routing)
                snapshots.append((tokens, logprobs, routing))
                # Match Molt's additional reset after the broadcast when caching is enabled.
                if prefix_cache:
                    reset = await asyncio.wait_for(engine.reset_prefix_cache(
                        reset_running_requests=True, reset_connector=True,
                    ), timeout)
                    assert reset is not False, "Engine refused the cache reset."
                await asyncio.sleep(0.1)
                assert state.detokenizer.output_token_ids == tokens, "Tokens advanced while paused."
                assert state.logprobs_processor.logprobs == logprobs
                np.testing.assert_array_equal(np.concatenate(state.routed_experts_chunks), routing)
                threshold = len(tokens) + 8
                await asyncio.wait_for(engine.resume_generation(), timeout)

            final = await asyncio.wait_for(generation, timeout)
            completion = final.outputs[0]
            assert final.finished and completion.finish_reason == "length"
            assert len(final.prompt_token_ids) > 512, "Prompt did not exercise chunked prefill."
            assert len(completion.token_ids) == len(completion.logprobs) == 256
            routed = completion.routed_experts
            assert routed is not None
            assert len(routed) == len(final.prompt_token_ids) + 256 - 1
            for tokens, logprobs, routing in snapshots:
                assert list(completion.token_ids[:len(tokens)]) == tokens
                assert completion.logprobs[:len(logprobs)] == logprobs
                np.testing.assert_array_equal(routed[:len(routing)], routing)
            print(json.dumps({
                "status": "PASS", "vllm_version": vllm.__version__,
                "prefix_cache": prefix_cache, "pauses": pauses,
                "tokens_at_pauses": [len(s[0]) for s in snapshots],
                "generated_tokens": len(completion.token_ids), "routing_shape": list(routed.shape),
                "cached_tokens": getattr(final, "num_cached_tokens", None),
                "weight_update_tested": False,
            }), flush=True)
    finally:
        if generation is not None and not generation.done():
            generation.cancel()
            await asyncio.gather(generation, return_exceptions=True)
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
