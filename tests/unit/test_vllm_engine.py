# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import sys
from dataclasses import dataclass
from types import ModuleType


def _install_vllm_test_stub():
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.21.0"

    @dataclass
    class AsyncEngineArgs:
        model: str
        dtype: str = "auto"
        enforce_eager: bool = False

    vllm.AsyncEngineArgs = AsyncEngineArgs
    vllm.AsyncLLMEngine = object

    inputs = ModuleType("vllm.inputs")

    class TokensPrompt(dict):
        pass

    inputs.TokensPrompt = TokensPrompt

    utils = ModuleType("vllm.utils")
    utils.random_uuid = lambda: "test-request-id"

    sys.modules["vllm"] = vllm
    sys.modules["vllm.inputs"] = inputs
    sys.modules["vllm.utils"] = utils


try:
    import vllm.inputs  # noqa: F401
except Exception:
    _install_vllm_test_stub()

import molt.trainer.vllm.vllm_engine as vllm_engine  # noqa: E402


def test_vllm_ray_executor_uses_worker_gpu_even_when_actor_is_cpu_only():
    assert vllm_engine._vllm_worker_num_gpus("ray", 0) == 1
    assert vllm_engine._vllm_worker_num_gpus("mp", 8) == 8
    assert vllm_engine._vllm_worker_num_gpus("uni", 1) == 1


def test_format_ray_gpu_ids_keeps_all_visible_devices():
    assert vllm_engine._format_ray_gpu_ids([0.0, 2.0]) == "0,2"
    assert vllm_engine._format_ray_gpu_ids(["GPU-abc"]) == "GPU-abc"


def test_filter_vllm_engine_kwargs_drops_unsupported_optional_args(monkeypatch):
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        dtype: str = "auto"
        enforce_eager: bool = False

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs(
        {
            "model": "model-path",
            "dtype": "bfloat16",
            "gdn_prefill_backend": "triton",
        }
    )

    assert filtered == {"model": "model-path", "dtype": "bfloat16"}


def test_filter_vllm_engine_kwargs_keeps_speculative_config(monkeypatch):
    # MTP rollout passes speculative_config; it is a real AsyncEngineArgs field,
    # so it must survive the whitelist filter (not be dropped like unknown kwargs).
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        dtype: str = "auto"
        speculative_config: object = None

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs(
        {"model": "m", "speculative_config": {"num_speculative_tokens": 1}}
    )

    assert filtered == {"model": "m", "speculative_config": {"num_speculative_tokens": 1}}


def test_filter_vllm_engine_kwargs_keeps_disable_custom_all_reduce(monkeypatch):
    # --vllm.disable_custom_all_reduce threads through as this kwarg; it is a real
    # AsyncEngineArgs field, so it must survive the whitelist filter (a rename/typo
    # would otherwise let it be silently dropped -> flag becomes a no-op).
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        disable_custom_all_reduce: bool = False

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs({"model": "m", "disable_custom_all_reduce": True})

    assert filtered == {"model": "m", "disable_custom_all_reduce": True}


def test_ray_visible_device_flag_is_cuda_only():
    assert vllm_engine.ray_noset_visible_devices({"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"})
    assert not vllm_engine.ray_noset_visible_devices({"RAY_EXPERIMENTAL_NOSET_OTHER_VISIBLE_DEVICES": "1"})


def test_rollout_ray_actor_init_is_synchronous():
    """Ray actor constructors are synchronous; an async __init__ would not be awaited."""
    actor_cls = getattr(vllm_engine.RolloutRayActor, "__ray_actor_class__", vllm_engine.RolloutRayActor)
    assert not inspect.iscoroutinefunction(actor_cls.__init__)


def test_partial_routing_audit_tracks_real_pause_boundary_and_detects_corruption(capsys):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import numpy as np
    import pytest

    actor_cls = getattr(vllm_engine.RolloutRayActor, "__ray_metadata__", None)
    actor_cls = actor_cls.modified_class if actor_cls else vllm_engine.RolloutRayActor
    routes = np.arange(12).reshape(3, 2, 2)

    async def scenario(corrupt=False, active=True):
        actor = object.__new__(actor_cls)
        actor._partial_audit = True
        actor._partial_snapshots = {}
        actor._partial_boundary = 0
        state = SimpleNamespace(
            detokenizer=SimpleNamespace(output_token_ids=[20, 21]),
            logprobs_processor=SimpleNamespace(logprobs=[{20: -0.1}, {21: -0.2}]),
            routed_experts_chunks=[routes.copy()], request_index=0,
        )
        actor.llm = SimpleNamespace(
            pause_generation=AsyncMock(),
            output_processor=SimpleNamespace(
                external_req_ids={"r": ["internal"]} if active else {},
                request_states={"internal": state},
            ),
        )
        await actor.pause_generation()
        actor.llm.pause_generation.assert_awaited_once_with(mode="keep")
        finished_routes = np.concatenate([routes, np.ones((1, 2, 2), dtype=int)])
        if corrupt:
            finished_routes[0, 0, 0] += 1
        completion = SimpleNamespace(
            index=0, token_ids=[20, 21, 22], logprobs=[{20: -0.1}, {21: -0.2}, {22: -0.3}],
            routed_experts=finished_routes, finish_reason="length",
        )

        async def generate(*args, **kwargs):
            yield SimpleNamespace(finished=True, request_id="r", prompt_token_ids=[1, 2], outputs=[completion])

        actor._original_generate = generate
        outputs = [o async for o in actor._generate_with_partial_audit(None, None, "r")]
        assert len(outputs) == 1 and not actor._partial_snapshots

    asyncio.run(scenario())
    log = capsys.readouterr().out
    assert '"generated_tokens": 2' in log and '"continued": true' in log
    assert '"prefix_preserved": true' in log
    with pytest.raises(AssertionError):
        asyncio.run(scenario(corrupt=True))
    assert "[partial_routing] FAIL" in capsys.readouterr().out
    asyncio.run(scenario(active=False))
    log = capsys.readouterr().out
    assert '"active_requests": 0' in log and '"event": "finished"' not in log
