# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest

from molt.cli.train_rl_ray import _ray_runtime_env_vars


def test_ray_runtime_env_forwards_wandb_settings(monkeypatch):
    expected = {
        "WANDB_API_KEY": "test-api-key",
        "WANDB_ENTITY": "test-entity",
        "WANDB_MODE": "offline",
        "MOLT_MOE_EXPERTS": "torch",
        "MOLT_LINEAR_BACKEND": "torch",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    env_vars = _ray_runtime_env_vars()

    assert {name: env_vars[name] for name in expected} == expected


def test_ray_runtime_env_omits_empty_optional_settings(monkeypatch):
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)

    env_vars = _ray_runtime_env_vars()

    assert not {"WANDB_API_KEY", "WANDB_ENTITY", "WANDB_MODE"} & env_vars.keys()


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--algo.advantage.score_centering_top_k", "1"], "must be 0 (disabled) or at least 2"),
        (["--algo.advantage.score_centering_top_k", "-1"], "must be 0 (disabled) or at least 2"),
        (["--rollout.top_p", "0.95"], "score centering needs --rollout.top_p 1.0"),
        (["--train.max_epochs", "2"], "needs one update per batch"),
        (["--fsdp.tp_size", "2"], "score centering requires --fsdp.tp_size 1"),
        (["--algo.advantage.is_correction_level", "seq"], "supports only IS level off/token"),
    ],
)
def test_score_centering_rejects_unsupported_config_before_starting_workers(flags, message):
    command = [
        sys.executable, "-m", "molt.cli.train_rl_ray",
        "--actor.model_name_or_path", "unused", "--train.agent_path", "unused.py",
        "--vllm.num_engines", "1", "--algo.advantage.score_centering_top_k", "32",
        "--algo.advantage.is_correction_level", "off", "--train.max_epochs", "1",
    ]
    result = subprocess.run(command + flags, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert message in result.stderr
    assert "ray.init" not in result.stderr


@pytest.mark.parametrize("surplus_batch", [False, True])
@pytest.mark.parametrize("unsupported", [None, "mtp", "uncorrected"])
def test_routing_replay_partial_rollout_cli(monkeypatch, surplus_batch, unsupported):
    import ast
    from pathlib import Path
    from types import ModuleType
    from unittest.mock import Mock

    from molt.cli import train_rl_ray

    # Run the real CLI body, replacing only worker startup and the optional backend import.
    source = Path(train_rl_ray.__file__)
    entrypoint = ast.parse(source.read_text()).body[-1]
    assert isinstance(entrypoint, ast.If)
    for name in ("nemo_automodel", "nemo_automodel.components", "nemo_automodel.components.moe",
                 "nemo_automodel.components.moe.router_replay"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    flags = [
        "train_rl_ray", "--actor.model_name_or_path", "unused", "--train.agent_path", "unused.py",
        "--vllm.num_engines", "1", "--train.routing_replay", "--rollout.top_p", "1.0",
        "--algo.advantage.is_correction_level", "off" if unsupported == "uncorrected" else "geo",
    ]
    flags += (["--rollout.batch_size", "8", "--rollout.vllm_generate_batch_size", "16"]
              if surplus_batch else ["--train.partial_rollout_enable"])
    if unsupported == "mtp":
        flags += ["--vllm.mtp_num_speculative_tokens", "1"]
    monkeypatch.setattr(sys, "argv", flags)
    start_workers = Mock()
    namespace = {**vars(train_rl_ray), "train": start_workers}
    code = compile(ast.Module(body=entrypoint.body, type_ignores=[]), str(source), "exec")
    if unsupported:
        message = "mtp_num_speculative_tokens is incompatible" if unsupported == "mtp" else "Off-policy rollout"
        with pytest.raises(ValueError, match=message):
            exec(code, namespace)
        start_workers.assert_not_called()
    else:
        exec(code, namespace)
        start_workers.assert_called_once()
        args = start_workers.call_args.args[0]
        assert args.train.routing_replay and args.train.partial_rollout_enable
        assert not args.train.force_sync_mode
        assert args.algo.advantage.is_correction_level == "geo"
