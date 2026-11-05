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
