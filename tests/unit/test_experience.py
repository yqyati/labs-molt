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

from types import SimpleNamespace

import pytest
import torch

import molt.trainer.algorithm.experience as experience_mod
from molt.trainer.algorithm.experience import (
    Experience,
    balance_experiences,
    get_model_parallel_size,
    make_experience_batch,
    split_experience_batch,
)
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker


def _args(cp=1, tp=1, ep=1, actor_gpus=1):
    return SimpleNamespace(
        actor=SimpleNamespace(num_nodes=1, num_gpus_per_node=actor_gpus),
        fsdp=SimpleNamespace(cp_size=cp, tp_size=tp, ep_size=ep),
    )


def test_model_parallel_size_excludes_ep():
    assert get_model_parallel_size(_args(cp=2, tp=3, ep=4)) == 6


def _len_sample(length, idx):
    return Experience(total_length=torch.tensor([length]), rollout_ids=[f"r{idx}"])


def test_balance_experiences_excludes_ep_and_returns_flat_samples():
    # DP size = actor_gpus // (cp*tp) = 4 (EP is excluded from the divisor). 6 per-sample
    # experiences -> keep (6 // 4) * 4 = 4 (drop the 2-sample remainder). If EP were wrongly in
    # the divisor, dp would be 2 and all 6 would be kept.
    samples = [_len_sample(6 - i, i) for i in range(6)]
    balanced = balance_experiences(samples, _args(ep=2, actor_gpus=4))

    assert len(balanced) == 4  # flat, equal count per rank (1×4), remainder dropped — not concatenated
    ids = {id(s) for s in samples}
    assert all(id(b) in ids for b in balanced)  # returns the input samples themselves (no concat)


def test_balance_experiences_equal_count_and_length_sorted():
    # 10 samples across 4 DP ranks: the 2-sample remainder is dropped so every rank receives the
    # SAME count (2) — unequal counts desync num_steps and deadlock the world all_reduce. The flat
    # result is contiguous per-rank blocks; within each block samples are sorted by length descending
    # (k-th microbatch size-matched across ranks so no straggler trips the NCCL watchdog).
    samples = [_len_sample(10 - i, i) for i in range(10)]
    balanced = balance_experiences(samples, _args(actor_gpus=4))

    assert len(balanced) == 8  # 4 ranks × 2 samples; trailing 2-sample remainder dropped
    assert len({id(b) for b in balanced}) == 8  # no duplicates
    for rank in range(4):
        block = balanced[rank * 2 : rank * 2 + 2]
        block_lengths = [int(b.total_length.item()) for b in block]
        assert block_lengths == sorted(block_lengths, reverse=True)


def _grpo_maker(length_penalty=False, n_samples=4):
    """A duck-typed RemoteExperienceMaker self with just the fields compute_advantages reads."""
    maker = SimpleNamespace(
        advantage_estimator="grpo",
        kl_ctl=SimpleNamespace(value=0.0),
        args=SimpleNamespace(
            reward=SimpleNamespace(
                clip_range=None,
                group_length_penalty=SimpleNamespace(
                    enable=length_penalty,
                    min_pass_rate=0.5,
                    quantile=0.5,
                    max_penalty=0.1,
                    tolerance=0.2,
                    saturation=1.0,
                ),
            ),
            algo=SimpleNamespace(advantage=SimpleNamespace(gamma=1.0, lam=1.0, no_whiten=False)),
            rollout=SimpleNamespace(n_samples_per_prompt=n_samples),
        ),
    )
    # Bind the two instance methods the estimator path dispatches through (each reads only its args).
    maker._merge_rollout_rewards = RemoteExperienceMaker._merge_rollout_rewards.__get__(maker)
    maker.compute_advantages_and_returns = RemoteExperienceMaker.compute_advantages_and_returns.__get__(maker)
    return maker


def _sample(idx, group_id, reward, length=6):
    # Equal length so the concatenated path stacks without padding — the two paths are
    # then directly comparable row-by-row.
    return Experience(
        action_mask=torch.ones(1, length, dtype=torch.bool),
        kl=torch.zeros(1, length),
        rewards=torch.tensor([float(reward)]),
        index=[idx],
        group_ids=[group_id],
        rollout_ids=[f"r{idx}"],
        info={},
    )


def test_distributed_advantages_match_materialized():
    # The distributed path runs compute_advantages_and_returns on per-sample "light"
    # Experiences; the materialized path runs it on the concatenated batch. Advantages are
    # a function of (reward, group, mask) only, so the per-sample results must be identical
    # — this is what lets the trainer skip gathering the heavy batch on the controller.
    rewards = [1.0, 0.0, 0.5, 0.25, 0.9, 0.1, 0.4, 0.6]
    groups = ["g0"] * 4 + ["g1"] * 4

    per_sample = [_sample(i, groups[i], rewards[i]) for i in range(8)]
    RemoteExperienceMaker.compute_advantages_and_returns(_grpo_maker(), per_sample)

    concat = make_experience_batch([_sample(i, groups[i], rewards[i]) for i in range(8)])
    RemoteExperienceMaker.compute_advantages_and_returns(_grpo_maker(), [concat])

    for i in range(8):
        assert torch.allclose(per_sample[i].advantages[0], concat.advantages[i])
        assert torch.allclose(per_sample[i].returns[0], concat.returns[i])


def test_group_length_penalty_success_median_and_raw_rewards():
    maker = _grpo_maker(length_penalty=True, n_samples=8)
    maker.advantage_estimator = "dr_grpo"
    raw_rewards = [1, 1, 1, 1, 1, 1, 0, 0]
    lengths = [80, 90, 100, 100, 160, 200, 400, 500]
    samples = [_sample(i, "g", reward, length) for i, (reward, length) in enumerate(zip(raw_rewards, lengths))]
    for sample in samples:
        sample.info["reward"] = sample.rewards.clone()
    maker.compute_advantages_and_returns(samples)

    expected_rewards = torch.tensor([1, 1, 1, 1, 0.95, 0.9, 0, 0])
    for i, sample in enumerate(samples):
        assert torch.allclose(sample.returns, torch.full_like(sample.returns, expected_rewards[i] - 0.73125))
        assert sample.rewards.item() == sample.info["reward"].item() == raw_rewards[i]
    assert maker.reward_shaping_metrics["rollout/length_shaped_reward_mean"] == pytest.approx(0.73125)
    assert maker.reward_shaping_metrics["rollout/length_penalty_mean"] == pytest.approx(0.01875)
    assert maker.reward_shaping_metrics["rollout/length_penalized_fraction"] == 0.25


@pytest.mark.parametrize("successes", [0, 1, 4])
def test_group_length_penalty_preserves_low_success_groups(successes):
    maker = _grpo_maker(length_penalty=True, n_samples=8)
    samples = [_sample(i, "g", i < successes, 200 if i == 0 else 100) for i in range(8)]
    maker.compute_advantages_and_returns(samples)
    assert maker.reward_shaping_metrics["rollout/length_penalty_mean"] == 0
    assert maker.reward_shaping_metrics["rollout/length_penalty_eligible_group_fraction"] == 0
    assert maker.reward_shaping_metrics["rollout/length_shaped_reward_mean"] == successes / 8


def test_group_length_penalty_strict_custom_gate_and_free_length_margin():
    maker = _grpo_maker(length_penalty=True, n_samples=5)
    maker.args.reward.group_length_penalty.min_pass_rate = 0.6
    samples = [_sample(i, "g", i < 3, 200 if i == 0 else 100) for i in range(5)]
    maker.compute_advantages_and_returns(samples)
    assert maker.reward_shaping_metrics["rollout/length_penalty_eligible_group_fraction"] == 0

    samples = [_sample(i, "g", 1, 120 if i == 0 else 100) for i in range(5)]
    maker.compute_advantages_and_returns(samples)
    assert maker.reward_shaping_metrics["rollout/length_penalized_fraction"] == 0
    assert all(torch.count_nonzero(sample.advantages) == 0 for sample in samples)


@pytest.mark.parametrize("estimator", ["grpo", "dr_grpo", "reinforce_baseline", "rloo"])
def test_group_length_penalty_all_successes_produce_length_advantages(estimator):
    maker = _grpo_maker(length_penalty=True)
    maker.advantage_estimator = estimator
    samples = [_sample(i, "g", 1, 200 if i == 3 else 100) for i in range(4)]
    maker.compute_advantages_and_returns(samples)
    assert all(torch.all(sample.advantages > 0) for sample in samples[:3])
    assert torch.all(samples[3].advantages < 0)
    assert maker.reward_shaping_metrics["rollout/length_shaped_reward_mean"] == pytest.approx(0.975)


def test_group_length_penalty_sums_segments_and_keeps_groups_independent():
    maker = _grpo_maker(length_penalty=True)
    maker.advantage_estimator = "dr_grpo"
    samples = [_sample(i, "short", 1, 100 if i else 80) for i in range(4)]
    second_segment = _sample(8, "short", 1, 120)
    second_segment.rollout_ids = samples[0].rollout_ids.copy()
    samples.append(second_segment)
    samples.extend(_sample(i + 4, "long", 1, 2000 if i == 0 else 1000) for i in range(4))
    # Inactive positions represent prompt/observation tokens and must not affect length.
    second_segment.action_mask = torch.cat([second_segment.action_mask, torch.zeros(1, 300, dtype=torch.bool)], -1)
    second_segment.kl = torch.zeros_like(second_segment.action_mask, dtype=torch.float)
    maker.compute_advantages_and_returns(samples)
    assert samples[0].returns[0, 0].item() == pytest.approx(-0.075)
    assert second_segment.returns[0, 0].item() == pytest.approx(-0.075)
    assert samples[5].returns[0, 0].item() == pytest.approx(-0.075)
    assert samples[1].returns[0, 0].item() == pytest.approx(0.025)
    assert maker.reward_shaping_metrics["rollout/length_penalty_mean"] == pytest.approx(0.025)
    assert maker.reward_shaping_metrics["rollout/length_penalized_fraction"] == 0.25


@pytest.mark.parametrize("enabled,lengths", [(False, [100, 100, 100, 200]), (True, [100] * 4)])
def test_group_length_penalty_disabled_or_equal_lengths_keeps_zero_advantages(enabled, lengths):
    maker = _grpo_maker(length_penalty=enabled)
    samples = [_sample(i, "g", 1, length) for i, length in enumerate(lengths)]
    maker.compute_advantages_and_returns(samples)
    assert all(torch.count_nonzero(sample.advantages) == 0 for sample in samples)
    if not enabled:
        assert maker.reward_shaping_metrics == {}


def test_group_length_penalty_skips_incomplete_groups():
    maker = _grpo_maker(length_penalty=True)
    maker.compute_advantages_and_returns([_sample(i, "g", 1, 200 if i == 0 else 100) for i in range(3)])
    assert maker.reward_shaping_metrics["rollout/length_penalty_mean"] == 0
    assert maker.reward_shaping_metrics["rollout/length_penalty_incomplete_group_fraction"] == 1


def test_group_length_penalty_rejects_nonbinary_rewards_and_flash():
    maker = _grpo_maker(length_penalty=True)
    with pytest.raises(ValueError, match="binary"):
        maker.compute_advantages_and_returns([_sample(0, "g", 0.5)])
    maker.advantage_estimator = "flash_reinforce"
    maker._per_sample_rewards = RemoteExperienceMaker._per_sample_rewards
    with pytest.raises(ValueError, match="group advantages"):
        maker.compute_advantages_and_returns([_sample(0, "g", 1)])


def test_text_info_survives_batching_and_splitting():
    items = [
        Experience(sequences=torch.tensor([1, 2]), info={"feedback": "wrong unit"}),
        Experience(sequences=torch.tensor([3]), info={"feedback": "syntax error"}),
    ]

    batch = make_experience_batch(items)
    assert batch.info["feedback"] == ["wrong unit", "syntax error"]
    assert [item.info["feedback"] for item in split_experience_batch(batch)] == ["wrong unit", "syntax error"]


def test_text_info_missing_on_some_samples_batches_as_none():
    # Env feedback is usually set on failures only; whichever sample comes first, the batch keeps
    # the key with None for the samples that lack it (no KeyError, no silently dropped key).
    with_text = Experience(sequences=torch.tensor([1, 2]), info={"feedback": "wrong unit"})
    without = Experience(sequences=torch.tensor([3]), info={})
    assert make_experience_batch([with_text, without]).info["feedback"] == ["wrong unit", None]
    batch = make_experience_batch([without, with_text])
    assert batch.info["feedback"] == [None, "wrong unit"]
    assert [item.info["feedback"] for item in split_experience_batch(batch)] == [None, "wrong unit"]


def test_experience_offload_reload_roundtrip(monkeypatch):
    # offload() moves the heavy fields into the object store (leaving a ref) and keeps the
    # lightweight ones in place; reload() restores them exactly. The controller only ever reads
    # the light fields, so a batch of these handles never fetches an image.
    store = {}

    def fake_put(obj):
        store[len(store)] = obj
        return len(store) - 1

    monkeypatch.setattr(experience_mod.ray, "put", fake_put)
    monkeypatch.setattr(experience_mod.ray, "get", lambda key: store[key])

    exp = Experience(
        sequences=torch.arange(6).view(1, 6),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        action_mask=torch.ones(1, 5, dtype=torch.bool),
        rewards=torch.tensor([1.0]),
        mm_train_inputs=[{"pixel_values": torch.zeros(2, 3)}],
    )
    exp.offload()
    assert exp.heavy_ref is not None
    assert exp.sequences is None and exp.attention_mask is None and exp.mm_train_inputs is None
    assert exp.action_mask is not None and exp.rewards is not None  # light fields untouched

    # offload() is idempotent: a second call must not re-put the now-nulled fields and clobber the
    # ref (the reload below would then restore Nones and lose the sample's tensors).
    first_ref = exp.heavy_ref
    exp.offload()
    assert exp.heavy_ref == first_ref

    exp.reload()
    assert exp.heavy_ref is None
    assert torch.equal(exp.sequences, torch.arange(6).view(1, 6))
    assert torch.equal(exp.mm_train_inputs[0]["pixel_values"], torch.zeros(2, 3))
    exp.reload()  # idempotent — second call is a no-op
