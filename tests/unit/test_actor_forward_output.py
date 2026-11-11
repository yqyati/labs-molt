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

"""Actor.forward returns a single named output dict.

Guards the concern that writing ``output["log_probs"]`` (and ``action_log_probs`` /
``entropy``) could clobber a tensor the underlying model already returned. It
cannot: HF causal-LM outputs and NeMo custom forwards never carry those keys, so
the writes are purely additive, and ``logits`` is the only field Actor.forward
intentionally replaces (with its gathered / chunk-sliced version).
"""

from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from molt.models.base import _AttrDict, _normalize_output
from molt.models.actor import Actor
from molt.models.base import BaseModel


def test_entropy_monitor_matches_dense_entropy_without_changing_policy_gradients():
    torch.manual_seed(42)
    logits = torch.randn(1, 2050, 11, dtype=torch.bfloat16, requires_grad=True)
    sequences = torch.randint(0, 11, (1, 2050))
    mask = torch.ones(1, 2049, dtype=torch.bool)
    actor = SimpleNamespace(temperature=0.7, packing_samples=False)
    actor._forward_backbone = lambda *args, **kwargs: (
        _AttrDict(logits=logits), sequences.roll(-1, dims=1), False, None, 1, 2050
    )
    actor._restore_full_sequence = lambda t, **kwargs: t
    baseline = Actor.forward(actor, sequences, mask)
    monitored = Actor.forward(actor, sequences, mask, return_entropy=True, entropy_no_grad=True)
    logp = torch.log_softmax(logits.float() / 0.7, dim=-1)
    expected = -(logp.exp() * logp).sum(-1)[:, :-1]
    torch.testing.assert_close(monitored.entropy, expected)
    assert not monitored.entropy.requires_grad
    torch.testing.assert_close(monitored.action_log_probs, baseline.action_log_probs)
    before = torch.autograd.grad(baseline.action_log_probs.sum(), logits, retain_graph=True)[0]
    after = torch.autograd.grad(monitored.action_log_probs.sum(), logits)[0]
    torch.testing.assert_close(before, after)


def test_hf_causal_lm_output_has_no_logprob_fields():
    # If any of these pre-existed on the model output, Actor.forward's
    # output[...] = ... assignments would overwrite a model-produced tensor.
    fields = set(CausalLMOutputWithPast.__dataclass_fields__)
    assert {"log_probs", "action_log_probs", "entropy"} & fields == set()


def test_adding_log_probs_to_hf_output_is_additive():
    logits = torch.randn(1, 5, 7)
    pkv = ("past",)
    hf_out = CausalLMOutputWithPast(logits=logits, past_key_values=pkv)

    out = _normalize_output(hf_out)
    assert isinstance(out, _AttrDict)
    assert "log_probs" not in out  # nothing to overwrite

    out["log_probs"] = logits[:, :-1, 0]
    out["action_log_probs"] = logits[:, -2:, 0]

    # Model-produced fields survive untouched; the new keys are additions.
    assert out["logits"] is logits
    assert out["past_key_values"] is pkv
    torch.testing.assert_close(out["log_probs"], logits[:, :-1, 0])
    assert out.log_probs is out["log_probs"]  # attribute access == key access

    # The original HF output object is not mutated (normalize copies into _AttrDict).
    assert not hasattr(hf_out, "log_probs")


def test_nemo_custom_raw_tensor_output_wraps_then_adds():
    # NeMo custom MoE/LLM return a raw logits Tensor (no dict, no log_probs).
    logits = torch.randn(1, 4, 9)
    out = _normalize_output(logits)
    assert isinstance(out, _AttrDict)
    assert out["logits"] is logits
    assert "log_probs" not in out

    out["log_probs"] = logits[:, :-1, 0]
    assert out["logits"] is logits  # logits unchanged by the new key


@pytest.mark.parametrize("packed", [False, True])
def test_score_centering_head_matches_dense_log_softmax_and_gradients(packed):
    torch.manual_seed(17)
    logits = torch.randn(2, 4, 9, dtype=torch.bfloat16, requires_grad=True)
    sequences = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    ids = torch.tensor([[[3, 8], [4, 2], [0, 0]], [[2, 6], [1, 5], [3, 7]]])
    indices = torch.tensor([0, 1, 2, 4, 5, 6, 7]) if packed else None
    labels = sequences.roll(-1, dims=1)
    model_logits = logits
    if packed:
        model_logits = logits.reshape(-1, 9)[indices].unsqueeze(0)
        labels = labels.reshape(-1)[indices].unsqueeze(0)
    calls = []

    def backbone(*args, **kwargs):
        calls.append(1)
        return _AttrDict(logits=model_logits), labels, False, indices, 2, 4

    actor = SimpleNamespace(temperature=0.7, packing_samples=packed, _forward_backbone=backbone)
    actor._restore_full_sequence = lambda t, **kwargs: BaseModel._restore_full_sequence(actor, t, **kwargs)
    output = Actor.forward(actor, sequences, mask, top_ids=ids)
    expected = torch.log_softmax(logits.float() / 0.7, dim=-1)[:, :-1].gather(-1, ids)
    valid = mask.unsqueeze(-1).expand_as(expected)
    torch.testing.assert_close(output["action_top_log_probs"][valid], expected[valid], atol=1e-6, rtol=1e-5)
    actual_grad = torch.autograd.grad(output["action_top_log_probs"][valid].sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected[valid].sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=0.02, rtol=0.02)
    assert len(calls) == 1
