#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU backend check inside the runtime image; no checkpoint/GPU required."""


def test_native_local_moe_replay_backward():
    import importlib.util
    import torch

    if importlib.util.find_spec("nemo_automodel") is None:
        import pytest
        pytest.skip("requires the runtime image with nemo_automodel")
    from nemo_automodel.components.models.common.utils import BackendConfig
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.layers import MoE
    from nemo_automodel.components.moe.experts import GroupedExperts
    from nemo_automodel.components.moe.router_replay import RouterReplay

    torch.manual_seed(17)
    RouterReplay.clear_registry()
    config = MoEConfig(
        n_routed_experts=4, n_shared_experts=0, n_activated_experts=2,
        n_expert_groups=1, n_limited_groups=1, train_gate=True,
        gate_bias_update_factor=0.0, aux_loss_coeff=0.0, score_func="softmax",
        route_scale=1.0, dim=8, inter_dim=16, moe_inter_dim=16,
        norm_topk_prob=True, dtype=torch.float32, enable_routing_replay=True,
    )
    model = MoE(config, BackendConfig(attn="sdpa", linear="torch", experts="torch", dispatcher="torch"))
    assert isinstance(model.experts, GroupedExperts)
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.1)
    x = torch.randn(6, 8, requires_grad=True)
    with RouterReplay.record():
        original = model(x).detach()
    targets = [t.clone() for t in RouterReplay.collect()]
    with RouterReplay.replay(targets):
        replayed = model(x)
    torch.testing.assert_close(replayed, original, rtol=0, atol=0)
    # Force a different selection, and verify the experts receive the overridden IDs.
    forced = [(targets[0] + 1) % 4]
    seen = []
    hook = model.experts.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[3].detach().clone()))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.experts.down_projs.detach().clone()
    try:
        with RouterReplay.replay(forced):
            loss = model(x).square().mean()
        loss.backward()
        assert torch.equal(seen[-1], forced[0])
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert model.experts.down_projs.grad.abs().sum() > 0
        optimizer.step()
        assert not torch.equal(before, model.experts.down_projs)
        print("PASS: native EP=1 local experts, forced routing replay, backward and optimizer update", flush=True)
    finally:
        hook.remove()
        RouterReplay.clear_registry()


if __name__ == "__main__":
    test_native_local_moe_replay_backward()
