"""Tests for prompt-level jointly trained LoRA mixtures."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from infl_ens.training.mixture_lora import (  # noqa: E402
    build_mixture_lora_model,
    gate_weights,
    load_balance_loss,
    mixture_lora_delta,
)


def test_delta_matches_manual_expert_weighting() -> None:
    inputs = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    a = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    b = torch.tensor([[[2.0]], [[3.0]]])
    gates = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
    got = mixture_lora_delta(inputs, a, b, gates, scale=1.0)
    expected = torch.tensor([[[5.0]], [[6.0]]])
    torch.testing.assert_close(got, expected)


def test_topk_gate_and_balance_loss() -> None:
    deployed, dense = gate_weights(torch.zeros(4, 3), top_k=2)
    assert torch.all((deployed > 0).sum(dim=1) == 2)
    torch.testing.assert_close(deployed.sum(dim=1), torch.ones(4))
    assert load_balance_loss(dense).item() == pytest.approx(1.0)


def test_wrapped_model_matches_manual_delta() -> None:
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(2, 1, bias=False)
            nn.init.zeros_(self.q_proj.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.q_proj(x)

    model = build_mixture_lora_model(
        Tiny(),
        trait_dim=1,
        n_experts=2,
        expert_rank=1,
        lora_alpha=1,
        target_modules=["q_proj"],
        top_k=None,
    )
    layer = model.base_model.q_proj
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]))
        layer.lora_b.copy_(torch.tensor([[[2.0]], [[3.0]]]))
    inputs = torch.tensor([[[1.0, 2.0]]])
    override = torch.tensor([[0.25, 0.75]])
    got = model(trait_coords=torch.zeros(1, 1), gate_override=override, x=inputs)
    torch.testing.assert_close(got, torch.tensor([[[5.0]]]))
