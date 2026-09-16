"""Unit tests for the MoLoRA wrapper (:mod:`infl_ens.training.molora`)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from infl_ens.training.molora import (  # noqa: E402
    MoLoRAConfig,
    balance_loss,
    gate_usage,
    inject_molora,
    is_molora_dir,
    load_molora,
    molora_linear_class,
    molora_modules,
    molora_param_count,
    molora_state_dict,
    save_molora,
)


class _Tiny(nn.Module):
    """Two-layer toy with PEFT-style attribute names."""

    def __init__(self, d: int = 8, hidden: int = 12) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.up_proj = nn.Linear(d, hidden)
        self.down_proj = nn.Linear(hidden, d)
        self.lm_head = nn.Linear(d, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.q_proj(x))
        h = self.down_proj(torch.relu(self.up_proj(h)))
        return self.lm_head(h)


def _cfg(**kw) -> MoLoRAConfig:
    base = dict(n_experts=3, expert_rank=2, expert_alpha=4,
                target_modules=("q_proj", "up_proj", "down_proj"))
    base.update(kw)
    return MoLoRAConfig(**base)


def test_config_validates() -> None:
    with pytest.raises(ValueError):
        MoLoRAConfig(n_experts=0)
    with pytest.raises(ValueError):
        MoLoRAConfig(balance_coef=-1.0)
    assert MoLoRAConfig(target_modules=["a", "b"]).target_modules == ("a", "b")


def test_inject_matches_targets_and_freezes_base() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    wrapped = inject_molora(model, _cfg())
    cls = molora_linear_class()
    assert len(wrapped) == 3
    assert all(isinstance(m, cls) for m in wrapped)
    assert isinstance(model.lm_head, nn.Linear) and not isinstance(model.lm_head, cls)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {
        f"{m}.{k}" for m in ("q_proj", "up_proj", "down_proj") for k in ("lora_A", "lora_B", "gate")
    }
    with pytest.raises(ValueError, match="no nn.Linear matched"):
        inject_molora(_Tiny(), _cfg(target_modules=("nope",)))


def test_forward_is_identity_at_init_and_shape_preserving() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        before = model(x)
    inject_molora(model, _cfg())
    with torch.no_grad():
        after = model(x)
    # lora_B and gate start at zero, so the wrapped model equals the base.
    assert after.shape == before.shape
    torch.testing.assert_close(after, before)


def test_gate_is_uniform_at_init_and_gradients_reach_everything() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    cfg = _cfg(balance_coef=0.1)
    mods = inject_molora(model, cfg)
    x = torch.randn(4, 3, 8)
    out = model(x)
    usage = gate_usage(mods)
    assert usage is not None and len(usage) == 3
    assert pytest.approx(sum(usage), abs=1e-6) == 1.0
    assert all(abs(u - 1.0 / 3.0) < 1e-6 for u in usage)
    bal = balance_loss(mods)
    assert bal is not None
    assert pytest.approx(bal.detach().item(), abs=1e-6) == 1.0  # uniform => E * sum (1/E)^2 = 1
    loss = out.pow(2).mean() + cfg.balance_coef * bal
    loss.backward()
    for m in mods:
        # lora_B is zero so lora_A receives no gradient from the output on the
        # first step (as with plain LoRA); lora_B and the gate must.
        assert m.lora_B.grad is not None and m.lora_B.grad.abs().sum() > 0
        assert m.gate.grad is not None
        assert m.base.weight.grad is None


def test_gate_gradient_flows_once_experts_are_nonzero() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    mods = inject_molora(model, _cfg())
    with torch.no_grad():
        for m in mods:
            m.lora_B.normal_()
    out = model(torch.randn(4, 3, 8))
    out.pow(2).mean().backward()
    for m in mods:
        assert m.lora_A.grad is not None and m.lora_A.grad.abs().sum() > 0
        assert m.gate.grad is not None and m.gate.grad.abs().sum() > 0


def test_param_count_matches_e_times_lora_plus_gate() -> None:
    model = _Tiny()
    cfg = _cfg()
    inject_molora(model, cfg)
    counts = molora_param_count(model)
    e, r = cfg.n_experts, cfg.expert_rank
    expected_experts = sum(
        e * r * (lin.in_features + lin.out_features)
        for lin in (nn.Linear(8, 8), nn.Linear(8, 12), nn.Linear(12, 8))
    )
    expected_gate = e * (8 + 8 + 12)
    assert counts == {
        "experts": expected_experts,
        "gate": expected_gate,
        "total": expected_experts + expected_gate,
    }


def test_save_load_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _Tiny()
    cfg = _cfg()
    mods = inject_molora(model, cfg)
    with torch.no_grad():
        for m in mods:
            m.lora_B.normal_()
            m.gate.normal_()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        ref = model(x)
    out_dir = save_molora(model, cfg, tmp_path / "round-00", base_model="org/base")
    assert is_molora_dir(out_dir)

    torch.manual_seed(0)
    fresh = _Tiny()
    load_molora(fresh, out_dir)
    assert not fresh.training
    with torch.no_grad():
        got = fresh(x)
    torch.testing.assert_close(got, ref)
    assert set(molora_state_dict(fresh)) == set(molora_state_dict(model))
    assert len(molora_modules(fresh)) == 3


def test_load_rejects_mismatched_checkpoint(tmp_path: Path) -> None:
    model = _Tiny()
    cfg = _cfg()
    inject_molora(model, cfg)
    out_dir = save_molora(model, cfg, tmp_path / "ckpt")
    with pytest.raises(FileNotFoundError):
        load_molora(_Tiny(), tmp_path / "missing")

    class _Other(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8)

    with pytest.raises(ValueError, match="mismatch"):
        load_molora(_Other(), out_dir)
