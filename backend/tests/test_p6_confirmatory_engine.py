"""Tests for P6ConfirmatoryEngine — all four conditions."""
import numpy as np
import pytest
import torch
from simulation.p6_confirmatory_engine import P6ConfirmatoryEngine
from simulation.p6_engine import P6Config


def _fast_config(seed: int = 0) -> P6Config:
    # grid_size must stay at default (20) — env obs_dim for AgentA is 11 only at grid_size=20.
    return P6Config(
        seed=seed,
        num_epochs=2,
        episodes_per_epoch=2,
        max_steps=4,
        hidden_dim=16,
        depth=1,
        device="cpu",
    )


FIXED_MULTIPLIER = 1.2287


# ── Initialization ─────────────────────────────────────────────────────────

def test_condition_a_obs_dim_13():
    eng = P6ConfirmatoryEngine(_fast_config(), condition="A", fixed_cost_multiplier=FIXED_MULTIPLIER)
    assert eng.agent_a.obs_encoder.in_features == 13


def test_condition_b_obs_dim_15():
    eng = P6ConfirmatoryEngine(_fast_config(), condition="B", fixed_cost_multiplier=FIXED_MULTIPLIER)
    assert eng.agent_a.obs_encoder.in_features == 15


def test_condition_c_obs_dim_13():
    eng = P6ConfirmatoryEngine(_fast_config(), condition="C", fixed_cost_multiplier=FIXED_MULTIPLIER)
    assert eng.agent_a.obs_encoder.in_features == 13


def test_condition_d_obs_dim_13():
    eng = P6ConfirmatoryEngine(_fast_config(), condition="D", fixed_cost_multiplier=FIXED_MULTIPLIER)
    assert eng.agent_a.obs_encoder.in_features == 13


def test_conditions_a_b_have_field():
    for cond in ("A", "B"):
        eng = P6ConfirmatoryEngine(_fast_config(), condition=cond, fixed_cost_multiplier=FIXED_MULTIPLIER)
        assert eng.field is not None, f"Condition {cond} should have a ConstraintField"


def test_conditions_c_d_no_field():
    for cond in ("C", "D"):
        eng = P6ConfirmatoryEngine(_fast_config(), condition=cond, fixed_cost_multiplier=FIXED_MULTIPLIER)
        assert eng.field is None, f"Condition {cond} should have no ConstraintField"


# ── Observation augmentation ───────────────────────────────────────────────

def test_condition_a_augmented_obs_shape_13():
    eng = P6ConfirmatoryEngine(_fast_config(), condition="A", fixed_cost_multiplier=FIXED_MULTIPLIER)
    raw = eng.env.reset(seed=0)
    aug = eng._augment_obs(raw)
    assert aug["A"].shape == (13,)


def test_condition_b_augmented_obs_shape_15():
    eng = P6ConfirmatoryEngine(_fast_config(), condition="B", fixed_cost_multiplier=FIXED_MULTIPLIER)
    raw = eng.env.reset(seed=0)
    aug = eng._augment_obs(raw)
    assert aug["A"].shape == (15,)


def test_condition_b_global_obs_contains_full_field():
    """Last 3 elements of Condition B AgentA obs should be the field values."""
    eng = P6ConfirmatoryEngine(_fast_config(), condition="B", fixed_cost_multiplier=FIXED_MULTIPLIER)
    eng.field.F = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    raw = eng.env.reset(seed=0)
    aug = eng._augment_obs(raw)
    assert aug["A"][-3].item() == pytest.approx(0.1, abs=1e-5)
    assert aug["A"][-2].item() == pytest.approx(0.2, abs=1e-5)
    assert aug["A"][-1].item() == pytest.approx(0.3, abs=1e-5)


def test_condition_c_augmented_obs_shape_13():
    eng = P6ConfirmatoryEngine(_fast_config(), condition="C", fixed_cost_multiplier=FIXED_MULTIPLIER)
    raw = eng.env.reset(seed=0)
    aug = eng._augment_obs(raw)
    assert aug["A"].shape == (13,)


def test_condition_c_field_slot_is_zero():
    """With no field, the F[i] slot in obs should be 0.0."""
    eng = P6ConfirmatoryEngine(_fast_config(), condition="C", fixed_cost_multiplier=FIXED_MULTIPLIER)
    raw = eng.env.reset(seed=0)
    aug = eng._augment_obs(raw)
    assert aug["A"][-1].item() == pytest.approx(0.0, abs=1e-5)


# ── Run sanity checks ──────────────────────────────────────────────────────

def test_all_conditions_run_and_return_epoch_series():
    for cond in ("A", "B", "C", "D"):
        eng = P6ConfirmatoryEngine(_fast_config(), condition=cond, fixed_cost_multiplier=FIXED_MULTIPLIER)
        result = eng.run()
        assert len(result) == 2, f"Condition {cond}: expected 2 epochs"
        assert "field_mean" in result[0]
        assert "avg_reward_A" in result[0]


def test_condition_d_field_mean_zero():
    """Condition D has no field — field_mean should be 0.0 every epoch."""
    eng = P6ConfirmatoryEngine(_fast_config(), condition="D", fixed_cost_multiplier=FIXED_MULTIPLIER)
    result = eng.run()
    for ep in result:
        assert ep["field_mean"] == pytest.approx(0.0, abs=1e-6)


def test_condition_a_field_nonzero_after_run():
    """Condition A should build a non-trivial field."""
    cfg = P6Config(
        seed=0, num_epochs=3, episodes_per_epoch=5,
        max_steps=8, hidden_dim=16, depth=1, device="cpu",
    )
    eng = P6ConfirmatoryEngine(cfg, condition="A", fixed_cost_multiplier=FIXED_MULTIPLIER)
    result = eng.run()
    assert result[-1]["field_mean"] > 0.0


def test_condition_c_higher_reward_cost_than_d():
    """Condition C has fixed tax; Condition D has none — C rewards should be lower on avg."""
    cfg = P6Config(
        seed=42, num_epochs=3, episodes_per_epoch=5,
        max_steps=8, hidden_dim=16, depth=1, device="cpu",
    )
    eng_c = P6ConfirmatoryEngine(cfg, condition="C", fixed_cost_multiplier=FIXED_MULTIPLIER)
    eng_d = P6ConfirmatoryEngine(cfg, condition="D", fixed_cost_multiplier=FIXED_MULTIPLIER)
    res_c = eng_c.run()
    res_d = eng_d.run()
    avg_c = np.mean([e["avg_reward_A"] for e in res_c])
    avg_d = np.mean([e["avg_reward_A"] for e in res_d])
    assert avg_c < avg_d, "Condition C (fixed tax) should produce lower rewards than D (no tax)"
