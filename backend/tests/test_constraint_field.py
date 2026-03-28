"""Tests for ConstraintField — Protocol 6 emergent constraint field."""
import numpy as np
import pytest
from simulation.constraint_field import ConstraintField


def test_initializes_to_zero():
    cf = ConstraintField(n_agents=3, diffusion_coefficient=0.1, decay_rate=0.05)
    assert np.allclose(cf.F, 0.0)


def test_reset_clears_field():
    cf = ConstraintField(n_agents=3, diffusion_coefficient=0.0, decay_rate=0.0)
    cf.update({"A": 0, "B": 0, "C": 0})
    cf.reset()
    assert np.allclose(cf.F, 0.0)


def test_emission_increments_local_cell_only():
    # With dc=0 and dr=0, only emission happens. Verify weights (Deviation 1: scaled 0.1x).
    cf = ConstraintField(n_agents=3, diffusion_coefficient=0.0, decay_rate=0.0)
    cf.update({"A": 0, "B": 1, "C": 2})  # DECLARE=0, QUERY=1, RESPOND=2
    assert cf.F[0] == pytest.approx(0.03)   # A: DECLARE weight
    assert cf.F[1] == pytest.approx(0.01)   # B: QUERY weight
    assert cf.F[2] == pytest.approx(0.02)   # C: RESPOND weight


def test_decay_reduces_field():
    cf = ConstraintField(n_agents=3, diffusion_coefficient=0.0, decay_rate=0.5)
    cf.update({"A": 0, "B": 0, "C": 0})   # F = [0.03, 0.03, 0.03] after decay: [0.015, 0.015, 0.015]
    assert np.allclose(cf.F, [0.015, 0.015, 0.015], atol=1e-6)


def test_decay_applied_after_emission():
    # Order: emit then decay. With dr=0.0, field stays at emission value.
    cf = ConstraintField(n_agents=3, diffusion_coefficient=0.0, decay_rate=0.0)
    cf.update({"A": 0, "B": 0, "C": 0})   # no decay: F = [0.03, 0.03, 0.03]
    assert np.allclose(cf.F, [0.03, 0.03, 0.03], atol=1e-6)


def test_diffusion_conserves_mass_approximately():
    # Diffusion redistributes but with outflow-only model, mass is conserved exactly.
    cf = ConstraintField(n_agents=3, diffusion_coefficient=0.3, decay_rate=0.0)
    cf.F = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    mass_before = cf.F.sum()
    cf._diffuse()
    # Mass is conserved (no creation/destruction in diffusion)
    assert cf.F.sum() == pytest.approx(mass_before, abs=1e-5)


def test_diffusion_spreads_to_neighbors():
    # Agent 0 has high field; diffusion should increase neighbors.
    cf = ConstraintField(n_agents=3, diffusion_coefficient=0.6, decay_rate=0.0)
    cf.F = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cf._diffuse()
    # Agent 0 loses 60% of 1.0 = 0.6, keeping 0.4
    # Neighbors (indices 1 and 2 in ring) each get 0.3
    assert cf.F[0] == pytest.approx(0.4, abs=1e-5)
    assert cf.F[1] == pytest.approx(0.3, abs=1e-5)
    assert cf.F[2] == pytest.approx(0.3, abs=1e-5)


def test_diffusion_is_symmetric():
    # Symmetric initial state should stay symmetric after diffusion.
    cf = ConstraintField(n_agents=3, diffusion_coefficient=0.5, decay_rate=0.0)
    cf.F = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    cf._diffuse()
    assert np.allclose(cf.F, [1.0, 1.0, 1.0], atol=1e-5)


def test_get_returns_own_field_only():
    cf = ConstraintField(n_agents=3, diffusion_coefficient=0.0, decay_rate=0.0)
    cf.F = np.array([0.5, 0.2, 0.8], dtype=np.float32)
    assert cf.get("A") == pytest.approx(0.5)
    assert cf.get("B") == pytest.approx(0.2)
    assert cf.get("C") == pytest.approx(0.8)


def test_effective_cost_linear_modulation():
    cf = ConstraintField(n_agents=3, diffusion_coefficient=0.0, decay_rate=0.0)
    cf.F = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    # effective = base * (1 + sensitivity * F[i]), sensitivity=1.0
    base = 0.01
    sensitivity = 1.0
    cost = cf.effective_cost("A", base_cost=base, sensitivity=sensitivity)
    assert cost == pytest.approx(base * (1 + sensitivity * 0.5))


def test_multiple_update_steps_accumulate():
    # Two DECLARE steps for agent A, no diffusion, no decay: F[0] = 0.06
    cf = ConstraintField(n_agents=3, diffusion_coefficient=0.0, decay_rate=0.0)
    cf.update({"A": 0, "B": 0, "C": 0})
    cf.update({"A": 0, "B": 0, "C": 0})
    assert cf.F[0] == pytest.approx(0.06, abs=1e-5)
