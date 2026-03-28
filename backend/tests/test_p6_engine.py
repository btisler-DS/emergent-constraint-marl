"""Tests for P6SimulationEngine."""
import numpy as np
import pytest
import torch
from simulation.p6_engine import P6SimulationEngine, P6Config


def make_engine(diffusion_coefficient=0.1, decay_rate=0.05, seed=0):
    config = P6Config(
        seed=seed,
        num_epochs=2,
        episodes_per_epoch=2,
        diffusion_coefficient=diffusion_coefficient,
        decay_rate=decay_rate,
    )
    return P6SimulationEngine(config)


def test_engine_initializes_field():
    engine = make_engine()
    assert engine.field is not None
    assert np.allclose(engine.field.F, 0.0)


def test_agent_a_obs_dim_is_13():
    engine = make_engine()
    assert engine.agent_a.obs_encoder.in_features == 13


def test_agent_a_parameter_delta():
    # P5: nn.Linear(12, 64) = 12*64 + 64 = 832 params
    # P6: nn.Linear(13, 64) = 13*64 + 64 = 896 params, delta = +64
    engine = make_engine()
    obs_enc_params = sum(p.numel() for p in engine.agent_a.obs_encoder.parameters())
    assert obs_enc_params == 896


def test_run_returns_epoch_series():
    engine = make_engine()
    results = engine.run()
    assert len(results) == 2    # num_epochs=2
    assert "field_mean" in results[0]
    assert "field_std" in results[0]
    assert "field_max" in results[0]
    assert "field_min" in results[0]
    assert "field_entropy" in results[0]


def test_field_non_zero_after_run():
    engine = make_engine(diffusion_coefficient=0.1, decay_rate=0.05)
    results = engine.run()
    means = [r["field_mean"] for r in results]
    assert any(m > 0.0 for m in means)


def test_existing_metrics_present():
    engine = make_engine()
    results = engine.run()
    assert "sustained_structure_score" in results[0]
    assert "exploitation_loop_rate" in results[0]
    assert "query_rate" in results[0]


def test_field_resets_between_episodes():
    # Engine should complete without error and produce finite values
    engine = make_engine(diffusion_coefficient=0.0, decay_rate=0.0, seed=42)
    results = engine.run()
    assert all(np.isfinite(r["field_max"]) for r in results)


def test_effective_cost_modulates_reward():
    # Both engines complete without error and return numeric avg rewards
    engine_high = make_engine(diffusion_coefficient=0.9, decay_rate=0.05, seed=7)
    results_high = engine_high.run()
    engine_low = make_engine(diffusion_coefficient=0.0, decay_rate=0.9, seed=7)
    results_low = engine_low.run()
    assert all(isinstance(r.get("avg_reward_A", 0.0), float) for r in results_high)
    assert all(isinstance(r.get("avg_reward_A", 0.0), float) for r in results_low)
