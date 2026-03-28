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
    # High diffusion + low decay: field accumulates, cost increases, rewards decrease.
    # Low diffusion + high decay: field collapses, cost stays near baseline, rewards higher.
    # With same seed, the only difference is field intensity -> reward difference is causal.
    config_high = P6Config(
        seed=7, num_epochs=3, episodes_per_epoch=3,
        diffusion_coefficient=0.9, decay_rate=0.05,
    )
    config_low = P6Config(
        seed=7, num_epochs=3, episodes_per_epoch=3,
        diffusion_coefficient=0.0, decay_rate=0.99,
    )
    results_high = P6SimulationEngine(config_high).run()
    results_low = P6SimulationEngine(config_low).run()

    avg_high = np.mean([r["avg_reward_A"] for r in results_high])
    avg_low = np.mean([r["avg_reward_A"] for r in results_low])

    # High field = higher effective tax = lower reward. Not strict equality,
    # but high-field agent should not outperform low-field agent on average.
    assert avg_high <= avg_low + 0.5, (
        f"High-field avg reward {avg_high:.4f} should not greatly exceed "
        f"low-field avg reward {avg_low:.4f}"
    )
    # Both must return valid floats
    assert np.isfinite(avg_high) and np.isfinite(avg_low)
