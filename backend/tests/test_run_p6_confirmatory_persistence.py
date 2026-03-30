"""Tests that run_one() persists SSS and ELR as per-seed scalar summaries.

These are the metrics that were missing from P6 confirmatory output and
prevented direct hypothesis testing against H1 and H2 as preregistered.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_p6_confirmatory import run_one
from simulation.p6_engine import P6Config


def _fast_cfg() -> dict:
    """Minimal config dict matching run_one() expectations."""
    return {
        "num_epochs": 3,
        "episodes_per_epoch": 2,
        "grid_size": 20,
        "num_obstacles": 8,
        "z_layers": 8,
        "max_steps": 4,
        "energy_budget": 100.0,
        "move_cost": 1.0,
        "collision_penalty": 5.0,
        "signal_dim": 8,
        "hidden_dim": 16,
        "depth": 1,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "communication_tax_rate": 0.01,
        "survival_bonus": 0.1,
        "declare_cost": 1.0,
        "query_cost": 1.5,
        "respond_cost": 0.8,
        "signal_cost_sensitivity": 1.0,
        "diffusion_coefficient": 0.1,
        "decay_rate": 0.05,
        "device": "cpu",
    }


FIXED_MULTIPLIER = 1.2287

REQUIRED_SSS_KEYS = {
    "mean_sss",
    "final_sss",
    "sss_window20",
}

REQUIRED_ELR_KEYS = {
    "mean_elr",
    "final_elr",
    "elr_window20",
}


@pytest.fixture(scope="module")
def result_a():
    return run_one("A", seed=0, base_config=_fast_cfg(), fixed_cost_multiplier=FIXED_MULTIPLIER)


# ── SSS scalars present ────────────────────────────────────────────────────

def test_mean_sss_present(result_a):
    assert "mean_sss" in result_a, "mean_sss missing from run_one() output"


def test_final_sss_present(result_a):
    assert "final_sss" in result_a, "final_sss missing from run_one() output"


def test_sss_window20_present(result_a):
    assert "sss_window20" in result_a, "sss_window20 missing from run_one() output"


# ── ELR scalars present ────────────────────────────────────────────────────

def test_mean_elr_present(result_a):
    assert "mean_elr" in result_a, "mean_elr missing from run_one() output"


def test_final_elr_present(result_a):
    assert "final_elr" in result_a, "final_elr missing from run_one() output"


def test_elr_window20_present(result_a):
    assert "elr_window20" in result_a, "elr_window20 missing from run_one() output"


# ── SSS time series present ────────────────────────────────────────────────

def test_sss_series_present(result_a):
    assert "sss_series" in result_a, "sss_series missing from run_one() output"


def test_elr_series_present(result_a):
    assert "elr_series" in result_a, "elr_series missing from run_one() output"


# ── Value correctness ──────────────────────────────────────────────────────

def test_mean_sss_matches_series(result_a):
    assert result_a["mean_sss"] == pytest.approx(
        float(np.mean(result_a["sss_series"])), abs=1e-5
    )


def test_final_sss_matches_series_last(result_a):
    assert result_a["final_sss"] == pytest.approx(result_a["sss_series"][-1], abs=1e-5)


def test_sss_window20_matches_series_tail(result_a):
    tail = result_a["sss_series"][-20:]
    assert result_a["sss_window20"] == pytest.approx(float(np.mean(tail)), abs=1e-5)


def test_mean_elr_matches_series(result_a):
    assert result_a["mean_elr"] == pytest.approx(
        float(np.mean(result_a["elr_series"])), abs=1e-5
    )


def test_series_length_equals_num_epochs(result_a):
    assert len(result_a["sss_series"]) == 3  # num_epochs in fast config
    assert len(result_a["elr_series"]) == 3


# ── Present for all four conditions ───────────────────────────────────────

@pytest.mark.parametrize("condition", ["A", "B", "C", "D"])
def test_sss_elr_present_all_conditions(condition):
    result = run_one(condition, seed=0, base_config=_fast_cfg(), fixed_cost_multiplier=FIXED_MULTIPLIER)
    for key in REQUIRED_SSS_KEYS | REQUIRED_ELR_KEYS:
        assert key in result, f"{key} missing for condition {condition}"
