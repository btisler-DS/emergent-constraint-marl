"""Tests for field diagnostic functions — Protocol 6."""
import math
import numpy as np
import pytest
from simulation.metrics.field_diagnostics import compute_field_diagnostics, compute_run_summary


def test_basic_statistics_uniform():
    F = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    d = compute_field_diagnostics(F)
    assert d["field_mean"] == pytest.approx(0.5)
    assert d["field_std"] == pytest.approx(0.0, abs=1e-6)
    assert d["field_max"] == pytest.approx(0.5)
    assert d["field_min"] == pytest.approx(0.5)


def test_entropy_is_zero_for_single_nonzero():
    # All mass on one cell -> entropy of [1,0,0] normalized = 0
    F = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    d = compute_field_diagnostics(F)
    # p = [1, ~0, ~0]; -sum(p*log(p)) ~= 0
    assert d["field_entropy"] == pytest.approx(0.0, abs=0.01)


def test_entropy_is_max_for_uniform():
    # Uniform distribution has maximum entropy = log(n)
    F = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    d = compute_field_diagnostics(F)
    expected_max_entropy = math.log(3)
    assert d["field_entropy"] == pytest.approx(expected_max_entropy, abs=1e-4)


def test_all_zero_field_returns_zero_stats():
    F = np.zeros(3, dtype=np.float32)
    d = compute_field_diagnostics(F)
    assert d["field_mean"] == pytest.approx(0.0)
    assert d["field_std"] == pytest.approx(0.0)
    assert d["field_max"] == pytest.approx(0.0)
    assert d["field_min"] == pytest.approx(0.0)
    # Entropy of uniform (fallback for zero field) = log(3)
    assert d["field_entropy"] == pytest.approx(math.log(3), abs=1e-4)


def test_field_std_is_nonzero_for_heterogeneous():
    F = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    d = compute_field_diagnostics(F)
    assert d["field_std"] > 0.05   # spatial heterogeneity threshold from spec


def test_run_summary_field_formed_true():
    # field_std > 0.05 sustained for > 50 consecutive epochs
    epoch_series = [{"field_std": 0.1}] * 60
    summary = compute_run_summary(epoch_series)
    assert summary["field_formed"] is True


def test_run_summary_field_formed_false_below_threshold():
    epoch_series = [{"field_std": 0.03}] * 200
    summary = compute_run_summary(epoch_series)
    assert summary["field_formed"] is False


def test_run_summary_field_formed_false_not_sustained():
    # std > 0.05 for only 40 consecutive epochs — below the 50-epoch requirement
    epoch_series = ([{"field_std": 0.03}] * 100
                    + [{"field_std": 0.1}] * 40
                    + [{"field_std": 0.03}] * 60)
    summary = compute_run_summary(epoch_series)
    assert summary["field_formed"] is False


def test_run_summary_field_saturated_true():
    epoch_series = ([{"field_std": 0.0, "field_max": 0.96}]
                    + [{"field_std": 0.0, "field_max": 0.3}] * 199)
    summary = compute_run_summary(epoch_series)
    assert summary["field_saturated"] is True


def test_run_summary_field_saturated_false():
    epoch_series = [{"field_std": 0.0, "field_max": 0.5}] * 200
    summary = compute_run_summary(epoch_series)
    assert summary["field_saturated"] is False


def test_run_summary_field_collapsed_true():
    # field_mean < 0.01 after epoch 50
    epoch_series = ([{"field_std": 0.1, "field_mean": 0.5, "field_max": 0.6}] * 50
                    + [{"field_std": 0.0, "field_mean": 0.005, "field_max": 0.01}] * 150)
    summary = compute_run_summary(epoch_series)
    assert summary["field_collapsed"] is True


def test_run_summary_field_collapsed_false():
    epoch_series = [{"field_std": 0.1, "field_mean": 0.1, "field_max": 0.5}] * 200
    summary = compute_run_summary(epoch_series)
    assert summary["field_collapsed"] is False


def test_run_summary_field_formed_exactly_50_consecutive():
    # Exactly 50 consecutive epochs — spec requires > 50, so this must be False
    epoch_series = [{"field_std": 0.1}] * 50
    summary = compute_run_summary(epoch_series)
    assert summary["field_formed"] is False


def test_run_summary_field_formed_exactly_51_consecutive():
    # 51 consecutive epochs — just over the threshold, must be True
    epoch_series = [{"field_std": 0.1}] * 51
    summary = compute_run_summary(epoch_series)
    assert summary["field_formed"] is True
