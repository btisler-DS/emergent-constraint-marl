"""Field diagnostic functions for Protocol 6 pilot.

Per-epoch diagnostics:
  field_mean, field_std, field_max, field_min, field_entropy

Per-run summary booleans (from spec Section 5.3):
  field_formed     -- field_std > 0.05 sustained for > 50 consecutive epochs
  field_saturated  -- field_max > 0.95 at any point
  field_collapsed  -- field_mean < 0.01 after epoch 50
"""

from __future__ import annotations

import numpy as np


def compute_field_diagnostics(F: np.ndarray) -> dict:
    """Compute per-epoch field statistics from field vector F.

    Parameters
    ----------
    F:
        Field vector of shape (n_agents,), values >= 0.

    Returns
    -------
    dict with keys: field_mean, field_std, field_max, field_min, field_entropy
    """
    field_mean = float(np.mean(F))
    field_std = float(np.std(F))
    field_max = float(np.max(F))
    field_min = float(np.min(F))

    # Shannon entropy of F normalized to sum-to-one.
    # Fallback to uniform distribution when all values are zero.
    total = float(F.sum())
    if total > 0:
        probs = F / total
    else:
        probs = np.ones(len(F), dtype=np.float32) / len(F)

    probs = np.clip(probs, 1e-10, 1.0)
    field_entropy = float(-np.sum(probs * np.log(probs)))

    return {
        "field_mean": field_mean,
        "field_std": field_std,
        "field_max": field_max,
        "field_min": field_min,
        "field_entropy": field_entropy,
    }


def compute_run_summary(epoch_series: list[dict]) -> dict:
    """Compute per-run summary booleans from epoch diagnostic series.

    Parameters
    ----------
    epoch_series:
        List of per-epoch dicts, each containing at minimum:
        field_std, field_max, field_mean.

    Returns
    -------
    dict with keys: field_formed, field_saturated, field_collapsed
    """
    # field_formed: field_std > 0.05 sustained for > 50 consecutive epochs
    sustained = 0
    max_sustained = 0
    for ep in epoch_series:
        if ep.get("field_std", 0.0) > 0.05:
            sustained += 1
            max_sustained = max(max_sustained, sustained)
        else:
            sustained = 0
    field_formed = max_sustained > 50

    # field_saturated: field_max > 0.95 at any point
    field_saturated = any(ep.get("field_max", 0.0) > 0.95 for ep in epoch_series)

    # field_collapsed: field_mean < 0.01 after epoch 50 (index 50+)
    field_collapsed = any(
        ep.get("field_mean", 1.0) < 0.01
        for ep in epoch_series[50:]
    )

    return {
        "field_formed": field_formed,
        "field_saturated": field_saturated,
        "field_collapsed": field_collapsed,
    }
