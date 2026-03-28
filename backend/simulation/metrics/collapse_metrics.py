"""Interrogative collapse metrics for Protocol 2: Ethical Constraints.

Measures whether agents maintain sustained interrogative structure over time or
collapse into exploitation loops that suppress novel question generation.

Functions
---------
interrogative_collapse_rate  — detect collapse in QUERY rate below threshold
exploitation_loop_detection  — detect exploitation via low target-selection entropy
sustained_structure_score    — composite metric: type entropy × QRC over final window
constrained_vs_unconstrained_divergence — Cohen's d + trajectory divergence
"""

from __future__ import annotations

import math
import statistics
from typing import Any


def interrogative_collapse_rate(
    query_rates_over_time: list[float],
    window: int = 10,
    collapse_threshold: float = 0.05,
) -> dict[str, Any]:
    """Detect when QUERY rate drops below threshold and stays there.

    Collapse is declared when the QUERY rate falls below `collapse_threshold`
    and remains there for `window` consecutive epochs.

    Parameters
    ----------
    query_rates_over_time:
        Per-epoch QUERY signal fraction (0.0–1.0), ordered chronologically.
    window:
        Number of consecutive below-threshold epochs required to confirm collapse.
    collapse_threshold:
        QUERY rate below which an epoch is considered 'collapsed'.

    Returns
    -------
    dict with keys:
        collapse_detected  — bool, True if sustained collapse was found
        epoch_of_collapse  — int or None, first epoch of the confirmed collapse window
        collapse_sustained — bool, True if collapse persists to end of series
        area_under_query_curve — float, trapezoidal AUC (proxy for total interrogative activity)
        collapse_speed     — float, slope of QUERY rate in final window (negative = declining)
    """
    n = len(query_rates_over_time)
    if n == 0:
        return {
            "collapse_detected": False,
            "epoch_of_collapse": None,
            "collapse_sustained": False,
            "area_under_query_curve": 0.0,
            "collapse_speed": 0.0,
        }

    # AUC via trapezoidal rule
    auc = sum(
        (query_rates_over_time[i] + query_rates_over_time[i + 1]) / 2.0
        for i in range(n - 1)
    )

    # Find first epoch of a sustained collapse window
    epoch_of_collapse: int | None = None
    streak = 0
    streak_start: int | None = None
    for i, rate in enumerate(query_rates_over_time):
        if rate < collapse_threshold:
            if streak == 0:
                streak_start = i
            streak += 1
            if streak >= window and epoch_of_collapse is None:
                epoch_of_collapse = streak_start
        else:
            streak = 0
            streak_start = None

    collapse_detected = epoch_of_collapse is not None
    # Sustained if collapse carries through to end of series
    collapse_sustained = (
        collapse_detected
        and all(r < collapse_threshold for r in query_rates_over_time[epoch_of_collapse:])
    )

    # Slope of QUERY rate over final window (negative = declining)
    tail = query_rates_over_time[-window:] if n >= window else query_rates_over_time
    if len(tail) >= 2:
        xs = list(range(len(tail)))
        x_mean = sum(xs) / len(xs)
        y_mean = sum(tail) / len(tail)
        numerator = sum((xs[i] - x_mean) * (tail[i] - y_mean) for i in range(len(tail)))
        denominator = sum((x - x_mean) ** 2 for x in xs)
        collapse_speed = numerator / denominator if denominator > 0 else 0.0
    else:
        collapse_speed = 0.0

    return {
        "collapse_detected": collapse_detected,
        "epoch_of_collapse": epoch_of_collapse,
        "collapse_sustained": collapse_sustained,
        "area_under_query_curve": round(auc, 6),
        "collapse_speed": round(collapse_speed, 6),
    }


def exploitation_loop_detection(
    target_selections: list[int],
    window: int = 10,
    entropy_threshold: float = 0.3,
) -> dict[str, Any]:
    """Detect exploitation loops via low Shannon entropy of target selection.

    An exploitation loop is active when agents repeatedly select the same
    resource (low diversity = low entropy) over a sliding window of steps.

    Parameters
    ----------
    target_selections:
        Sequence of target IDs chosen per step (or agent-action identifiers).
    window:
        Sliding window size for entropy computation.
    entropy_threshold:
        Shannon entropy below which behaviour is considered an exploitation loop.

    Returns
    -------
    dict with keys:
        loop_detected   — bool
        loop_start_epoch — int or None, index of first window below threshold
        loop_entropy    — float, minimum entropy observed across all windows
        diversity_score — float, mean entropy across all windows (0 = no diversity)
    """
    n = len(target_selections)
    if n < 2:
        return {
            "loop_detected": False,
            "loop_start_epoch": None,
            "loop_entropy": 0.0,
            "diversity_score": 0.0,
        }

    def _window_entropy(seq: list[int]) -> float:
        counts: dict[int, int] = {}
        for x in seq:
            counts[x] = counts.get(x, 0) + 1
        total = len(seq)
        return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

    entropies: list[float] = []
    loop_start_epoch: int | None = None

    for i in range(n - window + 1):
        w = target_selections[i : i + window]
        h = _window_entropy(w)
        entropies.append(h)
        if h < entropy_threshold and loop_start_epoch is None:
            loop_start_epoch = i

    loop_entropy = min(entropies) if entropies else 0.0
    diversity_score = sum(entropies) / len(entropies) if entropies else 0.0

    return {
        "loop_detected": loop_start_epoch is not None,
        "loop_start_epoch": loop_start_epoch,
        "loop_entropy": round(loop_entropy, 6),
        "diversity_score": round(diversity_score, 6),
    }


def sustained_structure_score(
    type_entropy_series: list[float],
    qrc_series: list[float],
    window: int = 20,
) -> float:
    """Composite metric: mean(type_entropy) × mean(QRC) over the final window epochs.

    High score = agents maintain both signal-type diversity and query-response
    coupling — i.e., sustained interrogative structure. A score near zero indicates
    collapse (either entropy or coupling dropped to zero).

    Parameters
    ----------
    type_entropy_series:
        Shannon entropy of signal-type distribution per epoch.
    qrc_series:
        Query-Response Coupling (P(RESPONSE within 3 steps | QUERY)) per epoch.
    window:
        Number of final epochs to average over.

    Returns
    -------
    float in [0, inf). 0 = fully collapsed, higher = more sustained.
    """
    if not type_entropy_series or not qrc_series:
        return 0.0

    te_tail = type_entropy_series[-window:]
    qrc_tail = qrc_series[-window:]

    mean_te = sum(te_tail) / len(te_tail)
    mean_qrc = sum(qrc_tail) / len(qrc_tail)

    return round(mean_te * mean_qrc, 6)


def constrained_vs_unconstrained_divergence(
    constrained_metrics: dict[str, Any],
    unconstrained_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Cohen's d and trajectory divergence between constrained and unconstrained conditions.

    Compares the two conditions on three key metrics:
        query_rate              — mean QUERY signal fraction (H1 proxy)
        sustained_structure_score — composite entropy × QRC (H2 metric)
        exploitation_loop_rate  — fraction of epochs in exploitation (H3 basis)

    Cohen's d > 0.8 is the H3 preregistered threshold for 'categorical difference'.

    Parameters
    ----------
    constrained_metrics:
        Dict with keys 'query_rates', 'structure_scores', 'loop_rates' (each a list[float]).
    unconstrained_metrics:
        Same structure as constrained_metrics.

    Returns
    -------
    dict with Cohen's d and mean difference for each metric, plus overall effect.
    """

    def _cohens_d(a: list[float], b: list[float]) -> float | None:
        if len(a) < 2 or len(b) < 2:
            return None
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        var_a = statistics.variance(a)
        var_b = statistics.variance(b)
        pooled_sd = math.sqrt((var_a + var_b) / 2.0)
        if pooled_sd == 0.0:
            return None
        return round((mean_a - mean_b) / pooled_sd, 4)

    def _mean_diff(a: list[float], b: list[float]) -> float | None:
        if not a or not b:
            return None
        return round(sum(a) / len(a) - sum(b) / len(b), 6)

    results = {}
    for metric_key in ("query_rates", "structure_scores", "loop_rates"):
        c_vals = constrained_metrics.get(metric_key, [])
        u_vals = unconstrained_metrics.get(metric_key, [])
        results[metric_key] = {
            "cohens_d": _cohens_d(c_vals, u_vals),
            "mean_diff_constrained_minus_unconstrained": _mean_diff(c_vals, u_vals),
            "mean_constrained": round(sum(c_vals) / len(c_vals), 6) if c_vals else None,
            "mean_unconstrained": round(sum(u_vals) / len(u_vals), 6) if u_vals else None,
        }

    # Summarise: is H3 threshold (d > 0.8) met for structure_scores?
    structure_d = results.get("structure_scores", {}).get("cohens_d")
    results["h3_threshold_met"] = (structure_d is not None and abs(structure_d) > 0.8)

    return results
