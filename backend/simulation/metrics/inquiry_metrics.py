"""Protocol 1 inquiry metrics: measuring the energetics of interrogative signals.

These metrics quantify whether the signal type taxonomy (DECLARATIVE /
INTERROGATIVE / RESPONSE) is doing thermodynamic work — i.e., whether
the cost asymmetry is driving meaningful query-response behavior.

Key metrics:
    type_entropy:           Shannon entropy of the type distribution.
                            Drops sharply at the "Pragmatic Phase Transition."
    inquiry_roi:            Coordination gain per query joule spent.
    query_response_coupling: P(RESPONSE at T+1 | QUERY at T) — the temporal
                            correlation that proves the interrogative calculus.
"""

from __future__ import annotations

import numpy as np
import torch


# Signal type constants (mirrors reward.py SIGNAL_TYPE_COSTS keys)
DECLARATIVE   = 0
INTERROGATIVE = 1
RESPONSE      = 2

QUERY_COST_MULTIPLIER = 1.5  # matches reward.py


def compute_inquiry_metrics(
    type_history: list[dict[str, int]],
    signal_history: list[dict[str, torch.Tensor]],
    target_reached_rate: float,
    communication_tax_rate: float,
) -> dict:
    """Compute Protocol 1 inquiry metrics from per-step type and signal logs.

    Args:
        type_history:         List of per-step dicts mapping agent name → signal_type int.
        signal_history:       List of per-step dicts mapping agent name → signal tensor.
                              Must be the same length as type_history.
        target_reached_rate:  Fraction of episodes where target was reached this epoch.
        communication_tax_rate: The base tax rate from SimulationConfig.

    Returns:
        Dict of inquiry metrics ready to merge into the epoch metrics dict.
    """
    if not type_history:
        return _empty_metrics()

    # --- Type counts per agent ---
    agents = ['A', 'B', 'C']
    type_counts = {a: {DECLARATIVE: 0, INTERROGATIVE: 0, RESPONSE: 0} for a in agents}
    for step in type_history:
        for agent, stype in step.items():
            if agent in type_counts and stype in type_counts[agent]:
                type_counts[agent][stype] += 1

    total = sum(sum(c.values()) for c in type_counts.values())
    total = max(total, 1)

    declare_frac  = sum(c[DECLARATIVE]   for c in type_counts.values()) / total
    query_frac    = sum(c[INTERROGATIVE] for c in type_counts.values()) / total
    response_frac = sum(c[RESPONSE]      for c in type_counts.values()) / total

    # --- Type entropy (measures crystallization) ---
    probs = np.array([declare_frac, query_frac, response_frac]) + 1e-10
    type_entropy = float(-np.sum(probs * np.log(probs)))

    # --- Total query energy cost across all steps ---
    query_energy = 0.0
    for i, step in enumerate(type_history):
        if i >= len(signal_history):
            break
        for agent, stype in step.items():
            if stype == INTERROGATIVE:
                sig = signal_history[i].get(agent)
                if sig is not None:
                    query_energy += (
                        communication_tax_rate
                        * QUERY_COST_MULTIPLIER
                        * sig.abs().sum().item()
                    )

    # --- Inquiry ROI: coordination gain per query joule ---
    # A rising ROI over epochs means queries are learning to earn their premium cost.
    inquiry_roi = target_reached_rate / max(query_energy, 1e-8)

    # --- Query-Response Coupling ---
    coupling = _compute_qr_coupling(type_history)

    query_count    = sum(c[INTERROGATIVE] for c in type_counts.values())
    response_count = sum(c[RESPONSE]     for c in type_counts.values())

    # Per-agent type fractions (for P4 substrate-independence analysis)
    per_agent_types: dict[str, dict] = {}
    for agent in agents:
        counts = type_counts[agent]
        agent_total = max(sum(counts.values()), 1)
        d = counts[DECLARATIVE]  / agent_total
        q = counts[INTERROGATIVE] / agent_total
        r = counts[RESPONSE]     / agent_total
        per_agent_types[agent] = {
            'DECLARE': round(d, 4),
            'QUERY':   round(q, 4),
            'RESPOND': round(r, 4),
        }

    return {
        'type_distribution': {
            'DECLARE':  round(declare_frac,  4),
            'QUERY':    round(query_frac,    4),
            'RESPOND':  round(response_frac, 4),
        },
        'per_agent_types':         per_agent_types,
        'type_entropy':            type_entropy,
        'query_energy':            float(query_energy),
        'inquiry_roi':             float(inquiry_roi),
        'query_response_coupling': float(coupling),
        'query_count':             int(query_count),
        'response_count':          int(response_count),
    }


def _compute_qr_coupling(type_history: list[dict[str, int]]) -> float:
    """P(any agent sends RESPONSE at T+1 | any agent sent QUERY at T).

    This is the temporal coupling metric. A value near 0 means queries
    are not being answered in the following step. A value rising toward 1
    means the system has discovered the query-response protocol.
    """
    query_steps = [
        i for i, step in enumerate(type_history)
        if any(t == INTERROGATIVE for t in step.values())
    ]
    if not query_steps:
        return 0.0

    followed_by_response = sum(
        1 for i in query_steps
        if i + 1 < len(type_history)
        and any(t == RESPONSE for t in type_history[i + 1].values())
    )
    return followed_by_response / len(query_steps)


def _empty_metrics() -> dict:
    """Return zeroed metrics when there is no history to analyze."""
    _zero_dist = {'DECLARE': 0.0, 'QUERY': 0.0, 'RESPOND': 0.0}
    return {
        'type_distribution': _zero_dist,
        'per_agent_types':   {'A': dict(_zero_dist), 'B': dict(_zero_dist), 'C': dict(_zero_dist)},
        'type_entropy':            0.0,
        'query_energy':            0.0,
        'inquiry_roi':             0.0,
        'query_response_coupling': 0.0,
        'query_count':             0,
        'response_count':          0,
    }
