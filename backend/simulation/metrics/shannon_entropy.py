"""Shannon entropy computation.

Bug Fix #3: Normalizes raw histogram counts to probabilities
before computing entropy, preventing incorrect entropy values.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy


def compute_signal_entropy(signals: torch.Tensor, num_bins: int = 20) -> float:
    """Compute Shannon entropy of signal distribution.

    Bug Fix #3: Converts histogram counts to probability distribution
    before calling scipy.stats.entropy.
    """
    values = signals.detach().cpu().numpy().flatten()
    if len(values) == 0:
        return 0.0

    counts, _ = np.histogram(values, bins=num_bins)

    # Bug Fix #3: normalize to probabilities
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts / total

    return float(scipy_entropy(probabilities, base=2))


def compute_per_agent_entropy(
    signal_history: list[dict[str, torch.Tensor]],
    num_bins: int = 20,
) -> dict[str, float]:
    """Compute entropy for each agent's signal history."""
    agent_signals: dict[str, list[torch.Tensor]] = {}

    for snapshot in signal_history:
        for name, signal in snapshot.items():
            if name not in agent_signals:
                agent_signals[name] = []
            agent_signals[name].append(signal)

    result = {}
    for name, sigs in agent_signals.items():
        if sigs:
            all_signals = torch.cat(sigs)
            result[name] = compute_signal_entropy(all_signals, num_bins)
        else:
            result[name] = 0.0

    return result
