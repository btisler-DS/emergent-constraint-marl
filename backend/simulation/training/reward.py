"""Landauer-inspired reward function.

The communication_tax_rate is a tunable hyperparameter (Commandment #2):
finding the "Critical Tax Rate" where the cost of silence exceeds
the cost of coordination.

Protocol 1 Extension: Signal type differential costs.
Each signal type carries a cost multiplier relative to the base tax rate:
    DECLARATIVE   (0): 1.0x — baseline broadcast cost
    INTERROGATIVE (1): 1.5x — premium: asking is expensive
    RESPONSE      (2): 0.8x — discount: answering is the cheapest signal

The RESPONSE discount is the metabolic "lubricant" — it incentivizes agents
to answer queries, which is the precondition for the interrogative calculus
to close the coordination loop.
"""

from __future__ import annotations

import torch
import numpy as np


# Protocol 1: Signal type cost multipliers.
# Defaults to 0 (DECLARATIVE) for backward compatibility.
SIGNAL_TYPE_COSTS: dict[int, float] = {
    0: 1.0,   # DECLARATIVE:   baseline broadcast
    1: 1.5,   # INTERROGATIVE: premium — asking costs more
    2: 0.8,   # RESPONSE:      discount — answering is cheap
}


def compute_reward(
    agent_name: str,
    env_reward: float,
    signal_sent: torch.Tensor,
    energy_remaining: float,
    energy_budget: float,
    communication_tax_rate: float = 0.01,
    reached_target: bool = False,
    survival_bonus: float = 0.1,
    signal_type: int = 0,
    declare_cost: float = 1.0,
    query_cost: float = 1.5,
    respond_cost: float = 0.8,
) -> float:
    """Compute Landauer-inspired reward with communication tax.

    Components:
    - Base environment reward (distance to target)
    - Communication tax: penalizes signal magnitude × type multiplier
    - Survival bonus: small reward for staying alive
    - Energy efficiency: scales with remaining energy fraction

    Args:
        signal_type: Protocol 1 signal type (0=DECLARE, 1=QUERY, 2=RESPOND).
                     Defaults to 0 (DECLARATIVE) — backward compatible with Run 10.
        declare_cost: Cost multiplier for DECLARE signals (default 1.0).
        query_cost:   Cost multiplier for QUERY signals (default 1.5).
        respond_cost: Cost multiplier for RESPOND signals (default 0.8).

    The communication_tax_rate is the key tunable parameter.
    Too low → agents yell forever. Too high → agents go mute.
    """
    # Base reward from environment
    reward = env_reward

    # Communication tax with per-condition type multiplier
    type_costs = {0: declare_cost, 1: query_cost, 2: respond_cost}
    type_multiplier = type_costs.get(signal_type, 1.0)
    signal_cost = communication_tax_rate * type_multiplier * signal_sent.abs().sum().item()
    reward -= signal_cost

    # Survival bonus (encourages energy conservation)
    energy_fraction = max(energy_remaining / energy_budget, 0.0)
    reward += survival_bonus * energy_fraction

    # Target bonus already included in env_reward

    return reward
