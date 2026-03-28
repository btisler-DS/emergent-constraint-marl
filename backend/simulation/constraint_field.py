"""Emergent constraint field for Protocol 6.

The field F is a shared vector of length N_agents. Each timestep:
  1. Each agent emits a signal-type-weighted increment to its own cell.
  2. The field diffuses symmetrically to ring-topology neighbors.
  3. The field decays uniformly.

The field lives in the environment (owned by P6SimulationEngine).
Each agent perceives only its own cell F[i] — local perception only.
"""

from __future__ import annotations

import numpy as np

# Signal type indices matching Protocol 1/2 convention
# 0 = DECLARE, 1 = QUERY (INTERROGATIVE), 2 = RESPOND
#
# Deviation 1 (2026-03-27): Scaled 0.1x from preregistered values
# (DECLARE: 0.3->0.03, QUERY: 0.1->0.01, RESPOND: 0.2->0.02).
# Original values produced F_eq = 2-6x above saturation threshold across all
# pilot parameter combinations. Scaled values keep F below saturation while
# preserving the relative weight structure. See preregistration Section 12.
_SIGNAL_WEIGHTS: dict[int, float] = {
    0: 0.03,  # DECLARE — assertion, high cost
    1: 0.01,  # QUERY   — question, low cost
    2: 0.02,  # RESPOND — reply, medium cost
}

_AGENT_INDEX: dict[str, int] = {"A": 0, "B": 1, "C": 2}


class ConstraintField:
    """Shared emergent constraint field for three-agent systems.

    Parameters
    ----------
    n_agents:
        Number of agents (must be 3 for Protocol 6 Condition A).
    diffusion_coefficient:
        Fraction of each cell's value distributed to neighbors each step.
        Range [0, 1]. Higher = more spatially homogeneous field.
    decay_rate:
        Fractional decay applied uniformly each step. Range [0, 1].
        Higher = faster decay, more transient field.
    """

    def __init__(
        self,
        n_agents: int,
        diffusion_coefficient: float,
        decay_rate: float,
    ) -> None:
        self.n = n_agents
        self.diffusion_coefficient = diffusion_coefficient
        self.decay_rate = decay_rate
        self.F = np.zeros(n_agents, dtype=np.float32)

    def reset(self) -> None:
        """Reset field to zero (call at episode start)."""
        self.F[:] = 0.0

    def _diffuse(self) -> None:
        """Symmetric ring diffusion: distribute dc fraction equally to neighbors."""
        dc = self.diffusion_coefficient
        if dc == 0.0:
            return
        new_F = self.F.copy()
        for i in range(self.n):
            outflow = dc * self.F[i]
            new_F[i] -= outflow
            new_F[(i - 1) % self.n] += outflow / 2.0
            new_F[(i + 1) % self.n] += outflow / 2.0
        self.F = new_F

    def update(self, signal_types: dict[str, int]) -> None:
        """Apply one timestep: emit -> diffuse -> decay.

        Parameters
        ----------
        signal_types:
            Map of agent name -> signal type int (0=DECLARE, 1=QUERY, 2=RESPOND).
        """
        # Step 1: Emission — each agent increments its own cell
        for name, stype in signal_types.items():
            idx = _AGENT_INDEX[name]
            self.F[idx] += _SIGNAL_WEIGHTS[stype]

        # Step 2: Symmetric diffusion
        self._diffuse()

        # Step 3: Uniform decay
        self.F *= (1.0 - self.decay_rate)

    def get(self, agent_name: str) -> float:
        """Return local field value for agent (local perception only)."""
        return float(self.F[_AGENT_INDEX[agent_name]])

    def effective_cost(
        self,
        agent_name: str,
        base_cost: float,
        sensitivity: float = 1.0,
    ) -> float:
        """Compute field-modulated signal cost for an agent.

        effective = base_cost * (1 + sensitivity * F[i])
        """
        return base_cost * (1.0 + sensitivity * self.get(agent_name))
