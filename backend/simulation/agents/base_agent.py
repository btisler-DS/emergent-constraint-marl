"""Abstract base agent interface for the MARL simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Normal


class BaseAgent(ABC, nn.Module):
    """Base class for all agents in the simulation.

    Centralises the three output heads (signal_mu / signal_log_std / action_head)
    and the type_head shared across all protocols. Subclasses implement only
    encode() — the observation+signal fusion trunk.
    """

    def __init__(self, name: str, signal_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.name = name
        self.signal_dim = signal_dim
        self.hidden_dim = hidden_dim
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []

        # Shared output heads — defined here for checkpoint key compatibility.
        # Moving these from subclasses preserves state_dict key names.
        self.signal_mu = nn.Linear(hidden_dim, signal_dim)
        self.signal_log_std = nn.Parameter(torch.zeros(signal_dim))
        self.action_head = nn.Linear(hidden_dim, 5)

        # Protocol 1: Signal type head — shared structure across all agents.
        # Taps the post-fusion trunk (hidden_dim,) and outputs 3-way logits.
        # DECLARATIVE=0, INTERROGATIVE=1, RESPONSE=2
        self.type_head = nn.Linear(hidden_dim, 3)

    @abstractmethod
    def encode(
        self,
        observation: torch.Tensor,
        incoming_signals: torch.Tensor,
    ) -> torch.Tensor:
        """Process observation and incoming signals into a trunk vector.

        Returns:
            trunk: Tensor of shape (hidden_dim,) — the post-fusion representation.
        """
        ...

    def forward(
        self,
        observation: torch.Tensor,
        incoming_signals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encode() then apply all three output heads.

        Returns:
            signal:        Tensor of shape (signal_dim,) to broadcast.
            action_logits: Tensor of shape (5,) — action distribution parameters.
            type_logits:   Tensor of shape (3,) — signal type head raw logits.
        """
        trunk = self.encode(observation, incoming_signals)
        mu = self.signal_mu(trunk)
        std = torch.exp(self.signal_log_std)
        signal = Normal(mu, std).rsample()
        action_logits = self.action_head(trunk)
        type_logits = self.type_head(trunk)
        return signal, action_logits, type_logits

    def freeze_type_head(self) -> None:
        """Freeze type_head parameters (used by Protocol 0)."""
        for p in self.type_head.parameters():
            p.requires_grad_(False)

    def store_outcome(self, log_prob: torch.Tensor, reward: float) -> None:
        """Store log probability and reward for REINFORCE update."""
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def clear_episode(self) -> None:
        """Clear stored log probs and rewards."""
        self.log_probs = []
        self.rewards = []

    def get_signal_dim(self) -> int:
        return self.signal_dim
