"""Agent C: Cross-modal attention (GNN-inspired) agent.

Perceives pairwise relational features between all entities (including token).
Uses attention over incoming signals — the "translator" agent.
Emits signals and actions from heads defined in BaseAgent.

Obs extension: pairwise entity set gains token as an additional entity.
  Before: 12 entities → 66 pairwise distances (obs_dim=66)
  After:  13 entities → 78 pairwise distances (obs_dim=78)
  The 12 new features are appended: (token, target), (token, obs1..8),
  (token, A), (token, B), (token, C). Set to 0.0 when token is inactive.

Depth support:
  depth=0 — feedforward baseline: obs_encoder (2-layer MLP) only, no attention,
            no signal encoder, no fusion. Agent-appropriate relational baseline.
  depth=1 — full pipeline: obs_encoder + cross-modal attention + fusion (default).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base_agent import BaseAgent


class AgentC(BaseAgent):
    """Cross-modal attention agent processing relational features."""

    def __init__(
        self,
        obs_dim: int,
        signal_dim: int = 8,
        hidden_dim: int = 64,
        num_incoming: int = 2,
        depth: int = 1,
    ):
        super().__init__(name="C", signal_dim=signal_dim, hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_incoming = num_incoming
        self.depth = depth

        # Relational feature encoder (2-layer MLP — agent-appropriate geometry)
        # Present at all depths.
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if depth >= 1:
            # Depth 1+: cross-modal attention over incoming signals
            self.signal_key = nn.Linear(signal_dim, hidden_dim)
            self.signal_value = nn.Linear(signal_dim, hidden_dim)
            self.obs_query = nn.Linear(hidden_dim, hidden_dim)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            )

    def encode(
        self,
        observation: torch.Tensor,
        incoming_signals: torch.Tensor,
    ) -> torch.Tensor:
        """Encode relational observation. At depth=0 returns MLP output directly.

        At depth=1+, attention-weighted context from both other agents' signals
        is fused with the relational encoding before the output heads.
        """
        obs_enc = self.obs_encoder(observation)

        if self.depth == 0:
            return obs_enc

        # Depth 1+: cross-modal attention
        signals = incoming_signals.view(self.num_incoming, self.signal_dim)

        query = self.obs_query(obs_enc).unsqueeze(0)  # (1, hidden)
        keys = self.signal_key(signals)               # (num_incoming, hidden)
        values = self.signal_value(signals)           # (num_incoming, hidden)

        scale = self.hidden_dim ** 0.5
        attn_weights = torch.softmax(
            (query @ keys.T) / scale, dim=-1
        )  # (1, num_incoming)
        attended = (attn_weights @ values).squeeze(0)  # (hidden,)

        return self.fusion(torch.cat([obs_enc, attended], dim=-1))
