"""Agent A: RNN-based sequential processor.

Perceives a 1D sequence (distances to entities).
Emits signals via Normal distribution (head defined in BaseAgent).

Depth support:
  depth=0 — feedforward baseline: obs_encoder only, no GRU, no signal encoder.
  depth=1 — full RNN: obs_encoder + signal_encoder + GRUCell (default).
  depth=2 — depth=1 + self_model_gru in parallel with primary GRU.

Observation vector: 12-dim (post obs-extension for token distance):
  [dist_target, dist_obs1..8, dist_agent_B, dist_agent_C, dist_token]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base_agent import BaseAgent


class AgentA(BaseAgent):
    """RNN agent processing 1D sequential observations."""

    def __init__(
        self,
        obs_dim: int,
        signal_dim: int = 8,
        hidden_dim: int = 64,
        num_incoming: int = 2,
        depth: int = 1,
        ablate_self_model_inputs: bool = False,
        freeze_self_model_gru: bool = False,
    ):
        super().__init__(name="A", signal_dim=signal_dim, hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self._ablate = ablate_self_model_inputs

        # Observation encoder — present at all depths
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)

        if depth >= 1:
            # Depth 1+: incoming signal encoder and primary GRU
            self.signal_encoder = nn.Linear(signal_dim * num_incoming, hidden_dim)
            self.rnn = nn.GRUCell(hidden_dim * 2, hidden_dim)

        if depth >= 2:
            # Depth 2: self_model_gru (parallel to primary GRU)
            # Inputs: prev_type_logits (3-dim) + energy_delta (1 scalar) = 4-dim
            # Output: self_state (64-dim) projected and added to primary hidden state
            self.self_model_gru = nn.GRUCell(4, hidden_dim)
            self.self_model_proj = nn.Linear(hidden_dim, hidden_dim)
            # Boundary condition: capacity present, no learning — freeze at random init
            if freeze_self_model_gru:
                for p in self.self_model_gru.parameters():
                    p.requires_grad_(False)
                for p in self.self_model_proj.parameters():
                    p.requires_grad_(False)

        # Hidden states (reset each episode)
        self.hidden = None
        self._self_state = None
        self._prev_type_logits = None
        self._energy_delta: float = 0.0

    def set_energy_delta(self, delta: float) -> None:
        """Called by engine after each env.step() with this agent's energy delta.

        Value is consumed by self_model_gru at depth=2 on the NEXT forward pass.
        """
        self._energy_delta = delta

    def reset_hidden(self) -> None:
        self.hidden = None
        self._self_state = None
        self._prev_type_logits = None
        self._energy_delta = 0.0

    def encode(
        self,
        observation: torch.Tensor,
        incoming_signals: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observation. Returns hidden state (or obs_enc at depth=0) as trunk."""
        obs_enc = torch.relu(self.obs_encoder(observation))

        if self.depth == 0:
            return obs_enc

        # Depth 1+: signal-fused GRU
        sig_enc = torch.relu(self.signal_encoder(incoming_signals))
        combined = torch.cat([obs_enc, sig_enc], dim=-1)

        if self.hidden is None:
            self.hidden = torch.zeros(self.hidden_dim, device=next(self.parameters()).device)

        self.hidden = self.rnn(combined.unsqueeze(0), self.hidden.unsqueeze(0)).squeeze(0)
        h = self.hidden

        if self.depth >= 2:
            dev = next(self.parameters()).device

            if self._ablate:
                sm_input = torch.zeros(4, device=dev)
            else:
                if self._prev_type_logits is None:
                    type_logits_prev = torch.zeros(3, device=dev)
                else:
                    type_logits_prev = self._prev_type_logits.to(dev)
                energy_delta_t = torch.tensor(
                    [self._energy_delta], dtype=torch.float32, device=dev
                )
                sm_input = torch.cat([type_logits_prev, energy_delta_t])  # (4,)

            if self._self_state is None:
                self._self_state = torch.zeros(self.hidden_dim, device=dev)

            self._self_state = self.self_model_gru(
                sm_input.unsqueeze(0), self._self_state.unsqueeze(0)
            ).squeeze(0)

            sm_proj = self.self_model_proj(self._self_state)
            h = h + sm_proj  # element-wise addition before value and action heads

        return h

    def forward(
        self,
        observation: torch.Tensor,
        incoming_signals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encode() then apply output heads. At depth=2 also stores prev type_logits."""
        signal, action_logits, type_logits = super().forward(observation, incoming_signals)
        if self.depth >= 2:
            # Store for next step's self_model_gru input (detached — no grad through time)
            self._prev_type_logits = type_logits.detach()
        return signal, action_logits, type_logits

    def clear_episode(self) -> None:
        super().clear_episode()
        self.hidden = None
        self._self_state = None
        self._prev_type_logits = None
        self._energy_delta = 0.0
