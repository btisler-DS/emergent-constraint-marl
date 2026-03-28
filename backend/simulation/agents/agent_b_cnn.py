"""Agent B: Conv3D spatial processor.

Perceives a 2-channel 3D volumetric density map:
  channel 0 — density map (Volumetric Echo, obstacles, target gradient)
  channel 1 — token presence map (binary, 1.0 at token position when active)

Emits signals via Normal distribution (head defined in BaseAgent).

Bug Fix #6: Uses AdaptiveAvgPool3d(4) instead of hardcoded grid=10 Linear,
making the agent size-independent.

Obs extension: Conv3d input channels 1→2 to accommodate token presence channel.

Depth support:
  depth=0 — volumetric baseline: conv3d + spatial_encoder only, no signal encoder,
            no fusion. Agent-appropriate feedforward baseline.
  depth=1 — full pipeline: conv3d + spatial_encoder + signal_encoder + fusion (default).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base_agent import BaseAgent


class AgentB(BaseAgent):
    """CNN agent processing 3D volumetric observations."""

    def __init__(
        self,
        signal_dim: int = 8,
        hidden_dim: int = 64,
        num_incoming: int = 2,
        depth: int = 1,
    ):
        super().__init__(name="B", signal_dim=signal_dim, hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.depth = depth

        # 3D convolution for volumetric input (2-channel: density + token presence)
        self.conv3d = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Bug Fix #6: AdaptiveAvgPool3d for size independence
            nn.AdaptiveAvgPool3d(4),
        )

        # Flatten 32 * 4 * 4 * 4 = 2048
        self.spatial_encoder = nn.Linear(32 * 4 * 4 * 4, hidden_dim)

        if depth >= 1:
            # Depth 1+: incoming signal encoder and fusion
            self.signal_encoder = nn.Linear(signal_dim * num_incoming, hidden_dim)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            )

    def encode(
        self,
        observation: torch.Tensor,
        incoming_signals: torch.Tensor,
    ) -> torch.Tensor:
        """Encode 2-channel volumetric observation. Returns spatial_enc or fused trunk."""
        # observation shape: (2, Z, H, W)
        if observation.dim() == 4:
            observation = observation.unsqueeze(0)  # add batch dim

        spatial = self.conv3d(observation)
        spatial = spatial.view(spatial.size(0), -1)
        spatial_enc = torch.relu(self.spatial_encoder(spatial)).squeeze(0)

        if self.depth == 0:
            return spatial_enc

        # Depth 1+: fuse spatial with incoming signals
        sig_enc = torch.relu(self.signal_encoder(incoming_signals))
        return self.fusion(torch.cat([spatial_enc, sig_enc], dim=-1))
