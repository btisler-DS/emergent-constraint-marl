"""Gumbel-Softmax temperature schedule for Protocol 1: Interrogative Emergence.

The temperature τ controls how "hard" the signal type decision is:
- τ = 1.0: Soft, exploratory — all three types equally sampled
- τ = 0.1: Hard, committed — one type strongly dominates

The warmup period keeps τ high through the expected coordination cliff
(~epoch 21) so we can observe whether the type distribution shifts at
the same phase boundary as the original entropy cliff.

A drop in type_entropy (logged per epoch) marks the "Pragmatic Phase
Transition" — the moment the system commits to a type hierarchy.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def get_tau(
    epoch: int,
    tau_max: float = 1.0,
    tau_min: float = 0.1,
    warmup_epochs: int = 20,
    decay_epochs: int = 60,
) -> float:
    """Exponential temperature decay with warmup period.

    Schedule:
        Epochs 0-19:  τ = 1.00  (warmup — pure exploration)
        Epoch  20:    τ = 1.00  (decay begins)
        Epoch  40:    τ ≈ 0.43
        Epoch  60:    τ ≈ 0.22
        Epoch  80:    τ ≈ 0.15
        Epoch  100+:  τ = 0.10  (floor — near-hard commitment)
    """
    if epoch < warmup_epochs:
        return tau_max
    progress = min((epoch - warmup_epochs) / decay_epochs, 1.0)
    tau = tau_min + (tau_max - tau_min) * math.exp(-3.0 * progress)
    return max(tau, tau_min)


def sample_signal_type(
    type_logits: torch.Tensor,
    tau: float,
    training: bool = True,
) -> tuple[torch.Tensor, int]:
    """Sample a signal type via Gumbel-Softmax.

    Uses Gumbel-Softmax during training for differentiable type selection.
    Uses plain softmax (argmax) during evaluation.

    Args:
        type_logits: Raw logits of shape (3,) from the agent's type_head.
        tau:         Current temperature from get_tau().
        training:    Whether the agent is in training mode.

    Returns:
        soft_type:  Soft one-hot vector (shape (3,)) — carries gradients.
        hard_type:  Integer label 0/1/2 — used for cost assignment and logging.
                    0 = DECLARATIVE, 1 = INTERROGATIVE, 2 = RESPONSE
    """
    if training:
        soft = F.gumbel_softmax(type_logits, tau=tau, hard=False)
    else:
        soft = F.softmax(type_logits, dim=-1)

    hard_type = int(soft.argmax().item())
    return soft, hard_type
