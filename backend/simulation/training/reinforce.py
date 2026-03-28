"""REINFORCE policy gradient training.

Bug Fix #1: Loss uses `torch.stack(losses).sum()` instead of starting
from `int 0`, which would break autograd.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from ..agents.base_agent import BaseAgent


def compute_reinforce_loss(agent: BaseAgent, gamma: float = 0.99) -> torch.Tensor:
    """Compute REINFORCE loss for a single agent.

    Bug Fix #1: Uses torch.stack to properly accumulate tensor losses
    instead of summing into a Python int.
    """
    if not agent.log_probs or not agent.rewards:
        device = next(agent.parameters()).device
        return torch.tensor(0.0, requires_grad=True, device=device)

    # Compute discounted returns
    returns = []
    G = 0.0
    for r in reversed(agent.rewards):
        G = r + gamma * G
        returns.insert(0, G)

    device = agent.log_probs[0].device
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    # Normalize returns for stability
    if len(returns_t) > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    # Bug Fix #1: torch.stack instead of int accumulation
    losses = []
    for log_prob, G_t in zip(agent.log_probs, returns_t):
        losses.append(-log_prob * G_t)

    return torch.stack(losses).sum()


def update_agents(
    agents: list[BaseAgent],
    optimizers: list[optim.Optimizer],
    gamma: float = 0.99,
) -> dict[str, float]:
    """Run one REINFORCE update for all agents.

    Returns dict of agent_name -> loss_value.
    """
    loss_values = {}

    for agent, optimizer in zip(agents, optimizers):
        loss = compute_reinforce_loss(agent, gamma)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()

        loss_values[agent.name] = loss.item()
        agent.clear_episode()

    return loss_values
