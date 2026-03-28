"""Protocol 6 Simulation Engine — Emergent Constraint Landscape, Condition A.

Standalone engine for Protocol 6 pilot. Reuses P5 components (agents,
environment, comm buffer, REINFORCE) without inheriting SimulationEngine.

Condition A only: local perception (each agent sees own F[i]).
No sacrifice-conflict, no counter-wave, no hysteresis logic.
Protocol 2 reward structure (all_constrained) as P5 baseline.
Depth 2 (self_model_gru trainable) matching P5 confirmatory default.
welfare_coupled=False (individual reward).
Field update: after action collection, before reward computation.

Observation augmentation per agent:
  AgentA: env_obs(11) + zero_token_dist(1) + F[A](1) = 13 dims
  AgentB: env_obs(1,8,20,20) + zero_token_channel → (2,8,20,20)
  AgentC: env_obs(66) — unchanged
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from .agents import AgentA, AgentB, AgentC
from .comm_buffer import CommBuffer, CommBufferConfig
from .constraint_field import ConstraintField
from .environment import Environment, EnvironmentConfig
from .metrics.collapse_metrics import exploitation_loop_detection
from .metrics.field_diagnostics import compute_field_diagnostics
from .protocols import create_protocol
from .training.reinforce import update_agents
from .utils.seeding import set_all_seeds

logger = logging.getLogger(__name__)

# Observation dimensions
_OBS_DIM_A = 13   # env(11) + zero_token_dist(1) + F[i](1)
_OBS_DIM_C = 66   # pairwise distances, no token, unchanged


@dataclass
class P6Config:
    seed: int = 0
    num_epochs: int = 200
    episodes_per_epoch: int = 10
    # Environment
    grid_size: int = 20
    num_obstacles: int = 8
    z_layers: int = 8
    max_steps: int = 64
    energy_budget: float = 100.0
    move_cost: float = 1.0
    collision_penalty: float = 5.0
    # Agent architecture
    signal_dim: int = 8
    hidden_dim: int = 64
    depth: int = 2
    # Training
    learning_rate: float = 1e-3
    gamma: float = 0.99
    # Protocol 2 (all_constrained)
    communication_tax_rate: float = 0.01
    survival_bonus: float = 0.1
    declare_cost: float = 1.0
    query_cost: float = 1.5
    respond_cost: float = 0.8
    # P6 constraint field
    diffusion_coefficient: float = 0.1
    decay_rate: float = 0.05
    signal_cost_sensitivity: float = 1.0
    # Runtime
    device: str = "cpu"


class P6SimulationEngine:
    """Protocol 6 engine for Condition A (emergent local constraint field)."""

    def __init__(self, config: P6Config | None = None) -> None:
        self.config = config or P6Config()
        set_all_seeds(self.config.seed)
        self.device = torch.device(self.config.device)

        # Protocol 2, all_constrained
        self.protocol = create_protocol(
            2,
            declare_cost=self.config.declare_cost,
            query_cost=self.config.query_cost,
            respond_cost=self.config.respond_cost,
            population_mode="all_constrained",
        )

        # Environment
        self.env = Environment(EnvironmentConfig(
            grid_size=self.config.grid_size,
            num_obstacles=self.config.num_obstacles,
            z_layers=self.config.z_layers,
            max_steps=self.config.max_steps,
            energy_budget=self.config.energy_budget,
            move_cost=self.config.move_cost,
            collision_penalty=self.config.collision_penalty,
        ))

        # Communication buffer
        self.comm_buffer = CommBuffer(CommBufferConfig(signal_dim=self.config.signal_dim))

        # Agents
        # AgentA: obs_dim=13 (env 11 + zero token dist 1 + F[i] 1)
        self.agent_a = AgentA(
            obs_dim=_OBS_DIM_A,
            signal_dim=self.config.signal_dim,
            hidden_dim=self.config.hidden_dim,
            depth=self.config.depth,
        ).to(self.device)

        # AgentB: volumetric, 2 channels (env provides 1, zero channel added for token)
        self.agent_b = AgentB(
            signal_dim=self.config.signal_dim,
            hidden_dim=self.config.hidden_dim,
            depth=self.config.depth,
        ).to(self.device)

        # AgentC: pairwise distances, obs_dim=66 (no token in P6)
        self.agent_c = AgentC(
            obs_dim=_OBS_DIM_C,
            signal_dim=self.config.signal_dim,
            hidden_dim=self.config.hidden_dim,
            depth=self.config.depth,
        ).to(self.device)

        self.agents = [self.agent_a, self.agent_b, self.agent_c]

        # Optimizers
        self.optimizers = [
            optim.Adam(a.parameters(), lr=self.config.learning_rate)
            for a in self.agents
        ]

        # Constraint field
        self.field = ConstraintField(
            n_agents=3,
            diffusion_coefficient=self.config.diffusion_coefficient,
            decay_rate=self.config.decay_rate,
        )

    def _augment_obs(self, raw_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Augment raw env observations for P6.

        AgentA: append zero token dist + F[A] → 13 dims
        AgentB: append zero second channel → (2, 8, 20, 20)
        AgentC: unchanged → 66 dims
        """
        dev = self.device

        # AgentA: (11,) → cat([obs, 0.0, F[A]]) → (13,)
        obs_a = raw_obs["A"]
        if not isinstance(obs_a, torch.Tensor):
            obs_a = torch.tensor(obs_a, dtype=torch.float32)
        token_dist = torch.zeros(1, dtype=torch.float32)
        fi_a = torch.tensor([self.field.get("A")], dtype=torch.float32)
        aug_a = torch.cat([obs_a, token_dist, fi_a]).to(dev)

        # AgentB: (1, 8, 20, 20) → cat along dim=0 with zeros → (2, 8, 20, 20)
        obs_b = raw_obs["B"]
        if not isinstance(obs_b, torch.Tensor):
            obs_b = torch.tensor(obs_b, dtype=torch.float32)
        obs_b = obs_b.to(dev)
        zero_channel = torch.zeros_like(obs_b)
        aug_b = torch.cat([obs_b, zero_channel], dim=0)

        # AgentC: (66,) — no change
        obs_c = raw_obs["C"]
        if not isinstance(obs_c, torch.Tensor):
            obs_c = torch.tensor(obs_c, dtype=torch.float32)
        aug_c = obs_c.to(dev)

        return {"A": aug_a, "B": aug_b, "C": aug_c}

    def _run_episode(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Run one episode. Returns per-episode summary dict."""
        self.field.reset()
        self.comm_buffer.clear()
        for agent in self.agents:
            agent.clear_episode()

        obs = self._augment_obs(raw_obs)

        total_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
        type_log: list[int] = []
        done = False
        tau = self.protocol.get_tau(0)

        while not done:
            signals: dict[str, torch.Tensor] = {}
            signal_types: dict[str, int] = {}
            action_logits: dict[str, torch.Tensor] = {}
            all_type_logits: dict[str, torch.Tensor] = {}

            # Collect signals and action logits from all agents
            for agent in self.agents:
                agent_obs = obs[agent.name].to(self.device)
                incoming = self.comm_buffer.receive_all(agent.name).to(self.device)
                signal, logits, type_logits = agent(agent_obs, incoming)

                _, hard_type = self.protocol.resolve_signal_type(
                    type_logits, tau, training=True
                )
                signals[agent.name] = signal
                signal_types[agent.name] = hard_type
                action_logits[agent.name] = logits
                all_type_logits[agent.name] = type_logits

            # Broadcast signals to comm buffer
            for name, signal in signals.items():
                self.comm_buffer.send(name, signal, signal_type=signal_types[name])
            self.comm_buffer.record_history()

            # Sample actions and store outcomes
            actions: dict[str, int] = {}
            for agent in self.agents:
                dist = Categorical(logits=action_logits[agent.name])
                action = dist.sample()
                log_prob = dist.log_prob(action)

                if self.protocol.should_train_type_head():
                    type_dist = Categorical(logits=all_type_logits[agent.name])
                    type_log_prob = type_dist.log_prob(
                        torch.tensor(signal_types[agent.name], device=self.device)
                    )
                    combined_log_prob = log_prob + type_log_prob
                else:
                    combined_log_prob = log_prob

                agent.store_outcome(combined_log_prob, 0.0)
                actions[agent.name] = action.item()

            # Environment step
            raw_obs_next, env_rewards, done, info = self.env.step(actions)

            # ----------------------------------------------------------------
            # P6: Update constraint field AFTER actions, BEFORE rewards.
            # ----------------------------------------------------------------
            self.field.update(signal_types)

            # Depth 2: feed energy delta to AgentA self_model_gru
            if self.config.depth >= 2:
                energy_now = info["energy"]
                # energy_prev tracked via env.energy before step — approximate as 0
                # (delta from previous step not tracked per-step; use 0 as fallback)
                self.agent_a.set_energy_delta(0.0)

            # Augment next observations (field values updated above)
            obs = self._augment_obs(raw_obs_next)

            # Compute rewards with field-modulated tax rate per agent
            for agent in self.agents:
                modulated_tax = self.field.effective_cost(
                    agent.name,
                    base_cost=self.config.communication_tax_rate,
                    sensitivity=self.config.signal_cost_sensitivity,
                )
                reward = self.protocol.compute_reward(
                    agent_name=agent.name,
                    env_reward=env_rewards[agent.name],
                    signal_sent=signals[agent.name],
                    energy_remaining=info["energy"][agent.name],
                    energy_budget=self.config.energy_budget,
                    communication_tax_rate=modulated_tax,
                    reached_target=info["reached_target"][agent.name],
                    survival_bonus=self.config.survival_bonus,
                    signal_type=signal_types[agent.name],
                )
                agent.rewards[-1] = reward
                total_rewards[agent.name] += reward

            type_log.extend(signal_types.values())

        return {
            "total_rewards": total_rewards,
            "type_log": type_log,
            "field_snapshot": self.field.F.copy(),
        }

    def _run_epoch(self, epoch: int) -> dict[str, Any]:
        """Run one epoch of episodes. Returns per-epoch metric dict."""
        tau = self.protocol.get_tau(epoch)
        epoch_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
        type_history_epoch: list[int] = []
        field_snapshots: list[np.ndarray] = []

        for ep_idx in range(self.config.episodes_per_epoch):
            ep_seed = self.config.seed + epoch * self.config.episodes_per_epoch + ep_idx
            raw_obs = self.env.reset(seed=ep_seed)
            result = self._run_episode(raw_obs)

            for name in ["A", "B", "C"]:
                epoch_rewards[name] += result["total_rewards"][name]
            type_history_epoch.extend(result["type_log"])
            field_snapshots.append(result["field_snapshot"])

            update_agents(self.agents, self.optimizers, gamma=self.config.gamma)

        # Field diagnostics: average field state over episodes in epoch
        mean_F = np.mean(np.stack(field_snapshots), axis=0)
        field_diag = compute_field_diagnostics(mean_F)

        # Signal type metrics
        n_signals = len(type_history_epoch)
        query_rate = (
            type_history_epoch.count(1) / n_signals if n_signals > 0 else 0.0
        )

        # Exploitation loop rate via diversity score
        if n_signals > 0:
            exploit_result = exploitation_loop_detection(type_history_epoch)
            exploitation_loop_rate = 1.0 - exploit_result["diversity_score"]
        else:
            exploitation_loop_rate = 0.0

        # sustained_structure_score: query_rate × (1 − exploitation_loop_rate)
        sustained_structure_score = query_rate * (1.0 - exploitation_loop_rate)

        n_ep = self.config.episodes_per_epoch
        return {
            **field_diag,
            "sustained_structure_score": float(sustained_structure_score),
            "exploitation_loop_rate": float(exploitation_loop_rate),
            "query_rate": float(query_rate),
            "avg_reward_A": float(epoch_rewards["A"] / n_ep),
            "avg_reward_B": float(epoch_rewards["B"] / n_ep),
            "avg_reward_C": float(epoch_rewards["C"] / n_ep),
            "epoch": epoch,
            "tau": float(tau),
        }

    def run(self) -> list[dict[str, Any]]:
        """Run all epochs. Returns list of per-epoch metric dicts."""
        results = []
        for epoch in range(self.config.num_epochs):
            metrics = self._run_epoch(epoch)
            results.append(metrics)
            logger.info(
                "Epoch %d/%d | field_std=%.4f field_mean=%.4f query_rate=%.3f",
                epoch + 1, self.config.num_epochs,
                metrics["field_std"], metrics["field_mean"], metrics["query_rate"],
            )
        return results
