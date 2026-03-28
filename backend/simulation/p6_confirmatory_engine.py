"""Protocol 6 Confirmatory Engine — four-condition harness.

Conditions
----------
A — Emergent local  : constraint field, each agent perceives only F[i]
B — Emergent global : constraint field, each agent perceives full F vector
C — Fixed external  : no dynamic field; fixed cost multiplier matched to
                      mean pilot field value (dc=0.1, dr=0.05)
D — No constraint   : no field, no communication tax (unconstrained baseline)

Observation dimensions
----------------------
A, C, D  AgentA: env(11) + zero_token_dist(1) + F[i](1) = 13  (F[i]=0 for C, D)
B        AgentA: env(11) + zero_token_dist(1) + F[A](1) + F[B](1) + F[C](1) = 15
         AgentB: (2, 8, 20, 20) — zero second channel, no field (all conditions)
         AgentC: 66 dims — unchanged (all conditions)
"""

from __future__ import annotations

import logging
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
from .p6_engine import P6Config
from .protocols import create_protocol
from .training.reinforce import update_agents
from .utils.seeding import set_all_seeds

logger = logging.getLogger(__name__)

_CONDITIONS = frozenset(("A", "B", "C", "D"))
_OBS_DIM_A_LOCAL = 13   # conditions A, C, D
_OBS_DIM_A_GLOBAL = 15  # condition B: full F vector appended
_OBS_DIM_C = 66


class P6ConfirmatoryEngine:
    """Four-condition Protocol 6 confirmatory engine.

    Parameters
    ----------
    config:
        Standard P6Config (dc/dr used only for conditions A and B).
    condition:
        One of 'A', 'B', 'C', 'D'.
    fixed_cost_multiplier:
        For Condition C: effective_tax = base_tax * fixed_cost_multiplier.
        Derived from pilot mean field: 1 + signal_cost_sensitivity * mean_F.
        Ignored for conditions A, B, D.
    """

    def __init__(
        self,
        config: P6Config,
        condition: str,
        fixed_cost_multiplier: float,
    ) -> None:
        if condition not in _CONDITIONS:
            raise ValueError(f"condition must be one of {sorted(_CONDITIONS)}, got {condition!r}")
        self.config = config
        self.condition = condition
        self.fixed_cost_multiplier = fixed_cost_multiplier
        set_all_seeds(config.seed)
        self.device = torch.device(config.device)

        # Protocol 2 (all_constrained) for all conditions
        self.protocol = create_protocol(
            2,
            declare_cost=config.declare_cost,
            query_cost=config.query_cost,
            respond_cost=config.respond_cost,
            population_mode="all_constrained",
        )

        self.env = Environment(EnvironmentConfig(
            grid_size=config.grid_size,
            num_obstacles=config.num_obstacles,
            z_layers=config.z_layers,
            max_steps=config.max_steps,
            energy_budget=config.energy_budget,
            move_cost=config.move_cost,
            collision_penalty=config.collision_penalty,
        ))

        self.comm_buffer = CommBuffer(CommBufferConfig(signal_dim=config.signal_dim))

        obs_dim_a = _OBS_DIM_A_GLOBAL if condition == "B" else _OBS_DIM_A_LOCAL

        self.agent_a = AgentA(
            obs_dim=obs_dim_a,
            signal_dim=config.signal_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        ).to(self.device)

        self.agent_b = AgentB(
            signal_dim=config.signal_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        ).to(self.device)

        self.agent_c = AgentC(
            obs_dim=_OBS_DIM_C,
            signal_dim=config.signal_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        ).to(self.device)

        self.agents = [self.agent_a, self.agent_b, self.agent_c]
        self.optimizers = [
            optim.Adam(a.parameters(), lr=config.learning_rate)
            for a in self.agents
        ]

        # Constraint field: only for conditions A and B
        if condition in ("A", "B"):
            self.field: ConstraintField | None = ConstraintField(
                n_agents=3,
                diffusion_coefficient=config.diffusion_coefficient,
                decay_rate=config.decay_rate,
            )
        else:
            self.field = None

    def _augment_obs(self, raw_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        dev = self.device

        # AgentA
        obs_a = raw_obs["A"]
        if not isinstance(obs_a, torch.Tensor):
            obs_a = torch.tensor(obs_a, dtype=torch.float32)
        token_dist = torch.zeros(1, dtype=torch.float32)

        if self.condition == "B":
            # Global perception: append full F vector
            f_vec = torch.tensor(self.field.F.copy(), dtype=torch.float32)
            aug_a = torch.cat([obs_a, token_dist, f_vec]).to(dev)
        else:
            # Local perception: append F[A] (0.0 if no field)
            fi_a = torch.tensor(
                [self.field.get("A") if self.field is not None else 0.0],
                dtype=torch.float32,
            )
            aug_a = torch.cat([obs_a, token_dist, fi_a]).to(dev)

        # AgentB: zero second channel
        obs_b = raw_obs["B"]
        if not isinstance(obs_b, torch.Tensor):
            obs_b = torch.tensor(obs_b, dtype=torch.float32)
        obs_b = obs_b.to(dev)
        aug_b = torch.cat([obs_b, torch.zeros_like(obs_b)], dim=0)

        # AgentC: unchanged
        obs_c = raw_obs["C"]
        if not isinstance(obs_c, torch.Tensor):
            obs_c = torch.tensor(obs_c, dtype=torch.float32)
        aug_c = obs_c.to(dev)

        return {"A": aug_a, "B": aug_b, "C": aug_c}

    def _effective_tax(self, agent_name: str) -> float:
        """Return the communication tax rate for this agent under this condition."""
        if self.condition in ("A", "B"):
            # Dynamic field modulation
            return self.field.effective_cost(
                agent_name,
                base_cost=self.config.communication_tax_rate,
                sensitivity=self.config.signal_cost_sensitivity,
            )
        elif self.condition == "C":
            # Fixed external tax
            return self.config.communication_tax_rate * self.fixed_cost_multiplier
        else:
            # Condition D: no constraint
            return 0.0

    def _run_episode(self, raw_obs: dict[str, Any], tau: float) -> dict[str, Any]:
        if self.field is not None:
            self.field.reset()
        self.comm_buffer.reset()
        self.protocol.reset_episode()
        for agent in self.agents:
            agent.clear_episode()

        obs = self._augment_obs(raw_obs)
        total_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
        type_log: list[int] = []
        done = False

        while not done:
            signals: dict[str, torch.Tensor] = {}
            signal_types: dict[str, int] = {}
            action_logits: dict[str, torch.Tensor] = {}
            all_type_logits: dict[str, torch.Tensor] = {}

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

            for name, signal in signals.items():
                self.comm_buffer.send(name, signal, signal_type=signal_types[name])
            self.comm_buffer.record_history()

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

            raw_obs_next, env_rewards, done, info = self.env.step(actions)

            # Update field for conditions A and B
            if self.field is not None:
                self.field.update(signal_types)

            if self.config.depth >= 2:
                self.agent_a.set_energy_delta(0.0)

            obs = self._augment_obs(raw_obs_next)

            for agent in self.agents:
                reward = self.protocol.compute_reward(
                    agent_name=agent.name,
                    env_reward=env_rewards[agent.name],
                    signal_sent=signals[agent.name],
                    energy_remaining=info["energy"][agent.name],
                    energy_budget=self.config.energy_budget,
                    communication_tax_rate=self._effective_tax(agent.name),
                    reached_target=info["reached_target"][agent.name],
                    survival_bonus=self.config.survival_bonus,
                    signal_type=signal_types[agent.name],
                )
                agent.rewards[-1] = reward
                total_rewards[agent.name] += reward

            type_log.extend(signal_types.values())

        field_snapshot = self.field.F.copy() if self.field is not None else np.zeros(3, dtype=np.float32)
        return {
            "total_rewards": total_rewards,
            "type_log": type_log,
            "field_snapshot": field_snapshot,
        }

    def _run_epoch(self, epoch: int) -> dict[str, Any]:
        tau = self.protocol.get_tau(epoch)
        epoch_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
        type_history_epoch: list[int] = []
        field_snapshots: list[np.ndarray] = []

        for ep_idx in range(self.config.episodes_per_epoch):
            ep_seed = self.config.seed + epoch * self.config.episodes_per_epoch + ep_idx
            raw_obs = self.env.reset(seed=ep_seed)
            result = self._run_episode(raw_obs, tau=tau)
            for name in ["A", "B", "C"]:
                epoch_rewards[name] += result["total_rewards"][name]
            type_history_epoch.extend(result["type_log"])
            field_snapshots.append(result["field_snapshot"])
            update_agents(self.agents, self.optimizers, gamma=self.config.gamma)

        mean_F = np.mean(np.stack(field_snapshots), axis=0)
        field_diag = compute_field_diagnostics(mean_F)

        n_signals = len(type_history_epoch)
        query_rate = type_history_epoch.count(1) / n_signals if n_signals > 0 else 0.0

        if n_signals > 0:
            exploit_result = exploitation_loop_detection(type_history_epoch)
            exploitation_loop_rate = 1.0 - exploit_result["diversity_score"]
        else:
            exploitation_loop_rate = 0.0

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
            "condition": self.condition,
        }

    def run(self) -> list[dict[str, Any]]:
        results = []
        for epoch in range(self.config.num_epochs):
            metrics = self._run_epoch(epoch)
            results.append(metrics)
            logger.info(
                "Cond %s Epoch %d/%d | field_mean=%.4f query_rate=%.3f avg_reward_A=%.3f",
                self.condition, epoch + 1, self.config.num_epochs,
                metrics["field_mean"], metrics["query_rate"], metrics["avg_reward_A"],
            )
        return results
