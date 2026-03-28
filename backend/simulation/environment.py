"""Grid environment with energy tracking and 3D volumetric density.

Bug Fix #4: Clamp density values to prevent negative gradients.
Hardware Friction Mandate: 3D density uses linear decay ("Volumetric Echo"),
NOT a perfect voxel grid. Obstacles are 1s across all Z; the target is a
gradient that increases as the agent approaches.
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class EnvironmentConfig:
    grid_size: int = 20
    num_obstacles: int = 8
    z_layers: int = 8
    max_steps: int = 100
    energy_budget: float = 100.0
    move_cost: float = 1.0
    collision_penalty: float = 5.0


class Environment:
    """2D grid world with energy budgets and 3D acoustic perception."""

    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = config or EnvironmentConfig()
        self.grid_size = self.config.grid_size
        self.z_layers = self.config.z_layers
        self.step_count = 0
        self.agents_pos: dict[str, np.ndarray] = {}
        self.target_pos = np.array([0, 0])
        self.obstacles: list[np.ndarray] = []
        self.energy: dict[str, float] = {}
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

    def reset(self, seed: int | None = None) -> dict:
        """Reset environment and return initial observations."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        self.step_count = 0

        # Place target
        self.target_pos = rng.randint(0, self.grid_size, size=2)

        # Place obstacles (not on target)
        self.obstacles = []
        for _ in range(self.config.num_obstacles):
            while True:
                pos = rng.randint(0, self.grid_size, size=2)
                if not np.array_equal(pos, self.target_pos):
                    self.obstacles.append(pos)
                    break

        # Place agents (not on obstacles or target)
        occupied = {tuple(self.target_pos)} | {tuple(o) for o in self.obstacles}
        for name in ["A", "B", "C"]:
            while True:
                pos = rng.randint(0, self.grid_size, size=2)
                if tuple(pos) not in occupied:
                    self.agents_pos[name] = pos
                    self.energy[name] = self.config.energy_budget
                    occupied.add(tuple(pos))
                    break

        self._update_grid()
        return self._get_observations()

    def _update_grid(self) -> None:
        """Rebuild the 2D grid representation."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = -1.0
        self.grid[self.target_pos[0], self.target_pos[1]] = 1.0
        for name, pos in self.agents_pos.items():
            idx = {"A": 0.3, "B": 0.5, "C": 0.7}[name]
            self.grid[pos[0], pos[1]] = idx

    def step(self, actions: dict[str, int]) -> tuple[dict, dict[str, float], bool, dict]:
        """Execute actions for all agents. Actions: 0=up,1=down,2=left,3=right,4=stay."""
        self.step_count += 1
        rewards = {}
        info: dict = {"collisions": {}, "reached_target": {}}

        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

        for name, action in actions.items():
            dx, dy = deltas.get(action, (0, 0))
            old_pos = self.agents_pos[name].copy()
            new_pos = old_pos + np.array([dx, dy])

            # Boundary check
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)

            # Collision check
            hit_obstacle = any(np.array_equal(new_pos, o) for o in self.obstacles)
            if hit_obstacle:
                self.energy[name] -= self.config.collision_penalty
                new_pos = old_pos
                info["collisions"][name] = True
            else:
                self.energy[name] -= self.config.move_cost

            self.agents_pos[name] = new_pos

            # Target check
            reached = np.array_equal(new_pos, self.target_pos)
            info["reached_target"][name] = reached

            # Distance-based reward
            dist = np.linalg.norm(new_pos - self.target_pos)
            max_dist = self.grid_size * np.sqrt(2)
            rewards[name] = 1.0 - (dist / max_dist)
            if reached:
                rewards[name] += 10.0

        self._update_grid()

        # Done conditions
        any_reached = any(info["reached_target"].values())
        all_dead = all(e <= 0 for e in self.energy.values())
        timeout = self.step_count >= self.config.max_steps
        done = any_reached or all_dead or timeout

        info["energy"] = dict(self.energy)
        info["step"] = self.step_count
        info["done_reason"] = (
            "target_reached" if any_reached
            else "energy_depleted" if all_dead
            else "timeout" if timeout
            else "running"
        )

        return self._get_observations(), rewards, done, info

    def _get_observations(self) -> dict:
        """Return per-agent observations."""
        return {
            "A": self._get_1d_sequence("A"),
            "B": self._get_3d_density_map("B"),
            "C": self._get_relational_features(),
            "grid": self.grid.copy(),
        }

    def _get_1d_sequence(self, agent_name: str) -> torch.Tensor:
        """Agent A sees a 1D sequence: distances to all entities."""
        pos = self.agents_pos[agent_name].astype(np.float32)
        features = []
        # Distance to target
        features.append(np.linalg.norm(pos - self.target_pos))
        # Distance to each obstacle
        for obs in self.obstacles:
            features.append(np.linalg.norm(pos - obs))
        # Distance to other agents
        for name in ["A", "B", "C"]:
            if name != agent_name:
                features.append(np.linalg.norm(pos - self.agents_pos[name]))
        return torch.tensor(features, dtype=torch.float32)

    def get_3d_density_map(self, agent_name: str) -> torch.Tensor:
        """Agent B's volumetric perception — Hardware Friction Mandate.

        NOT a perfect voxel grid. Uses linear decay gradient:
        - Obstacles: value 1.0 across ALL z-layers (acoustic wall)
        - Target: gradient that increases as agent approaches (echo)
        - Bug Fix #4: clamp all values to >= 0.0
        """
        density = torch.zeros(1, self.z_layers, self.grid_size, self.grid_size)
        pos = self.agents_pos[agent_name].astype(np.float32)

        # Obstacles: hard 1s across all z-layers
        for obs in self.obstacles:
            density[0, :, obs[0], obs[1]] = 1.0

        # Target: linear decay gradient ("Volumetric Echo")
        dist_to_target = np.linalg.norm(pos - self.target_pos)
        max_dist = self.grid_size * np.sqrt(2)
        # Intensity increases as agent gets closer (inverted distance)
        base_intensity = 1.0 - (dist_to_target / max_dist)

        tx, ty = self.target_pos
        for z in range(self.z_layers):
            # Decay factor: signal fades across z-layers
            z_decay = 1.0 - (z / self.z_layers)
            density[0, z, tx, ty] = base_intensity * z_decay

        # Bug Fix #4: clamp to prevent negative values
        density = torch.clamp(density, min=0.0)
        return density

    def _get_3d_density_map(self, agent_name: str) -> torch.Tensor:
        """Internal wrapper for observations."""
        return self.get_3d_density_map(agent_name)

    def _get_relational_features(self) -> torch.Tensor:
        """Agent C sees pairwise distances between all entities (relational)."""
        entities = [self.target_pos.astype(np.float32)]
        entities.extend(o.astype(np.float32) for o in self.obstacles)
        for name in ["A", "B", "C"]:
            entities.append(self.agents_pos[name].astype(np.float32))

        n = len(entities)
        features = []
        for i in range(n):
            for j in range(i + 1, n):
                features.append(np.linalg.norm(entities[i] - entities[j]))
        return torch.tensor(features, dtype=torch.float32)

    @property
    def total_energy_spent(self) -> float:
        """Total energy consumed across all agents."""
        return sum(
            self.config.energy_budget - e for e in self.energy.values()
        )
