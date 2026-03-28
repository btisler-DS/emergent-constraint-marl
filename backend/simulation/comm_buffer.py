"""Communication buffer with gradient isolation and kill switch.

Bug Fix #2: All signals are `.detach()`ed before storing to prevent
gradient leakage between agents.

Ablation Kill Switch: Communication can be severed mid-run to prove
the emergent protocol is load-bearing.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field


@dataclass
class CommBufferConfig:
    signal_dim: int = 8
    num_agents: int = 3


class CommBuffer:
    """Stores inter-agent signals with gradient isolation."""

    def __init__(self, config: CommBufferConfig | None = None):
        self.config = config or CommBufferConfig()
        self.signal_dim = self.config.signal_dim
        self._buffer: dict[str, torch.Tensor] = {}
        self._history: list[dict[str, torch.Tensor]] = []
        self._killed = False
        # Protocol 1: parallel type buffer (DECLARATIVE=0, INTERROGATIVE=1, RESPONSE=2)
        self._type_buffer: dict[str, int] = {}
        self._type_history: list[dict[str, int]] = []

    def send(self, agent_name: str, signal: torch.Tensor, signal_type: int = 0) -> None:
        """Store a signal and its type from an agent.

        Args:
            signal_type: 0=DECLARATIVE (default), 1=INTERROGATIVE, 2=RESPONSE.
                         Defaults to 0 for backward compatibility with pre-Protocol-1 runs.
        """
        self._buffer[agent_name] = signal.detach().clone()
        self._type_buffer[agent_name] = signal_type

    def receive(self, agent_name: str) -> dict[str, torch.Tensor]:
        """Get signals from all OTHER agents. Returns zeros if killed."""
        signals = {}
        for name, sig in self._buffer.items():
            if name != agent_name:
                if self._killed:
                    signals[name] = torch.zeros_like(sig)
                else:
                    signals[name] = sig.clone()
        return signals

    def receive_all(self, agent_name: str) -> torch.Tensor:
        """Get concatenated signals from all other agents."""
        signals = self.receive(agent_name)
        if not signals:
            return torch.zeros(self.signal_dim * (self.config.num_agents - 1))
        return torch.cat(list(signals.values()))

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Return a copy of the current buffer state."""
        return {k: v.clone() for k, v in self._buffer.items()}

    def record_history(self) -> None:
        """Save current buffer state to history for analysis."""
        self._history.append(self.snapshot())
        self._type_history.append(dict(self._type_buffer))  # Protocol 1

    @property
    def history(self) -> list[dict[str, torch.Tensor]]:
        return self._history

    @property
    def type_history(self) -> list[dict[str, int]]:
        """Per-step signal type log for Protocol 1 inquiry metrics."""
        return self._type_history

    def kill(self) -> None:
        """Sever all communication (ablation kill switch)."""
        self._killed = True

    def restore(self) -> None:
        """Restore communication after kill."""
        self._killed = False

    @property
    def is_killed(self) -> bool:
        return self._killed

    def clear(self) -> None:
        """Clear the buffer (called between steps)."""
        self._buffer.clear()

    def reset(self) -> None:
        """Full reset including history."""
        self._buffer.clear()
        self._history.clear()
        self._type_buffer.clear()
        self._type_history.clear()
        self._killed = False
