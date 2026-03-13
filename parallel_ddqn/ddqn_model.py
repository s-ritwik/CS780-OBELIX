"""Shared DDQN model utilities for OBELIX."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


class QNetwork(nn.Module):
    """Small MLP used by both trainer and submission-time policy."""

    def __init__(
        self,
        obs_dim: int = 18,
        action_dim: int = 5,
        hidden_dims: Sequence[int] = (128, 128),
    ):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers: list[nn.Module] = []
        last = int(obs_dim)
        for h in hidden_dims:
            h_i = int(h)
            if h_i <= 0:
                raise ValueError("All hidden_dims values must be > 0")
            layers.append(nn.Linear(last, h_i))
            layers.append(nn.ReLU())
            last = h_i
        layers.append(nn.Linear(last, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
