"""Submission-time PPO policy for OBELIX."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = 18,
        action_dim: int = 5,
        hidden_dims: tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers: list[nn.Module] = []
        last_dim = int(obs_dim)
        for h in hidden_dims:
            h_i = int(h)
            layers.append(nn.Linear(last_dim, h_i))
            layers.append(nn.Tanh())
            last_dim = h_i

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, int(action_dim))
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value


_MODEL: Optional[ActorCritic] = None


def _infer_hidden_dims(state_dict: dict):
    backbone_keys = []
    for key, value in state_dict.items():
        if key.startswith("backbone.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                backbone_keys.append((int(parts[1]), int(value.shape[0])))
    backbone_keys.sort(key=lambda x: x[0])
    hidden = [dim for _, dim in backbone_keys]
    return tuple(hidden) if hidden else (128, 64)


def _load_once() -> None:
    global _MODEL
    if _MODEL is not None:
        return

    here = os.path.dirname(__file__)
    env_override = os.environ.get("OBELIX_WEIGHTS")
    if env_override:
        wpath = env_override
        if not os.path.isabs(wpath):
            wpath = os.path.join(here, wpath)
    else:
        wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Weights file not found: {wpath}")

    raw = torch.load(wpath, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = raw["state_dict"]
        hidden_dims = tuple(int(x) for x in raw.get("hidden_dims", _infer_hidden_dims(state_dict)))
    elif isinstance(raw, dict):
        state_dict = raw
        hidden_dims = _infer_hidden_dims(state_dict)
    else:
        raise RuntimeError("Unsupported checkpoint format")

    model = ActorCritic(hidden_dims=hidden_dims)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _MODEL = model


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, _ = _MODEL(x)
    action = int(torch.argmax(logits, dim=1).item())
    return ACTIONS[action]
