"""Submission-time policy template for weights trained by train_ddqn_parallel.py.

Usage:
1) Copy this file as `agent.py`
2) Place `weights.pth` next to it
3) Zip and submit: agent.py + weights.pth
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = 18, action_dim: int = 5, hidden_dims=(128, 128)):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")
        layers = []
        last = int(obs_dim)
        for h in hidden_dims:
            h_i = int(h)
            layers.append(nn.Linear(last, h_i))
            layers.append(nn.ReLU())
            last = h_i
        layers.append(nn.Linear(last, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_MODEL: Optional[QNetwork] = None


def _infer_hidden_dims(state_dict: dict):
    # Collect hidden layer out-dims from all linear layers except the final output layer.
    linear_keys = []
    for k in state_dict.keys():
        if k.startswith("net.") and k.endswith(".weight"):
            parts = k.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                linear_keys.append((int(parts[1]), k))
    if not linear_keys:
        return (128, 128)

    linear_keys.sort(key=lambda x: x[0])
    if len(linear_keys) == 1:
        return (128, 128)

    hidden = []
    for _, key in linear_keys[:-1]:
        hidden.append(int(state_dict[key].shape[0]))
    if len(hidden) == 0:
        return (128, 128)
    return tuple(hidden)


def _load_once() -> None:
    global _MODEL
    if _MODEL is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py")

    raw = torch.load(wpath, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = raw["state_dict"]
        if "hidden_dims" in raw:
            hidden_dims = tuple(int(x) for x in raw["hidden_dims"])
        elif "hidden_dim" in raw:
            hd = int(raw["hidden_dim"])
            hidden_dims = (hd, hd)
        else:
            hidden_dims = _infer_hidden_dims(state_dict)
    elif isinstance(raw, dict):
        state_dict = raw
        hidden_dims = _infer_hidden_dims(state_dict)
    else:
        raise RuntimeError("Unsupported checkpoint format")

    model = QNetwork(hidden_dims=hidden_dims)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _MODEL = model


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q = _MODEL(x).squeeze(0)
    act = int(torch.argmax(q).item())
    return ACTIONS[act]
