"""Submission-time agent for teacher-student PPO checkpoints."""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PPO_DIR = os.path.join(os.path.dirname(THIS_DIR), "PPO")
if PPO_DIR not in sys.path:
    sys.path.insert(0, PPO_DIR)

from obs_encoder import ENCODED_OBS_DIM, RAW_OBS_DIM, encode_obs_tensor, infer_use_rec_encoder_from_state_dict


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
STOCHASTIC_POLICY = False
USE_REC_ENCO = True


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = RAW_OBS_DIM,
        action_dim: int = 5,
        hidden_dims: tuple[int, ...] = (128, 64),
        use_rec_encoder: bool = False,
    ):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers: list[nn.Module] = []
        self.use_rec_encoder = bool(use_rec_encoder)
        last_dim = ENCODED_OBS_DIM if self.use_rec_encoder else int(obs_dim)
        for h in hidden_dims:
            h_i = int(h)
            layers.append(nn.Linear(last_dim, h_i))
            layers.append(nn.Tanh())
            last_dim = h_i

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, int(action_dim))
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor):
        if self.use_rec_encoder:
            x = encode_obs_tensor(x)
        feat = self.backbone(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value


_MODEL: Optional[ActorCritic] = None


def _use_stochastic_policy() -> bool:
    env_override = os.environ.get("OBELIX_STOCHASTIC")
    if env_override is None:
        return bool(STOCHASTIC_POLICY)
    return env_override.strip().lower() in {"1", "true", "yes", "on"}


def _infer_hidden_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
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

    env_override = os.environ.get("OBELIX_WEIGHTS")
    if env_override:
        wpath = env_override
        if not os.path.isabs(wpath):
            wpath = os.path.join(THIS_DIR, wpath)
    else:
        wpath = os.path.join(THIS_DIR, "weights_teacher_student.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Weights file not found: {wpath}")

    raw = torch.load(wpath, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = raw["state_dict"]
        hidden_dims = tuple(int(x) for x in raw.get("hidden_dims", _infer_hidden_dims(state_dict)))
        ckpt_use_rec_enco = raw.get("use_rec_encoder")
    elif isinstance(raw, dict):
        state_dict = raw
        hidden_dims = _infer_hidden_dims(state_dict)
        ckpt_use_rec_enco = None
    else:
        raise RuntimeError("Unsupported checkpoint format")

    if ckpt_use_rec_enco is None:
        ckpt_use_rec_enco = infer_use_rec_encoder_from_state_dict(state_dict)
    if bool(ckpt_use_rec_enco) != bool(USE_REC_ENCO):
        raise ValueError(
            "Checkpoint encoder setting does not match agent toggle: "
            f"checkpoint_use_rec_encoder={bool(ckpt_use_rec_enco)} USE_REC_ENCO={bool(USE_REC_ENCO)}"
        )

    model = ActorCritic(hidden_dims=hidden_dims, use_rec_encoder=USE_REC_ENCO)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _MODEL = model


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, _ = _MODEL(x)
    if _use_stochastic_policy():
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64, copy=False)
        probs /= probs.sum()
        action = int(rng.choice(len(ACTIONS), p=probs))
    else:
        action = int(torch.argmax(logits, dim=1).item())
    return ACTIONS[action]
