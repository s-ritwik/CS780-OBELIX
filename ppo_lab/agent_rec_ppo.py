from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


RAW_OBS_DIM = 18
ENCODED_OBS_DIM = 32
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


def encode_obs_tensor(obs: torch.Tensor) -> torch.Tensor:
    raw = obs.to(dtype=torch.float32)
    left = raw[..., 0:4]
    front = raw[..., 4:12]
    right = raw[..., 12:16]
    ir = raw[..., 16:17]
    stuck = raw[..., 17:18]

    left_count = left.sum(dim=-1, keepdim=True)
    front_count = front.sum(dim=-1, keepdim=True)
    right_count = right.sum(dim=-1, keepdim=True)
    front_far_count = front[..., ::2].sum(dim=-1, keepdim=True)
    front_near_count = front[..., 1::2].sum(dim=-1, keepdim=True)
    side_mean_count = 0.5 * (left_count + right_count)
    blind = (raw[..., :16].sum(dim=-1, keepdim=True) == 0.0).to(dtype=torch.float32)

    derived = torch.cat(
        [
            (left_count > 0.0).to(dtype=torch.float32),
            (front_count > 0.0).to(dtype=torch.float32),
            (right_count > 0.0).to(dtype=torch.float32),
            ir,
            stuck,
            blind,
            left_count,
            front_count,
            right_count,
            front_far_count,
            front_near_count,
            left_count - right_count,
            right_count - left_count,
            front_count - side_mean_count,
        ],
        dim=-1,
    )
    return torch.cat([raw, derived], dim=-1)


def infer_use_rec_encoder_from_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    first_weight = state_dict.get("backbone.0.weight")
    if first_weight is None:
        return False
    in_features = int(first_weight.shape[1])
    if in_features == RAW_OBS_DIM:
        return False
    if in_features == ENCODED_OBS_DIM:
        return True
    raise ValueError(f"Unexpected first-layer input dim in checkpoint: {in_features}")


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = RAW_OBS_DIM,
        action_dim: int = 5,
        hidden_dims: tuple[int, ...] = (128, 64),
        use_rec_encoder: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        self.use_rec_encoder = bool(use_rec_encoder)
        last_dim = ENCODED_OBS_DIM if self.use_rec_encoder else int(obs_dim)
        for hidden_dim in hidden_dims:
            h = int(hidden_dim)
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h

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
_STOCHASTIC = os.environ.get("OBELIX_STOCHASTIC", "1") != "0"


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


def _checkpoint_path() -> str:
    here = os.path.dirname(__file__)
    override = os.environ.get("OBELIX_WEIGHTS")
    if override:
        return override if os.path.isabs(override) else os.path.join(here, override)
    return os.path.join(here, "weights_best.pth")


def _load_once() -> None:
    global _MODEL
    if _MODEL is not None:
        return

    raw = torch.load(_checkpoint_path(), map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = raw["state_dict"]
        hidden_dims = tuple(int(x) for x in raw.get("hidden_dims", _infer_hidden_dims(state_dict)))
        use_rec_encoder = raw.get("use_rec_encoder")
    elif isinstance(raw, dict):
        state_dict = raw
        hidden_dims = _infer_hidden_dims(state_dict)
        use_rec_encoder = None
    else:
        raise RuntimeError("Unsupported checkpoint format")

    if use_rec_encoder is None:
        use_rec_encoder = infer_use_rec_encoder_from_state_dict(state_dict)

    model = ActorCritic(hidden_dims=hidden_dims, use_rec_encoder=bool(use_rec_encoder))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _MODEL = model


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, _ = _MODEL(x)
    if _STOCHASTIC:
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64, copy=False)
        probs /= probs.sum()
        action_idx = int(rng.choice(len(ACTIONS), p=probs))
    else:
        action_idx = int(torch.argmax(logits, dim=1).item())
    return ACTIONS[action_idx]
