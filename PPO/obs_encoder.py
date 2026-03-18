"""Structured observation encoder for PPO on OBELIX."""

from __future__ import annotations

import torch


RAW_OBS_DIM = 18
ENCODED_OBS_DIM = 32


def encode_obs_tensor(obs: torch.Tensor) -> torch.Tensor:
    """Augment the raw 18-bit observation with grouped directional features."""
    if obs.shape[-1] != RAW_OBS_DIM:
        raise ValueError(f"Expected obs last dim {RAW_OBS_DIM}, got {obs.shape[-1]}")

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
