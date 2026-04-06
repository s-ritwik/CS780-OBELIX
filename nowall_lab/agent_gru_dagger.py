from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
GRU_DIR = os.path.join(os.path.dirname(HERE), "gru_pose")
if GRU_DIR not in sys.path:
    sys.path.insert(0, GRU_DIR)

from common import ACTIONS, FeatureConfig, GRUPolicy, PoseGRUFeatureTracker, load_checkpoint


DEFAULT_WEIGHT_CANDIDATES = (
    "weights_best.pth",
    "weights_final.pth",
    "weights_nowall_gru_dagger.pth",
    "weights_nowall_gru_dagger_final.pth",
)

_MODEL: Optional[GRUPolicy] = None
_TRACKER: Optional[PoseGRUFeatureTracker] = None
_HIDDEN: Optional[torch.Tensor] = None
_LAST_RNG_ID: Optional[int] = None
_PENDING_ACTION: Optional[int] = None


def _checkpoint_path() -> str:
    override = os.environ.get("OBELIX_WEIGHTS")
    if override:
        return override if os.path.isabs(override) else os.path.join(HERE, override)
    for candidate in DEFAULT_WEIGHT_CANDIDATES:
        path = os.path.join(HERE, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Weights file not found")


def _load_once() -> None:
    global _MODEL, _TRACKER, _HIDDEN
    if _MODEL is not None and _TRACKER is not None and _HIDDEN is not None:
        return

    checkpoint = load_checkpoint(_checkpoint_path(), device=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]
    encoder_dims = tuple(int(x) for x in checkpoint.get("encoder_dims", [128, 128]))
    gru_hidden_dim = int(checkpoint.get("gru_hidden_dim", 128))
    feature_payload = checkpoint.get("feature_config", {})
    feature_config = FeatureConfig(
        max_steps=int(feature_payload.get("max_steps", 2000)),
        pose_clip=float(feature_payload.get("pose_clip", 500.0)),
        blind_clip=float(feature_payload.get("blind_clip", 200.0)),
        stuck_clip=float(feature_payload.get("stuck_clip", 20.0)),
        contact_clip=float(feature_payload.get("contact_clip", 20.0)),
        same_obs_clip=float(feature_payload.get("same_obs_clip", 100.0)),
        wall_hit_clip=float(feature_payload.get("wall_hit_clip", 20.0)),
        last_action_hist=int(feature_payload.get("last_action_hist", 5)),
        heading_bins=int(feature_payload.get("heading_bins", 8)),
    )

    model = GRUPolicy(
        input_dim=feature_config.feature_dim,
        encoder_dims=encoder_dims,
        gru_hidden_dim=gru_hidden_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _MODEL = model
    _TRACKER = PoseGRUFeatureTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
    _HIDDEN = _MODEL.initial_state(1, torch.device("cpu"))


def _reset_episode(obs: np.ndarray) -> None:
    global _TRACKER, _HIDDEN, _PENDING_ACTION
    _TRACKER.reset_all(obs[None, :])
    _HIDDEN = _MODEL.initial_state(1, torch.device("cpu"))
    _PENDING_ACTION = None


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_RNG_ID, _PENDING_ACTION, _HIDDEN
    _load_once()

    obs_arr = np.asarray(obs, dtype=np.float32)
    rng_id = id(rng)
    if _LAST_RNG_ID != rng_id:
        _reset_episode(obs_arr)
        _LAST_RNG_ID = rng_id
        starts = torch.ones((1,), dtype=torch.float32)
    else:
        if _PENDING_ACTION is not None:
            _TRACKER.post_step(
                actions=torch.tensor([_PENDING_ACTION], dtype=torch.long),
                next_obs=obs_arr[None, :],
                dones=np.asarray([False]),
            )
        starts = torch.zeros((1,), dtype=torch.float32)

    features = _TRACKER.features()
    logits, _, _HIDDEN = _MODEL.forward_step(features, _HIDDEN, starts)
    action_idx = int(torch.argmax(logits, dim=1).item())
    _PENDING_ACTION = action_idx
    return ACTIONS[action_idx]
