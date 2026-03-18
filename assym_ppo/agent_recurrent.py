from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch


HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from recurrent import ACTIONS, PoseMemoryTracker, RecurrentAsymmetricActorCritic, RecurrentFeatureConfig, load_checkpoint


DEFAULT_WEIGHT_CANDIDATES = (
    "recurrent_wall_best.pth",
    "recurrent_wall_best_final.pth",
    "weights_best.pth",
)

_MODEL: Optional[RecurrentAsymmetricActorCritic] = None
_TRACKER: Optional[PoseMemoryTracker] = None
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
    raise FileNotFoundError(
        "Weights file not found. Expected one of: " + ", ".join(DEFAULT_WEIGHT_CANDIDATES)
    )


def _load_once() -> None:
    global _MODEL, _TRACKER, _HIDDEN
    if _MODEL is not None and _TRACKER is not None and _HIDDEN is not None:
        return

    checkpoint = load_checkpoint(_checkpoint_path(), device=torch.device("cpu"))
    encoder_dims = tuple(int(x) for x in checkpoint.get("encoder_dims", [256, 128]))
    critic_hidden_dims = tuple(int(x) for x in checkpoint.get("critic_hidden_dims", [512, 256]))
    feature_payload = checkpoint.get("feature_config", {})
    feature_config = RecurrentFeatureConfig(
        max_steps=int(feature_payload.get("max_steps", 500)),
        pose_clip=float(feature_payload.get("pose_clip", 500.0)),
        blind_clip=float(feature_payload.get("blind_clip", 100.0)),
        stuck_clip=float(feature_payload.get("stuck_clip", 20.0)),
        contact_clip=float(feature_payload.get("contact_clip", 20.0)),
        same_obs_clip=float(feature_payload.get("same_obs_clip", 50.0)),
        wall_hit_clip=float(feature_payload.get("wall_hit_clip", 20.0)),
        last_action_hist=int(feature_payload.get("last_action_hist", 6)),
        heading_bins=int(feature_payload.get("heading_bins", 8)),
    )
    model = RecurrentAsymmetricActorCritic(
        actor_dim=feature_config.feature_dim,
        privileged_dim=int(checkpoint.get("privileged_dim", 34)),
        encoder_dims=encoder_dims,
        gru_hidden_dim=int(checkpoint.get("gru_hidden_dim", 128)),
        critic_hidden_dims=critic_hidden_dims,
        gru_layers=int(checkpoint.get("gru_layers", 1)),
        gru_dropout=float(checkpoint.get("gru_dropout", 0.0)),
        fw_bias_init=0.0,
    )
    model.actor_encoder.load_state_dict(checkpoint["actor_encoder_state_dict"], strict=True)
    model.actor_rnn.load_state_dict(checkpoint["actor_rnn_state_dict"], strict=True)
    model.policy_head.load_state_dict(checkpoint["policy_head_state_dict"], strict=True)
    model.eval()

    _MODEL = model
    _TRACKER = PoseMemoryTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
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
    logits, _HIDDEN = _MODEL.actor_step(features, _HIDDEN, starts)
    action_idx = int(torch.argmax(logits, dim=1).item())
    _PENDING_ACTION = action_idx
    return ACTIONS[action_idx]
