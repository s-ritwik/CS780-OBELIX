from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch


HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from recurrent_v2 import ACTIONS, PoseMemoryTracker, RecurrentAsymmetricActorCritic, RecurrentFeatureConfig, load_checkpoint


DEFAULT_WEIGHT = "mixed_rnn_v2_ft2000_seed0_final.pth"

_MODEL: Optional[RecurrentAsymmetricActorCritic] = None
_TRACKER: Optional[PoseMemoryTracker] = None
_STATE = None
_LAST_RNG_ID: Optional[int] = None
_PENDING_ACTION: Optional[int] = None


def _checkpoint_path() -> str:
    override = os.environ.get("OBELIX_WEIGHTS")
    if override:
        return override if os.path.isabs(override) else os.path.join(HERE, override)
    return os.path.join(HERE, DEFAULT_WEIGHT)


def _load_once() -> None:
    global _MODEL, _TRACKER, _STATE
    if _MODEL is not None and _TRACKER is not None and _STATE is not None:
        return

    checkpoint = load_checkpoint(_checkpoint_path(), device=torch.device("cpu"))
    encoder_dims = tuple(int(x) for x in checkpoint.get("encoder_dims", [256, 128]))
    critic_hidden_dims = tuple(int(x) for x in checkpoint.get("critic_hidden_dims", [512, 256]))
    feature_payload = checkpoint.get("feature_config", {})
    feature_config = RecurrentFeatureConfig(**feature_payload)
    model = RecurrentAsymmetricActorCritic(
        actor_dim=feature_config.feature_dim,
        privileged_dim=int(checkpoint.get("privileged_dim", 34)),
        encoder_dims=encoder_dims,
        rnn_hidden_dim=int(checkpoint.get("rnn_hidden_dim", 192)),
        critic_hidden_dims=critic_hidden_dims,
        rnn_layers=int(checkpoint.get("rnn_layers", 1)),
        rnn_dropout=float(checkpoint.get("rnn_dropout", 0.0)),
        actor_dropout=float(checkpoint.get("actor_dropout", 0.0)),
        critic_dropout=float(checkpoint.get("critic_dropout", 0.0)),
        feature_dropout=float(checkpoint.get("feature_dropout", 0.0)),
        aux_target_dim=int(checkpoint.get("aux_target_dim", 0)),
        aux_hidden_dim=int(checkpoint.get("aux_hidden_dim", 0)),
        fw_bias_init=0.0,
        rnn_type=str(checkpoint.get("rnn_type", "gru")),
    )
    model.load_state_dict(checkpoint["full_state_dict"], strict=True)
    model.eval()

    _MODEL = model
    _TRACKER = PoseMemoryTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
    _STATE = _MODEL.initial_state(1, torch.device("cpu"))


def _reset_episode(obs: np.ndarray) -> None:
    global _TRACKER, _STATE, _PENDING_ACTION
    _TRACKER.reset_all(obs[None, :])
    _STATE = _MODEL.initial_state(1, torch.device("cpu"))
    _PENDING_ACTION = None


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_RNG_ID, _PENDING_ACTION, _STATE
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
    logits, _STATE = _MODEL.actor_step(features, _STATE, starts)
    action_idx = int(torch.argmax(logits, dim=1).item())
    _PENDING_ACTION = action_idx
    return ACTIONS[action_idx]
