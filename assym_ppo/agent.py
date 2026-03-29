from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch


HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from features import ACTIONS, FeatureConfig, FeatureTracker
from model import ActorOnly, load_checkpoint


_MODEL: Optional[ActorOnly] = None
_TRACKER: Optional[FeatureTracker] = None
_LAST_RNG_ID: Optional[int] = None
_PENDING_ACTION: Optional[int] = None
_STOCHASTIC = os.environ.get("OBELIX_STOCHASTIC", "0") != "0"


def _checkpoint_path() -> str:
    override = os.environ.get("OBELIX_WEIGHTS")
    if override:
        return override if os.path.isabs(override) else os.path.join(HERE, override)
    return os.path.join(HERE, "weights_best.pth")


def _load_once() -> None:
    global _MODEL, _TRACKER
    if _MODEL is not None and _TRACKER is not None:
        return

    checkpoint = load_checkpoint(_checkpoint_path(), device=torch.device("cpu"))
    actor_hidden_dims = tuple(int(x) for x in checkpoint.get("actor_hidden_dims", [256, 128]))
    feature_payload = checkpoint.get("feature_config", {})
    feature_config = FeatureConfig(**feature_payload)

    model = ActorOnly(
        actor_dim=feature_config.feature_dim,
        actor_hidden_dims=actor_hidden_dims,
        fw_bias_init=0.0,
    )
    model.actor_backbone.load_state_dict(checkpoint["actor_backbone_state_dict"], strict=True)
    model.policy_head.load_state_dict(checkpoint["policy_head_state_dict"], strict=True)
    model.eval()

    _MODEL = model
    _TRACKER = FeatureTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_RNG_ID, _PENDING_ACTION
    _load_once()

    obs_arr = np.asarray(obs, dtype=np.float32)
    rng_id = id(rng)
    if _LAST_RNG_ID != rng_id:
        _LAST_RNG_ID = rng_id
        _PENDING_ACTION = None
        _TRACKER.reset_all(obs_arr[None, :])
    elif _PENDING_ACTION is not None:
        _TRACKER.post_step(
            actions=torch.tensor([_PENDING_ACTION], dtype=torch.long),
            next_obs=obs_arr[None, :],
            dones=np.asarray([False]),
        )

    x = _TRACKER.features()
    logits = _MODEL(x)
    if _STOCHASTIC:
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64, copy=False)
        probs /= probs.sum()
        action_idx = int(rng.choice(len(ACTIONS), p=probs))
    else:
        action_idx = int(torch.argmax(logits, dim=1).item())
    _PENDING_ACTION = action_idx
    return ACTIONS[action_idx]
