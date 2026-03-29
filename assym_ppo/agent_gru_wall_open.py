from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np
import torch


HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
GRU_AGENT_PATH = os.path.join(ROOT, "submission_gru_pose_wall", "agent.py")
GRU_WEIGHT_PATH = os.path.join(ROOT, "submission_gru_pose_wall", "weights.pth")

_MOD: ModuleType | None = None
_LAST_EPISODE_KEY: int | None = None
_PREV_ACTION: int | None = None
_OPEN_SEQ: list[str] = []
_STEP = 0


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wrapped_gru_wall", GRU_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {GRU_AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._checkpoint_path = lambda: GRU_WEIGHT_PATH
    module._load_once()
    return module


def _mod() -> ModuleType:
    global _MOD
    if _MOD is None:
        _MOD = _load_module()
    return _MOD


def _episode_key(rng: np.random.Generator) -> int:
    seed_seq = getattr(rng.bit_generator, "_seed_seq", None)
    if seed_seq is not None and hasattr(seed_seq, "entropy"):
        return int(seed_seq.entropy)
    return id(rng)


def _reset_episode(obs: np.ndarray, rng: np.random.Generator) -> None:
    global _LAST_EPISODE_KEY, _PREV_ACTION, _OPEN_SEQ, _STEP
    mod = _mod()
    obs_arr = np.asarray(obs, dtype=np.float32)
    mod._reset_episode(obs_arr)
    _LAST_EPISODE_KEY = _episode_key(rng)
    _PREV_ACTION = None
    _STEP = 0

    front = int(np.sum(obs_arr[4:12] > 0.5))
    if front >= 6:
        _OPEN_SEQ = ["FW"] * 6
    elif front > 0:
        _OPEN_SEQ = ["FW"] * 4
    else:
        _OPEN_SEQ = []


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _PREV_ACTION, _STEP
    mod = _mod()
    obs_arr = np.asarray(obs, dtype=np.float32)
    if _LAST_EPISODE_KEY != _episode_key(rng):
        _reset_episode(obs_arr, rng)

    if _PREV_ACTION is not None:
        mod._TRACKER.post_step(
            actions=torch.tensor([_PREV_ACTION], dtype=torch.long),
            next_obs=obs_arr[None, :],
            dones=np.asarray([False]),
        )
        starts = torch.zeros((1,), dtype=torch.float32)
    else:
        starts = torch.ones((1,), dtype=torch.float32)

    features = mod._TRACKER.features()
    logits, _, mod._HIDDEN = mod._MODEL.forward_step(features, mod._HIDDEN, starts)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64, copy=False)
    probs /= probs.sum()
    action_idx = int(rng.choice(len(mod.ACTIONS), p=probs))
    action = mod.ACTIONS[action_idx]

    if _STEP < len(_OPEN_SEQ):
        action = _OPEN_SEQ[_STEP]
        action_idx = mod.ACTIONS.index(action)

    _PREV_ACTION = action_idx
    _STEP += 1
    return action
