from __future__ import annotations

import importlib.util
import os
from typing import Optional

import numpy as np


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

HERE = os.path.dirname(__file__)
REPO_DIR = os.path.dirname(HERE)
EXPERT_PATH = os.path.join(REPO_DIR, "ppo_lab", "expert_conservative.py")
NOWALL_PATH = os.path.join(REPO_DIR, "ppo_lab", "agent_nowall_submission.py")
SAFE_PATH = os.path.join(REPO_DIR, "assym_ppo", "agent_mixed_rnn_v2_best.py")

EARLY_STEPS = 16
NOWALL_VISIBLE_MAX = 4
SEARCH_LIMIT = 160
NOWALL_TRANSFER_STEP = 300

_EXPERT = None
_NOWALL = None
_SAFE = None


def _load_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_once() -> None:
    global _EXPERT, _NOWALL, _SAFE
    if _EXPERT is None:
        _EXPERT = _load_module(EXPERT_PATH, "search_mix_expert")
    if _NOWALL is None:
        _NOWALL = _load_module(NOWALL_PATH, "search_mix_nowall")
    if _SAFE is None:
        _SAFE = _load_module(SAFE_PATH, "search_mix_safe")


class EpisodeState:
    def __init__(self) -> None:
        self.rng_id: Optional[int] = None
        self.step_idx = 0
        self.early_visible = 0
        self.probable_nowall: Optional[bool] = None
        self.visible_seen = False
        self.ir_seen = False
        self.last_stuck = False
        self.search_dir = 1
        self.search_plan: list[str] = []

    def reset(self, rng_id: int) -> None:
        self.rng_id = rng_id
        self.step_idx = 0
        self.early_visible = 0
        self.probable_nowall = None
        self.visible_seen = False
        self.ir_seen = False
        self.last_stuck = False
        self.search_dir = 1
        self.search_plan = []


_STATE = EpisodeState()


def _visible(obs: np.ndarray) -> bool:
    return bool(np.any(obs[:16] > 0.5))


def _contact(obs: np.ndarray) -> bool:
    return bool(obs[16] > 0.5)


def _next_search_action(stuck: bool) -> str:
    if stuck:
        _STATE.search_dir *= -1
        _STATE.search_plan = ["L45" if _STATE.search_dir > 0 else "R45", "FW"]
    elif not _STATE.search_plan:
        turn = "L22" if _STATE.search_dir > 0 else "R22"
        _STATE.search_plan = ["FW", "FW", turn]
        _STATE.search_dir *= -1
    return _STATE.search_plan.pop(0)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    obs_arr = np.asarray(obs, dtype=np.float32)
    rng_id = id(rng)
    if _STATE.rng_id != rng_id:
        _STATE.reset(rng_id)

    visible = _visible(obs_arr)
    contact = _contact(obs_arr)
    stuck = bool(obs_arr[17] > 0.5)

    if _STATE.step_idx < EARLY_STEPS and visible:
        _STATE.early_visible += 1
    if _STATE.step_idx == EARLY_STEPS and _STATE.probable_nowall is None:
        _STATE.probable_nowall = _STATE.early_visible <= NOWALL_VISIBLE_MAX
    if _STATE.probable_nowall is None and _STATE.step_idx > EARLY_STEPS:
        _STATE.probable_nowall = _STATE.early_visible <= NOWALL_VISIBLE_MAX

    _STATE.visible_seen = _STATE.visible_seen or visible
    _STATE.ir_seen = _STATE.ir_seen or contact

    if _STATE.probable_nowall:
        if (not _STATE.visible_seen) and _STATE.step_idx < SEARCH_LIMIT:
            action = _next_search_action(stuck)
        elif (not _STATE.ir_seen) and _STATE.step_idx >= NOWALL_TRANSFER_STEP:
            action = _NOWALL.policy(obs_arr, rng)
        else:
            action = _EXPERT.policy(obs_arr, rng)
    else:
        action = _SAFE.policy(obs_arr, rng)

    _STATE.last_stuck = stuck
    _STATE.step_idx += 1
    return action
