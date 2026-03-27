from __future__ import annotations

import importlib.util
import os
from types import ModuleType
from typing import Optional

import numpy as np


HERE = os.path.dirname(__file__)
EXPERT_PATH = os.path.join(HERE, "expert_conservative.py")
NOWALL_PATH = os.path.join(HERE, "agent_nowall_submission.py")
SAFE_PATH = os.path.join(HERE, "agent_seenmask_submission.py")
WALL_DETECT_STEPS = 16
WALL_VISIBLE_THRESHOLD = 8
WALL_FALLBACK_STEP = 96
NOWALL_FALLBACK_STEP = 260

_EXPERT: ModuleType | None = None
_NOWALL: ModuleType | None = None
_SAFE: ModuleType | None = None


def _load_module(path: str, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_once() -> tuple[ModuleType, ModuleType, ModuleType]:
    global _EXPERT, _NOWALL, _SAFE
    if _EXPERT is None:
        _EXPERT = _load_module(EXPERT_PATH, "hybrid_curriculum_expert")
    if _NOWALL is None:
        _NOWALL = _load_module(NOWALL_PATH, "hybrid_curriculum_nowall")
    if _SAFE is None:
        _SAFE = _load_module(SAFE_PATH, "hybrid_curriculum_safe")
    return _EXPERT, _NOWALL, _SAFE


class EpisodeState:
    def __init__(self) -> None:
        self.rng_id: Optional[int] = None
        self.step_idx = 0
        self.contact_steps = 0
        self.visible_steps = 0
        self.early_visible_steps = 0
        self.stuck_events = 0
        self.last_stuck = False
        self.fallback_mode: Optional[str] = None
        self.probable_wall: Optional[bool] = None
        self.ir_seen = False

    def reset(self, rng_id: int) -> None:
        self.rng_id = rng_id
        self.step_idx = 0
        self.contact_steps = 0
        self.visible_steps = 0
        self.early_visible_steps = 0
        self.stuck_events = 0
        self.last_stuck = False
        self.fallback_mode = None
        self.probable_wall = None
        self.ir_seen = False


_STATE = EpisodeState()


def _front_near_count(obs: np.ndarray) -> int:
    return int(np.sum(obs[5:12:2] > 0.5))


def _visible(obs: np.ndarray) -> bool:
    return bool(np.any(obs[:16] > 0.5))


def _contact_like(obs: np.ndarray) -> bool:
    return bool(obs[16] > 0.5)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    expert, nowall, safe = _load_once()
    obs_arr = np.asarray(obs, dtype=np.float32)
    rng_id = id(rng)
    if _STATE.rng_id != rng_id:
        _STATE.reset(rng_id)

    _STATE.step_idx += 1
    visible = _visible(obs_arr)
    contact_like = _contact_like(obs_arr)
    stuck = bool(obs_arr[17] > 0.5)

    if visible:
        _STATE.visible_steps += 1
        if _STATE.step_idx <= WALL_DETECT_STEPS:
            _STATE.early_visible_steps += 1
    if contact_like:
        _STATE.contact_steps += 1
        _STATE.ir_seen = True
    if stuck and (not _STATE.last_stuck):
        _STATE.stuck_events += 1
    _STATE.last_stuck = stuck

    if _STATE.probable_wall is None and _STATE.step_idx >= WALL_DETECT_STEPS:
        _STATE.probable_wall = _STATE.early_visible_steps >= WALL_VISIBLE_THRESHOLD

    if _STATE.fallback_mode is None:
        if _STATE.probable_wall:
            if _STATE.step_idx >= WALL_FALLBACK_STEP:
                _STATE.fallback_mode = "safe"
        else:
            no_progress = (not _STATE.ir_seen) and (_STATE.step_idx >= NOWALL_FALLBACK_STEP)
            if no_progress:
                _STATE.fallback_mode = "nowall"

    if _STATE.fallback_mode == "nowall":
        return nowall.policy(obs_arr, rng)
    if _STATE.fallback_mode == "safe":
        return safe.policy(obs_arr, rng)
    return expert.policy(obs_arr, rng)
