from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np


HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
HANDMADE_PATH = os.path.join(ROOT, "handmade", "agent.py")
WALL_PATH = os.path.join(HERE, "agent_gru_wall_open.py")

EARLY_STEPS = int(os.environ.get("OBELIX_SWITCH_EARLY_STEPS", "120"))
FRONT_TRIGGER = int(os.environ.get("OBELIX_SWITCH_FRONT_TRIGGER", "6"))
SIDE_TRIGGER = int(os.environ.get("OBELIX_SWITCH_SIDE_TRIGGER", "4"))
STUCK_TRIGGER = int(os.environ.get("OBELIX_SWITCH_STUCK_TRIGGER", "1"))
BLIND_TRIGGER = int(os.environ.get("OBELIX_SWITCH_BLIND_TRIGGER", "999999"))
REPEAT_TRIGGER = int(os.environ.get("OBELIX_SWITCH_REPEAT_TRIGGER", "999999"))
VISIBLE_REPEAT_TRIGGER = int(os.environ.get("OBELIX_SWITCH_VISIBLE_REPEAT_TRIGGER", "999999"))
MIN_SWITCH_STEP = int(os.environ.get("OBELIX_SWITCH_MIN_STEP", "0"))

_HANDMADE: ModuleType | None = None
_WALL: ModuleType | None = None
_LAST_EPISODE_KEY: int | None = None
_STEP = 0
_WALL_MODE = False
_TOTAL_STUCK = 0
_BLIND_STREAK = 0
_REPEAT_STREAK = 0
_LAST_OBS: np.ndarray | None = None


def _load_module(path: str, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_once() -> tuple[ModuleType, ModuleType]:
    global _HANDMADE, _WALL
    if _HANDMADE is None:
        _HANDMADE = _load_module(HANDMADE_PATH, "handmade_switch")
    if _WALL is None:
        _WALL = _load_module(WALL_PATH, "gru_wall_switch")
    return _HANDMADE, _WALL


def _episode_key(rng: np.random.Generator) -> int:
    seed_seq = getattr(rng.bit_generator, "_seed_seq", None)
    if seed_seq is not None and hasattr(seed_seq, "entropy"):
        return int(seed_seq.entropy)
    return id(rng)


def _reset_episode(rng: np.random.Generator) -> None:
    global _LAST_EPISODE_KEY, _STEP, _WALL_MODE, _TOTAL_STUCK, _BLIND_STREAK, _REPEAT_STREAK, _LAST_OBS
    _LAST_EPISODE_KEY = _episode_key(rng)
    _STEP = 0
    _WALL_MODE = False
    _TOTAL_STUCK = 0
    _BLIND_STREAK = 0
    _REPEAT_STREAK = 0
    _LAST_OBS = None


def _should_switch(obs_arr: np.ndarray) -> bool:
    if _STEP < MIN_SWITCH_STEP:
        return False
    if _TOTAL_STUCK >= STUCK_TRIGGER:
        return True
    front_count = int(np.sum(obs_arr[4:12] > 0.5))
    left_count = int(np.sum(obs_arr[:4] > 0.5))
    right_count = int(np.sum(obs_arr[12:16] > 0.5))
    if _STEP < EARLY_STEPS and front_count >= FRONT_TRIGGER:
        return True
    if _STEP < EARLY_STEPS and max(left_count, right_count) >= SIDE_TRIGGER:
        return True
    if (front_count + left_count + right_count) > 0 and _REPEAT_STREAK >= VISIBLE_REPEAT_TRIGGER:
        return True
    if _BLIND_STREAK >= BLIND_TRIGGER and _REPEAT_STREAK >= REPEAT_TRIGGER:
        return True
    return False


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _STEP, _WALL_MODE, _TOTAL_STUCK, _BLIND_STREAK, _REPEAT_STREAK, _LAST_OBS
    handmade, wall = _load_once()

    if _LAST_EPISODE_KEY != _episode_key(rng):
        _reset_episode(rng)

    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_arr[17] > 0.5:
        _TOTAL_STUCK += 1
    if float(np.sum(obs_arr[:16])) == 0.0:
        _BLIND_STREAK += 1
    else:
        _BLIND_STREAK = 0
    if _LAST_OBS is not None and np.array_equal(obs_arr, _LAST_OBS):
        _REPEAT_STREAK += 1
    else:
        _REPEAT_STREAK = 0
    _LAST_OBS = obs_arr.copy()

    if not _WALL_MODE and _should_switch(obs_arr):
        _WALL_MODE = True

    action = wall.policy(obs_arr, rng) if _WALL_MODE else handmade.policy(obs_arr, rng)
    _STEP += 1
    return action
