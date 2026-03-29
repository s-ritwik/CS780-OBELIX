from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np


HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
HANDMADE_PATH = os.path.join(ROOT, "handmade", "agent.py")
WALL_PATH = os.path.join(HERE, "agent_gru_wall_open.py")

_HANDMADE: ModuleType | None = None
_WALL: ModuleType | None = None
_LAST_EPISODE_KEY: int | None = None
_MODE = "handmade"
_INITIAL_CLASSIFIED = False
_BLIND_START = False
_SEEN_VISIBLE = False
_STEP = 0
BLIND_STUCK_SWITCH_STEP = 12
BLIND_VISIBLE_SWITCH_STEP = 60
_BLIND_STUCK_BEFORE_VISIBLE = False


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
        _HANDMADE = _load_module(HANDMADE_PATH, "initial_classifier_handmade")
    if _WALL is None:
        _WALL = _load_module(WALL_PATH, "initial_classifier_wall")
    return _HANDMADE, _WALL


def _episode_key(rng: np.random.Generator) -> int:
    seed_seq = getattr(rng.bit_generator, "_seed_seq", None)
    if seed_seq is not None and hasattr(seed_seq, "entropy"):
        return int(seed_seq.entropy)
    return id(rng)


def _reset_episode(rng: np.random.Generator) -> None:
    global _LAST_EPISODE_KEY, _MODE, _INITIAL_CLASSIFIED, _BLIND_START, _SEEN_VISIBLE, _STEP, _BLIND_STUCK_BEFORE_VISIBLE
    _LAST_EPISODE_KEY = _episode_key(rng)
    _MODE = "handmade"
    _INITIAL_CLASSIFIED = False
    _BLIND_START = False
    _SEEN_VISIBLE = False
    _STEP = 0
    _BLIND_STUCK_BEFORE_VISIBLE = False


def _classify_initial(obs_arr: np.ndarray) -> None:
    global _MODE, _INITIAL_CLASSIFIED, _BLIND_START
    left_count = int(np.sum(obs_arr[:4] > 0.5))
    front_count = int(np.sum(obs_arr[4:12] > 0.5))
    right_count = int(np.sum(obs_arr[12:16] > 0.5))

    if front_count > 0 or right_count > 0 or left_count >= 4 or (left_count > 0 and right_count > 0):
        _MODE = "wall"
    else:
        _MODE = "handmade"
    _BLIND_START = (left_count + front_count + right_count) == 0
    _INITIAL_CLASSIFIED = True


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _MODE, _SEEN_VISIBLE, _STEP, _BLIND_STUCK_BEFORE_VISIBLE
    handmade, wall = _load_once()

    if _LAST_EPISODE_KEY != _episode_key(rng):
        _reset_episode(rng)

    obs_arr = np.asarray(obs, dtype=np.float32)
    if not _INITIAL_CLASSIFIED:
        _classify_initial(obs_arr)

    currently_visible = bool(np.any(obs_arr[:16] > 0.5))
    if currently_visible and _BLIND_START and _BLIND_STUCK_BEFORE_VISIBLE and _STEP <= BLIND_VISIBLE_SWITCH_STEP:
        _MODE = "wall"
    if currently_visible:
        _SEEN_VISIBLE = True

    if (
        _MODE == "handmade"
        and _BLIND_START
        and not _SEEN_VISIBLE
        and obs_arr[17] > 0.5
        and _STEP <= BLIND_STUCK_SWITCH_STEP
    ):
        _MODE = "wall"
    elif _MODE == "handmade" and _BLIND_START and not _SEEN_VISIBLE and obs_arr[17] > 0.5:
        _BLIND_STUCK_BEFORE_VISIBLE = True

    action = wall.policy(obs_arr, rng) if _MODE == "wall" else handmade.policy(obs_arr, rng)
    _STEP += 1
    return action
