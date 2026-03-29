from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np


HERE = os.path.dirname(__file__)
WALL_MODULE_PATH = os.path.join(HERE, "agent.py")
WALL_WEIGHTS = os.path.join(HERE, "wall_tuned_v1_final.pth")
NOWALL_MODULE_PATH = os.path.join(os.path.dirname(HERE), "ppo_lab", "agent_nowall.py")

PROBE_STEPS = 80
CONTACT_THRESHOLD = 15
BLIND_THRESHOLD = 45
FRONT_TOTAL_THRESHOLD = 20

_WALL: ModuleType | None = None
_NOWALL: ModuleType | None = None
_LAST_RNG_ID: int | None = None
_STEP_COUNT = 0
_CONTACT_COUNT = 0
_BLIND_COUNT = 0
_FRONT_TOTAL = 0
_SWITCHED = False
_DECIDED = False


def _load_module(path: str, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_once() -> tuple[ModuleType, ModuleType]:
    global _WALL, _NOWALL
    if _WALL is None:
        _WALL = _load_module(WALL_MODULE_PATH, "probe_switch_wall")
        _WALL._checkpoint_path = lambda: WALL_WEIGHTS  # type: ignore[attr-defined]
    if _NOWALL is None:
        _NOWALL = _load_module(NOWALL_MODULE_PATH, "probe_switch_nowall")
    return _WALL, _NOWALL


def _reset_episode(rng: np.random.Generator) -> None:
    global _LAST_RNG_ID, _STEP_COUNT, _CONTACT_COUNT, _BLIND_COUNT, _FRONT_TOTAL, _SWITCHED, _DECIDED
    _LAST_RNG_ID = id(rng)
    _STEP_COUNT = 0
    _CONTACT_COUNT = 0
    _BLIND_COUNT = 0
    _FRONT_TOTAL = 0
    _SWITCHED = False
    _DECIDED = False


def _update_probe_stats(obs_arr: np.ndarray) -> None:
    global _CONTACT_COUNT, _BLIND_COUNT, _FRONT_TOTAL
    front_count = int(np.sum(obs_arr[4:12] > 0.5))
    front_near = int(np.sum(obs_arr[5:12:2] > 0.5))
    _CONTACT_COUNT += int(front_near >= 1 or front_count >= 4)
    _BLIND_COUNT += int(np.sum(obs_arr[:16]) == 0.0)
    _FRONT_TOTAL += front_count


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _STEP_COUNT, _SWITCHED, _DECIDED
    wall, nowall = _load_once()

    if _LAST_RNG_ID != id(rng):
        _reset_episode(rng)

    obs_arr = np.asarray(obs, dtype=np.float32)

    if _STEP_COUNT < PROBE_STEPS:
        _update_probe_stats(obs_arr)
        action = wall.policy(obs_arr, rng)
    else:
        if not _DECIDED:
            _SWITCHED = (
                (_CONTACT_COUNT <= CONTACT_THRESHOLD)
                and (_BLIND_COUNT >= BLIND_THRESHOLD)
                and (_FRONT_TOTAL >= FRONT_TOTAL_THRESHOLD)
            )
            _DECIDED = True
        action = nowall.policy(obs_arr, rng) if _SWITCHED else wall.policy(obs_arr, rng)

    _STEP_COUNT += 1
    return action
