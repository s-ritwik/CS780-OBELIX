from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np


HERE = os.path.dirname(__file__)
REC_MODULE_PATH = os.path.join(HERE, "agent_rec_ppo.py")
WALL_WEIGHTS = os.path.join(HERE, "weights_wall.pth")
_WALL: ModuleType | None = None


def _load_once() -> ModuleType:
    global _WALL
    if _WALL is not None:
        return _WALL

    spec = importlib.util.spec_from_file_location("wall_reactive_base", REC_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {REC_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _fixed_checkpoint_path() -> str:
        return WALL_WEIGHTS

    module._checkpoint_path = _fixed_checkpoint_path  # type: ignore[attr-defined]
    module._STOCHASTIC = False  # type: ignore[attr-defined]
    _WALL = module
    return _WALL


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    wall = _load_once()
    obs_arr = np.asarray(obs, dtype=np.float32)

    left_count = int(np.sum(obs_arr[:4] > 0.5))
    front_count = int(np.sum(obs_arr[4:12] > 0.5))
    right_count = int(np.sum(obs_arr[12:16] > 0.5))
    ir = bool(obs_arr[16] > 0.5)
    stuck = bool(obs_arr[17] > 0.5)

    if not stuck:
        if ir or front_count >= 3:
            return "FW"
        if front_count == 0 and left_count > right_count and left_count > 0:
            return "L45" if (left_count - right_count) >= 2 else "L22"
        if front_count == 0 and right_count > left_count and right_count > 0:
            return "R45" if (right_count - left_count) >= 2 else "R22"

    return wall.policy(obs_arr, rng)
