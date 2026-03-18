from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np


HERE = os.path.dirname(__file__)
BASE_MODULE_PATH = os.path.join(HERE, "agent.py")
BASE_WEIGHTS = os.environ.get("OBELIX_REACTIVE_WEIGHTS", "/tmp/ppo_runs/static_warm_wall_best.pth")
_BASE: ModuleType | None = None


def _load_once() -> ModuleType:
    global _BASE
    if _BASE is not None:
        return _BASE

    spec = importlib.util.spec_from_file_location("reactive_base_agent", BASE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {BASE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _fixed_checkpoint_path() -> str:
        return BASE_WEIGHTS

    module._checkpoint_path = _fixed_checkpoint_path  # type: ignore[attr-defined]
    module._STOCHASTIC = False  # type: ignore[attr-defined]
    _BASE = module
    return _BASE


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    base = _load_once()
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

    return base.policy(obs_arr, rng)
