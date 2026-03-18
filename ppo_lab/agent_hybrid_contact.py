from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np


HERE = os.path.dirname(__file__)
REC_MODULE_PATH = os.path.join(HERE, "agent_rec_ppo.py")
NOWALL_WEIGHTS = os.path.join(HERE, "weights_nowall.pth")
WALL_WEIGHTS = os.path.join(HERE, "weights_wall.pth")

_NOWALL: ModuleType | None = None
_WALL: ModuleType | None = None


def _load_rec_policy(module_name: str, weights_path: str, stochastic: bool) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, REC_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {REC_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _fixed_checkpoint_path() -> str:
        return weights_path

    module._checkpoint_path = _fixed_checkpoint_path  # type: ignore[attr-defined]
    module._STOCHASTIC = bool(stochastic)  # type: ignore[attr-defined]
    return module


def _load_once() -> tuple[ModuleType, ModuleType]:
    global _NOWALL, _WALL
    if _NOWALL is None:
        _NOWALL = _load_rec_policy("contact_nowall_policy", NOWALL_WEIGHTS, True)
    if _WALL is None:
        _WALL = _load_rec_policy("contact_wall_policy", WALL_WEIGHTS, False)
    return _NOWALL, _WALL


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    nowall, wall = _load_once()
    obs_arr = np.asarray(obs, dtype=np.float32)

    front_count = int(np.sum(obs_arr[4:12] > 0.5))
    front_near = int(np.sum(obs_arr[5:12:2] > 0.5))
    ir = bool(obs_arr[16] > 0.5)
    stuck = bool(obs_arr[17] > 0.5)

    if not stuck and (ir or front_near >= 1 or front_count >= 4):
        return nowall.policy(obs_arr, rng)
    return wall.policy(obs_arr, rng)
