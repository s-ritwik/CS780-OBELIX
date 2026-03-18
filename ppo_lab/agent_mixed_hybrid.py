from __future__ import annotations

import importlib.util
import os
from typing import Optional

import numpy as np


_EXPLORER = None
_CONSERVATIVE = None
_LAST_RNG_ID: Optional[int] = None
_LAST_ACTION: Optional[str] = None
_USE_CONSERVATIVE = False
_STUCK_EVENTS = 0
_SENSOR_SEEN = np.zeros((17,), dtype=bool)
_PREV_BITS: Optional[np.ndarray] = None
_REPEAT_STEPS = 0


def _import_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import agent module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_agents() -> None:
    global _EXPLORER, _CONSERVATIVE
    if _EXPLORER is not None and _CONSERVATIVE is not None:
        return

    here = os.path.dirname(__file__)
    explorer_path = os.path.join(here, "agent_mixed_submission.py")
    conservative_path = os.path.join(here, "agent.py")

    os.environ["OBELIX_STOCHASTIC"] = "1"

    os.environ["OBELIX_WEIGHTS"] = os.path.join(here, "mixed_scaled_ft1.pth")
    _EXPLORER = _import_module(explorer_path, "explorer_agent")
    if hasattr(_EXPLORER, "_load_once"):
        _EXPLORER._load_once()

    conservative_weights = os.path.join(here, "mixed_seenmask_ft1_snapshot.pth")
    if not os.path.exists(conservative_weights):
        conservative_weights = os.path.join(here, "mixed_seenmask_ft1.pth")
    os.environ["OBELIX_WEIGHTS"] = conservative_weights
    _CONSERVATIVE = _import_module(conservative_path, "conservative_agent")
    if hasattr(_CONSERVATIVE, "_load_once"):
        _CONSERVATIVE._load_once()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_RNG_ID, _LAST_ACTION, _USE_CONSERVATIVE, _STUCK_EVENTS, _SENSOR_SEEN, _PREV_BITS, _REPEAT_STEPS
    _load_agents()

    rng_id = id(rng)
    obs_arr = np.asarray(obs, dtype=np.float32)
    obs_bits = (obs_arr > 0.5).astype(np.int8, copy=False)
    if _LAST_RNG_ID != rng_id:
        _LAST_RNG_ID = rng_id
        _LAST_ACTION = None
        _USE_CONSERVATIVE = False
        _STUCK_EVENTS = 0
        _SENSOR_SEEN = np.zeros((17,), dtype=bool)
        _PREV_BITS = None
        _REPEAT_STEPS = 0

    if _PREV_BITS is not None and np.array_equal(obs_bits[:17], _PREV_BITS[:17]):
        _REPEAT_STEPS += 1
    else:
        _REPEAT_STEPS = 0

    new_bits = np.logical_and(obs_bits[:17].astype(bool), ~_SENSOR_SEEN)
    _SENSOR_SEEN |= obs_bits[:17].astype(bool)

    if obs_arr[17] > 0.5 and _LAST_ACTION == "FW":
        _STUCK_EVENTS += 1
        _USE_CONSERVATIVE = True

    front_near = int(np.sum(obs_bits[5:12:2]))
    any_visible = bool(np.any(obs_bits[:16]))
    front_visible = int(np.sum(obs_bits[4:12]))
    temporary_conservative = (
        any_visible
        and front_near == 0
        and front_visible > 0
        and (not np.any(new_bits[4:17]))
        and _REPEAT_STEPS >= 1
    )

    if _USE_CONSERVATIVE or temporary_conservative:
        action = _CONSERVATIVE.policy(obs_arr, rng)
    else:
        action = _EXPLORER.policy(obs_arr, rng)

    _PREV_BITS = obs_bits.copy()
    _LAST_ACTION = action
    return action
