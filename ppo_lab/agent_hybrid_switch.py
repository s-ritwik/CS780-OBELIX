from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np


HERE = os.path.dirname(__file__)
NOWALL_WEIGHTS = os.path.join(HERE, "weights_nowall.pth")
SAFE_WEIGHTS = os.environ.get("OBELIX_SAFE_WEIGHTS", "/tmp/ppo_runs/static_warm_wall_best.pth")
NOWALL_MODULE = os.path.join(HERE, "agent_rec_ppo.py")
SAFE_MODULE = os.path.join(HERE, "agent.py")

_AGGRESSIVE: ModuleType | None = None
_SAFE: ModuleType | None = None
_LAST_RNG_ID: int | None = None
_USE_SAFE = True
_STEP_COUNT = 0
_BLIND_STEPS = 0
_STUCK_STEPS = 0
_EVER_STUCK = False


def _load_policy_module(path: str, module_name: str, weights_path: str, stochastic: bool) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _fixed_checkpoint_path() -> str:
        return weights_path

    module._checkpoint_path = _fixed_checkpoint_path  # type: ignore[attr-defined]
    if hasattr(module, "_STOCHASTIC"):
        module._STOCHASTIC = bool(stochastic)  # type: ignore[attr-defined]
    return module


def _load_once() -> tuple[ModuleType, ModuleType]:
    global _AGGRESSIVE, _SAFE
    if _AGGRESSIVE is None:
        _AGGRESSIVE = _load_policy_module(
            path=NOWALL_MODULE,
            module_name="hybrid_nowall_policy",
            weights_path=NOWALL_WEIGHTS,
            stochastic=True,
        )
    if _SAFE is None:
        _SAFE = _load_policy_module(
            path=SAFE_MODULE,
            module_name="hybrid_safe_policy",
            weights_path=SAFE_WEIGHTS,
            stochastic=False,
        )
    return _AGGRESSIVE, _SAFE


def _reset_episode(rng: np.random.Generator) -> None:
    global _LAST_RNG_ID, _USE_SAFE, _STEP_COUNT, _BLIND_STEPS, _STUCK_STEPS, _EVER_STUCK
    _LAST_RNG_ID = id(rng)
    _USE_SAFE = True
    _STEP_COUNT = 0
    _BLIND_STEPS = 0
    _STUCK_STEPS = 0
    _EVER_STUCK = False


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _USE_SAFE, _STEP_COUNT, _BLIND_STEPS, _STUCK_STEPS, _EVER_STUCK
    aggressive, safe = _load_once()

    if _LAST_RNG_ID != id(rng):
        _reset_episode(rng)

    obs_arr = np.asarray(obs, dtype=np.float32)
    visible = bool(np.any(obs_arr[:16] > 0.5))
    stuck = bool(obs_arr[17] > 0.5)

    _STEP_COUNT += 1
    _BLIND_STEPS = 0 if visible else (_BLIND_STEPS + 1)
    _STUCK_STEPS = (_STUCK_STEPS + 1) if stuck else 0
    _EVER_STUCK = _EVER_STUCK or stuck

    # Stay safe by default. If the episode has shown no stuck events for a while,
    # hand control to the stronger no-wall specialist. Any later stuck event
    # immediately returns control to the safe policy permanently.
    if _USE_SAFE:
        if (not _EVER_STUCK) and _STEP_COUNT >= 80 and _BLIND_STEPS <= 30:
            _USE_SAFE = False
    elif _STUCK_STEPS >= 1:
        _USE_SAFE = True

    if _USE_SAFE:
        return safe.policy(obs_arr, rng)
    return aggressive.policy(obs_arr, rng)
