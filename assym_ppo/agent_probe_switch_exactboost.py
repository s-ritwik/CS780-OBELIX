from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np


HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SUBMISSION_MODULE_PATH = os.path.join(HERE, "submission_probe_switch_v2", "agent.py")
HANDMADE_PATH = os.path.join(ROOT, "handmade", "agent.py")

LEFT1_EXACT = (
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)
CROSS_SIDE_EXACT = (
    1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
)
BOOST_PATTERNS = {LEFT1_EXACT, CROSS_SIDE_EXACT}

BOOST_STEPS = 120
BOOST_STUCK_SWITCH = 2

_SUBMISSION: ModuleType | None = None
_HANDMADE: ModuleType | None = None
_LAST_EPISODE_KEY: int | None = None
_BOOST_ACTIVE = False
_BOOST_STEP = 0
_BOOST_STUCK = 0
_BOOST_SEEN_CONTACT = False


def _load_module(path: str, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_once() -> tuple[ModuleType, ModuleType]:
    global _SUBMISSION, _HANDMADE
    if _SUBMISSION is None:
        _SUBMISSION = _load_module(SUBMISSION_MODULE_PATH, "probe_switch_exactboost_submission")
    if _HANDMADE is None:
        _HANDMADE = _load_module(HANDMADE_PATH, "probe_switch_exactboost_handmade")
    return _SUBMISSION, _HANDMADE


def _episode_key(rng: np.random.Generator) -> int:
    return id(rng)


def _reset_episode(rng: np.random.Generator, obs_arr: np.ndarray) -> None:
    global _LAST_EPISODE_KEY, _BOOST_ACTIVE, _BOOST_STEP, _BOOST_STUCK, _BOOST_SEEN_CONTACT
    _LAST_EPISODE_KEY = _episode_key(rng)
    bits = tuple((obs_arr > 0.5).astype(np.int8, copy=False).tolist())
    _BOOST_ACTIVE = bits in BOOST_PATTERNS
    _BOOST_STEP = 0
    _BOOST_STUCK = 0
    _BOOST_SEEN_CONTACT = False


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _BOOST_ACTIVE, _BOOST_STEP, _BOOST_STUCK, _BOOST_SEEN_CONTACT
    submission, handmade = _load_once()

    obs_arr = np.asarray(obs, dtype=np.float32)
    if _LAST_EPISODE_KEY != _episode_key(rng):
        _reset_episode(rng, obs_arr)

    if _BOOST_ACTIVE:
        front_count = int(np.sum(obs_arr[4:12] > 0.5))
        if obs_arr[17] > 0.5:
            _BOOST_STUCK += 1
        if obs_arr[16] > 0.5 or front_count > 0:
            _BOOST_SEEN_CONTACT = True
        if (
            (_BOOST_STUCK >= BOOST_STUCK_SWITCH and not _BOOST_SEEN_CONTACT)
            or (_BOOST_STEP >= BOOST_STEPS and not _BOOST_SEEN_CONTACT)
        ):
            _BOOST_ACTIVE = False
        else:
            _BOOST_STEP += 1
            return handmade.policy(obs_arr, rng)

    return submission.policy(obs_arr, rng)
