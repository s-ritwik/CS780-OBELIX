from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np


HERE = os.path.dirname(__file__)
WALL_MODULE_PATH = os.path.join(HERE, "agent_initial_classifier.py")
NOWALL_SUBMISSION_PATH = os.path.join(HERE, "submission_probe_switch_v2", "agent.py")
NOWALL_WEIGHTS = os.path.join(HERE, "submission_probe_switch_v2", "weights.pth")

PROBE_STEPS = 80
CONTACT_THRESHOLD = 15
BLIND_THRESHOLD = 45
FRONT_TOTAL_THRESHOLD = 20

_WALL: ModuleType | None = None
_NOWALL_SUBMISSION: ModuleType | None = None
_NOWALL = None
_LAST_EPISODE_KEY: int | None = None
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


def _load_once():
    global _WALL, _NOWALL_SUBMISSION, _NOWALL
    if _WALL is None:
        _WALL = _load_module(WALL_MODULE_PATH, "probe_switch_initialwall_wall")
    if _NOWALL_SUBMISSION is None:
        _NOWALL_SUBMISSION = _load_module(NOWALL_SUBMISSION_PATH, "probe_switch_initialwall_bundle")
    if _NOWALL is None:
        bundle = _NOWALL_SUBMISSION._load_checkpoint(NOWALL_WEIGHTS)
        nowall_ckpt = bundle.get("nowall")
        if not isinstance(nowall_ckpt, dict):
            raise RuntimeError("weights.pth must contain a dict entry 'nowall'")
        _NOWALL = _NOWALL_SUBMISSION.NoWallPolicy(nowall_ckpt)
    return _WALL, _NOWALL


def _episode_key(rng: np.random.Generator) -> int:
    seed_seq = getattr(rng.bit_generator, "_seed_seq", None)
    if seed_seq is not None and hasattr(seed_seq, "entropy"):
        return int(seed_seq.entropy)
    return id(rng)


def _reset_episode(rng: np.random.Generator) -> None:
    global _LAST_EPISODE_KEY, _STEP_COUNT, _CONTACT_COUNT, _BLIND_COUNT, _FRONT_TOTAL, _SWITCHED, _DECIDED
    _LAST_EPISODE_KEY = _episode_key(rng)
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

    if _LAST_EPISODE_KEY != _episode_key(rng):
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
        action = nowall.act(obs_arr, rng) if _SWITCHED else wall.policy(obs_arr, rng)

    _STEP_COUNT += 1
    return action
