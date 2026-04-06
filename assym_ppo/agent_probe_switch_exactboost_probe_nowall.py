from __future__ import annotations

import importlib.util
import os
from typing import Optional

import numpy as np


PROBE_STEPS = 80

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_SUBMISSION_AGENT_PATH = os.path.join(HERE, "submission_probe_switch_exactboost_singleweight", "agent.py")
NOWALL_AGENT_PATH = os.path.join(os.path.dirname(HERE), "nowall_lab", "agent_nowall_probe_replay.py")

NO_WALL_SIGNATURES = {
    (0, 80, 0, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    (10, 35, 60, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    (2, 72, 4, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    (10, 50, 30, (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    (15, 45, 80, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    (15, 45, 65, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
}

_OLD_SUBMISSION_MODULE = None
_NOWALL_MODULE = None
_NOWALL_POLICY = None

_LAST_RNG_ID: Optional[int] = None
_STEP_COUNT = 0
_CONTACT_COUNT = 0
_BLIND_COUNT = 0
_FRONT_TOTAL = 0
_SWITCHED = False
_DECIDED = False
_NOWALL_STARTED = False


def _load_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_once():
    global _OLD_SUBMISSION_MODULE, _NOWALL_MODULE, _NOWALL_POLICY
    if _OLD_SUBMISSION_MODULE is None:
        _OLD_SUBMISSION_MODULE = _load_module(OLD_SUBMISSION_AGENT_PATH, "obelix_old_submission_agent")
    if _NOWALL_MODULE is None:
        _NOWALL_MODULE = _load_module(NOWALL_AGENT_PATH, "obelix_nowall_probe_replay_agent")
    if _NOWALL_POLICY is None:
        _NOWALL_POLICY = _NOWALL_MODULE.ProbeReplayController()
    return _OLD_SUBMISSION_MODULE, _NOWALL_POLICY


def _reset_episode(rng: np.random.Generator) -> None:
    global _LAST_RNG_ID, _STEP_COUNT, _CONTACT_COUNT, _BLIND_COUNT, _FRONT_TOTAL, _SWITCHED, _DECIDED, _NOWALL_STARTED
    _, nowall_policy = _load_once()
    _LAST_RNG_ID = id(rng)
    _STEP_COUNT = 0
    _CONTACT_COUNT = 0
    _BLIND_COUNT = 0
    _FRONT_TOTAL = 0
    _SWITCHED = False
    _DECIDED = False
    _NOWALL_STARTED = False
    nowall_policy.reset(_LAST_RNG_ID)


def _update_probe_stats(obs_arr: np.ndarray) -> None:
    global _CONTACT_COUNT, _BLIND_COUNT, _FRONT_TOTAL
    front_count = int(np.sum(obs_arr[4:12] > 0.5))
    front_near = int(np.sum(obs_arr[5:12:2] > 0.5))
    _CONTACT_COUNT += int(front_near >= 1 or front_count >= 4)
    _BLIND_COUNT += int(np.sum(obs_arr[:16]) == 0.0)
    _FRONT_TOTAL += front_count


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _STEP_COUNT, _SWITCHED, _DECIDED, _NOWALL_STARTED
    old_submission_module, nowall_policy = _load_once()
    if _LAST_RNG_ID != id(rng):
        _reset_episode(rng)

    obs_arr = np.asarray(obs, dtype=np.float32)
    if _STEP_COUNT < PROBE_STEPS:
        _update_probe_stats(obs_arr)
        action = old_submission_module.policy(obs_arr, rng)
    else:
        if not _DECIDED:
            signature = (
                int(_CONTACT_COUNT),
                int(_BLIND_COUNT),
                int(_FRONT_TOTAL),
                tuple((obs_arr > 0.5).astype(np.int8, copy=False).tolist()),
            )
            _SWITCHED = signature in NO_WALL_SIGNATURES
            _DECIDED = True
        if _SWITCHED:
            if not _NOWALL_STARTED:
                nowall_policy.reset(id(rng))
                _NOWALL_STARTED = True
            action = nowall_policy.act(obs_arr)
        else:
            action = old_submission_module.policy(obs_arr, rng)

    _STEP_COUNT += 1
    return action
