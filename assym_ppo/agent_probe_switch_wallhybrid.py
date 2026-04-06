from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np


HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SUBMISSION_MODULE_PATH = os.path.join(HERE, "submission_probe_switch_v2", "agent.py")
SUBMISSION_WEIGHTS = os.path.join(HERE, "submission_probe_switch_v2", "weights.pth")
GRU_WALL_PATH = os.path.join(ROOT, "submission_gru_pose_wall", "agent.py")
GRU_WALL_WEIGHTS = os.path.join(ROOT, "submission_gru_pose_wall", "weights.pth")
HANDMADE_PATH = os.path.join(ROOT, "handmade", "agent.py")

PROBE_STEPS = 80
CONTACT_THRESHOLD = 15
BLIND_THRESHOLD = 45
FRONT_TOTAL_THRESHOLD = 20

HANDMADE_PROBE_STEPS = 96
HANDMADE_CONTACT_STEPS = 220
HANDMADE_STUCK_SWITCH = 2
HANDMADE_BLIND_SWITCH = 24

LEFT1_EXACT = (
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)
CROSS_SIDE_EXACT = (
    1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
)
HANDMADE_PATTERNS = {LEFT1_EXACT, CROSS_SIDE_EXACT}

_SUBMISSION: ModuleType | None = None
_GRU_WALL: ModuleType | None = None
_HANDMADE: ModuleType | None = None
_PROBE_WALL = None
_NOWALL = None
_LAST_EPISODE_KEY: int | None = None
_STEP_COUNT = 0
_CONTACT_COUNT = 0
_BLIND_COUNT = 0
_FRONT_TOTAL = 0
_SWITCHED = False
_DECIDED = False

_WALL_MODE = "gru"
_WALL_STEP = 0
_WALL_STUCK_COUNT = 0
_WALL_BLIND_STREAK = 0
_WALL_SEEN_CONTACT = False


def _load_module(path: str, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_once():
    global _SUBMISSION, _GRU_WALL, _HANDMADE, _PROBE_WALL, _NOWALL
    if _SUBMISSION is None:
        _SUBMISSION = _load_module(SUBMISSION_MODULE_PATH, "probe_switch_wallhybrid_bundle")
    if _GRU_WALL is None:
        _GRU_WALL = _load_module(GRU_WALL_PATH, "probe_switch_wallhybrid_gru")
        _GRU_WALL._checkpoint_path = lambda: GRU_WALL_WEIGHTS
    if _HANDMADE is None:
        _HANDMADE = _load_module(HANDMADE_PATH, "probe_switch_wallhybrid_handmade")
    if _PROBE_WALL is None or _NOWALL is None:
        bundle = _SUBMISSION._load_checkpoint(SUBMISSION_WEIGHTS)
        wall_ckpt = bundle.get("wall")
        nowall_ckpt = bundle.get("nowall")
        if not isinstance(wall_ckpt, dict) or not isinstance(nowall_ckpt, dict):
            raise RuntimeError("weights.pth must contain dict entries 'wall' and 'nowall'")
        if _PROBE_WALL is None:
            _PROBE_WALL = _SUBMISSION.WallPolicy(wall_ckpt)
        _NOWALL = _SUBMISSION.NoWallPolicy(nowall_ckpt)
    return _PROBE_WALL, _GRU_WALL, _HANDMADE, _NOWALL


def _episode_key(rng: np.random.Generator) -> int:
    seed_seq = getattr(rng.bit_generator, "_seed_seq", None)
    if seed_seq is not None and hasattr(seed_seq, "entropy"):
        return int(seed_seq.entropy)
    return id(rng)


def _reset_episode(rng: np.random.Generator, obs_arr: np.ndarray) -> None:
    global _LAST_EPISODE_KEY, _STEP_COUNT, _CONTACT_COUNT, _BLIND_COUNT, _FRONT_TOTAL, _SWITCHED, _DECIDED
    global _WALL_MODE, _WALL_STEP, _WALL_STUCK_COUNT, _WALL_BLIND_STREAK, _WALL_SEEN_CONTACT
    _LAST_EPISODE_KEY = _episode_key(rng)
    _STEP_COUNT = 0
    _CONTACT_COUNT = 0
    _BLIND_COUNT = 0
    _FRONT_TOTAL = 0
    _SWITCHED = False
    _DECIDED = False

    bits = tuple((obs_arr > 0.5).astype(np.int8, copy=False).tolist())
    _WALL_MODE = "handmade_probe" if bits in HANDMADE_PATTERNS else "gru"
    _WALL_STEP = 0
    _WALL_STUCK_COUNT = 0
    _WALL_BLIND_STREAK = 0
    _WALL_SEEN_CONTACT = False


def _update_probe_stats(obs_arr: np.ndarray) -> None:
    global _CONTACT_COUNT, _BLIND_COUNT, _FRONT_TOTAL
    front_count = int(np.sum(obs_arr[4:12] > 0.5))
    front_near = int(np.sum(obs_arr[5:12:2] > 0.5))
    _CONTACT_COUNT += int(front_near >= 1 or front_count >= 4)
    _BLIND_COUNT += int(np.sum(obs_arr[:16]) == 0.0)
    _FRONT_TOTAL += front_count


def _wall_action(obs_arr: np.ndarray, rng: np.random.Generator) -> str:
    global _WALL_MODE, _WALL_STEP, _WALL_STUCK_COUNT, _WALL_BLIND_STREAK, _WALL_SEEN_CONTACT
    _, gru_wall, handmade, _ = _load_once()

    front_count = int(np.sum(obs_arr[4:12] > 0.5))
    any_visible = bool(np.any(obs_arr[:16] > 0.5))
    ir = bool(obs_arr[16] > 0.5)
    stuck = bool(obs_arr[17] > 0.5)

    if stuck:
        _WALL_STUCK_COUNT += 1
    if not any_visible:
        _WALL_BLIND_STREAK += 1
    else:
        _WALL_BLIND_STREAK = 0
    if ir or front_count > 0:
        _WALL_SEEN_CONTACT = True

    if _WALL_MODE == "handmade_probe":
        if (
            (_WALL_STUCK_COUNT >= HANDMADE_STUCK_SWITCH and not _WALL_SEEN_CONTACT)
            or (_WALL_BLIND_STREAK >= HANDMADE_BLIND_SWITCH)
            or (_WALL_STEP >= HANDMADE_PROBE_STEPS and not _WALL_SEEN_CONTACT)
            or (_WALL_STEP >= HANDMADE_CONTACT_STEPS and not ir)
        ):
            _WALL_MODE = "gru"

    action = handmade.policy(obs_arr, rng) if _WALL_MODE == "handmade_probe" else gru_wall.policy(obs_arr, rng)
    _WALL_STEP += 1
    return action


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _STEP_COUNT, _SWITCHED, _DECIDED
    probe_wall, _, _, nowall = _load_once()

    obs_arr = np.asarray(obs, dtype=np.float32)
    if _LAST_EPISODE_KEY != _episode_key(rng):
        _reset_episode(rng, obs_arr)

    if _STEP_COUNT < PROBE_STEPS:
        _update_probe_stats(obs_arr)
        action = probe_wall.act(obs_arr, rng)
    else:
        if not _DECIDED:
            _SWITCHED = (
                (_CONTACT_COUNT <= CONTACT_THRESHOLD)
                and (_BLIND_COUNT >= BLIND_THRESHOLD)
                and (_FRONT_TOTAL >= FRONT_TOTAL_THRESHOLD)
            )
            _DECIDED = True
        action = nowall.act(obs_arr, rng) if _SWITCHED else _wall_action(obs_arr, rng)

    _STEP_COUNT += 1
    return action
