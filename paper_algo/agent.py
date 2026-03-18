from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
BASE_OBS_DIM = 18
POLICY_STATE_DIM = 19
PREV_STUCK_MEMORY_INDEX = 18
FORWARD_ACTION = ACTIONS.index("FW")
L45_ACTION = ACTIONS.index("L45")
R45_ACTION = ACTIONS.index("R45")
TURN_ACTIONS = frozenset((ACTIONS.index("L45"), ACTIONS.index("L22"), ACTIONS.index("R22"), ACTIONS.index("R45")))
BUMP_BIT = 16
STUCK_BIT = 17

_MODEL: dict[str, Any] | None = None
_LAST_RNG_ID: int | None = None
_PUSH_TIMER = 0
_UNWEDGE_TIMER = 0
_BLIND_TURN_STREAK = 0
_LAST_OBS_STUCK_BIT = 0
_UNWEDGE_TURN_ACTION: int | None = None
_UNWEDGE_PROBE_FORWARD = False
_LAST_UNWEDGE_TURN_SIGN = 0


def _binarize(obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float32) > 0.5).astype(np.int8, copy=False)


def _load_once() -> dict[str, Any]:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    here = os.path.dirname(__file__)
    candidate_paths = [
        os.environ.get("PAPER_POLICY_PATH"),
        os.path.join(here, "paper_policy.json"),
        os.path.join(here, "paper_algo", "paper_policy.json"),
    ]
    path = next((candidate for candidate in candidate_paths if candidate and os.path.exists(candidate)), None)
    if path is None:
        raise FileNotFoundError(
            "paper_policy.json not found. Expected it next to agent.py, "
            "in paper_algo/paper_policy.json, or at $PAPER_POLICY_PATH."
        )

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for behavior, module in payload["modules"].items():
        for action_model in module["actions"]:
            for cluster in action_model["clusters"]:
                cluster["probs"] = np.asarray(cluster["probs"], dtype=np.float32)
    _MODEL = payload
    return _MODEL


def _refresh_episode_state(rng: np.random.Generator) -> None:
    global _LAST_RNG_ID, _PUSH_TIMER, _UNWEDGE_TIMER, _BLIND_TURN_STREAK, _LAST_OBS_STUCK_BIT
    global _UNWEDGE_TURN_ACTION, _UNWEDGE_PROBE_FORWARD, _LAST_UNWEDGE_TURN_SIGN
    rng_id = id(rng)
    if _LAST_RNG_ID != rng_id:
        _LAST_RNG_ID = rng_id
        _PUSH_TIMER = 0
        _UNWEDGE_TIMER = 0
        _BLIND_TURN_STREAK = 0
        _LAST_OBS_STUCK_BIT = 0
        _UNWEDGE_TURN_ACTION = None
        _UNWEDGE_PROBE_FORWARD = False
        _LAST_UNWEDGE_TURN_SIGN = 0


def _update_timers(obs_bits: np.ndarray, model: dict[str, Any]) -> None:
    global _PUSH_TIMER, _UNWEDGE_TIMER

    if bool(obs_bits[BUMP_BIT]):
        _PUSH_TIMER = int(model.get("push_persistence_steps", 5))
    elif _PUSH_TIMER > 0:
        _PUSH_TIMER -= 1

    if bool(obs_bits[STUCK_BIT]):
        _UNWEDGE_TIMER = int(model.get("unwedge_persistence_steps", 5))
    elif _UNWEDGE_TIMER > 0:
        _UNWEDGE_TIMER -= 1


def _select_behavior(obs_bits: np.ndarray) -> str:
    if bool(obs_bits[STUCK_BIT]) or _UNWEDGE_TIMER > 0:
        return "unwedge"
    if bool(obs_bits[BUMP_BIT]) or _PUSH_TIMER > 0:
        return "push"
    return "find"


def _policy_obs_dim(model: dict[str, Any], behavior: str) -> int:
    module = model["modules"][behavior]
    return int(module.get("obs_dim", BASE_OBS_DIM))


def _encode_policy_state(obs_bits: np.ndarray, policy_obs_dim: int) -> np.ndarray:
    if policy_obs_dim <= BASE_OBS_DIM:
        return np.asarray(obs_bits[:policy_obs_dim], dtype=np.int8)

    state_bits = np.zeros((policy_obs_dim,), dtype=np.int8)
    copy_dim = min(BASE_OBS_DIM, policy_obs_dim)
    state_bits[:copy_dim] = np.asarray(obs_bits[:copy_dim], dtype=np.int8)
    if policy_obs_dim > PREV_STUCK_MEMORY_INDEX:
        state_bits[PREV_STUCK_MEMORY_INDEX] = 1 if _LAST_OBS_STUCK_BIT else 0
    return state_bits


def _choose_unwedge_turn_action(obs_bits: np.ndarray) -> tuple[int, int]:
    global _LAST_UNWEDGE_TURN_SIGN
    left_score = int(np.sum(obs_bits[:4]))
    right_score = int(np.sum(obs_bits[12:16]))
    if left_score > right_score:
        return R45_ACTION, -1
    if right_score > left_score:
        return L45_ACTION, 1
    if _LAST_UNWEDGE_TURN_SIGN >= 0:
        return R45_ACTION, -1
    return L45_ACTION, 1


def _unwedge_override(obs_bits: np.ndarray) -> int | None:
    global _UNWEDGE_TURN_ACTION, _UNWEDGE_PROBE_FORWARD, _LAST_UNWEDGE_TURN_SIGN

    if not bool(obs_bits[STUCK_BIT]):
        _UNWEDGE_TURN_ACTION = None
        _UNWEDGE_PROBE_FORWARD = False
        return None

    if _UNWEDGE_TURN_ACTION is None:
        _UNWEDGE_TURN_ACTION, _LAST_UNWEDGE_TURN_SIGN = _choose_unwedge_turn_action(obs_bits)

    if not _UNWEDGE_PROBE_FORWARD:
        _UNWEDGE_PROBE_FORWARD = True
        return int(_UNWEDGE_TURN_ACTION)

    _UNWEDGE_PROBE_FORWARD = False
    return FORWARD_ACTION


def _match_probability(cluster_probs: np.ndarray, state_bits: np.ndarray) -> float:
    probs = np.clip(cluster_probs, 1e-6, 1.0 - 1e-6)
    terms = np.where(state_bits > 0, probs, 1.0 - probs)
    return float(np.exp(np.sum(np.log(terms.astype(np.float64)))))


def _estimate_action_value(action_model: dict[str, Any], state_bits: np.ndarray) -> float:
    clusters = action_model["clusters"]
    if not clusters:
        return 0.0

    weights = np.asarray(
        [_match_probability(cluster["probs"], state_bits) for cluster in clusters],
        dtype=np.float64,
    )
    denom = float(weights.sum())
    if denom <= 0.0:
        return 0.0

    q_values = np.asarray([cluster["q_value"] for cluster in clusters], dtype=np.float64)
    return float(np.dot(weights, q_values) / denom)


def _action_values(module: dict[str, Any], state_bits: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_estimate_action_value(action_model, state_bits) for action_model in module["actions"]],
        dtype=np.float32,
    )


def _greedy_action(module: dict[str, Any], state_bits: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    q_values = _action_values(module, state_bits)
    max_q = float(np.max(q_values))
    candidates = np.flatnonzero(np.isclose(q_values, max_q))
    return int(rng.choice(candidates)), q_values


def _anti_spin_override(
    behavior: str,
    obs_bits: np.ndarray,
    action_idx: int,
    q_values: np.ndarray,
    state_bits: np.ndarray,
) -> int:
    global _BLIND_TURN_STREAK, _LAST_OBS_STUCK_BIT

    if behavior != "find" or bool(np.any(obs_bits[:16] > 0)):
        _BLIND_TURN_STREAK = 0
        return action_idx

    recent_stuck = (
        bool(state_bits[PREV_STUCK_MEMORY_INDEX])
        if state_bits.shape[0] > PREV_STUCK_MEMORY_INDEX
        else bool(_LAST_OBS_STUCK_BIT)
    )
    if recent_stuck and action_idx == FORWARD_ACTION:
        turn_choices = np.asarray([0, 1, 3, 4], dtype=np.int64)
        turn_q = q_values[turn_choices]
        return int(turn_choices[int(np.argmax(turn_q))])

    if action_idx in TURN_ACTIONS:
        _BLIND_TURN_STREAK += 1
    else:
        _BLIND_TURN_STREAK = 0
        return action_idx

    if recent_stuck:
        return action_idx

    if _BLIND_TURN_STREAK >= 4 and float(q_values[FORWARD_ACTION]) >= (float(np.max(q_values)) - 0.35):
        _BLIND_TURN_STREAK = 0
        return FORWARD_ACTION

    return action_idx


def _visible_find_override(obs_bits: np.ndarray, action_idx: int) -> int:
    left_count = int(np.sum(obs_bits[:4]))
    front_count = int(np.sum(obs_bits[4:12]))
    right_count = int(np.sum(obs_bits[12:16]))

    if front_count == 0 and left_count > right_count:
        if (left_count - right_count) >= 2:
            return L45_ACTION
        return ACTIONS.index("L22")
    if front_count == 0 and right_count > left_count:
        if (right_count - left_count) >= 2:
            return R45_ACTION
        return ACTIONS.index("R22")
    if front_count >= 2 and action_idx in TURN_ACTIONS:
        return FORWARD_ACTION
    return action_idx


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_OBS_STUCK_BIT
    model = _load_once()
    _refresh_episode_state(rng)
    obs_bits = _binarize(obs)
    _update_timers(obs_bits, model)
    unwedge_action = _unwedge_override(obs_bits)
    if unwedge_action is not None:
        _LAST_OBS_STUCK_BIT = int(obs_bits[STUCK_BIT])
        return ACTIONS[unwedge_action]

    behavior = _select_behavior(obs_bits)
    state_bits = _encode_policy_state(obs_bits, _policy_obs_dim(model, behavior))
    action_idx, q_values = _greedy_action(model["modules"][behavior], state_bits, rng)
    if behavior == "find" and bool(np.any(obs_bits[:16] > 0)):
        action_idx = _visible_find_override(obs_bits, action_idx)
    action_idx = _anti_spin_override(behavior, obs_bits, action_idx, q_values, state_bits)
    _LAST_OBS_STUCK_BIT = int(obs_bits[STUCK_BIT])
    return ACTIONS[action_idx]
