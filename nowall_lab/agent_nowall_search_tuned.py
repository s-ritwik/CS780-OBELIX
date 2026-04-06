from __future__ import annotations

from typing import Optional

import numpy as np


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
TURN_DELTAS = {"L45": 45.0, "L22": 22.5, "FW": 0.0, "R22": -22.5, "R45": -45.0}
FORWARD_STEP = 5.0

SWEEP_FW_STEPS = 50
REACQUIRE_TURNS = 4
PUSH_COMMIT_STEPS = 8


def _binarize(obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float32) > 0.5).astype(np.int8, copy=False)


def _wrap_angle(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


class ObsFeatures:
    def __init__(
        self,
        *,
        bits: np.ndarray,
        left_far: int,
        left_near: int,
        front_far: int,
        front_near: int,
        right_far: int,
        right_near: int,
        ir: int,
        stuck: int,
    ) -> None:
        self.bits = bits
        self.left_far = left_far
        self.left_near = left_near
        self.front_far = front_far
        self.front_near = front_near
        self.right_far = right_far
        self.right_near = right_near
        self.ir = ir
        self.stuck = stuck
        self.left_count = left_far + left_near
        self.front_count = front_far + front_near
        self.right_count = right_far + right_near
        self.any_visible = bool(np.any(bits[:16]))


def _extract_features(bits: np.ndarray) -> ObsFeatures:
    left = bits[0:4]
    front = bits[4:12]
    right = bits[12:16]
    return ObsFeatures(
        bits=bits,
        left_far=int(left[[0, 2]].sum()),
        left_near=int(left[[1, 3]].sum()),
        front_far=int(front[::2].sum()),
        front_near=int(front[1::2].sum()),
        right_far=int(right[[0, 2]].sum()),
        right_near=int(right[[1, 3]].sum()),
        ir=int(bits[16]),
        stuck=int(bits[17]),
    )


class NoWallSearchController:
    def __init__(self) -> None:
        self.episode_key: Optional[int] = None
        self.prev_bits: Optional[np.ndarray] = None
        self.last_action: Optional[str] = None
        self.heading_deg = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.recovery_plan: list[str] = []
        self.push_commit = 0
        self.last_seen_side = 0
        self.reacquire_steps = 0
        self.search_turn_left = True
        self.sweep_remaining = SWEEP_FW_STEPS
        self.sweep_index = 0
        self.turn_plan: list[str] = []

    def reset(self, episode_key: int) -> None:
        self.episode_key = episode_key
        self.prev_bits = None
        self.last_action = None
        self.heading_deg = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.recovery_plan = []
        self.push_commit = 0
        self.last_seen_side = 0
        self.reacquire_steps = 0
        self.search_turn_left = True
        self.sweep_remaining = SWEEP_FW_STEPS
        self.sweep_index = 0
        self.turn_plan = []

    def _integrate_last_action(self, stuck: int) -> None:
        if self.last_action is None:
            return
        if self.last_action == "FW":
            if not stuck:
                rad = np.deg2rad(self.heading_deg)
                self.pose_x += FORWARD_STEP * float(np.cos(rad))
                self.pose_y += FORWARD_STEP * float(np.sin(rad))
        else:
            self.heading_deg = _wrap_angle(self.heading_deg + TURN_DELTAS[self.last_action])

    def _set_action(self, bits: np.ndarray, action: str) -> str:
        self.prev_bits = bits.copy()
        self.last_action = action
        return action

    def _turn_towards(self, side: int, hard: bool) -> str:
        if side < 0:
            return "L45" if hard else "L22"
        if side > 0:
            return "R45" if hard else "R22"
        return "FW"

    def _begin_next_search_turn(self) -> None:
        turn = "L45" if self.search_turn_left else "R45"
        self.sweep_index += 1
        if self.sweep_index % 6 == 0:
            self.turn_plan = [turn, turn]
        else:
            self.turn_plan = [turn]
        self.sweep_remaining = SWEEP_FW_STEPS

    def _search_action(self) -> str:
        if self.reacquire_steps > 0 and self.last_seen_side != 0:
            self.reacquire_steps -= 1
            return self._turn_towards(self.last_seen_side, hard=False)

        if self.turn_plan:
            return self.turn_plan.pop(0)

        if self.sweep_remaining > 0:
            self.sweep_remaining -= 1
            return "FW"

        self._begin_next_search_turn()
        return self.turn_plan.pop(0)

    def _start_blind_boundary_recovery(self) -> str:
        self.search_turn_left = not self.search_turn_left
        turn = "L45" if self.search_turn_left else "R45"
        self.recovery_plan = [turn, turn, turn, turn]
        self.sweep_remaining = SWEEP_FW_STEPS
        return self.recovery_plan.pop(0)

    def _sensor_action(self, feat: ObsFeatures) -> Optional[str]:
        left_score = 1.2 * feat.left_far + 2.0 * feat.left_near
        right_score = 1.2 * feat.right_far + 2.0 * feat.right_near
        front_score = 1.4 * feat.front_far + 2.5 * feat.front_near

        if feat.ir:
            return "FW"
        if feat.front_near >= 2:
            return "FW"
        if front_score >= max(left_score, right_score) and feat.front_count > 0:
            if abs(left_score - right_score) >= 2.0:
                return self._turn_towards(-1 if left_score > right_score else 1, hard=False)
            return "FW"
        if left_score > right_score + 0.5:
            return self._turn_towards(-1, hard=feat.left_near > 0)
        if right_score > left_score + 0.5:
            return self._turn_towards(1, hard=feat.right_near > 0)
        if feat.front_far > 0:
            return "FW"
        return None

    def act(self, obs: np.ndarray) -> str:
        bits = _binarize(obs)
        self._integrate_last_action(int(bits[17]))
        feat = _extract_features(bits)

        if feat.left_count > feat.right_count and feat.left_count > 0:
            self.last_seen_side = -1
        elif feat.right_count > feat.left_count and feat.right_count > 0:
            self.last_seen_side = 1
        elif feat.front_count > 0:
            self.last_seen_side = 0

        if feat.ir or feat.front_near > 0:
            self.push_commit = PUSH_COMMIT_STEPS
        elif self.push_commit > 0:
            self.push_commit -= 1

        if feat.stuck and self.last_action == "FW":
            if feat.any_visible or self.push_commit > 0:
                side = self.last_seen_side
                if side == 0:
                    side = -1 if feat.left_count >= feat.right_count else 1
                turn = self._turn_towards(side, hard=True)
                self.recovery_plan = [turn, "FW", "FW"]
                return self._set_action(bits, self.recovery_plan.pop(0))
            return self._set_action(bits, self._start_blind_boundary_recovery())

        if self.recovery_plan:
            return self._set_action(bits, self.recovery_plan.pop(0))

        if not feat.any_visible:
            if self.last_seen_side != 0:
                self.reacquire_steps = max(self.reacquire_steps, REACQUIRE_TURNS)
            return self._set_action(bits, self._search_action())

        self.reacquire_steps = 0

        if self.push_commit > 0:
            if abs(feat.left_count - feat.right_count) >= 3 and feat.front_count == 0:
                action = self._turn_towards(-1 if feat.left_count > feat.right_count else 1, hard=False)
            else:
                action = "FW"
            return self._set_action(bits, action)

        action = self._sensor_action(feat)
        if action is None:
            if self.last_seen_side != 0:
                action = self._turn_towards(self.last_seen_side, hard=False)
            else:
                action = "FW"
        return self._set_action(bits, action)


_CONTROLLER = NoWallSearchController()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    episode_key = id(rng)
    if _CONTROLLER.episode_key != episode_key:
        _CONTROLLER.reset(episode_key)
    return _CONTROLLER.act(obs)
