from __future__ import annotations

from typing import Optional

import numpy as np


TURN_DELTAS = {"L45": 45.0, "L22": 22.5, "FW": 0.0, "R22": -22.5, "R45": -45.0}
FORWARD_STEP = 5.0
LANE_SHIFT_STEPS = 14
PUSH_COMMIT_STEPS = 10
REACQUIRE_TURNS = 5


def _bits(obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float32) > 0.5).astype(np.int8, copy=False)


class Features:
    def __init__(self, bits: np.ndarray) -> None:
        self.bits = bits
        left = bits[0:4]
        front = bits[4:12]
        right = bits[12:16]
        self.left_far = int(left[[0, 2]].sum())
        self.left_near = int(left[[1, 3]].sum())
        self.front_far = int(front[::2].sum())
        self.front_near = int(front[1::2].sum())
        self.right_far = int(right[[0, 2]].sum())
        self.right_near = int(right[[1, 3]].sum())
        self.left_count = self.left_far + self.left_near
        self.front_count = self.front_far + self.front_near
        self.right_count = self.right_far + self.right_near
        self.ir = int(bits[16])
        self.stuck = int(bits[17])
        self.any_visible = bool(np.any(bits[:16]))


class ZigZagNoWallController:
    def __init__(self) -> None:
        self.episode_key: Optional[int] = None
        self.prev_bits: Optional[np.ndarray] = None
        self.last_action: Optional[str] = None
        self.heading_deg = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.recovery_plan: list[str] = []
        self.boundary_turn_left = True
        self.boundary_initialized = False
        self.push_commit = 0
        self.last_seen_side = 0
        self.reacquire_turns = 0

    def reset(self, episode_key: int) -> None:
        self.episode_key = episode_key
        self.prev_bits = None
        self.last_action = None
        self.heading_deg = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.recovery_plan = []
        self.boundary_turn_left = True
        self.boundary_initialized = False
        self.push_commit = 0
        self.last_seen_side = 0
        self.reacquire_turns = 0

    def _set_action(self, bits: np.ndarray, action: str) -> str:
        self.prev_bits = bits.copy()
        self.last_action = action
        return action

    def _integrate(self, stuck: int) -> None:
        if self.last_action is None:
            return
        if self.last_action == "FW":
            if not stuck:
                rad = np.deg2rad(self.heading_deg)
                self.pose_x += FORWARD_STEP * float(np.cos(rad))
                self.pose_y += FORWARD_STEP * float(np.sin(rad))
        else:
            self.heading_deg = ((self.heading_deg + TURN_DELTAS[self.last_action] + 180.0) % 360.0) - 180.0

    def _turn_towards(self, side: int, hard: bool) -> str:
        if side < 0:
            return "L45" if hard else "L22"
        if side > 0:
            return "R45" if hard else "R22"
        return "FW"

    def _sensor_guided_action(self, feat: Features) -> str:
        left_score = 1.0 * feat.left_far + 2.0 * feat.left_near
        right_score = 1.0 * feat.right_far + 2.0 * feat.right_near
        front_score = 1.25 * feat.front_far + 2.5 * feat.front_near

        if feat.ir or feat.front_near >= 2:
            return "FW"
        if feat.front_count > 0 and front_score >= max(left_score, right_score):
            if abs(left_score - right_score) >= 2.0:
                return self._turn_towards(-1 if left_score > right_score else 1, hard=False)
            return "FW"
        if left_score > right_score + 0.5:
            return self._turn_towards(-1, hard=feat.left_near > 0)
        if right_score > left_score + 0.5:
            return self._turn_towards(1, hard=feat.right_near > 0)
        if feat.front_far > 0:
            return "FW"
        return self._turn_towards(self.last_seen_side, hard=False) if self.last_seen_side != 0 else "FW"

    def _boundary_plan(self) -> list[str]:
        turn = "L45" if self.boundary_turn_left else "R45"
        self.boundary_turn_left = not self.boundary_turn_left
        self.boundary_initialized = True
        return [turn, turn] + ["FW"] * LANE_SHIFT_STEPS + [turn, turn]

    def act(self, obs: np.ndarray) -> str:
        bits = _bits(obs)
        self._integrate(int(bits[17]))
        feat = Features(bits)

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

        if feat.any_visible:
            self.reacquire_turns = 0
            self.recovery_plan = []
            if feat.stuck and self.last_action == "FW":
                side = self.last_seen_side
                if side == 0:
                    side = -1 if feat.left_count >= feat.right_count else 1
                self.recovery_plan = [self._turn_towards(side, hard=True), "FW", "FW"]
                return self._set_action(bits, self.recovery_plan.pop(0))
            if self.push_commit > 0:
                if abs(feat.left_count - feat.right_count) >= 3 and feat.front_count == 0:
                    return self._set_action(
                        bits,
                        self._turn_towards(-1 if feat.left_count > feat.right_count else 1, hard=False),
                    )
                return self._set_action(bits, "FW")
            return self._set_action(bits, self._sensor_guided_action(feat))

        if self.recovery_plan:
            return self._set_action(bits, self.recovery_plan.pop(0))

        if self.last_seen_side != 0 and self.reacquire_turns < REACQUIRE_TURNS:
            self.reacquire_turns += 1
            return self._set_action(bits, self._turn_towards(self.last_seen_side, hard=False))

        self.reacquire_turns = 0

        if feat.stuck and self.last_action == "FW":
            self.recovery_plan = self._boundary_plan()
            return self._set_action(bits, self.recovery_plan.pop(0))

        if not self.boundary_initialized:
            return self._set_action(bits, "FW")
        return self._set_action(bits, "FW")


_CONTROLLER = ZigZagNoWallController()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    episode_key = id(rng)
    if _CONTROLLER.episode_key != episode_key:
        _CONTROLLER.reset(episode_key)
    return _CONTROLLER.act(obs)
