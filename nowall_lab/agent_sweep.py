from __future__ import annotations

from typing import Optional

import numpy as np


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
TURN_DELTAS = {"L45": 45.0, "L22": 22.5, "FW": 0.0, "R22": -22.5, "R45": -45.0}
FORWARD_STEP = 5.0

SHIFT_STEPS = 6
TURN_90_LEFT = ["L45", "L45"]
TURN_90_RIGHT = ["R45", "R45"]


def _binarize(obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float32) > 0.5).astype(np.int8, copy=False)


def _wrap_angle(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


class ObsFeatures:
    def __init__(
        self,
        *,
        bits: np.ndarray,
        sector_far: np.ndarray,
        sector_near: np.ndarray,
        sector_active: np.ndarray,
        left_far: int,
        left_near: int,
        front_far: int,
        front_near: int,
        right_far: int,
        right_near: int,
        ir: int,
        stuck: int,
        left_count: int,
        front_count: int,
        right_count: int,
        any_visible: bool,
    ) -> None:
        self.bits = bits
        self.sector_far = sector_far
        self.sector_near = sector_near
        self.sector_active = sector_active
        self.left_far = left_far
        self.left_near = left_near
        self.front_far = front_far
        self.front_near = front_near
        self.right_far = right_far
        self.right_near = right_near
        self.ir = ir
        self.stuck = stuck
        self.left_count = left_count
        self.front_count = front_count
        self.right_count = right_count
        self.any_visible = any_visible


def _extract_features(bits: np.ndarray) -> ObsFeatures:
    sector_far = bits[:16:2]
    sector_near = bits[1:16:2]
    sector_active = np.logical_or(sector_far, sector_near).astype(np.int8, copy=False)

    left = bits[0:4]
    front = bits[4:12]
    right = bits[12:16]

    left_far = int(left[[0, 2]].sum())
    left_near = int(left[[1, 3]].sum())
    front_far = int(front[::2].sum())
    front_near = int(front[1::2].sum())
    right_far = int(right[[0, 2]].sum())
    right_near = int(right[[1, 3]].sum())

    left_count = left_far + left_near
    front_count = front_far + front_near
    right_count = right_far + right_near

    return ObsFeatures(
        bits=bits,
        sector_far=sector_far,
        sector_near=sector_near,
        sector_active=sector_active,
        left_far=left_far,
        left_near=left_near,
        front_far=front_far,
        front_near=front_near,
        right_far=right_far,
        right_near=right_near,
        ir=int(bits[16]),
        stuck=int(bits[17]),
        left_count=left_count,
        front_count=front_count,
        right_count=right_count,
        any_visible=bool(np.any(bits[:16])),
    )


class SweepController:
    def __init__(self) -> None:
        self.episode_key: Optional[int] = None
        self.prev_bits: Optional[np.ndarray] = None
        self.last_action: Optional[str] = None
        self.plan: list[str] = []
        self.commit_fw_steps = 0
        self.last_seen_side = 0
        self.sweep_turn_sign = 1
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading_deg = 0.0
        self.boundary_hits = 0

    def reset(self, episode_key: int) -> None:
        self.episode_key = episode_key
        self.prev_bits = None
        self.last_action = None
        self.plan = []
        self.commit_fw_steps = 0
        self.last_seen_side = 0
        self.sweep_turn_sign = 1
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading_deg = 0.0
        self.boundary_hits = 0

    def _integrate_last_action(self, current_stuck: int) -> None:
        if self.last_action is None:
            return
        if self.last_action == "FW":
            if not current_stuck:
                rad = np.deg2rad(self.heading_deg)
                self.pose_x += FORWARD_STEP * float(np.cos(rad))
                self.pose_y += FORWARD_STEP * float(np.sin(rad))
        else:
            self.heading_deg = _wrap_angle(self.heading_deg + TURN_DELTAS[self.last_action])

    def _queue_boundary_plan(self) -> str:
        self.boundary_hits += 1
        if self.sweep_turn_sign > 0:
            turns = TURN_90_LEFT
        else:
            turns = TURN_90_RIGHT
        self.plan = list(turns) + (["FW"] * SHIFT_STEPS) + list(turns)
        self.sweep_turn_sign *= -1
        return self.plan.pop(0)

    def _search_action(self) -> str:
        if self.plan:
            return self.plan.pop(0)
        return "FW"

    def _turn_towards(self, side: int, hard: bool) -> str:
        if side < 0:
            return "R45" if hard else "R22"
        if side > 0:
            return "L45" if hard else "L22"
        return "FW"

    def _sensor_guided_action(self, feat: ObsFeatures) -> Optional[str]:
        active = feat.sector_active
        near = feat.sector_near

        left_side = bool(active[0] or active[1])
        right_side = bool(active[6] or active[7])
        front_left = bool(active[2] or active[3])
        front_right = bool(active[4] or active[5])
        front_left_near = bool(near[2] or near[3])
        front_right_near = bool(near[4] or near[5])

        if feat.ir:
            return "FW"

        if (front_left_near and front_right_near) or (feat.front_near >= 2):
            return "FW"

        if front_left and not front_right:
            return "R22"
        if front_right and not front_left:
            return "L22"

        if left_side and not right_side and feat.front_count == 0:
            return "R22"
        if right_side and not left_side and feat.front_count == 0:
            return "L22"

        if feat.front_count > 0 and not (front_left or front_right):
            return "FW"

        if feat.front_count > 0 and front_left and front_right:
            return "FW"

        if left_side and right_side:
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

        if feat.stuck and self.last_action == "FW":
            self.plan = []
            action = self._queue_boundary_plan()
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if feat.ir or feat.front_near > 0:
            self.commit_fw_steps = max(self.commit_fw_steps, 3 if feat.ir else 2)
        elif self.commit_fw_steps > 0 and feat.front_count == 0:
            self.commit_fw_steps = max(0, self.commit_fw_steps - 1)

        if self.commit_fw_steps > 0:
            self.commit_fw_steps -= 1
            action = "FW"
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        sensor_action = self._sensor_guided_action(feat)
        if sensor_action is not None:
            self.plan = []
            self.prev_bits = bits.copy()
            self.last_action = sensor_action
            return sensor_action

        if not feat.any_visible:
            action = self._search_action()
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        left_score = 1.0 * feat.left_far + 1.8 * feat.left_near
        right_score = 1.0 * feat.right_far + 1.8 * feat.right_near
        front_score = 1.2 * feat.front_far + 2.0 * feat.front_near

        if feat.front_count > 0 and front_score >= max(left_score, right_score):
            action = "FW"
        elif left_score > right_score + 0.5:
            action = self._turn_towards(-1, hard=(left_score - right_score) >= 2.0)
        elif right_score > left_score + 0.5:
            action = self._turn_towards(1, hard=(right_score - left_score) >= 2.0)
        elif self.last_seen_side < 0:
            action = "R22"
        elif self.last_seen_side > 0:
            action = "L22"
        else:
            action = "FW"

        self.prev_bits = bits.copy()
        self.last_action = action
        return action


_CONTROLLER = SweepController()


def _episode_key(rng: np.random.Generator) -> int:
    return id(rng)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    episode_key = _episode_key(rng)
    if _CONTROLLER.episode_key != episode_key:
        _CONTROLLER.reset(episode_key)
    return _CONTROLLER.act(obs)
