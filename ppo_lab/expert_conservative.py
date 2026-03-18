from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
TURN_DELTAS = {"L45": 45.0, "L22": 22.5, "FW": 0.0, "R22": -22.5, "R45": -45.0}
FORWARD_STEP = 5.0
WALL_MEMORY_RADIUS = 55.0
WALL_MEMORY_ANGLE = 45.0
POSE_BIN = 30.0


def _binarize(obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float32) > 0.5).astype(np.int8, copy=False)


def _wrap_angle(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _angle_diff(a: float, b: float) -> float:
    return abs(_wrap_angle(a - b))


def _turn_action(sign: int, degrees: int) -> str:
    if degrees >= 45:
        return "L45" if sign > 0 else "R45"
    return "L22" if sign > 0 else "R22"


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


class ConservativeController:
    def __init__(self) -> None:
        self.episode_key: Optional[int] = None
        self.prev_bits: Optional[np.ndarray] = None
        self.last_action: Optional[str] = None
        self.blind_steps = 0
        self.scan_dir = 1
        self.escape_bias = 1
        self.recovery_plan: list[str] = []
        self.search_plan: list[str] = []
        self.commit_fw_steps = 0
        self.contact_memory = 0
        self.last_seen_side = 0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading_deg = 0.0
        self.wall_hits: list[tuple[float, float, float, int]] = []
        self.visit_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        self.search_cycle = 0
        self.last_progress_step = 0
        self.step_idx = 0

    def reset(self, episode_key: int) -> None:
        self.episode_key = episode_key
        self.prev_bits = None
        self.last_action = None
        self.blind_steps = 0
        self.scan_dir = 1
        self.escape_bias = 1
        self.recovery_plan = []
        self.search_plan = []
        self.commit_fw_steps = 0
        self.contact_memory = 0
        self.last_seen_side = 0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading_deg = 0.0
        self.wall_hits = []
        self.visit_counts = defaultdict(int)
        self.search_cycle = 0
        self.last_progress_step = 0
        self.step_idx = 0

    def _transition_counts(self, bits: np.ndarray) -> tuple[int, int, int]:
        if self.prev_bits is None:
            return 0, 0, 0
        prev = self.prev_bits
        new_left = int(np.logical_and(bits[0:4] == 1, prev[0:4] == 0).sum())
        new_front = int(np.logical_and(bits[4:12] == 1, prev[4:12] == 0).sum())
        new_right = int(np.logical_and(bits[12:16] == 1, prev[12:16] == 0).sum())
        return new_left, new_front, new_right

    def _pose_key(self) -> tuple[int, int, int]:
        return (
            int(np.round(self.pose_x / POSE_BIN)),
            int(np.round(self.pose_y / POSE_BIN)),
            int(np.round(_wrap_angle(self.heading_deg) / 45.0)) % 8,
        )

    def _remember_wall_hit(self, turn_sign: int) -> None:
        sign = 1 if turn_sign >= 0 else -1
        self.wall_hits.append((self.pose_x, self.pose_y, self.heading_deg, sign))
        if len(self.wall_hits) > 16:
            self.wall_hits = self.wall_hits[-16:]

    def _integrate_last_action(self, current_stuck: int) -> None:
        if self.last_action is None:
            return
        if self.last_action == "FW":
            if current_stuck:
                self._remember_wall_hit(self.escape_bias)
            else:
                rad = np.deg2rad(self.heading_deg)
                self.pose_x += FORWARD_STEP * float(np.cos(rad))
                self.pose_y += FORWARD_STEP * float(np.sin(rad))
                self.visit_counts[self._pose_key()] += 1
        else:
            self.heading_deg = _wrap_angle(self.heading_deg + TURN_DELTAS[self.last_action])

    def _wall_ahead_turn(self) -> int:
        for hit_x, hit_y, hit_heading, turn_sign in reversed(self.wall_hits):
            dx = self.pose_x - hit_x
            dy = self.pose_y - hit_y
            if (dx * dx + dy * dy) > (WALL_MEMORY_RADIUS * WALL_MEMORY_RADIUS):
                continue
            if _angle_diff(self.heading_deg, hit_heading) <= WALL_MEMORY_ANGLE:
                return int(turn_sign)
        return 0

    def _start_recovery(self, feat: ObsFeatures) -> str:
        wall_turn = self._wall_ahead_turn()
        if wall_turn != 0:
            sign = wall_turn
        elif self.last_seen_side != 0 and feat.front_count == 0:
            sign = -self.last_seen_side
        else:
            sign = self.escape_bias
        self.escape_bias *= -1
        self.search_plan = []
        self.commit_fw_steps = 0
        self.recovery_plan = [
            _turn_action(sign, 45),
            _turn_action(sign, 22),
            "FW",
        ]
        return self.recovery_plan.pop(0)

    def _build_search_plan(self) -> list[str]:
        ahead_turn = self._wall_ahead_turn()
        revisit_count = self.visit_counts[self._pose_key()]

        if ahead_turn != 0:
            sign = ahead_turn
            return [_turn_action(sign, 45), "FW"]

        if revisit_count >= 3:
            sign = -self.scan_dir
            return [_turn_action(sign, 45), "FW"]

        if revisit_count >= 2:
            sign = self.scan_dir
            return [_turn_action(sign, 22), "FW"]

        if self.last_seen_side != 0 and (self.step_idx - self.last_progress_step) < 40:
            sign = -self.last_seen_side
            return [_turn_action(sign, 22), "FW", "FW"]

        if self.blind_steps < 30:
            burst = 2
            turn_degrees = 22
        elif self.blind_steps < 80:
            burst = 3
            turn_degrees = 22
        else:
            burst = 2
            turn_degrees = 45

        sign = self.scan_dir
        self.search_cycle += 1
        if self.search_cycle % 4 == 0:
            self.scan_dir *= -1

        return ["FW"] * burst + [_turn_action(sign, turn_degrees)]

    def _search_action(self) -> str:
        self.blind_steps += 1
        if not self.search_plan:
            self.search_plan = self._build_search_plan()
        return self.search_plan.pop(0)

    def _sensor_guided_action(self, feat: ObsFeatures) -> Optional[str]:
        active = feat.sector_active
        near = feat.sector_near

        left_side = bool(active[0] or active[1])
        right_side = bool(active[6] or active[7])
        front_left = bool(active[2] or active[3])
        front_right = bool(active[4] or active[5])
        front_left_inner = bool(active[3])
        front_right_inner = bool(active[4])
        front_left_near = bool(near[2] or near[3])
        front_right_near = bool(near[4] or near[5])

        if feat.ir:
            return "FW"

        if (front_left_inner and front_right_inner) or (
            front_left_near and front_right_near and abs(feat.left_count - feat.right_count) <= 1
        ):
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
        self.step_idx += 1
        bits = _binarize(obs)
        self._integrate_last_action(int(bits[17]))
        feat = _extract_features(bits)
        new_left, new_front, new_right = self._transition_counts(bits)

        if feat.stuck and self.last_action == "FW":
            action = self._start_recovery(feat)
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if self.recovery_plan:
            action = self.recovery_plan.pop(0)
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if feat.ir or feat.front_near > 0:
            self.contact_memory = min(16, self.contact_memory + 3)
            self.commit_fw_steps = max(self.commit_fw_steps, 3 if feat.ir else 2)
        else:
            self.contact_memory = max(0, self.contact_memory - 1)

        if feat.left_count > feat.right_count and feat.left_count > 0:
            self.last_seen_side = -1
            self.last_progress_step = self.step_idx
        elif feat.right_count > feat.left_count and feat.right_count > 0:
            self.last_seen_side = 1
            self.last_progress_step = self.step_idx
        elif feat.front_count > 0 or feat.ir:
            self.last_seen_side = 0
            self.last_progress_step = self.step_idx

        if not feat.any_visible:
            action = self._search_action()
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        self.blind_steps = 0
        self.search_plan = []

        if self.last_action == "FW" and new_front > 0 and (feat.front_count > 0 or feat.ir):
            self.commit_fw_steps = max(self.commit_fw_steps, 2)

        if self.commit_fw_steps > 0:
            severe_side_pull = abs(feat.left_count - feat.right_count) >= 3 and feat.front_count == 0
            if not severe_side_pull:
                self.commit_fw_steps -= 1
                action = "FW"
                self.prev_bits = bits.copy()
                self.last_action = action
                return action
            self.commit_fw_steps = 0

        if feat.ir:
            action = "FW"
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if feat.front_near >= 2:
            self.commit_fw_steps = 2
            action = "FW"
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        sensor_action = self._sensor_guided_action(feat)
        if sensor_action is not None:
            self.prev_bits = bits.copy()
            self.last_action = sensor_action
            return sensor_action

        left_score = 1.0 * feat.left_far + 2.0 * feat.left_near + 0.8 * new_left
        right_score = 1.0 * feat.right_far + 2.0 * feat.right_near + 0.8 * new_right
        front_score = 1.2 * feat.front_far + 2.2 * feat.front_near + 0.8 * new_front

        if feat.front_count > 0 and front_score >= max(left_score, right_score):
            if abs(left_score - right_score) >= 1.5:
                action = "R22" if left_score > right_score else "L22"
            else:
                action = "FW"
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if left_score > right_score + 0.5:
            action = "R45" if feat.left_near > 0 or (left_score - right_score) >= 2.0 else "R22"
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if right_score > left_score + 0.5:
            action = "L45" if feat.right_near > 0 or (right_score - left_score) >= 2.0 else "L22"
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if feat.front_far > 0:
            action = "FW"
        elif self.last_seen_side < 0:
            action = "L22"
        elif self.last_seen_side > 0:
            action = "R22"
        else:
            action = _turn_action(self.scan_dir, 22)

        self.prev_bits = bits.copy()
        self.last_action = action
        return action


_CONTROLLER = ConservativeController()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    episode_key = id(rng)
    if _CONTROLLER.episode_key != episode_key:
        _CONTROLLER.reset(episode_key)
    return _CONTROLLER.act(obs)
