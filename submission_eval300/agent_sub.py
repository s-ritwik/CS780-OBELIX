"""Combined submission agent for the 300-step setting.

Strategy:
- Default to the stronger no-wall handmade controller.
- If the sensor pattern looks like a broad obstacle face for several consecutive
  observations, switch into a conservative wall-safe scan policy.
- Switch back to the handmade controller when the observation becomes compact /
  contact-like again.

This keeps the good no-wall behavior while avoiding the large wall-collision
penalties that hurt the original handmade controller under obstacles.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
TURN_DELTAS = {"L45": 45.0, "L22": 22.5, "FW": 0.0, "R22": -22.5, "R45": -45.0}
FORWARD_STEP = 5.0
WALL_MEMORY_RADIUS = 45.0
WALL_MEMORY_ANGLE = 40.0
WALL_SCAN_ACTION = "L22"


def _binarize(obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float32) > 0.5).astype(np.int8, copy=False)


def _wrap_angle(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _angle_diff(a: float, b: float) -> float:
    return abs(_wrap_angle(a - b))


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


class HandmadeController:
    def __init__(self) -> None:
        self.episode_key: Optional[int] = None
        self.prev_bits: Optional[np.ndarray] = None
        self.last_action: Optional[str] = None
        self.blind_steps = 0
        self.scan_dir = 1
        self.stuck_streak = 0
        self.escape_side = 0
        self.escape_cycles = 0
        self.recovery_plan: list[str] = []
        self.commit_fw_steps = 0
        self.contact_memory = 0
        self.last_seen_side = 0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading_deg = 0.0
        self.wall_hits: list[tuple[float, float, float, int]] = []

    def reset(self, episode_key: int) -> None:
        self.episode_key = episode_key
        self.prev_bits = None
        self.last_action = None
        self.blind_steps = 0
        self.scan_dir = 1
        self.stuck_streak = 0
        self.escape_side = 0
        self.escape_cycles = 0
        self.recovery_plan = []
        self.commit_fw_steps = 0
        self.contact_memory = 0
        self.last_seen_side = 0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading_deg = 0.0
        self.wall_hits = []

    def _turn_towards(self, side: int, hard: bool) -> str:
        if side < 0:
            return "R22"
        if side > 0:
            return "L45" if hard else "L22"
        return "FW"

    def _transition_counts(self, bits: np.ndarray) -> tuple[int, int, int]:
        if self.prev_bits is None:
            return 0, 0, 0
        prev = self.prev_bits
        new_left = int(np.logical_and(bits[0:4] == 1, prev[0:4] == 0).sum())
        new_front = int(np.logical_and(bits[4:12] == 1, prev[4:12] == 0).sum())
        new_right = int(np.logical_and(bits[12:16] == 1, prev[12:16] == 0).sum())
        return new_left, new_front, new_right

    def _remember_wall_hit(self, preferred_turn_side: int) -> None:
        side = preferred_turn_side if preferred_turn_side != 0 else self.scan_dir
        self.wall_hits.append((self.pose_x, self.pose_y, self.heading_deg, side))
        if len(self.wall_hits) > 12:
            self.wall_hits = self.wall_hits[-12:]
        self.scan_dir = side

    def _integrate_last_action(self, current_stuck: int) -> None:
        if self.last_action is None:
            return
        if self.last_action == "FW":
            if current_stuck:
                self._remember_wall_hit(self.escape_side)
            else:
                rad = np.deg2rad(self.heading_deg)
                self.pose_x += FORWARD_STEP * float(np.cos(rad))
                self.pose_y += FORWARD_STEP * float(np.sin(rad))
        else:
            self.heading_deg = _wrap_angle(self.heading_deg + TURN_DELTAS[self.last_action])

    def _start_recovery(self) -> str:
        self.escape_side = 1
        self.stuck_streak += 1
        self.escape_cycles += 1
        self.recovery_plan = ["L22", "L45", "FW"]
        self.commit_fw_steps = 0
        return self.recovery_plan.pop(0)

    def _search_action(self) -> str:
        self.blind_steps += 1
        return "FW"

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
        bits = _binarize(obs)
        self._integrate_last_action(int(bits[17]))
        feat = _extract_features(bits)
        new_left, new_front, new_right = self._transition_counts(bits)

        if feat.stuck and self.last_action == "FW":
            self.recovery_plan = []
            action = self._start_recovery()
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if self.recovery_plan:
            action = self.recovery_plan.pop(0)
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if feat.ir or feat.front_near > 0:
            self.contact_memory = min(12, self.contact_memory + 3)
            self.commit_fw_steps = max(self.commit_fw_steps, 3 if feat.ir else 2)
        else:
            self.contact_memory = max(0, self.contact_memory - 1)

        if feat.left_count > feat.right_count and feat.left_count > 0:
            self.last_seen_side = -1
        elif feat.right_count > feat.left_count and feat.right_count > 0:
            self.last_seen_side = 1
        elif feat.front_count > 0:
            self.last_seen_side = 0

        if not feat.stuck:
            self.stuck_streak = 0
            self.escape_cycles = 0
            self.escape_side = 0
            self.recovery_plan = []
            if self.last_action in {"R22", "R45"}:
                action = "FW"
                self.prev_bits = bits.copy()
                self.last_action = action
                return action

        if not feat.any_visible:
            action = self._search_action()
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        self.blind_steps = 0

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

        left_score = 1.0 * feat.left_far + 1.8 * feat.left_near + 0.7 * new_left
        right_score = 1.0 * feat.right_far + 1.8 * feat.right_near + 0.7 * new_right
        front_score = 1.2 * feat.front_far + 2.0 * feat.front_near + 0.8 * new_front

        if feat.front_count > 0 and front_score >= max(left_score, right_score):
            if abs(left_score - right_score) >= 1.5:
                action = self._turn_towards(-1 if left_score > right_score else 1, hard=False)
            else:
                action = "FW"
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if left_score > right_score + 0.5:
            hard = feat.left_near > 0 or (left_score - right_score) >= 2.0
            action = self._turn_towards(-1, hard=hard)
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if right_score > left_score + 0.5:
            hard = feat.right_near > 0 or (right_score - left_score) >= 2.0
            action = self._turn_towards(1, hard=hard)
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if feat.front_far > 0:
            action = "FW"
        elif self.last_seen_side < 0:
            action = "R22"
        elif self.last_seen_side > 0:
            action = "L22"
        else:
            action = "FW"

        self.prev_bits = bits.copy()
        self.last_action = action
        return action


class WallTeacher:
    def __init__(self) -> None:
        self.episode_key: Optional[int] = None

    def reset(self, episode_key: int) -> None:
        self.episode_key = episode_key

    def act(self, obs: np.ndarray) -> str:
        _ = obs
        return WALL_SCAN_ACTION


class HybridController:
    def __init__(self) -> None:
        self.episode_key: Optional[int] = None
        self.handmade = HandmadeController()
        self.wall_teacher = WallTeacher()
        self.mode = "handmade"
        self.wall_evidence = 0
        self.compact_evidence = 0

    def reset(self, episode_key: int) -> None:
        self.episode_key = episode_key
        self.handmade.reset(episode_key)
        self.wall_teacher.reset(episode_key)
        self.mode = "handmade"
        self.wall_evidence = 0
        self.compact_evidence = 0

    def _is_wall_like(self, feat: ObsFeatures) -> bool:
        if feat.ir:
            return False
        active_count = int(feat.sector_active.sum())
        broad_front = feat.front_count >= 4
        broad_far_front = feat.front_far >= 3 and feat.front_near == 0
        wide_face = feat.front_count >= 2 and feat.left_count >= 2 and feat.right_count >= 2 and feat.front_near == 0
        very_broad = active_count >= 6
        return broad_front or broad_far_front or wide_face or very_broad

    def _is_box_like(self, feat: ObsFeatures) -> bool:
        active_count = int(feat.sector_active.sum())
        compact_front = active_count <= 3 and feat.front_count > 0 and max(feat.left_count, feat.right_count) <= 2
        near_compact = feat.front_near >= 1 and feat.front_count <= 2 and feat.left_count <= 1 and feat.right_count <= 1
        return bool(feat.ir or compact_front or near_compact)

    def act(self, obs: np.ndarray) -> str:
        bits = _binarize(obs)
        feat = _extract_features(bits)
        wall_like = self._is_wall_like(feat)
        box_like = self._is_box_like(feat)

        if self.mode == "handmade":
            if (not feat.stuck) and wall_like and (not box_like):
                self.wall_evidence = min(8, self.wall_evidence + 1)
            else:
                self.wall_evidence = max(0, self.wall_evidence - 1)

            if self.wall_evidence >= 3:
                self.mode = "wall"
                self.compact_evidence = 0

        else:
            if feat.stuck:
                self.mode = "handmade"
                self.handmade.reset(self.episode_key if self.episode_key is not None else 0)
                self.wall_evidence = 0
                self.compact_evidence = 0
            else:
                if box_like and (not wall_like):
                    self.compact_evidence = min(6, self.compact_evidence + 1)
                else:
                    self.compact_evidence = max(0, self.compact_evidence - 1)

                if self.compact_evidence >= 2:
                    self.mode = "handmade"
                    self.handmade.reset(self.episode_key if self.episode_key is not None else 0)
                    self.wall_evidence = 0
                    self.compact_evidence = 0

        if self.mode == "wall":
            return self.wall_teacher.act(obs)
        return self.handmade.act(obs)


_CONTROLLER = HybridController()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    episode_key = id(rng)
    if _CONTROLLER.episode_key != episode_key:
        _CONTROLLER.reset(episode_key)
    return _CONTROLLER.act(obs)
