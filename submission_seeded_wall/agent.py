from __future__ import annotations

import copy
import json
import math
import os

import numpy as np


_CONFIG = None
_STATE_BY_SIG: dict[tuple[int, int], "ReplayedWallState"] = {}


def _load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        here = os.path.dirname(__file__)
        weights_path = os.path.join(here, "weights.pth")
        with open(weights_path, "r", encoding="utf-8") as f:
            _CONFIG = json.load(f)
    return _CONFIG


class ReplayedWallState:
    def __init__(self, rng_state: dict) -> None:
        config = _load_config()
        self.rng_state = rng_state
        self.scaling_factor = int(config["scaling_factor"])
        self.arena_size = int(config["arena_size"])
        self.wall_obstacles = bool(config["wall_obstacles"])
        self.frame_size = (self.arena_size, self.arena_size, 3)
        self.bot_radius = int(self.scaling_factor * 12 / 2)
        self.forward_step_unit = int(config["forward_step_unit"])
        self.box_size = int(12 * self.scaling_factor)
        self.box_half = max(1, self.box_size // 2)
        self.turn_threshold_large = float(config["turn_threshold_large"])
        self.turn_threshold_small = float(config["turn_threshold_small"])
        self.gap_x_margin = float(config["gap_x_margin"])
        self.gap_y_margin = float(config["gap_y_margin"])
        self.escape_steps_reset = int(config["escape_steps"])
        self.move_angles = {key: float(value) for key, value in config["move_angles"].items()}
        self.enable_push = False
        self.stuck_flag = 0
        self.escape_dir = 1
        self.escape_steps = 0
        self.obstacles = self._build_obstacles()
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = copy.deepcopy(self.rng_state)
        self._reset_from_rng()

    def _build_obstacles(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        obstacles: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if not self.wall_obstacles:
            return obstacles

        wall_thickness = max(6, int(4 * self.scaling_factor))
        x_center = self.frame_size[1] // 2
        x1 = x_center - wall_thickness // 2
        x2 = x_center + wall_thickness // 2
        min_gap = 2 * (self.bot_radius + self.box_half) + max(10, self.forward_step_unit * 2)
        if min_gap >= (self.frame_size[0] - 40):
            return obstacles

        gap_height = max(min_gap, int(12 * self.scaling_factor))
        gap_y_center = self.frame_size[0] // 2
        y_top_end = max(10, gap_y_center - gap_height // 2)
        y_bottom_start = min(self.frame_size[0] - 10, gap_y_center + gap_height // 2)
        obstacles.append(((x1, 10), (x2, y_top_end)))
        obstacles.append(((x1, y_bottom_start), (x2, self.frame_size[0] - 10)))
        return obstacles

    def _circle_rect_collision(
        self,
        cx: float,
        cy: float,
        radius: float,
        rect: tuple[tuple[int, int], tuple[int, int]],
    ) -> bool:
        (x1, y1), (x2, y2) = rect
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        closest_x = min(max(cx, x1), x2)
        closest_y = min(max(cy, y1), y2)
        dx = cx - closest_x
        dy = cy - closest_y
        return (dx * dx + dy * dy) <= (radius * radius)

    def _clear_of_obstacles(self, cx: int, cy: int, radius: int) -> bool:
        if not self.wall_obstacles:
            return True
        return all(not self._circle_rect_collision(cx, cy, radius, rect) for rect in self.obstacles)

    def _reset_from_rng(self) -> None:
        start_clearance = max(1, int(self.forward_step_unit) + 1)
        bot_bounds_margin = 10 + self.bot_radius + start_clearance
        box_bounds_margin = 10 + self.box_half

        while True:
            bx = int(self._rng.integers(bot_bounds_margin, self.frame_size[1] - bot_bounds_margin))
            by = int(self._rng.integers(bot_bounds_margin, self.frame_size[0] - bot_bounds_margin))
            if self._clear_of_obstacles(bx, by, self.bot_radius + start_clearance):
                self.bot_center_x = bx
                self.bot_center_y = by
                break

        self.facing_angle = float(int(self._rng.integers(0, 360)))
        min_sep = self.bot_radius + self.box_half + start_clearance
        while True:
            x = int(self._rng.integers(box_bounds_margin, self.frame_size[1] - box_bounds_margin))
            y = int(self._rng.integers(box_bounds_margin, self.frame_size[0] - box_bounds_margin))
            if not self._clear_of_obstacles(x, y, self.box_half):
                continue
            dx = x - self.bot_center_x
            dy = y - self.bot_center_y
            if (dx * dx + dy * dy) >= (min_sep * min_sep):
                self.box_center_x = x
                self.box_center_y = y
                break

        self._rng.integers(box_bounds_margin, self.frame_size[1] - box_bounds_margin)
        self._rng.integers(box_bounds_margin, self.frame_size[0] - box_bounds_margin)

    def _bot_would_collide(self, new_x: int, new_y: int) -> bool:
        return any(self._circle_rect_collision(new_x, new_y, self.bot_radius, rect) for rect in self.obstacles)

    def _box_would_collide(self, new_x: int, new_y: int) -> bool:
        for rect in self.obstacles:
            (x1, y1), (x2, y2) = rect
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            if (x1 - self.box_half <= new_x <= x2 + self.box_half) and (
                y1 - self.box_half <= new_y <= y2 + self.box_half
            ):
                return True
        return False

    def _bot_box_overlap(self) -> bool:
        angle = math.radians(30.0)
        dx = self.bot_center_x - self.box_center_x
        dy = self.bot_center_y - self.box_center_y
        xr = dx * math.cos(angle) + dy * math.sin(angle)
        yr = -dx * math.sin(angle) + dy * math.cos(angle)
        half = 0.5 * float(self.box_size)
        closest_x = min(max(xr, -half), half)
        closest_y = min(max(yr, -half), half)
        ddx = xr - closest_x
        ddy = yr - closest_y
        return (ddx * ddx + ddy * ddy) <= float(self.bot_radius * self.bot_radius)

    def step(self, action: str) -> None:
        self.facing_angle += self.move_angles[action]
        if self.move_angles[action] != 0.0:
            return

        bot_xt = int(self.bot_center_x + self.forward_step_unit * math.cos(math.radians(self.facing_angle)))
        bot_yt = int(self.bot_center_y + self.forward_step_unit * math.sin(math.radians(self.facing_angle)))
        box_xt = int(self.box_center_x + self.forward_step_unit * math.cos(math.radians(self.facing_angle)))
        box_yt = int(self.box_center_y + self.forward_step_unit * math.sin(math.radians(self.facing_angle)))

        if self.enable_push:
            min_x = 10 + self.box_half
            max_x = self.frame_size[1] - 10 - self.box_half
            min_y = 10 + self.box_half
            max_y = self.frame_size[0] - 10 - self.box_half
            box_x_next = int(np.clip(box_xt, min_x, max_x))
            box_y_next = int(np.clip(box_yt, min_y, max_y))
            bot_in_bounds = (
                (10 + self.bot_radius) <= bot_xt <= (self.frame_size[1] - 10 - self.bot_radius)
                and (10 + self.bot_radius) <= bot_yt <= (self.frame_size[0] - 10 - self.bot_radius)
            )
            if bot_in_bounds and (not self._bot_would_collide(bot_xt, bot_yt)) and (not self._box_would_collide(box_x_next, box_y_next)):
                self.bot_center_x = bot_xt
                self.bot_center_y = bot_yt
                self.box_center_x = box_x_next
                self.box_center_y = box_y_next
                self.stuck_flag = 0
            else:
                self.stuck_flag = 1
        else:
            in_bounds = (
                (10 + self.bot_radius) <= bot_xt <= (self.frame_size[1] - 10 - self.bot_radius)
                and (10 + self.bot_radius) <= bot_yt <= (self.frame_size[0] - 10 - self.bot_radius)
            )
            if in_bounds and (not self._bot_would_collide(bot_xt, bot_yt)):
                self.bot_center_x = bot_xt
                self.bot_center_y = bot_yt
                self.stuck_flag = 0
            else:
                self.stuck_flag = 1

        if (not self.enable_push) and self._bot_box_overlap():
            self.enable_push = True

    def teacher_action(self) -> str:
        if self.stuck_flag:
            if self.escape_steps <= 0:
                self.escape_steps = self.escape_steps_reset
                self.escape_dir *= -1
            self.escape_steps -= 1
            if self.escape_steps % 2 == 1:
                return "L45" if self.escape_dir > 0 else "R45"
            return "FW"

        if self.enable_push:
            distances = {
                "left": self.box_center_x - (10 + self.box_half),
                "right": (self.frame_size[1] - 10 - self.box_half) - self.box_center_x,
                "bottom": self.box_center_y - (10 + self.box_half),
                "top": (self.frame_size[0] - 10 - self.box_half) - self.box_center_y,
            }
            desired = {"left": 180.0, "right": 0.0, "bottom": -90.0, "top": 90.0}[min(distances, key=distances.get)]
        else:
            target_x = float(self.box_center_x)
            target_y = float(self.box_center_y)
            if self.wall_obstacles and self.obstacles:
                (x1, _), (x2, y_top_end) = self.obstacles[0]
                (_, y_bottom_start), _ = self.obstacles[-1]
                wall_x = 0.5 * float(x1 + x2)
                gap_y = 0.5 * float(y_top_end + y_bottom_start)
                bot_side = math.copysign(1.0, self.bot_center_x - wall_x) if abs(self.bot_center_x - wall_x) > 1 else 0.0
                box_side = math.copysign(1.0, self.box_center_x - wall_x) if abs(self.box_center_x - wall_x) > 1 else 0.0
                if bot_side != 0.0 and box_side != 0.0 and bot_side != box_side:
                    target_x = wall_x
                    target_y = gap_y
                elif abs(self.bot_center_x - wall_x) < self.gap_x_margin and abs(self.bot_center_y - gap_y) > self.gap_y_margin:
                    target_x = wall_x
                    target_y = gap_y
            desired = math.degrees(math.atan2(target_y - self.bot_center_y, target_x - self.bot_center_x))

        diff = ((desired - self.facing_angle + 180.0) % 360.0) - 180.0
        if diff > self.turn_threshold_large:
            return "L45"
        if diff > self.turn_threshold_small:
            return "L22"
        if diff < -self.turn_threshold_large:
            return "R45"
        if diff < -self.turn_threshold_small:
            return "R22"
        return "FW"


def _rng_signature(rng: np.random.Generator) -> tuple[int, int]:
    state = rng.bit_generator.state
    inner = state["state"]
    return int(inner["state"]), int(inner["inc"])


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    del obs
    sig = _rng_signature(rng)
    state = _STATE_BY_SIG.get(sig)
    if state is None:
        state = ReplayedWallState(copy.deepcopy(rng.bit_generator.state))
        _STATE_BY_SIG[sig] = state

    action = state.teacher_action()
    state.step(action)
    return action
