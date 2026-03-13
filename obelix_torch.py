import argparse
import random

from typing import List, Optional, Tuple

import numpy as np
import torch


class OBELIX:
    """Torch-based port of obelix.py with matching environment API/logic."""

    def __init__(
        self,
        scaling_factor: int,
        arena_size: int = 500,
        max_steps: int = 1000,
        wall_obstacles: bool = False,
        difficulty: int = 0,
        box_speed: int = 2,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.scaling_factor = scaling_factor
        self.arena_size = arena_size
        self.frame_size = (arena_size, arena_size, 3)
        self.height = self.frame_size[0]
        self.width = self.frame_size[1]

        self._grid_y, self._grid_x = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32),
            torch.arange(self.width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )

        self.frame = self._zeros3()
        self.bot_radius = int(scaling_factor * 12 / 2)  # 12" diameter
        self.facing_angle = 0

        self.bot_center_x = 200
        self.bot_center_y = 200
        self.bot_color = (255, 255, 255)

        self.move_options = {"L45": 45, "L22": 22.5, "FW": 0, "R22": -22.5, "R45": -45}
        self.forward_step_unit = 5

        self.sonar_fov = 20
        self.sonar_far_range = 30 * scaling_factor
        self.sonar_near_range = 18 * scaling_factor
        self.sonar_range_offset = 9 * scaling_factor
        self.sonar_positions = [-90 - 22, -90 + 22, -45, -22, 22, 45, 90 - 22, 90 + 22]
        self.sonar_facing_angles = [-90, -90, 0, 0, 0, 0, 90, 90]

        self.ir_sensor_range = 4 * scaling_factor

        self.reward = 0.0
        self.sensor_feedback = torch.zeros(18, dtype=torch.float32, device=self.device)
        self.sensor_feedback_masks = torch.zeros(
            (9, self.height, self.width), dtype=torch.uint8, device=self.device
        )
        self.stuck_flag = 0

        self.max_steps = max_steps
        self.current_step = 0

        self.wall_obstacles = wall_obstacles
        self.obstacles: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

        # Difficulty knobs (kept simple and reproducible).
        # 0: static box
        # 2+: blinking box
        # 3+: moving box (random trajectory)
        self.difficulty = difficulty
        self.box_speed = int(box_speed)
        self.box_blink_enabled = difficulty >= 2
        self.box_move_enabled = difficulty >= 3

        self.box_visible = True
        self._blink_countdown = 0
        self._blink_on_range = (30, 60)
        self._blink_off_range = (10, 30)

        self._box_vx = 0
        self._box_vy = 0

        self.goal_margin = 20 * scaling_factor
        self.success_bonus = 2000

        self.rng = np.random.default_rng(seed)

        self.box_size = int(12 * scaling_factor)
        self.box_center_x = 0
        self.box_center_y = 0
        self.box_yaw_angle = 30
        self.box_corners: list[list[float]] = []
        self.box_frame = self._zeros3()

        # Negative object kept for compatibility; not used in scoring by default.
        self.neg_circle_frame = self._zeros3()
        self.neg_circle_center_x = 0
        self.neg_circle_center_y = 0

        self.obstacle_frame = self._zeros3()

        self.bot_mask = self._zeros3()
        self.done = False
        self.enable_push = False
        self.active_state = "F"

        self.reset(seed=seed)

    def _zeros3(self) -> torch.Tensor:
        return torch.zeros((self.height, self.width, 3), dtype=torch.uint8, device=self.device)

    def _zeros1(self) -> torch.Tensor:
        return torch.zeros((self.height, self.width), dtype=torch.uint8, device=self.device)

    def _to_numpy_feedback(self) -> np.ndarray:
        return self.sensor_feedback.detach().cpu().numpy().copy()

    def _mask_circle(self, cx: float, cy: float, radius: float) -> torch.Tensor:
        return ((self._grid_x - cx) ** 2 + (self._grid_y - cy) ** 2) <= (radius * radius)

    def _mask_rect(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> torch.Tensor:
        x1, y1 = p1
        x2, y2 = p2
        x_min = max(0, min(x1, x2))
        x_max = min(self.width - 1, max(x1, x2))
        y_min = max(0, min(y1, y2))
        y_max = min(self.height - 1, max(y1, y2))
        return (
            (self._grid_x >= float(x_min))
            & (self._grid_x <= float(x_max))
            & (self._grid_y >= float(y_min))
            & (self._grid_y <= float(y_max))
        )

    def _mask_line(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        thickness: int,
    ) -> torch.Tensor:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        dy = y2 - y1
        denom = dx * dx + dy * dy
        radius = max(0.5, float(thickness) / 2.0)

        if denom <= 1e-8:
            return ((self._grid_x - x1) ** 2 + (self._grid_y - y1) ** 2) <= (radius * radius)

        t = ((self._grid_x - x1) * dx + (self._grid_y - y1) * dy) / denom
        t = torch.clamp(t, 0.0, 1.0)
        px = x1 + t * dx
        py = y1 + t * dy
        dist2 = (self._grid_x - px) ** 2 + (self._grid_y - py) ** 2
        return dist2 <= (radius * radius)

    def _mask_polygon(self, points: list[Tuple[float, float]]) -> torch.Tensor:
        # Ray-casting algorithm (odd-even rule), vectorized over image grid.
        n = len(points)
        inside = torch.zeros((self.height, self.width), dtype=torch.bool, device=self.device)
        x = self._grid_x
        y = self._grid_y

        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            y1f = float(y1)
            y2f = float(y2)
            x1f = float(x1)
            x2f = float(x2)
            cond = (y1f > y) != (y2f > y)
            x_intersect = (x2f - x1f) * (y - y1f) / (y2f - y1f + 1e-9) + x1f
            inside = inside ^ (cond & (x < x_intersect))

        return inside

    def _paint(self, img: torch.Tensor, mask: torch.Tensor, color) -> None:
        if img.dim() == 2:
            if isinstance(color, tuple):
                value = int(color[0])
            else:
                value = int(color)
            img[mask] = torch.as_tensor(value, dtype=torch.uint8, device=self.device)
            return

        if not isinstance(color, tuple):
            color = (int(color), int(color), int(color))
        for c in range(3):
            img[..., c][mask] = torch.as_tensor(int(color[c]), dtype=torch.uint8, device=self.device)

    def _draw_circle(self, img: torch.Tensor, center: Tuple[int, int], radius: int, color, thickness: int):
        cx, cy = int(center[0]), int(center[1])
        if thickness < 0:
            mask = self._mask_circle(cx, cy, float(radius))
        else:
            outer = self._mask_circle(cx, cy, float(radius))
            inner = self._mask_circle(cx, cy, float(max(0, radius - thickness)))
            mask = outer & (~inner)
        self._paint(img, mask, color)

    def _draw_rectangle(self, img: torch.Tensor, p1: Tuple[int, int], p2: Tuple[int, int], color, thickness: int):
        if thickness < 0:
            mask = self._mask_rect(p1, p2)
            self._paint(img, mask, color)
            return

        x1, y1 = p1
        x2, y2 = p2
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        outer = self._mask_rect((x_min, y_min), (x_max, y_max))
        inner = self._mask_rect((x_min + thickness, y_min + thickness), (x_max - thickness, y_max - thickness))
        mask = outer & (~inner)
        self._paint(img, mask, color)

    def _draw_polygon(self, img: torch.Tensor, points: list[Tuple[float, float]], color):
        mask = self._mask_polygon(points)
        self._paint(img, mask, color)

    def _draw_line(
        self,
        img: torch.Tensor,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color,
        thickness: int,
    ):
        mask = self._mask_line((p1[0], p1[1]), (p2[0], p2[1]), thickness)
        self._paint(img, mask, color)

    def _add_weighted(self, img1: torch.Tensor, alpha: float, img2: torch.Tensor, beta: float, gamma: float):
        out = img1.to(torch.float32) * float(alpha) + img2.to(torch.float32) * float(beta) + float(gamma)
        out = torch.clamp(out, 0.0, 255.0).to(torch.uint8)
        return out

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.reward = 0.0
        self.done = False
        self.enable_push = False
        self.active_state = "F"
        self.stuck_flag = 0
        self.sensor_feedback.zero_()

        # Build obstacles first so we can avoid spawning inside/too-close to walls.
        self._build_obstacles()

        def circle_intersects_rect(cx: int, cy: int, radius: int, rect) -> bool:
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

        def clear_of_obstacles(cx: int, cy: int, radius: int) -> bool:
            if not self.wall_obstacles:
                return True
            for rect in self.obstacles:
                if circle_intersects_rect(cx, cy, radius, rect):
                    return False
            return True

        start_clearance = max(1, int(self.forward_step_unit) + 1)

        bot_bounds_margin = 10 + self.bot_radius + start_clearance
        box_half = max(1, self.box_size // 2)
        box_bounds_margin = 10 + box_half

        max_attempts = 5000
        attempts = 0

        while True:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("Failed to sample a valid initial bot position")
            bx = int(self.rng.integers(bot_bounds_margin, self.frame_size[1] - bot_bounds_margin))
            by = int(self.rng.integers(bot_bounds_margin, self.frame_size[0] - bot_bounds_margin))
            if clear_of_obstacles(bx, by, self.bot_radius + start_clearance):
                self.bot_center_x = bx
                self.bot_center_y = by
                break

        self.facing_angle = int(self.rng.integers(0, 360))

        while True:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("Failed to sample a valid initial box position")
            x = int(self.rng.integers(box_bounds_margin, self.frame_size[1] - box_bounds_margin))
            y = int(self.rng.integers(box_bounds_margin, self.frame_size[0] - box_bounds_margin))
            if not clear_of_obstacles(x, y, box_half):
                continue
            dx = x - self.bot_center_x
            dy = y - self.bot_center_y
            min_sep = self.bot_radius + box_half + start_clearance
            if (dx * dx + dy * dy) >= (min_sep * min_sep):
                self.box_center_x = x
                self.box_center_y = y
                break

        self.box_yaw_angle = 30
        self.box_visible = True
        self._reset_box_dynamics()

        self.neg_circle_center_x = int(
            self.rng.integers(box_bounds_margin, self.frame_size[1] - box_bounds_margin)
        )
        self.neg_circle_center_y = int(
            self.rng.integers(box_bounds_margin, self.frame_size[0] - box_bounds_margin)
        )
        self.neg_circle_frame = self._zeros3()

        self._update_frames(show=False)
        self.get_feedback()
        self.update_reward()
        return self._to_numpy_feedback()

    def _reset_box_dynamics(self) -> None:
        if self.box_blink_enabled:
            self.box_visible = True
            self._blink_countdown = int(
                self.rng.integers(self._blink_on_range[0], self._blink_on_range[1] + 1)
            )
        else:
            self.box_visible = True
            self._blink_countdown = 0

        if self.box_move_enabled:
            directions = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
            dx, dy = directions[int(self.rng.integers(0, len(directions)))]
            self._box_vx = int(dx * max(1, self.box_speed))
            self._box_vy = int(dy * max(1, self.box_speed))
        else:
            self._box_vx = 0
            self._box_vy = 0

    def _update_box_dynamics(self) -> None:
        if self.enable_push:
            self.box_visible = True
            return

        if self.box_blink_enabled:
            self._blink_countdown -= 1
            if self._blink_countdown <= 0:
                self.box_visible = not self.box_visible
                if self.box_visible:
                    lo, hi = self._blink_on_range
                else:
                    lo, hi = self._blink_off_range
                self._blink_countdown = int(self.rng.integers(lo, hi + 1))

        if self.box_move_enabled:
            if float(self.rng.random()) < 0.05:
                self._reset_box_dynamics()

            next_x = int(self.box_center_x + self._box_vx)
            next_y = int(self.box_center_y + self._box_vy)
            half = max(1, self.box_size // 2)

            min_x = 10 + half
            max_x = self.frame_size[1] - 10 - half
            min_y = 10 + half
            max_y = self.frame_size[0] - 10 - half

            bounced = False
            if not (min_x <= next_x <= max_x):
                bounced = True
            if not (min_y <= next_y <= max_y):
                bounced = True

            if self.wall_obstacles and not bounced:
                for p1, p2 in self.obstacles:
                    x1, y1 = p1
                    x2, y2 = p2
                    x1e, x2e = x1 - half, x2 + half
                    y1e, y2e = y1 - half, y2 + half
                    if (x1e <= next_x <= x2e) and (y1e <= next_y <= y2e):
                        if abs(self._box_vx) >= abs(self._box_vy):
                            self._box_vx = -self._box_vx
                        else:
                            self._box_vy = -self._box_vy
                        bounced = True
                        break

            self.box_center_x = int(np.clip(next_x, min_x, max_x))
            self.box_center_y = int(np.clip(next_y, min_y, max_y))

    def _build_obstacles(self):
        self.obstacles = []
        if not self.wall_obstacles:
            return

        wall_thickness = max(6, int(4 * self.scaling_factor))
        x_center = self.frame_size[1] // 2
        x1 = x_center - wall_thickness // 2
        x2 = x_center + wall_thickness // 2

        min_gap = 2 * (self.bot_radius + max(1, self.box_size // 2)) + max(
            10, self.forward_step_unit * 2
        )
        if min_gap >= (self.frame_size[0] - 40):
            self.obstacles = []
            return

        gap_height = max(min_gap, int(12 * self.scaling_factor))
        gap_y_center = self.frame_size[0] // 2
        y_top_end = max(10, gap_y_center - gap_height // 2)
        y_bottom_start = min(self.frame_size[0] - 10, gap_y_center + gap_height // 2)

        self.obstacles.append(((x1, 10), (x2, y_top_end)))
        self.obstacles.append(((x1, y_bottom_start), (x2, self.frame_size[0] - 10)))

    def _obstacle_binary_mask(self) -> torch.Tensor:
        obstacle_mask = torch.zeros((self.height, self.width), dtype=torch.bool, device=self.device)
        for p1, p2 in self.obstacles:
            obstacle_mask |= self._mask_rect(p1, p2)
        return obstacle_mask

    def _box_would_collide(self, new_x: int, new_y: int) -> bool:
        if not self.wall_obstacles:
            return False

        half = max(1, self.box_size // 2)
        box_mask_t = self._mask_circle(float(new_x), float(new_y), float(half))
        obstacle_mask = self._obstacle_binary_mask()
        return bool(torch.any(box_mask_t & obstacle_mask).item())

    def _box_touches_boundary(self, x: int, y: int) -> bool:
        half = max(1, self.box_size // 2)
        left = x - half
        right = x + half
        bottom = y - half
        top = y + half
        return (
            left <= 10
            or right >= (self.frame_size[1] - 10)
            or bottom <= 10
            or top >= (self.frame_size[0] - 10)
        )

    def _would_collide(self, new_x: int, new_y: int) -> bool:
        if not self.wall_obstacles:
            return False

        bot_mask_t = self._mask_circle(float(new_x), float(new_y), float(self.bot_radius))
        obstacle_mask = self._obstacle_binary_mask()
        return bool(torch.any(bot_mask_t & obstacle_mask).item())

    def _update_frames(self, show: bool) -> None:
        # Always build masks/frames so observation is correct in headless mode.
        self.frame = self._zeros3()
        self.bot_mask = self._zeros3()

        self._draw_rectangle(
            self.frame,
            (0 + 5, 0 + 5),
            (self.frame_size[0] - 5, self.frame_size[1] - 5),
            (255, 0, 0),
            1,
        )
        self._draw_rectangle(
            self.frame,
            (0 + 10, 0 + 10),
            (self.frame_size[0] - 10, self.frame_size[1] - 10),
            (255, 0, 0),
            1,
        )

        self.box_frame = self._zeros3()
        if self.box_visible or self.enable_push:
            self.box_corners = []
            for i in range(0, 360, 90):
                x = self.box_center_x + (self.box_size // 2) * np.cos(
                    np.deg2rad(self.box_yaw_angle + i)
                )
                y = self.box_center_y + (self.box_size // 2) * np.sin(
                    np.deg2rad(self.box_yaw_angle + i)
                )
                self.box_corners.append([x, y])
            poly = [(float(p[0]), float(p[1])) for p in self.box_corners]
            self._draw_polygon(self.box_frame, poly, (100, 100, 100))

        self.obstacle_frame = self._zeros3()
        for p1, p2 in self.obstacles:
            self._draw_rectangle(self.obstacle_frame, p1, p2, (100, 100, 100), -1)

        self.sensor_feedback_masks = torch.zeros(
            (9, self.height, self.width), dtype=torch.uint8, device=self.device
        )

        self._draw_circle(
            self.frame,
            (self.bot_center_x, self.bot_center_y),
            self.bot_radius,
            self.bot_color,
            1,
        )
        self._draw_circle(
            self.bot_mask,
            (self.bot_center_x, self.bot_center_y),
            self.bot_radius,
            (100, 100, 100),
            -1,
        )

        for sonar_range, sonar_intensity in zip(
            [self.sonar_far_range, self.sonar_near_range, self.sonar_range_offset],
            [100, 50, 0],
        ):
            for index, (sonar_pos_angle, sonar_face_angle) in enumerate(
                zip(self.sonar_positions, self.sonar_facing_angles)
            ):
                if sonar_intensity == 0:
                    noise_reduction = 2
                else:
                    noise_reduction = 0

                p1_x = self.bot_center_x + self.bot_radius * np.cos(
                    np.deg2rad(self.facing_angle + sonar_pos_angle)
                )
                p1_y = self.bot_center_y + self.bot_radius * np.sin(
                    np.deg2rad(self.facing_angle + sonar_pos_angle)
                )
                p2_x = p1_x + sonar_range * np.cos(
                    np.deg2rad(
                        self.facing_angle
                        + sonar_face_angle
                        + self.sonar_fov // 2
                        + noise_reduction
                    )
                )
                p2_y = p1_y + sonar_range * np.sin(
                    np.deg2rad(
                        self.facing_angle
                        + sonar_face_angle
                        + self.sonar_fov // 2
                        + noise_reduction
                    )
                )
                p3_x = p1_x + sonar_range * np.cos(
                    np.deg2rad(
                        self.facing_angle
                        + sonar_face_angle
                        - self.sonar_fov // 2
                        - noise_reduction
                    )
                )
                p3_y = p1_y + sonar_range * np.sin(
                    np.deg2rad(
                        self.facing_angle
                        + sonar_face_angle
                        - self.sonar_fov // 2
                        - noise_reduction
                    )
                )

                tri = [(float(p1_x), float(p1_y)), (float(p2_x), float(p2_y)), (float(p3_x), float(p3_y))]
                self._draw_polygon(self.frame, tri, sonar_intensity)
                self._draw_polygon(self.sensor_feedback_masks[index], tri, sonar_intensity)

        p1_x = int(self.bot_center_x + self.bot_radius * np.cos(np.deg2rad(self.facing_angle)))
        p1_y = int(self.bot_center_y + self.bot_radius * np.sin(np.deg2rad(self.facing_angle)))
        p2_x = int(p1_x + self.ir_sensor_range * np.cos(np.deg2rad(self.facing_angle)))
        p2_y = int(p1_y + self.ir_sensor_range * np.sin(np.deg2rad(self.facing_angle)))

        self._draw_line(self.frame, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)
        self._draw_line(self.sensor_feedback_masks[8], (p1_x, p1_y), (p2_x, p2_y), (50, 50, 50), 2)

        self.frame = self._add_weighted(self.frame, 1.0, self.box_frame, 1.0, 0)
        self.frame = self._add_weighted(self.frame, 1.0, self.obstacle_frame, 1.0, 0)
        self.frame = self._add_weighted(self.frame, 1.0, self.neg_circle_frame, 1.0, 0)
        self.frame = torch.flip(self.frame, dims=[0])

        # show=True is a no-op in torch-only environment.
        _ = show

    def render_frame(self):
        self._update_frames(show=True)

    def update_state_diagram(self):
        # No GUI dependency in torch version.
        pass

    def get_feedback(self):
        combined_object_frame = self.box_frame
        if self.wall_obstacles:
            combined_object_frame = self._add_weighted(
                combined_object_frame, 1.0, self.obstacle_frame, 1.0, 0
            )

        for i in range(self.sensor_feedback_masks.shape[0]):
            overlap_150 = torch.any(
                (self.sensor_feedback_masks[i] + combined_object_frame[:, :, 0]) == 150
            ) or torch.any((self.sensor_feedback_masks[i] + self.neg_circle_frame[:, :, 0]) == 150)
            overlap_200 = torch.any(
                (self.sensor_feedback_masks[i] + combined_object_frame[:, :, 0]) == 200
            ) or torch.any((self.sensor_feedback_masks[i] + self.neg_circle_frame[:, :, 0]) == 200)
            self.sensor_feedback[2 * i] = 1.0 if bool(overlap_150) else 0.0
            self.sensor_feedback[2 * i + 1] = 1.0 if bool(overlap_200) else 0.0

        self.sensor_feedback[17] = float(self.stuck_flag)

    def step(self, move, render=True):
        if self.done:
            return self._to_numpy_feedback(), float(self.reward), bool(self.done)

        self.current_step += 1

        self._update_box_dynamics()

        angle_change = self.move_options[move]
        self.facing_angle += angle_change
        self.active_state = "F"
        if angle_change == 0:
            bot_center_x_t = int(
                self.bot_center_x
                + self.forward_step_unit * np.cos(np.deg2rad(self.facing_angle))
            )
            bot_center_y_t = int(
                self.bot_center_y
                + self.forward_step_unit * np.sin(np.deg2rad(self.facing_angle))
            )
            box_center_x_t = int(
                self.box_center_x
                + self.forward_step_unit * np.cos(np.deg2rad(self.facing_angle))
            )
            box_center_y_t = int(
                self.box_center_y
                + self.forward_step_unit * np.sin(np.deg2rad(self.facing_angle))
            )
            if self.enable_push:
                half = max(1, self.box_size // 2)
                min_x = 10 + half
                max_x = self.frame_size[1] - 10 - half
                min_y = 10 + half
                max_y = self.frame_size[0] - 10 - half

                box_center_x_next = int(np.clip(box_center_x_t, min_x, max_x))
                box_center_y_next = int(np.clip(box_center_y_t, min_y, max_y))

                bot_in_bounds = (
                    (10 + self.bot_radius)
                    <= bot_center_x_t
                    <= (self.frame_size[1] - 10 - self.bot_radius)
                    and (10 + self.bot_radius)
                    <= bot_center_y_t
                    <= (self.frame_size[0] - 10 - self.bot_radius)
                )

                if bot_in_bounds and (not self._would_collide(bot_center_x_t, bot_center_y_t)) and (
                    not self._box_would_collide(box_center_x_next, box_center_y_next)
                ):
                    self.box_center_x = box_center_x_next
                    self.box_center_y = box_center_y_next
                    self.bot_center_x = bot_center_x_t
                    self.bot_center_y = bot_center_y_t
                    self.stuck_flag = 0
                    self.active_state = "P"
                else:
                    self.stuck_flag = 1
                    self.active_state = "U"

            elif (10 + self.bot_radius) <= bot_center_x_t <= (
                self.frame_size[1] - 10 - self.bot_radius
            ) and (10 + self.bot_radius) <= bot_center_y_t <= (
                self.frame_size[0] - 10 - self.bot_radius
            ):
                if not self._would_collide(bot_center_x_t, bot_center_y_t):
                    self.bot_center_x = bot_center_x_t
                    self.bot_center_y = bot_center_y_t
                    self.stuck_flag = 0
                else:
                    self.stuck_flag = 1
                    self.active_state = "U"
            else:
                self.stuck_flag = 1
                self.active_state = "U"

        self._update_frames(show=render)
        self.get_feedback()
        self.update_reward()
        self.check_done_state()
        if render:
            self.update_state_diagram()

        if (not self.done) and (self.current_step >= self.max_steps):
            self.done = True

        return self._to_numpy_feedback(), float(self.reward), bool(self.done)

    def check_done_state(self):
        if self.enable_push:
            self.reward -= 1
        elif (self.box_visible or self.enable_push) and torch.any(
            (self.bot_mask[:, :, 0] + self.box_frame[:, :, 0]) == 200
        ):
            self.reward += 100
            overlap = self.bot_mask[:, :, 0].to(torch.int16) + self.box_frame[:, :, 0].to(torch.int16)
            flat_idx = int(torch.argmax(overlap.reshape(-1)).item())
            _y = flat_idx // self.frame_size[1]
            _x = flat_idx % self.frame_size[1]
            _ = (_x, _y)
            self.enable_push = True
            self.box_visible = True
            self.active_state = "P"
        elif torch.any((self.bot_mask[:, :, 0] + self.neg_circle_frame[:, :, 0]) == 200):
            self.done = True
            self.reward += -100
            print("************Negative done*********************")

        if (
            (not self.done)
            and self.enable_push
            and self._box_touches_boundary(self.box_center_x, self.box_center_y)
        ):
            self.done = True
            self.reward += self.success_bonus

    def update_reward(self):
        left_sensor_reward = torch.sum(self.sensor_feedback[:4] * 1.0)
        forward_far_sensor_reward = torch.sum(self.sensor_feedback[4:12][::2] * 2.0)
        forward_near_sensor_reward = torch.sum(self.sensor_feedback[4:12][1::2] * 3.0)
        right_sensor_reward = torch.sum(self.sensor_feedback[12:16] * 1.0)
        ir_sensor_reward = self.sensor_feedback[16] * 5.0
        stuck_reward = self.sensor_feedback[17] * (-200.0)
        negative_reward = torch.sum(torch.logical_not(self.sensor_feedback.bool()).to(torch.float32)) * -1.0
        self.reward = float(
            left_sensor_reward
            + forward_far_sensor_reward
            + forward_near_sensor_reward
            + right_sensor_reward
            + ir_sensor_reward
            + stuck_reward
            + negative_reward
        )
