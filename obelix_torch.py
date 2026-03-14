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


class OBELIXVectorized:
    """Single-process batched torch environment for OBELIX."""

    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
    MOVE_ANGLES = torch.tensor([45.0, 22.5, 0.0, -22.5, -45.0], dtype=torch.float32)

    def __init__(
        self,
        num_envs: int,
        scaling_factor: int,
        arena_size: int = 500,
        max_steps: int = 1000,
        wall_obstacles: bool = False,
        difficulty: int = 0,
        box_speed: int = 2,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be > 0")

        if device == "auto" or device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.scaling_factor = scaling_factor
        self.arena_size = arena_size
        self.frame_size = (arena_size, arena_size, 3)

        self.bot_radius = int(scaling_factor * 12 / 2)
        self.forward_step_unit = 5

        self.sonar_fov = 20
        self.sonar_far_range = 30 * scaling_factor
        self.sonar_near_range = 18 * scaling_factor
        self.sonar_range_offset = 9 * scaling_factor
        self.sonar_positions = torch.tensor(
            [-90 - 22, -90 + 22, -45, -22, 22, 45, 90 - 22, 90 + 22],
            dtype=torch.float32,
            device=self.device,
        )
        self.sonar_facing_angles = torch.tensor(
            [-90, -90, 0, 0, 0, 0, 90, 90], dtype=torch.float32, device=self.device
        )
        self.ir_sensor_range = 4 * scaling_factor

        self.max_steps = int(max_steps)
        self.wall_obstacles = bool(wall_obstacles)

        self.difficulty = int(difficulty)
        self.box_speed = int(box_speed)
        self.box_blink_enabled = self.difficulty >= 2
        self.box_move_enabled = self.difficulty >= 3
        self._blink_on_range = (30, 60)
        self._blink_off_range = (10, 30)

        self.goal_margin = 20 * scaling_factor
        self.success_bonus = 2000

        self.box_size = int(12 * scaling_factor)
        self.box_half = max(1, self.box_size // 2)

        self.move_angles = self.MOVE_ANGLES.to(self.device)
        self.rng = np.random.default_rng(seed)

        # Vector state.
        self.bot_center_x = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.bot_center_y = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.facing_angle = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.box_center_x = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.box_center_y = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.box_visible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._blink_countdown = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._box_vx = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._box_vy = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.current_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.enable_push = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stuck_flag = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.sensor_feedback = torch.zeros((self.num_envs, 18), dtype=torch.float32, device=self.device)
        self.reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.obstacles: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self._build_obstacles()

        self.reset_all(seed=seed)

    @staticmethod
    def _angle_diff_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.remainder(a - b + 180.0, 360.0) - 180.0

    def _batch_circle_rect_collision(
        self, cx: torch.Tensor, cy: torch.Tensor, radius: float
    ) -> torch.Tensor:
        if (not self.wall_obstacles) or len(self.obstacles) == 0:
            return torch.zeros_like(cx, dtype=torch.bool)
        collided = torch.zeros_like(cx, dtype=torch.bool)
        rr = float(radius) * float(radius)
        for p1, p2 in self.obstacles:
            x1, y1 = p1
            x2, y2 = p2
            x_min = float(min(x1, x2))
            x_max = float(max(x1, x2))
            y_min = float(min(y1, y2))
            y_max = float(max(y1, y2))
            closest_x = torch.clamp(cx, min=x_min, max=x_max)
            closest_y = torch.clamp(cy, min=y_min, max=y_max)
            dx = cx - closest_x
            dy = cy - closest_y
            collided |= (dx * dx + dy * dy) <= rr
        return collided

    def _box_touches_boundary(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        half = float(self.box_half)
        left = x - half
        right = x + half
        bottom = y - half
        top = y + half
        return (
            (left <= 10.0)
            | (right >= float(self.frame_size[1] - 10))
            | (bottom <= 10.0)
            | (top >= float(self.frame_size[0] - 10))
        )

    def _build_obstacles(self):
        self.obstacles = []
        if not self.wall_obstacles:
            return

        wall_thickness = max(6, int(4 * self.scaling_factor))
        x_center = self.frame_size[1] // 2
        x1 = x_center - wall_thickness // 2
        x2 = x_center + wall_thickness // 2

        min_gap = 2 * (self.bot_radius + self.box_half) + max(10, self.forward_step_unit * 2)
        if min_gap >= (self.frame_size[0] - 40):
            self.obstacles = []
            return

        gap_height = max(min_gap, int(12 * self.scaling_factor))
        gap_y_center = self.frame_size[0] // 2
        y_top_end = max(10, gap_y_center - gap_height // 2)
        y_bottom_start = min(self.frame_size[0] - 10, gap_y_center + gap_height // 2)

        self.obstacles.append(((x1, 10), (x2, y_top_end)))
        self.obstacles.append(((x1, y_bottom_start), (x2, self.frame_size[0] - 10)))

    def _sample_one_env(self, env_id: int, rng_i: np.random.Generator) -> None:
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
        box_bounds_margin = 10 + self.box_half

        max_attempts = 5000
        attempts = 0

        while True:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("Failed to sample a valid initial bot position")
            bx = int(rng_i.integers(bot_bounds_margin, self.frame_size[1] - bot_bounds_margin))
            by = int(rng_i.integers(bot_bounds_margin, self.frame_size[0] - bot_bounds_margin))
            if clear_of_obstacles(bx, by, self.bot_radius + start_clearance):
                self.bot_center_x[env_id] = float(bx)
                self.bot_center_y[env_id] = float(by)
                break

        self.facing_angle[env_id] = float(int(rng_i.integers(0, 360)))

        while True:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("Failed to sample a valid initial box position")
            x = int(rng_i.integers(box_bounds_margin, self.frame_size[1] - box_bounds_margin))
            y = int(rng_i.integers(box_bounds_margin, self.frame_size[0] - box_bounds_margin))
            if not clear_of_obstacles(x, y, self.box_half):
                continue
            dx = x - int(self.bot_center_x[env_id].item())
            dy = y - int(self.bot_center_y[env_id].item())
            min_sep = self.bot_radius + self.box_half + start_clearance
            if (dx * dx + dy * dy) >= (min_sep * min_sep):
                self.box_center_x[env_id] = float(x)
                self.box_center_y[env_id] = float(y)
                break

    def _reset_box_dynamics(self, idx_t: torch.Tensor, rng_list: list[np.random.Generator]) -> None:
        if idx_t.numel() == 0:
            return
        ids = idx_t.detach().cpu().numpy().tolist()
        for local_i, env_id in enumerate(ids):
            rng_i = rng_list[local_i]
            if self.box_blink_enabled:
                self.box_visible[env_id] = True
                self._blink_countdown[env_id] = int(
                    rng_i.integers(self._blink_on_range[0], self._blink_on_range[1] + 1)
                )
            else:
                self.box_visible[env_id] = True
                self._blink_countdown[env_id] = 0

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
                dx, dy = directions[int(rng_i.integers(0, len(directions)))]
                self._box_vx[env_id] = float(dx * max(1, self.box_speed))
                self._box_vy[env_id] = float(dy * max(1, self.box_speed))
            else:
                self._box_vx[env_id] = 0.0
                self._box_vy[env_id] = 0.0

    def reset(
        self,
        env_indices: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        if len(env_indices) == 0:
            return {}

        idx_t = torch.as_tensor(env_indices, dtype=torch.long, device=self.device)

        if seed is not None:
            rng_list = [np.random.default_rng(int(seed + int(i))) for i in env_indices]
        else:
            rng_list = [self.rng for _ in env_indices]

        self.current_step[idx_t] = 0
        self.done[idx_t] = False
        self.enable_push[idx_t] = False
        self.stuck_flag[idx_t] = 0.0
        self.sensor_feedback[idx_t] = 0.0
        self.reward[idx_t] = 0.0

        for local_i, env_id in enumerate(env_indices):
            self._sample_one_env(int(env_id), rng_list[local_i])

        self._reset_box_dynamics(idx_t, rng_list)
        self._compute_feedback()
        self._update_reward()

        obs = self.sensor_feedback[idx_t].detach().cpu().numpy().astype(np.float32, copy=True)
        return {int(env_indices[i]): obs[i] for i in range(len(env_indices))}

    def reset_all(self, seed: Optional[int] = None) -> np.ndarray:
        obs_map = self.reset(env_indices=list(range(self.num_envs)), seed=seed)
        return np.stack([obs_map[i] for i in range(self.num_envs)], axis=0)

    def _update_box_dynamics(self, active: torch.Tensor) -> None:
        if active.numel() == 0:
            return

        movable = active & (~self.enable_push)
        if movable.any():
            if self.box_blink_enabled:
                self._blink_countdown[movable] -= 1
                toggles = movable & (self._blink_countdown <= 0)
                if toggles.any():
                    ids = torch.nonzero(toggles, as_tuple=False).squeeze(1).detach().cpu().numpy().tolist()
                    for env_id in ids:
                        self.box_visible[env_id] = ~self.box_visible[env_id]
                        if bool(self.box_visible[env_id]):
                            lo, hi = self._blink_on_range
                        else:
                            lo, hi = self._blink_off_range
                        self._blink_countdown[env_id] = int(self.rng.integers(lo, hi + 1))

            if self.box_move_enabled:
                ids = torch.nonzero(movable, as_tuple=False).squeeze(1).detach().cpu().numpy().tolist()
                for env_id in ids:
                    if float(self.rng.random()) < 0.05:
                        idx_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                        self._reset_box_dynamics(idx_t, [self.rng])

                    next_x = float(self.box_center_x[env_id] + self._box_vx[env_id])
                    next_y = float(self.box_center_y[env_id] + self._box_vy[env_id])

                    min_x = 10 + self.box_half
                    max_x = self.frame_size[1] - 10 - self.box_half
                    min_y = 10 + self.box_half
                    max_y = self.frame_size[0] - 10 - self.box_half

                    bounced = False
                    if not (min_x <= next_x <= max_x):
                        bounced = True
                    if not (min_y <= next_y <= max_y):
                        bounced = True

                    if self.wall_obstacles and (not bounced):
                        for p1, p2 in self.obstacles:
                            x1, y1 = p1
                            x2, y2 = p2
                            x1e, x2e = x1 - self.box_half, x2 + self.box_half
                            y1e, y2e = y1 - self.box_half, y2 + self.box_half
                            if (x1e <= next_x <= x2e) and (y1e <= next_y <= y2e):
                                if abs(float(self._box_vx[env_id])) >= abs(float(self._box_vy[env_id])):
                                    self._box_vx[env_id] = -self._box_vx[env_id]
                                else:
                                    self._box_vy[env_id] = -self._box_vy[env_id]
                                bounced = True
                                break

                    self.box_center_x[env_id] = float(np.clip(next_x, min_x, max_x))
                    self.box_center_y[env_id] = float(np.clip(next_y, min_y, max_y))

        attached = active & self.enable_push
        if attached.any():
            self.box_visible[attached] = True

    def _compute_feedback(self) -> None:
        self.sensor_feedback.zero_()

        visible = self.box_visible | self.enable_push
        bot_x = self.bot_center_x
        bot_y = self.bot_center_y
        facing = self.facing_angle

        face_rad = torch.deg2rad(facing[:, None] + self.sonar_positions[None, :])
        p1_x = bot_x[:, None] + float(self.bot_radius) * torch.cos(face_rad)
        p1_y = bot_y[:, None] + float(self.bot_radius) * torch.sin(face_rad)

        dir_deg = facing[:, None] + self.sonar_facing_angles[None, :]

        dx = self.box_center_x[:, None] - p1_x
        dy = self.box_center_y[:, None] - p1_y
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        bearing = torch.rad2deg(torch.atan2(dy, dx))
        ang = torch.abs(self._angle_diff_deg(bearing, dir_deg))

        box_margin = torch.rad2deg(
            torch.atan2(torch.full_like(dist, float(self.box_half)), torch.clamp(dist, min=1e-6))
        )
        within = ang <= (float(self.sonar_fov) / 2.0 + box_margin)

        in_offset = dist <= float(self.sonar_range_offset)
        in_near = (dist <= float(self.sonar_near_range)) & (~in_offset)
        in_far = (dist <= float(self.sonar_far_range)) & (dist > float(self.sonar_near_range))

        near_bits = visible[:, None] & within & in_near
        far_bits = visible[:, None] & within & in_far

        if self.wall_obstacles and len(self.obstacles) > 0:
            near_obs = torch.zeros_like(near_bits)
            far_obs = torch.zeros_like(far_bits)
            for p1, p2 in self.obstacles:
                x1, y1 = p1
                x2, y2 = p2
                x_min = float(min(x1, x2))
                x_max = float(max(x1, x2))
                y_min = float(min(y1, y2))
                y_max = float(max(y1, y2))
                qx = torch.clamp(p1_x, min=x_min, max=x_max)
                qy = torch.clamp(p1_y, min=y_min, max=y_max)

                odx = qx - p1_x
                ody = qy - p1_y
                odist = torch.sqrt(odx * odx + ody * ody + 1e-6)
                obearing = torch.rad2deg(torch.atan2(ody, odx))
                oang = torch.abs(self._angle_diff_deg(obearing, dir_deg))
                owithin = oang <= (float(self.sonar_fov) / 2.0)

                onear = owithin & (odist > float(self.sonar_range_offset)) & (
                    odist <= float(self.sonar_near_range)
                )
                ofar = owithin & (odist > float(self.sonar_near_range)) & (
                    odist <= float(self.sonar_far_range)
                )
                near_obs |= onear
                far_obs |= ofar
            near_bits |= near_obs
            far_bits |= far_obs

        for i in range(8):
            self.sensor_feedback[:, 2 * i] = near_bits[:, i].to(torch.float32)
            self.sensor_feedback[:, 2 * i + 1] = far_bits[:, i].to(torch.float32)

        # IR sensor (index 16) from segment intersection with box.
        face = torch.deg2rad(facing)
        ir_p1_x = bot_x + float(self.bot_radius) * torch.cos(face)
        ir_p1_y = bot_y + float(self.bot_radius) * torch.sin(face)
        ir_p2_x = ir_p1_x + float(self.ir_sensor_range) * torch.cos(face)
        ir_p2_y = ir_p1_y + float(self.ir_sensor_range) * torch.sin(face)

        vx = ir_p2_x - ir_p1_x
        vy = ir_p2_y - ir_p1_y
        wx = self.box_center_x - ir_p1_x
        wy = self.box_center_y - ir_p1_y
        denom = vx * vx + vy * vy + 1e-6
        t = torch.clamp((wx * vx + wy * vy) / denom, 0.0, 1.0)
        proj_x = ir_p1_x + t * vx
        proj_y = ir_p1_y + t * vy
        dist2 = (self.box_center_x - proj_x) ** 2 + (self.box_center_y - proj_y) ** 2
        ir_hit = visible & (dist2 <= float((self.box_half + 1) * (self.box_half + 1)))
        self.sensor_feedback[:, 16] = ir_hit.to(torch.float32)

        self.sensor_feedback[:, 17] = self.stuck_flag

    def _update_reward(self) -> None:
        left_sensor_reward = torch.sum(self.sensor_feedback[:, :4] * 1.0, dim=1)
        forward_far_sensor_reward = torch.sum(self.sensor_feedback[:, 4:12][:, ::2] * 2.0, dim=1)
        forward_near_sensor_reward = torch.sum(self.sensor_feedback[:, 4:12][:, 1::2] * 3.0, dim=1)
        right_sensor_reward = torch.sum(self.sensor_feedback[:, 12:16] * 1.0, dim=1)
        ir_sensor_reward = self.sensor_feedback[:, 16] * 5.0
        stuck_reward = self.sensor_feedback[:, 17] * (-200.0)
        negative_reward = torch.sum(
            torch.logical_not(self.sensor_feedback.bool()).to(torch.float32), dim=1
        ) * -1.0
        self.reward = (
            left_sensor_reward
            + forward_far_sensor_reward
            + forward_near_sensor_reward
            + right_sensor_reward
            + ir_sensor_reward
            + stuck_reward
            + negative_reward
        )

    def _check_done_state(self) -> None:
        push_mask = self.enable_push & (~self.done)
        if push_mask.any():
            self.reward[push_mask] -= 1.0

        can_attach = (~self.enable_push) & (~self.done) & self.box_visible
        if can_attach.any():
            dx = self.box_center_x - self.bot_center_x
            dy = self.box_center_y - self.bot_center_y
            attach = can_attach & ((dx * dx + dy * dy) <= float((self.bot_radius + self.box_half) ** 2))
            if attach.any():
                self.reward[attach] += 100.0
                self.enable_push[attach] = True
                self.box_visible[attach] = True

        boundary_touch = (~self.done) & self.enable_push & self._box_touches_boundary(
            self.box_center_x, self.box_center_y
        )
        if boundary_touch.any():
            self.done[boundary_touch] = True
            self.reward[boundary_touch] += float(self.success_bonus)

    def _actions_to_idx(self, actions) -> torch.Tensor:
        if isinstance(actions, torch.Tensor):
            idx = actions.to(device=self.device, dtype=torch.long)
        elif isinstance(actions, np.ndarray):
            idx = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        elif isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], str):
            lut = {a: i for i, a in enumerate(self.ACTIONS)}
            idx = torch.as_tensor([lut[a] for a in actions], dtype=torch.long, device=self.device)
        else:
            idx = torch.as_tensor(actions, dtype=torch.long, device=self.device)

        if idx.numel() != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {idx.numel()}")
        return idx

    def step(self, actions):
        action_idx = self._actions_to_idx(actions)
        active = ~self.done

        if active.any():
            self.current_step[active] += 1
            self._update_box_dynamics(active)

            angle_change = self.move_angles[action_idx]
            self.facing_angle[active] += angle_change[active]

            fw_mask = active & (angle_change == 0.0)
            if fw_mask.any():
                face = torch.deg2rad(self.facing_angle)
                bot_xt = torch.trunc(
                    self.bot_center_x + float(self.forward_step_unit) * torch.cos(face)
                )
                bot_yt = torch.trunc(
                    self.bot_center_y + float(self.forward_step_unit) * torch.sin(face)
                )
                box_xt = torch.trunc(
                    self.box_center_x + float(self.forward_step_unit) * torch.cos(face)
                )
                box_yt = torch.trunc(
                    self.box_center_y + float(self.forward_step_unit) * torch.sin(face)
                )

                push_mask = fw_mask & self.enable_push
                if push_mask.any():
                    min_x = 10 + self.box_half
                    max_x = self.frame_size[1] - 10 - self.box_half
                    min_y = 10 + self.box_half
                    max_y = self.frame_size[0] - 10 - self.box_half
                    box_x_next = torch.clamp(box_xt, min=float(min_x), max=float(max_x))
                    box_y_next = torch.clamp(box_yt, min=float(min_y), max=float(max_y))

                    bot_in_bounds = (
                        (bot_xt >= float(10 + self.bot_radius))
                        & (bot_xt <= float(self.frame_size[1] - 10 - self.bot_radius))
                        & (bot_yt >= float(10 + self.bot_radius))
                        & (bot_yt <= float(self.frame_size[0] - 10 - self.bot_radius))
                    )
                    bot_col = self._batch_circle_rect_collision(
                        bot_xt, bot_yt, float(self.bot_radius)
                    )
                    box_col = self._batch_circle_rect_collision(
                        box_x_next, box_y_next, float(self.box_half)
                    )

                    can_move = push_mask & bot_in_bounds & (~bot_col) & (~box_col)
                    cant_move = push_mask & (~can_move)

                    self.box_center_x[can_move] = box_x_next[can_move]
                    self.box_center_y[can_move] = box_y_next[can_move]
                    self.bot_center_x[can_move] = bot_xt[can_move]
                    self.bot_center_y[can_move] = bot_yt[can_move]
                    self.stuck_flag[can_move] = 0.0
                    self.stuck_flag[cant_move] = 1.0

                nonpush = fw_mask & (~self.enable_push)
                if nonpush.any():
                    bot_in_bounds = (
                        (bot_xt >= float(10 + self.bot_radius))
                        & (bot_xt <= float(self.frame_size[1] - 10 - self.bot_radius))
                        & (bot_yt >= float(10 + self.bot_radius))
                        & (bot_yt <= float(self.frame_size[0] - 10 - self.bot_radius))
                    )
                    bot_col = self._batch_circle_rect_collision(
                        bot_xt, bot_yt, float(self.bot_radius)
                    )
                    can_move = nonpush & bot_in_bounds & (~bot_col)
                    cant_move = nonpush & (~can_move)
                    self.bot_center_x[can_move] = bot_xt[can_move]
                    self.bot_center_y[can_move] = bot_yt[can_move]
                    self.stuck_flag[can_move] = 0.0
                    self.stuck_flag[cant_move] = 1.0

            self._compute_feedback()
            self._update_reward()
            self._check_done_state()

            timeout = (~self.done) & (self.current_step >= int(self.max_steps))
            self.done[timeout] = True

        obs = self.sensor_feedback.detach().cpu().numpy().astype(np.float32, copy=True)
        rew = self.reward.detach().cpu().numpy().astype(np.float32, copy=True)
        done = self.done.detach().cpu().numpy().astype(bool, copy=True)
        return obs, rew, done

    def close(self) -> None:
        return
