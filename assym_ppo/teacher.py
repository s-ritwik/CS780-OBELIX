from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from rnn_model import ACTIONS


ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
TURN45_DEG = 35.0
TURN22_DEG = 15.0


def angle_diff_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a - b + 180.0, 360.0) - 180.0


def _to_float_tensor(value, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _to_bool_tensor(value, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.bool)
    return torch.as_tensor(value, device=device, dtype=torch.bool)


def wall_geometry(vec_env) -> tuple[float, float] | None:
    if (not bool(vec_env.wall_obstacles)) or len(vec_env.obstacles) == 0:
        return None
    (x1, _), (x2, y_top_end) = vec_env.obstacles[0]
    (_, y_bottom_start), _ = vec_env.obstacles[-1]
    return 0.5 * float(x1 + x2), 0.5 * float(y_top_end + y_bottom_start)


def _choose_push_heading_t(
    *,
    box_x: torch.Tensor,
    box_y: torch.Tensor,
    frame_w: float,
    frame_h: float,
    box_half: float,
    wall_x: torch.Tensor | None,
) -> torch.Tensor:
    left = box_x - (10.0 + box_half)
    right = (frame_w - 10.0 - box_half) - box_x
    bottom = box_y - (10.0 + box_half)
    top = (frame_h - 10.0 - box_half) - box_y
    vertical_dist = torch.minimum(bottom, top)
    vertical_heading = torch.where(bottom < top, torch.full_like(box_x, -90.0), torch.full_like(box_x, 90.0))

    if wall_x is None:
        horizontal_dist = torch.minimum(left, right)
        horizontal_heading = torch.where(left < right, torch.full_like(box_x, 180.0), torch.zeros_like(box_x))
    else:
        horizontal_dist = torch.where(box_x <= wall_x, left, right)
        horizontal_heading = torch.where(box_x <= wall_x, torch.full_like(box_x, 180.0), torch.zeros_like(box_x))

    use_vertical = vertical_dist + 20.0 < horizontal_dist
    return torch.where(use_vertical, vertical_heading, horizontal_heading)


def _stage_target_t(
    *,
    box_x: torch.Tensor,
    box_y: torch.Tensor,
    push_heading_deg: torch.Tensor,
    bot_radius: float,
    box_half: float,
    frame_w: float,
    frame_h: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    stage_dist = float(bot_radius + box_half + 22.0)
    push_rad = torch.deg2rad(push_heading_deg)
    min_x = 10.0 + bot_radius
    max_x = frame_w - 10.0 - bot_radius
    min_y = 10.0 + bot_radius
    max_y = frame_h - 10.0 - bot_radius
    stage_x = torch.clamp(box_x - stage_dist * torch.cos(push_rad), min=min_x, max=max_x)
    stage_y = torch.clamp(box_y - stage_dist * torch.sin(push_rad), min=min_y, max=max_y)
    return stage_x, stage_y


def _route_target_via_gap_t(
    *,
    bot_x: torch.Tensor,
    bot_y: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    wall_x: torch.Tensor | None,
    gap_y: torch.Tensor | None,
    clearance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if wall_x is None or gap_y is None:
        return target_x, target_y

    bot_side = torch.sign(bot_x - wall_x)
    target_side = torch.sign(target_x - wall_x)
    opposite_side = (bot_side != 0.0) & (target_side != 0.0) & (bot_side != target_side)
    near_wall_misaligned = (torch.abs(bot_x - wall_x) < clearance) & (torch.abs(bot_y - gap_y) > 20.0)
    route_gap = opposite_side | near_wall_misaligned
    if not bool(torch.any(route_gap)):
        return target_x, target_y

    route_idx = torch.nonzero(route_gap, as_tuple=False).squeeze(1)
    side = torch.where(bot_side[route_idx] == 0.0, torch.ones_like(bot_side[route_idx]), bot_side[route_idx])
    pre_gap_x = wall_x[route_idx] + side * clearance
    post_gap_x = wall_x[route_idx] - side * (clearance + 18.0)
    aligned = (torch.abs(bot_x[route_idx] - pre_gap_x) < 18.0) & (torch.abs(bot_y[route_idx] - gap_y[route_idx]) < 18.0)

    routed_x = torch.where(aligned, post_gap_x, pre_gap_x)
    routed_y = gap_y[route_idx]
    target_x = target_x.clone()
    target_y = target_y.clone()
    target_x[route_idx] = routed_x
    target_y[route_idx] = routed_y
    return target_x, target_y


class PrivilegedTeacher:
    def __init__(self, num_envs: int, device: torch.device) -> None:
        self.num_envs = int(num_envs)
        self.device = device
        self.escape_dir = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.escape_steps = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)

    def reset_all(self) -> None:
        self.escape_dir.fill_(1.0)
        self.escape_steps.zero_()

    def reset_indices(self, env_indices: list[int] | torch.Tensor) -> None:
        if isinstance(env_indices, list):
            if len(env_indices) == 0:
                return
            idx = torch.as_tensor(env_indices, dtype=torch.long, device=self.device)
        else:
            if env_indices.numel() == 0:
                return
            idx = env_indices.to(device=self.device, dtype=torch.long)
        self.escape_dir[idx] = 1.0
        self.escape_steps[idx] = 0

    def act(self, vec_env) -> torch.Tensor:
        device = self.device
        actions = torch.full(
            (self.num_envs,),
            ACTION_TO_INDEX["FW"],
            dtype=torch.long,
            device=device,
        )
        done = _to_bool_tensor(vec_env.done, device=device)
        active = ~done
        if not bool(torch.any(active)):
            return actions

        stuck_flag = _to_float_tensor(vec_env.stuck_flag, device=device)
        stuck = active & (stuck_flag > 0.5)
        if bool(torch.any(stuck)):
            new_escape = stuck & (self.escape_steps <= 0)
            if bool(torch.any(new_escape)):
                self.escape_steps[new_escape] = 6
                self.escape_dir[new_escape] = -self.escape_dir[new_escape]

            odd_step = (self.escape_steps % 2) == 1
            turn_left = stuck & odd_step & (self.escape_dir > 0)
            turn_right = stuck & odd_step & (self.escape_dir < 0)
            actions[turn_left] = ACTION_TO_INDEX["L45"]
            actions[turn_right] = ACTION_TO_INDEX["R45"]
            self.escape_steps[stuck] = torch.clamp(self.escape_steps[stuck] - 1, min=0)

        nonstuck = active & (~stuck)
        if not bool(torch.any(nonstuck)):
            return actions

        bot_x = _to_float_tensor(vec_env.bot_center_x, device=device)
        bot_y = _to_float_tensor(vec_env.bot_center_y, device=device)
        box_x = _to_float_tensor(vec_env.box_center_x, device=device)
        box_y = _to_float_tensor(vec_env.box_center_y, device=device)
        facing = _to_float_tensor(vec_env.facing_angle, device=device)
        enable_push = _to_bool_tensor(vec_env.enable_push, device=device)
        frame_w = float(vec_env.frame_size[1])
        frame_h = float(vec_env.frame_size[0])
        box_half = float(vec_env.box_half)
        wall_geom = wall_geometry(vec_env)
        wall_x_t = None
        gap_y_t = None
        if wall_geom is not None:
            wall_x_t = torch.full_like(bot_x, wall_geom[0])
            gap_y_t = torch.full_like(bot_y, wall_geom[1])
        push_heading = _choose_push_heading_t(
            box_x=box_x,
            box_y=box_y,
            frame_w=frame_w,
            frame_h=frame_h,
            box_half=box_half,
            wall_x=wall_x_t,
        )

        desired_heading = torch.zeros_like(facing)

        pushing = nonstuck & enable_push
        if bool(torch.any(pushing)):
            desired_heading[pushing] = push_heading[pushing]

        seeking = nonstuck & (~enable_push)
        if bool(torch.any(seeking)):
            target_x, target_y = _stage_target_t(
                box_x=box_x,
                box_y=box_y,
                push_heading_deg=push_heading,
                bot_radius=float(vec_env.bot_radius),
                box_half=box_half,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            target_x, target_y = _route_target_via_gap_t(
                bot_x=bot_x,
                bot_y=bot_y,
                target_x=target_x,
                target_y=target_y,
                wall_x=wall_x_t,
                gap_y=gap_y_t,
                clearance=float(vec_env.bot_radius + box_half + 20.0),
            )
            stage_dx = target_x - bot_x
            stage_dy = target_y - bot_y
            stage_dist = torch.sqrt(stage_dx * stage_dx + stage_dy * stage_dy + 1e-6)
            stage_heading = torch.rad2deg(torch.atan2(stage_dy, stage_dx))
            desired_heading[seeking] = torch.where(stage_dist < 18.0, push_heading, stage_heading)[seeking]

        diff = angle_diff_deg(desired_heading, facing)
        left45 = nonstuck & (diff > TURN45_DEG)
        left22 = nonstuck & (diff > TURN22_DEG) & (~left45)
        right45 = nonstuck & (diff < -TURN45_DEG)
        right22 = nonstuck & (diff < -TURN22_DEG) & (~right45)

        actions[left45] = ACTION_TO_INDEX["L45"]
        actions[left22] = ACTION_TO_INDEX["L22"]
        actions[right45] = ACTION_TO_INDEX["R45"]
        actions[right22] = ACTION_TO_INDEX["R22"]
        return actions


def cpu_wall_geometry(env) -> tuple[float, float] | None:
    if (not env.wall_obstacles) or not env.obstacles:
        return None
    (x1, _), (x2, y_top_end) = env.obstacles[0]
    (_, y_bottom_start), _ = env.obstacles[-1]
    return 0.5 * float(x1 + x2), 0.5 * float(y_top_end + y_bottom_start)


def _choose_push_heading_scalar(env) -> float:
    half = max(1, int(env.box_size // 2))
    left = float(env.box_center_x - (10 + half))
    right = float((env.frame_size[1] - 10 - half) - env.box_center_x)
    bottom = float(env.box_center_y - (10 + half))
    top = float((env.frame_size[0] - 10 - half) - env.box_center_y)
    vertical_dist = min(bottom, top)
    vertical_heading = -90.0 if bottom < top else 90.0

    geom = cpu_wall_geometry(env)
    if geom is None:
        horizontal_dist = min(left, right)
        horizontal_heading = 180.0 if left < right else 0.0
    else:
        wall_x, _ = geom
        if env.box_center_x <= wall_x:
            horizontal_dist = left
            horizontal_heading = 180.0
        else:
            horizontal_dist = right
            horizontal_heading = 0.0

    return vertical_heading if vertical_dist + 20.0 < horizontal_dist else horizontal_heading


def _stage_target_scalar(env, push_heading: float) -> tuple[float, float]:
    half = max(1, int(env.box_size // 2))
    stage_dist = float(env.bot_radius + half + 22.0)
    push_rad = math.radians(push_heading)
    min_x = float(10 + env.bot_radius)
    max_x = float(env.frame_size[1] - 10 - env.bot_radius)
    min_y = float(10 + env.bot_radius)
    max_y = float(env.frame_size[0] - 10 - env.bot_radius)
    tx = float(env.box_center_x - stage_dist * math.cos(push_rad))
    ty = float(env.box_center_y - stage_dist * math.sin(push_rad))
    return min(max(tx, min_x), max_x), min(max(ty, min_y), max_y)


def _route_target_via_gap_scalar(env, tx: float, ty: float) -> tuple[float, float]:
    geom = cpu_wall_geometry(env)
    if geom is None:
        return tx, ty

    wall_x, gap_y = geom
    clearance = float(env.bot_radius + max(1, int(env.box_size // 2)) + 20)
    bot_side = math.copysign(1.0, env.bot_center_x - wall_x) if abs(env.bot_center_x - wall_x) > 1 else 0.0
    target_side = math.copysign(1.0, tx - wall_x) if abs(tx - wall_x) > 1 else 0.0
    opposite_side = bot_side != 0.0 and target_side != 0.0 and bot_side != target_side
    near_wall_misaligned = abs(env.bot_center_x - wall_x) < clearance and abs(env.bot_center_y - gap_y) > 20.0
    if not (opposite_side or near_wall_misaligned):
        return tx, ty

    side = bot_side if bot_side != 0.0 else 1.0
    pre_gap_x = wall_x + side * clearance
    post_gap_x = wall_x - side * (clearance + 18.0)
    aligned = abs(env.bot_center_x - pre_gap_x) < 18.0 and abs(env.bot_center_y - gap_y) < 18.0
    return (post_gap_x if aligned else pre_gap_x), gap_y


@dataclass
class ScriptedTeacherState:
    escape_dir: int = 1
    escape_steps: int = 0

    def reset(self) -> None:
        self.escape_dir = 1
        self.escape_steps = 0


def scripted_teacher_action(env, state: ScriptedTeacherState) -> str:
    if env.stuck_flag:
        if state.escape_steps <= 0:
            state.escape_steps = 6
            state.escape_dir *= -1
        state.escape_steps -= 1
        if state.escape_steps % 2 == 1:
            return "L45" if state.escape_dir > 0 else "R45"
        return "FW"

    push_heading = _choose_push_heading_scalar(env)
    if env.enable_push:
        desired = push_heading
    else:
        tx, ty = _stage_target_scalar(env, push_heading)
        tx, ty = _route_target_via_gap_scalar(env, tx, ty)
        stage_dist = math.hypot(tx - float(env.bot_center_x), ty - float(env.bot_center_y))
        desired = push_heading if stage_dist < 18.0 else math.degrees(
            math.atan2(ty - float(env.bot_center_y), tx - float(env.bot_center_x))
        )

    diff = ((desired - float(env.facing_angle) + 180.0) % 360.0) - 180.0
    if diff > TURN45_DEG:
        return "L45"
    if diff > TURN22_DEG:
        return "L22"
    if diff < -TURN45_DEG:
        return "R45"
    if diff < -TURN22_DEG:
        return "R22"
    return "FW"
