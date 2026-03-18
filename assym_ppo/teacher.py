from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rnn_model import ACTIONS


ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}


def angle_diff_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a - b + 180.0, 360.0) - 180.0


def wall_geometry(vec_env) -> tuple[float, float] | None:
    if (not bool(vec_env.wall_obstacles)) or len(vec_env.obstacles) == 0:
        return None
    (x1, _), (x2, y_top_end) = vec_env.obstacles[0]
    (_, y_bottom_start), _ = vec_env.obstacles[-1]
    return 0.5 * float(x1 + x2), 0.5 * float(y_top_end + y_bottom_start)


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
        active = ~vec_env.done
        if not bool(torch.any(active)):
            return actions

        stuck = active & (vec_env.stuck_flag > 0.5)
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

        bot_x = vec_env.bot_center_x.to(dtype=torch.float32)
        bot_y = vec_env.bot_center_y.to(dtype=torch.float32)
        box_x = vec_env.box_center_x.to(dtype=torch.float32)
        box_y = vec_env.box_center_y.to(dtype=torch.float32)
        facing = vec_env.facing_angle.to(dtype=torch.float32)

        desired_heading = torch.zeros_like(facing)

        pushing = nonstuck & vec_env.enable_push
        if bool(torch.any(pushing)):
            half = float(vec_env.box_half)
            left = box_x - (10.0 + half)
            right = (float(vec_env.frame_size[1]) - 10.0 - half) - box_x
            bottom = box_y - (10.0 + half)
            top = (float(vec_env.frame_size[0]) - 10.0 - half) - box_y
            push_heading = torch.where(
                box_x < (float(vec_env.frame_size[1]) * 0.5),
                torch.full_like(facing, 180.0),
                torch.zeros_like(facing),
            )
            vertical_better = torch.minimum(bottom, top) + 30.0 < torch.minimum(left, right)
            push_heading = torch.where(
                vertical_better & (bottom < top),
                torch.full_like(push_heading, -90.0),
                push_heading,
            )
            push_heading = torch.where(
                vertical_better & (top <= bottom),
                torch.full_like(push_heading, 90.0),
                push_heading,
            )
            desired_heading[pushing] = push_heading[pushing]

        seeking = nonstuck & (~vec_env.enable_push)
        if bool(torch.any(seeking)):
            target_x = box_x.clone()
            target_y = box_y.clone()
            geom = wall_geometry(vec_env)
            if geom is not None:
                wall_x, gap_y = geom
                wall_x_t = torch.full_like(bot_x, wall_x)
                gap_y_t = torch.full_like(bot_y, gap_y)
                clearance = float(vec_env.bot_radius + vec_env.box_half + 20)

                bot_dx = bot_x - wall_x_t
                box_dx = box_x - wall_x_t
                bot_side = torch.sign(bot_dx)
                box_side = torch.sign(box_dx)
                opposite_side = (bot_side != 0.0) & (box_side != 0.0) & (bot_side != box_side)
                near_wall_misaligned = (torch.abs(bot_dx) < clearance) & (torch.abs(bot_y - gap_y_t) > 20.0)
                route_gap = seeking & (opposite_side | near_wall_misaligned)
                if bool(torch.any(route_gap)):
                    route_idx = torch.nonzero(route_gap, as_tuple=False).squeeze(1)
                    side = torch.where(bot_side[route_idx] == 0.0, torch.ones_like(bot_side[route_idx]), bot_side[route_idx])
                    pre_gap_x = wall_x_t[route_idx] + side * clearance
                    post_gap_x = wall_x_t[route_idx] - side * (clearance + 20.0)
                    aligned = (
                        torch.abs(bot_x[route_idx] - pre_gap_x) < 18.0
                    ) & (
                        torch.abs(bot_y[route_idx] - gap_y_t[route_idx]) < 18.0
                    )
                    target_x[route_idx] = torch.where(aligned, post_gap_x, pre_gap_x)
                    target_y[route_idx] = gap_y_t[route_idx]

            desired_heading[seeking] = torch.rad2deg(torch.atan2(target_y - bot_y, target_x - bot_x))[seeking]

        diff = angle_diff_deg(desired_heading, facing)
        left45 = nonstuck & (diff > 35.0)
        left22 = nonstuck & (diff > 10.0) & (~left45)
        right45 = nonstuck & (diff < -35.0)
        right22 = nonstuck & (diff < -10.0) & (~right45)

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

    if env.enable_push:
        half = max(1, int(env.box_size // 2))
        left = env.box_center_x - (10 + half)
        right = (env.frame_size[1] - 10 - half) - env.box_center_x
        bottom = env.box_center_y - (10 + half)
        top = (env.frame_size[0] - 10 - half) - env.box_center_y
        desired = 180.0 if env.box_center_x < (env.frame_size[1] * 0.5) else 0.0
        if min(bottom, top) + 30.0 < min(left, right):
            desired = -90.0 if bottom < top else 90.0
    else:
        tx = float(env.box_center_x)
        ty = float(env.box_center_y)
        geom = cpu_wall_geometry(env)
        if geom is not None:
            wall_x, gap_y = geom
            clearance = float(env.bot_radius + max(1, int(env.box_size // 2)) + 20)
            bot_side = math.copysign(1.0, env.bot_center_x - wall_x) if abs(env.bot_center_x - wall_x) > 1 else 0.0
            box_side = math.copysign(1.0, env.box_center_x - wall_x) if abs(env.box_center_x - wall_x) > 1 else 0.0
            if bot_side != 0.0 and box_side != 0.0 and bot_side != box_side:
                pre_gap_x = wall_x + bot_side * clearance
                post_gap_x = wall_x - bot_side * (clearance + 20.0)
                aligned = abs(env.bot_center_x - pre_gap_x) < 18.0 and abs(env.bot_center_y - gap_y) < 18.0
                tx = post_gap_x if aligned else pre_gap_x
                ty = gap_y
            elif abs(env.bot_center_x - wall_x) < clearance and abs(env.bot_center_y - gap_y) > 20.0:
                tx = wall_x + (bot_side if bot_side != 0.0 else 1.0) * clearance
                ty = gap_y
        desired = math.degrees(math.atan2(ty - float(env.bot_center_y), tx - float(env.bot_center_x)))

    diff = ((desired - float(env.facing_angle) + 180.0) % 360.0) - 180.0
    if diff > 35.0:
        return "L45"
    if diff > 10.0:
        return "L22"
    if diff < -35.0:
        return "R45"
    if diff < -10.0:
        return "R22"
    return "FW"
