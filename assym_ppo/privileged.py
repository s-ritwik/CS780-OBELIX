from __future__ import annotations

import torch


PRIVILEGED_DIM = 34


def privileged_obs_dim() -> int:
    return PRIVILEGED_DIM


def _wall_geometry(vec_env, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (not bool(vec_env.wall_obstacles)) or len(vec_env.obstacles) == 0:
        zeros = torch.zeros_like(ref, dtype=torch.float32)
        return zeros, zeros, zeros

    x1, _ = vec_env.obstacles[0][0]
    x2, y_top_end = vec_env.obstacles[0][1]
    _, y_bottom_start = vec_env.obstacles[-1][0]

    wall_x_center = 0.5 * float(x1 + x2)
    gap_center_y = 0.5 * float(y_top_end + y_bottom_start)
    gap_half = 0.5 * float(max(1, y_bottom_start - y_top_end))
    return (
        torch.full_like(ref, wall_x_center, dtype=torch.float32),
        torch.full_like(ref, gap_center_y, dtype=torch.float32),
        torch.full_like(ref, gap_half, dtype=torch.float32),
    )


def box_goal_distance(vec_env, *, target_device: torch.device) -> torch.Tensor:
    box_x = vec_env.box_center_x.to(dtype=torch.float32)
    box_y = vec_env.box_center_y.to(dtype=torch.float32)
    half = float(vec_env.box_half)
    arena = float(vec_env.arena_size)

    left = box_x - (10.0 + half)
    right = (arena - 10.0 - half) - box_x
    bottom = box_y - (10.0 + half)
    top = (arena - 10.0 - half) - box_y
    goal_dist = torch.min(torch.stack([left, right, bottom, top], dim=1), dim=1).values
    return goal_dist.to(device=target_device, dtype=torch.float32)


def extract_shaping_metrics(vec_env, *, target_device: torch.device) -> dict[str, torch.Tensor]:
    arena = float(vec_env.arena_size)

    bot_x = vec_env.bot_center_x.to(dtype=torch.float32)
    bot_y = vec_env.bot_center_y.to(dtype=torch.float32)
    facing_deg = vec_env.facing_angle.to(dtype=torch.float32)
    facing_rad = torch.deg2rad(facing_deg)

    box_x = vec_env.box_center_x.to(dtype=torch.float32)
    box_y = vec_env.box_center_y.to(dtype=torch.float32)
    rel_x = box_x - bot_x
    rel_y = box_y - bot_y
    bot_box_distance = torch.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-6)

    target_heading = torch.atan2(rel_y, rel_x)
    rel_heading = torch.atan2(
        torch.sin(target_heading - facing_rad),
        torch.cos(target_heading - facing_rad),
    )
    heading_alignment = torch.cos(rel_heading)

    gap_wall_x, gap_center_y, gap_half = _wall_geometry(vec_env, bot_x)
    goal_distance = box_goal_distance(vec_env, target_device=target_device)
    gap_dx = gap_wall_x - bot_x
    gap_dy = gap_center_y - bot_y
    gap_target_distance = torch.sqrt(gap_dx * gap_dx + gap_dy * gap_dy + 1e-6)
    gap_heading = torch.atan2(gap_dy, gap_dx)
    gap_rel_heading = torch.atan2(
        torch.sin(gap_heading - facing_rad),
        torch.cos(gap_heading - facing_rad),
    )
    opposite_wall_side = (((bot_x - gap_wall_x) * (box_x - gap_wall_x)) < 0.0).to(torch.float32)

    return {
        "bot_box_distance": (bot_box_distance / arena).to(device=target_device),
        "heading_alignment": heading_alignment.to(device=target_device),
        "goal_distance": (goal_distance / arena).to(device=target_device),
        "push_active": vec_env.enable_push.to(device=target_device, dtype=torch.float32),
        "box_visible": vec_env.box_visible.to(device=target_device, dtype=torch.float32),
        "stuck": vec_env.stuck_flag.to(device=target_device, dtype=torch.float32),
        "gap_wall_x": (gap_wall_x / arena).to(device=target_device),
        "gap_center_y": (gap_center_y / arena).to(device=target_device),
        "gap_half": (gap_half / arena).to(device=target_device),
        "bot_y": (bot_y / arena).to(device=target_device),
        "box_y": (box_y / arena).to(device=target_device),
        "gap_target_distance": (gap_target_distance / arena).to(device=target_device),
        "gap_alignment": torch.cos(gap_rel_heading).to(device=target_device),
        "opposite_wall_side": opposite_wall_side.to(device=target_device),
    }


def extract_privileged_obs(vec_env, *, target_device: torch.device) -> torch.Tensor:
    arena = float(vec_env.arena_size)
    max_steps = max(1.0, float(vec_env.max_steps))
    bot_radius = float(vec_env.bot_radius)

    bot_x = vec_env.bot_center_x.to(dtype=torch.float32)
    bot_y = vec_env.bot_center_y.to(dtype=torch.float32)
    facing_deg = vec_env.facing_angle.to(dtype=torch.float32)
    facing_rad = torch.deg2rad(facing_deg)

    box_x = vec_env.box_center_x.to(dtype=torch.float32)
    box_y = vec_env.box_center_y.to(dtype=torch.float32)
    rel_x = box_x - bot_x
    rel_y = box_y - bot_y
    distance = torch.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-6)

    target_heading = torch.atan2(rel_y, rel_x)
    rel_heading = torch.atan2(torch.sin(target_heading - facing_rad), torch.cos(target_heading - facing_rad))

    left_margin = (bot_x - (10.0 + bot_radius)) / arena
    right_margin = ((arena - 10.0 - bot_radius) - bot_x) / arena
    bottom_margin = (bot_y - (10.0 + bot_radius)) / arena
    top_margin = ((arena - 10.0 - bot_radius) - bot_y) / arena

    wall_flag = torch.full_like(bot_x, 1.0 if bool(vec_env.wall_obstacles) else 0.0)
    difficulty = torch.full_like(bot_x, float(vec_env.difficulty) / 3.0)
    box_speed = torch.full_like(bot_x, float(vec_env.box_speed) / 5.0)
    gap_wall_x, gap_center_y, gap_half = _wall_geometry(vec_env, bot_x)
    goal_distance = box_goal_distance(vec_env, target_device=bot_x.device)

    privileged = torch.stack(
        [
            bot_x / arena,
            bot_y / arena,
            torch.sin(facing_rad),
            torch.cos(facing_rad),
            box_x / arena,
            box_y / arena,
            rel_x / arena,
            rel_y / arena,
            distance / arena,
            torch.sin(rel_heading),
            torch.cos(rel_heading),
            vec_env.enable_push.to(dtype=torch.float32),
            vec_env.box_visible.to(dtype=torch.float32),
            vec_env.stuck_flag.to(dtype=torch.float32),
            vec_env.current_step.to(dtype=torch.float32) / max_steps,
            wall_flag,
            difficulty,
            box_speed,
            vec_env._box_vx.to(dtype=torch.float32) / max(1.0, float(vec_env.box_speed)),
            vec_env._box_vy.to(dtype=torch.float32) / max(1.0, float(vec_env.box_speed)),
            left_margin,
            right_margin,
            bottom_margin,
            top_margin,
            goal_distance / arena,
            gap_wall_x / arena,
            gap_center_y / arena,
            gap_half / arena,
            (bot_y - gap_center_y) / arena,
            (box_y - gap_center_y) / arena,
            (bot_x - gap_wall_x) / arena,
            (box_x - gap_wall_x) / arena,
            (((bot_x - gap_wall_x) * (box_x - gap_wall_x)) >= 0.0).to(torch.float32),
            torch.cos(rel_heading) * vec_env.enable_push.to(dtype=torch.float32),
        ],
        dim=1,
    )
    return privileged.to(device=target_device)
