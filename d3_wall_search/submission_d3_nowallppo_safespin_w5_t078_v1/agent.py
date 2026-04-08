from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
OBS_DIM = 18
ACTION_DIM = len(ACTIONS)
FW_ACTION_INDEX = ACTION_TO_INDEX["FW"]
TURN_DELTAS = torch.tensor([45.0, 22.5, 0.0, -22.5, -45.0], dtype=torch.float32)
FORWARD_STEP = 5.0
DERIVED_OBS_DIM = 16


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def apply_forward_bias(layer: nn.Linear, fw_bias_init: float) -> None:
    if fw_bias_init == 0.0:
        return
    with torch.no_grad():
        layer.bias[FW_ACTION_INDEX] += float(fw_bias_init)


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def wrap_angle_deg(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + 180.0, 360.0) - 180.0


class RecurrentFeatureConfig:
    def __init__(
        self,
        max_steps: int = 500,
        pose_clip: float = 500.0,
        blind_clip: float = 120.0,
        stuck_clip: float = 24.0,
        contact_clip: float = 24.0,
        same_obs_clip: float = 64.0,
        wall_hit_clip: float = 24.0,
        blind_turn_clip: float = 24.0,
        stuck_memory_clip: float = 24.0,
        turn_streak_clip: float = 24.0,
        forward_streak_clip: float = 24.0,
        last_action_hist: int = 6,
        heading_bins: int = 8,
        use_current_obs: bool = True,
        use_delta_obs: bool = True,
        use_derived_obs: bool = True,
        use_pose_features: bool = True,
        use_counter_features: bool = True,
        use_action_history: bool = True,
        use_same_obs_feature: bool = True,
        use_wall_hit_memory: bool = True,
        use_last_seen_features: bool = True,
        use_sensor_seen_mask: bool = True,
    ) -> None:
        self.max_steps = int(max_steps)
        self.pose_clip = float(pose_clip)
        self.blind_clip = float(blind_clip)
        self.stuck_clip = float(stuck_clip)
        self.contact_clip = float(contact_clip)
        self.same_obs_clip = float(same_obs_clip)
        self.wall_hit_clip = float(wall_hit_clip)
        self.blind_turn_clip = float(blind_turn_clip)
        self.stuck_memory_clip = float(stuck_memory_clip)
        self.turn_streak_clip = float(turn_streak_clip)
        self.forward_streak_clip = float(forward_streak_clip)
        self.last_action_hist = int(last_action_hist)
        self.heading_bins = int(heading_bins)
        self.use_current_obs = bool(use_current_obs)
        self.use_delta_obs = bool(use_delta_obs)
        self.use_derived_obs = bool(use_derived_obs)
        self.use_pose_features = bool(use_pose_features)
        self.use_counter_features = bool(use_counter_features)
        self.use_action_history = bool(use_action_history)
        self.use_same_obs_feature = bool(use_same_obs_feature)
        self.use_wall_hit_memory = bool(use_wall_hit_memory)
        self.use_last_seen_features = bool(use_last_seen_features)
        self.use_sensor_seen_mask = bool(use_sensor_seen_mask)

    @property
    def feature_dim(self) -> int:
        dim = 0
        if self.use_current_obs:
            dim += OBS_DIM
        if self.use_delta_obs:
            dim += OBS_DIM
        if self.use_derived_obs:
            dim += DERIVED_OBS_DIM
        if self.use_pose_features:
            dim += 4
        if self.use_counter_features:
            dim += 7
        if self.use_action_history:
            dim += self.last_action_hist * ACTION_DIM
        if self.use_same_obs_feature:
            dim += 1
        if self.use_wall_hit_memory:
            dim += self.heading_bins
        if self.use_last_seen_features:
            dim += 2
        if self.use_sensor_seen_mask:
            dim += 17
        return dim

    def enabled_feature_groups(self) -> tuple[str, ...]:
        groups: list[str] = []
        if self.use_current_obs:
            groups.append("obs")
        if self.use_delta_obs:
            groups.append("delta")
        if self.use_derived_obs:
            groups.append("derived")
        if self.use_pose_features:
            groups.append("pose")
        if self.use_counter_features:
            groups.append("counters")
        if self.use_action_history:
            groups.append("action_hist")
        if self.use_same_obs_feature:
            groups.append("same_obs")
        if self.use_wall_hit_memory:
            groups.append("wall_hits")
        if self.use_last_seen_features:
            groups.append("last_seen")
        if self.use_sensor_seen_mask:
            groups.append("sensor_seen")
        return tuple(groups)


class PoseMemoryTracker:
    def __init__(self, num_envs: int, config: RecurrentFeatureConfig, device: torch.device) -> None:
        self.num_envs = int(num_envs)
        self.config = config
        self.device = device
        self.turn_deltas = TURN_DELTAS.to(device=self.device)

        self.current_obs = torch.zeros((self.num_envs, OBS_DIM), dtype=torch.float32, device=self.device)
        self.prev_obs = torch.zeros((self.num_envs, OBS_DIM), dtype=torch.float32, device=self.device)
        self.last_action_hist = torch.zeros(
            (self.num_envs, self.config.last_action_hist, ACTION_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self.x_rel = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.y_rel = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.theta_deg = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.blind_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.stuck_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.contact_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.same_obs_count = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.wall_hit_count_by_heading_bin = torch.zeros(
            (self.num_envs, self.config.heading_bins),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_seen_side = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.last_seen_front_strength = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.sensor_seen = torch.zeros((self.num_envs, 17), dtype=torch.float32, device=self.device)
        self.blind_turn_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.stuck_memory = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.turn_streak = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.forward_streak = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

    def _obs_to_tensor(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _heading_bins_for(self, theta_deg: torch.Tensor) -> torch.Tensor:
        width = 360.0 / float(self.config.heading_bins)
        shifted = torch.remainder(theta_deg + 180.0, 360.0)
        bins = torch.floor(shifted / width).to(torch.long)
        return torch.clamp(bins, 0, self.config.heading_bins - 1)

    def _front_strength(self, obs_t: torch.Tensor) -> torch.Tensor:
        front_far = torch.sum(obs_t[:, 4:12:2], dim=1)
        front_near = torch.sum(obs_t[:, 5:12:2], dim=1)
        strength = front_far + 2.0 * front_near
        return torch.clamp(strength / 12.0, 0.0, 1.0)

    def _derived_obs(self) -> torch.Tensor:
        obs = self.current_obs
        left = torch.sum(obs[:, 0:4], dim=1)
        front = torch.sum(obs[:, 4:12], dim=1)
        right = torch.sum(obs[:, 12:16], dim=1)
        front_far = torch.sum(obs[:, 4:12:2], dim=1)
        front_near = torch.sum(obs[:, 5:12:2], dim=1)
        blind = (torch.sum(obs[:, :16], dim=1) == 0.0).to(torch.float32)
        ir = obs[:, 16]
        stuck = obs[:, 17]
        any_left = (left > 0.0).to(torch.float32)
        any_front = (front > 0.0).to(torch.float32)
        any_right = (right > 0.0).to(torch.float32)
        side_balance = torch.clamp((left - right) / 4.0, min=-1.0, max=1.0)
        front_bias = torch.clamp((front - 0.5 * (left + right)) / 8.0, min=-1.0, max=1.0)
        contact_like = ((ir > 0.5) | (front_near > 0.0)).to(torch.float32)
        far_bias = torch.clamp((front_far - front_near) / 4.0, min=-1.0, max=1.0)
        total_visible = torch.clamp((left + front + right) / 16.0, 0.0, 1.0)

        return torch.stack(
            [
                left / 4.0,
                front / 8.0,
                right / 4.0,
                front_far / 4.0,
                front_near / 4.0,
                any_left,
                any_front,
                any_right,
                blind,
                ir,
                stuck,
                side_balance,
                front_bias,
                contact_like,
                far_bias,
                total_visible,
            ],
            dim=1,
        )

    def _update_meta(self, env_indices: torch.Tensor, obs_t: torch.Tensor) -> None:
        if env_indices.numel() == 0:
            return

        visible = torch.any(obs_t[:, :17] > 0.5, dim=1)
        stuck = obs_t[:, 17] > 0.5
        front_near = torch.sum(obs_t[:, 5:12:2], dim=1)
        contact_like = (obs_t[:, 16] > 0.5) | (front_near > 0.0)
        left = torch.sum(obs_t[:, :4], dim=1)
        right = torch.sum(obs_t[:, 12:16], dim=1)
        front_strength = self._front_strength(obs_t)

        self.blind_steps[env_indices] = torch.where(
            visible,
            torch.zeros_like(self.blind_steps[env_indices]),
            torch.clamp(self.blind_steps[env_indices] + 1.0, max=float(self.config.blind_clip)),
        )
        self.stuck_steps[env_indices] = torch.where(
            stuck,
            torch.clamp(self.stuck_steps[env_indices] + 1.0, max=float(self.config.stuck_clip)),
            torch.zeros_like(self.stuck_steps[env_indices]),
        )
        self.contact_steps[env_indices] = torch.where(
            contact_like,
            torch.clamp(self.contact_steps[env_indices] + 1.0, max=float(self.config.contact_clip)),
            torch.zeros_like(self.contact_steps[env_indices]),
        )

        side = self.last_seen_side[env_indices]
        side = torch.where((left > right) & (left > 0.0), torch.full_like(side, -1.0), side)
        side = torch.where((right > left) & (right > 0.0), torch.full_like(side, 1.0), side)
        side = torch.where(front_strength > 0.0, torch.zeros_like(side), side)
        self.last_seen_side[env_indices] = side

        self.last_seen_front_strength[env_indices] = torch.where(
            front_strength > 0.0,
            front_strength,
            torch.clamp(self.last_seen_front_strength[env_indices] - 0.05, min=0.0),
        )
        self.sensor_seen[env_indices] = torch.maximum(
            self.sensor_seen[env_indices],
            (obs_t[:, :17] > 0.5).to(torch.float32),
        )
        self.stuck_memory[env_indices] = torch.where(
            stuck,
            torch.clamp(self.stuck_memory[env_indices] + 1.0, max=float(self.config.stuck_memory_clip)),
            torch.clamp(self.stuck_memory[env_indices] - 1.0, min=0.0),
        )

    def reset_all(self, obs: np.ndarray | torch.Tensor) -> None:
        obs_t = self._obs_to_tensor(obs)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        self.current_obs.zero_()
        self.prev_obs.zero_()
        self.last_action_hist.zero_()
        self.x_rel.zero_()
        self.y_rel.zero_()
        self.theta_deg.zero_()
        self.blind_steps.zero_()
        self.stuck_steps.zero_()
        self.contact_steps.zero_()
        self.same_obs_count.zero_()
        self.wall_hit_count_by_heading_bin.zero_()
        self.last_seen_side.zero_()
        self.last_seen_front_strength.zero_()
        self.sensor_seen.zero_()
        self.blind_turn_steps.zero_()
        self.stuck_memory.zero_()
        self.turn_streak.zero_()
        self.forward_streak.zero_()

        self.current_obs.copy_(obs_t)
        self.prev_obs.copy_(obs_t)
        all_idx = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._update_meta(all_idx, obs_t)

    def reset_indices(self, env_indices: torch.Tensor, obs: np.ndarray | torch.Tensor) -> None:
        if env_indices.numel() == 0:
            return
        obs_t = self._obs_to_tensor(obs)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        self.current_obs[env_indices] = 0.0
        self.prev_obs[env_indices] = 0.0
        self.last_action_hist[env_indices] = 0.0
        self.x_rel[env_indices] = 0.0
        self.y_rel[env_indices] = 0.0
        self.theta_deg[env_indices] = 0.0
        self.blind_steps[env_indices] = 0.0
        self.stuck_steps[env_indices] = 0.0
        self.contact_steps[env_indices] = 0.0
        self.same_obs_count[env_indices] = 0.0
        self.wall_hit_count_by_heading_bin[env_indices] = 0.0
        self.last_seen_side[env_indices] = 0.0
        self.last_seen_front_strength[env_indices] = 0.0
        self.sensor_seen[env_indices] = 0.0
        self.blind_turn_steps[env_indices] = 0.0
        self.stuck_memory[env_indices] = 0.0
        self.turn_streak[env_indices] = 0.0
        self.forward_streak[env_indices] = 0.0

        self.current_obs[env_indices] = obs_t
        self.prev_obs[env_indices] = obs_t
        self._update_meta(env_indices, obs_t)

    def features(self) -> torch.Tensor:
        delta_obs = self.current_obs - self.prev_obs
        pose = torch.stack(
            [
                torch.clamp(self.x_rel / max(1.0, float(self.config.pose_clip)), min=-1.0, max=1.0),
                torch.clamp(self.y_rel / max(1.0, float(self.config.pose_clip)), min=-1.0, max=1.0),
                torch.sin(torch.deg2rad(self.theta_deg)),
                torch.cos(torch.deg2rad(self.theta_deg)),
            ],
            dim=1,
        )
        counters = torch.stack(
            [
                torch.clamp(self.blind_steps / max(1.0, float(self.config.blind_clip)), 0.0, 1.0),
                torch.clamp(self.stuck_steps / max(1.0, float(self.config.stuck_clip)), 0.0, 1.0),
                torch.clamp(self.contact_steps / max(1.0, float(self.config.contact_clip)), 0.0, 1.0),
                torch.clamp(
                    self.blind_turn_steps / max(1.0, float(self.config.blind_turn_clip)),
                    0.0,
                    1.0,
                ),
                torch.clamp(
                    self.stuck_memory / max(1.0, float(self.config.stuck_memory_clip)),
                    0.0,
                    1.0,
                ),
                torch.clamp(
                    self.turn_streak / max(1.0, float(self.config.turn_streak_clip)),
                    0.0,
                    1.0,
                ),
                torch.clamp(
                    self.forward_streak / max(1.0, float(self.config.forward_streak_clip)),
                    0.0,
                    1.0,
                ),
            ],
            dim=1,
        )
        last_actions = self.last_action_hist.reshape(self.num_envs, -1)
        same_obs = torch.clamp(
            self.same_obs_count / max(1.0, float(self.config.same_obs_clip)),
            0.0,
            1.0,
        ).unsqueeze(1)
        wall_hits = torch.clamp(
            self.wall_hit_count_by_heading_bin / max(1.0, float(self.config.wall_hit_clip)),
            0.0,
            1.0,
        )
        last_seen = torch.stack(
            [
                self.last_seen_side,
                torch.clamp(self.last_seen_front_strength, 0.0, 1.0),
            ],
            dim=1,
        )

        parts: list[torch.Tensor] = []
        if self.config.use_current_obs:
            parts.append(self.current_obs)
        if self.config.use_delta_obs:
            parts.append(delta_obs)
        if self.config.use_derived_obs:
            parts.append(self._derived_obs())
        if self.config.use_pose_features:
            parts.append(pose)
        if self.config.use_counter_features:
            parts.append(counters)
        if self.config.use_action_history:
            parts.append(last_actions)
        if self.config.use_same_obs_feature:
            parts.append(same_obs)
        if self.config.use_wall_hit_memory:
            parts.append(wall_hits)
        if self.config.use_last_seen_features:
            parts.append(last_seen)
        if self.config.use_sensor_seen_mask:
            parts.append(self.sensor_seen)
        if not parts:
            raise RuntimeError("At least one actor feature group must be enabled")
        return torch.cat(parts, dim=1)

    def post_step(
        self,
        actions: torch.Tensor,
        next_obs: np.ndarray | torch.Tensor,
        dones: np.ndarray | torch.Tensor,
    ) -> None:
        next_obs_t = self._obs_to_tensor(next_obs)
        if next_obs_t.ndim == 1:
            next_obs_t = next_obs_t.unsqueeze(0)
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        actions_t = actions.to(device=self.device, dtype=torch.long)

        active_mask = ~done_t
        if bool(torch.any(active_mask)):
            active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
            prev_obs = self.current_obs[active_idx].clone()
            obs_after = next_obs_t[active_idx]
            act = actions_t[active_idx]

            rolled_actions = torch.roll(self.last_action_hist[active_idx], shifts=-1, dims=1)
            rolled_actions[:, -1] = 0.0
            rolled_actions[:, -1, :] = F.one_hot(act, num_classes=ACTION_DIM).to(torch.float32)
            self.last_action_hist[active_idx] = rolled_actions

            theta_prev = self.theta_deg[active_idx]
            theta_next = wrap_angle_deg(theta_prev + self.turn_deltas[act])
            self.theta_deg[active_idx] = theta_next

            turning = act != FW_ACTION_INDEX
            self.turn_streak[active_idx] = torch.where(
                turning,
                torch.clamp(
                    self.turn_streak[active_idx] + 1.0,
                    max=float(self.config.turn_streak_clip),
                ),
                torch.zeros_like(self.turn_streak[active_idx]),
            )

            fw_mask = act == FW_ACTION_INDEX
            self.forward_streak[active_idx] = torch.where(
                fw_mask,
                torch.clamp(
                    self.forward_streak[active_idx] + 1.0,
                    max=float(self.config.forward_streak_clip),
                ),
                torch.zeros_like(self.forward_streak[active_idx]),
            )
            if bool(torch.any(fw_mask)):
                fw_idx = active_idx[fw_mask]
                fw_theta = theta_next[fw_mask]
                fw_success = obs_after[fw_mask, 17] <= 0.5
                if bool(torch.any(fw_success)):
                    move_idx = fw_idx[fw_success]
                    move_theta = fw_theta[fw_success]
                    rad = torch.deg2rad(move_theta)
                    self.x_rel[move_idx] += float(FORWARD_STEP) * torch.cos(rad)
                    self.y_rel[move_idx] += float(FORWARD_STEP) * torch.sin(rad)
                if bool(torch.any(~fw_success)):
                    stuck_idx = fw_idx[~fw_success]
                    hit_bins = self._heading_bins_for(fw_theta[~fw_success])
                    self.wall_hit_count_by_heading_bin[stuck_idx, hit_bins] = torch.clamp(
                        self.wall_hit_count_by_heading_bin[stuck_idx, hit_bins] + 1.0,
                        max=float(self.config.wall_hit_clip),
                    )

            blind = ~torch.any(obs_after[:, :16] > 0.5, dim=1)
            self.blind_turn_steps[active_idx] = torch.where(
                blind & turning,
                torch.clamp(
                    self.blind_turn_steps[active_idx] + 1.0,
                    max=float(self.config.blind_turn_clip),
                ),
                torch.zeros_like(self.blind_turn_steps[active_idx]),
            )

            same_obs = torch.all((prev_obs[:, :17] > 0.5) == (obs_after[:, :17] > 0.5), dim=1)
            self.same_obs_count[active_idx] = torch.where(
                same_obs,
                torch.clamp(
                    self.same_obs_count[active_idx] + 1.0,
                    max=float(self.config.same_obs_clip),
                ),
                torch.zeros_like(self.same_obs_count[active_idx]),
            )

            self.prev_obs[active_idx] = prev_obs
            self.current_obs[active_idx] = obs_after
            self._update_meta(active_idx, obs_after)

        if bool(torch.any(done_t)):
            done_idx = torch.nonzero(done_t, as_tuple=False).squeeze(1)
            self.reset_indices(done_idx, next_obs_t[done_idx])


RecurrentState = tuple[torch.Tensor, Optional[torch.Tensor]]


class RecurrentActor(nn.Module):
    def __init__(
        self,
        actor_dim: int,
        privileged_dim: int,
        encoder_dims: tuple[int, ...] = (256, 128),
        rnn_hidden_dim: int = 192,
        critic_hidden_dims: tuple[int, ...] = (768, 384, 192),
        rnn_layers: int = 1,
        rnn_dropout: float = 0.0,
        actor_dropout: float = 0.0,
        critic_dropout: float = 0.0,
        feature_dropout: float = 0.0,
        aux_target_dim: int = 0,
        aux_hidden_dim: int = 0,
        fw_bias_init: float = 0.0,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        self.actor_dim = int(actor_dim)
        self.privileged_dim = int(privileged_dim)
        self.encoder_dims = tuple(int(x) for x in encoder_dims)
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.critic_hidden_dims = tuple(int(x) for x in critic_hidden_dims)
        self.rnn_layers = int(rnn_layers)
        self.rnn_type = str(rnn_type).lower()
        if self.rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be one of {'gru', 'lstm'}")
        self.feature_dropout = float(feature_dropout)
        self.actor_dropout = float(actor_dropout)
        self.critic_dropout = float(critic_dropout)
        self.aux_target_dim = int(aux_target_dim)
        self.aux_hidden_dim = int(aux_hidden_dim)

        actor_layers: list[nn.Module] = []
        last_dim = self.actor_dim
        for hidden_dim in self.encoder_dims:
            h = int(hidden_dim)
            actor_layers.append(layer_init(nn.Linear(last_dim, h)))
            actor_layers.append(nn.Tanh())
            if self.actor_dropout > 0.0:
                actor_layers.append(nn.Dropout(self.actor_dropout))
            last_dim = h
        self.actor_input_dropout = nn.Dropout(self.feature_dropout) if self.feature_dropout > 0.0 else nn.Identity()
        self.actor_encoder = nn.Sequential(*actor_layers)
        if self.rnn_type == "lstm":
            self.actor_rnn: nn.Module = nn.LSTM(
                input_size=last_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=self.rnn_layers,
                dropout=float(rnn_dropout) if self.rnn_layers > 1 else 0.0,
            )
        else:
            self.actor_rnn = nn.GRU(
                input_size=last_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=self.rnn_layers,
                dropout=float(rnn_dropout) if self.rnn_layers > 1 else 0.0,
            )
        self.policy_head = layer_init(nn.Linear(self.rnn_hidden_dim, ACTION_DIM), std=0.01)
        apply_forward_bias(self.policy_head, fw_bias_init)

        critic_layers: list[nn.Module] = []
        critic_in = self.rnn_hidden_dim + self.privileged_dim
        for hidden_dim in self.critic_hidden_dims:
            h = int(hidden_dim)
            critic_layers.append(layer_init(nn.Linear(critic_in, h)))
            critic_layers.append(nn.Tanh())
            if self.critic_dropout > 0.0:
                critic_layers.append(nn.Dropout(self.critic_dropout))
            critic_in = h
        self.critic_backbone = nn.Sequential(*critic_layers)
        self.value_head = layer_init(nn.Linear(critic_in, 1), std=1.0)
        self.aux_head: nn.Module | None = None
        if self.aux_target_dim > 0:
            aux_layers: list[nn.Module] = []
            aux_in = self.rnn_hidden_dim
            if self.aux_hidden_dim > 0:
                aux_layers.append(layer_init(nn.Linear(aux_in, self.aux_hidden_dim)))
                aux_layers.append(nn.Tanh())
                if self.actor_dropout > 0.0:
                    aux_layers.append(nn.Dropout(self.actor_dropout))
                aux_in = self.aux_hidden_dim
            aux_layers.append(layer_init(nn.Linear(aux_in, self.aux_target_dim), std=0.01))
            self.aux_head = nn.Sequential(*aux_layers)

    def initial_state(self, batch_size: int, device: torch.device) -> RecurrentState:
        h = torch.zeros((self.rnn_layers, int(batch_size), self.rnn_hidden_dim), dtype=torch.float32, device=device)
        c = None
        if self.rnn_type == "lstm":
            c = torch.zeros_like(h)
        return h, c

    def _mask_state(self, state: RecurrentState, starts: torch.Tensor | None) -> RecurrentState:
        if starts is None:
            return state
        mask = 1.0 - starts.to(dtype=torch.float32, device=state[0].device).view(1, -1, 1)
        h = state[0] * mask
        c = state[1] * mask if state[1] is not None else None
        return h, c

    def _rnn_forward(self, encoded_seq: torch.Tensor, state: RecurrentState) -> tuple[torch.Tensor, RecurrentState]:
        if self.rnn_type == "lstm":
            out, (h, c) = self.actor_rnn(encoded_seq, (state[0], state[1]))
            return out, (h, c)
        out, h = self.actor_rnn(encoded_seq, state[0])
        return out, (h, None)

    def _value_from_state(self, actor_state: torch.Tensor, privileged_obs: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat([actor_state, privileged_obs], dim=1)
        critic_feat = self.critic_backbone(critic_input)
        return self.value_head(critic_feat).squeeze(-1)

    def predict_aux(self, actor_state: torch.Tensor) -> torch.Tensor | None:
        if self.aux_head is None:
            return None
        return self.aux_head(actor_state)

    def forward_step(
        self,
        actor_obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        state: RecurrentState,
        starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RecurrentState, torch.Tensor, torch.Tensor | None]:
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        state = self._mask_state(state, starts)
        encoded = self.actor_encoder(self.actor_input_dropout(actor_obs))
        out, next_state = self._rnn_forward(encoded.unsqueeze(0), state)
        actor_state = out.squeeze(0)
        logits = self.policy_head(actor_state)
        value = self._value_from_state(actor_state, privileged_obs)
        aux_pred = self.predict_aux(actor_state)
        return logits, value, next_state, actor_state, aux_pred

    def actor_step(
        self,
        actor_obs: torch.Tensor,
        state: RecurrentState,
        starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        state = self._mask_state(state, starts)
        encoded = self.actor_encoder(self.actor_input_dropout(actor_obs))
        out, next_state = self._rnn_forward(encoded.unsqueeze(0), state)
        logits = self.policy_head(out.squeeze(0))
        return logits, next_state

    def act(
        self,
        actor_obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        state: RecurrentState,
        starts: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, RecurrentState]:
        logits, value, next_state, _, _ = self.forward_step(actor_obs, privileged_obs, state, starts)
        temp = max(1e-4, float(temperature))
        dist = Categorical(logits=logits / temp)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, logits, next_state

    def evaluate_sequence(
        self,
        actor_obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        actions: torch.Tensor,
        state0: RecurrentState,
        starts: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, RecurrentState]:
        seq_len, _, _ = actor_obs.shape
        state = state0
        logits_list = []
        values_list = []
        log_prob_list = []
        entropy_list = []
        aux_pred_list = [] if self.aux_head is not None else None

        for t in range(seq_len):
            logits_t, value_t, state, _, aux_pred_t = self.forward_step(
                actor_obs[t],
                privileged_obs[t],
                state,
                starts[t],
            )
            temp = max(1e-4, float(temperature))
            dist_t = Categorical(logits=logits_t / temp)
            action_t = actions[t]
            logits_list.append(logits_t)
            values_list.append(value_t)
            log_prob_list.append(dist_t.log_prob(action_t))
            entropy_list.append(dist_t.entropy())
            if aux_pred_list is not None and aux_pred_t is not None:
                aux_pred_list.append(aux_pred_t)

        logits = torch.stack(logits_list, dim=0)
        values = torch.stack(values_list, dim=0)
        log_probs = torch.stack(log_prob_list, dim=0)
        entropy = torch.stack(entropy_list, dim=0)
        aux_preds = torch.stack(aux_pred_list, dim=0) if aux_pred_list is not None else None
        return log_probs, entropy, values, logits, aux_preds, state



RAW_OBS_DIM = 18
ENCODED_OBS_DIM = 32


def encode_obs_tensor(obs: torch.Tensor) -> torch.Tensor:
    raw = obs.to(dtype=torch.float32)
    left = raw[..., 0:4]
    front = raw[..., 4:12]
    right = raw[..., 12:16]
    ir = raw[..., 16:17]
    stuck = raw[..., 17:18]
    left_count = left.sum(dim=-1, keepdim=True)
    front_count = front.sum(dim=-1, keepdim=True)
    right_count = right.sum(dim=-1, keepdim=True)
    front_far_count = front[..., ::2].sum(dim=-1, keepdim=True)
    front_near_count = front[..., 1::2].sum(dim=-1, keepdim=True)
    side_mean_count = 0.5 * (left_count + right_count)
    blind = (raw[..., :16].sum(dim=-1, keepdim=True) == 0.0).to(dtype=torch.float32)
    derived = torch.cat([
        (left_count > 0.0).to(dtype=torch.float32),
        (front_count > 0.0).to(dtype=torch.float32),
        (right_count > 0.0).to(dtype=torch.float32),
        ir,
        stuck,
        blind,
        left_count,
        front_count,
        right_count,
        front_far_count,
        front_near_count,
        left_count - right_count,
        right_count - left_count,
        front_count - side_mean_count,
    ], dim=-1)
    return torch.cat([raw, derived], dim=-1)


class PPOActorCritic(nn.Module):
    def __init__(self, hidden_dims: tuple[int, ...], use_rec_encoder: bool) -> None:
        super().__init__()
        self.use_rec_encoder = bool(use_rec_encoder)
        last_dim = ENCODED_OBS_DIM if self.use_rec_encoder else RAW_OBS_DIM
        layers: list[nn.Module] = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.Tanh())
            last_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, ACTION_DIM)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = encode_obs_tensor(obs) if self.use_rec_encoder else obs
        return self.policy_head(self.backbone(x))


class PPOPolicy:
    def __init__(self, checkpoint: dict) -> None:
        state_dict = checkpoint['state_dict']
        hidden_dims = tuple(int(x) for x in checkpoint.get('hidden_dims', (128, 64)))
        use_rec_encoder = bool(checkpoint.get('use_rec_encoder', True))
        self.model = PPOActorCritic(hidden_dims=hidden_dims, use_rec_encoder=use_rec_encoder)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> str:
        del rng
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None, :], dtype=torch.float32)
        logits = self.model(obs_t)
        return ACTIONS[int(torch.argmax(logits, dim=1).item())]


class RecurrentPolicy:
    def __init__(self, checkpoint: dict) -> None:
        feature_config = RecurrentFeatureConfig(**checkpoint.get('feature_config', {}))
        self.model = RecurrentActor(
            actor_dim=feature_config.feature_dim,
            privileged_dim=int(checkpoint.get('privileged_dim', 34)),
            encoder_dims=tuple(int(x) for x in checkpoint.get('encoder_dims', [256, 128])),
            rnn_hidden_dim=int(checkpoint.get('rnn_hidden_dim', 192)),
            critic_hidden_dims=tuple(int(x) for x in checkpoint.get('critic_hidden_dims', [512, 256])),
            rnn_layers=int(checkpoint.get('rnn_layers', 1)),
            rnn_dropout=float(checkpoint.get('rnn_dropout', 0.0)),
            actor_dropout=float(checkpoint.get('actor_dropout', 0.0)),
            critic_dropout=float(checkpoint.get('critic_dropout', 0.0)),
            feature_dropout=float(checkpoint.get('feature_dropout', 0.0)),
            aux_target_dim=int(checkpoint.get('aux_target_dim', 0)),
            aux_hidden_dim=int(checkpoint.get('aux_hidden_dim', 0)),
            fw_bias_init=0.0,
            rnn_type=str(checkpoint.get('rnn_type', 'gru')),
        )
        self.model.actor_encoder.load_state_dict(checkpoint['actor_encoder_state_dict'], strict=True)
        self.model.actor_rnn.load_state_dict(checkpoint['actor_rnn_state_dict'], strict=True)
        self.model.policy_head.load_state_dict(checkpoint['policy_head_state_dict'], strict=True)
        self.model.eval()
        self.tracker = PoseMemoryTracker(num_envs=1, config=feature_config, device=torch.device('cpu'))
        self.state = self.model.initial_state(1, torch.device('cpu'))
        self.last_rng_id: Optional[int] = None
        self.pending_action: Optional[int] = None

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> str:
        obs_arr = np.asarray(obs, dtype=np.float32)
        rng_id = id(rng)
        if self.last_rng_id != rng_id:
            self.last_rng_id = rng_id
            self.pending_action = None
            self.tracker.reset_all(obs_arr[None, :])
            self.state = self.model.initial_state(1, torch.device('cpu'))
            starts = torch.ones((1,), dtype=torch.float32)
        else:
            if self.pending_action is not None:
                self.tracker.post_step(
                    actions=torch.tensor([self.pending_action], dtype=torch.long),
                    next_obs=obs_arr[None, :],
                    dones=np.asarray([False]),
                )
            starts = torch.zeros((1,), dtype=torch.float32)
        logits, self.state = self.model.actor_step(self.tracker.features(), self.state, starts)
        action_idx = int(torch.argmax(logits, dim=1).item())
        self.pending_action = action_idx
        return ACTIONS[action_idx]


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.Tanh())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_WALL: Optional[RecurrentPolicy] = None
_NOWALL: Optional[PPOPolicy] = None
_NOWALL_V1: Optional[PPOPolicy] = None
_ROUTER: Optional[RouterMLP] = None
_ROUTER_THRESHOLD = 0.9
_PROBE_STEPS = 20
_LONG_PROBE_TARGET = 75

_LAST_RNG_ID: Optional[int] = None
_STEP = 0
_MODE: Optional[str] = None
_FIRST_OBS: Optional[np.ndarray] = None
_STATS = np.zeros((12,), dtype=np.float32)
_PROBE_REWARD_PROXY = 0.0
_SPIN_COUNT = 0
_SPIN_SEEN = 0
_SPIN_LAST_SUM = 0.0


def _weights_path() -> str:
    return os.path.join(os.path.dirname(__file__), 'weights.pth')


def _load_once() -> None:
    global _WALL, _NOWALL, _NOWALL_V1, _ROUTER, _ROUTER_THRESHOLD, _PROBE_STEPS
    if _WALL is not None and _NOWALL is not None and _NOWALL_V1 is not None and _ROUTER is not None:
        return
    bundle = torch.load(_weights_path(), map_location='cpu')
    _WALL = RecurrentPolicy(bundle['wall'])
    _NOWALL = PPOPolicy(bundle['nowall'])
    _NOWALL_V1 = PPOPolicy(bundle['nowall_v1'])
    router_ckpt = bundle['router']
    _ROUTER = RouterMLP(
        input_dim=int(router_ckpt['input_dim']),
        hidden_dims=tuple(int(x) for x in router_ckpt['hidden_dims']),
    )
    router_state = router_ckpt['state_dict']
    if router_state and not next(iter(router_state)).startswith('net.'):
        router_state = {f'net.{key}': value for key, value in router_state.items()}
    _ROUTER.load_state_dict(router_state, strict=True)
    _ROUTER.eval()
    _ROUTER_THRESHOLD = float(router_ckpt.get('threshold', 0.9))
    _PROBE_STEPS = int(router_ckpt.get('probe_steps', 20))


def _reset(obs_arr: np.ndarray, rng: np.random.Generator) -> None:
    global _LAST_RNG_ID, _STEP, _MODE, _FIRST_OBS, _STATS, _PROBE_REWARD_PROXY, _SPIN_COUNT, _SPIN_SEEN, _SPIN_LAST_SUM
    _LAST_RNG_ID = id(rng)
    _STEP = 0
    _MODE = None
    _FIRST_OBS = obs_arr.astype(np.float32, copy=True)
    _STATS = np.zeros((12,), dtype=np.float32)
    _PROBE_REWARD_PROXY = 0.0
    _SPIN_COUNT = 0
    _SPIN_SEEN = 0
    _SPIN_LAST_SUM = 0.0


def _obs_stats(obs: np.ndarray) -> np.ndarray:
    left = float(np.sum(obs[:4]))
    front = float(np.sum(obs[4:12]))
    right = float(np.sum(obs[12:16]))
    return np.asarray([
        left,
        front,
        right,
        float(np.sum(obs[:16]) == 0.0),
        float(obs[16]),
        float(obs[17]),
        float(left > 0.0),
        float(front > 0.0),
        float(right > 0.0),
        float(np.sum(obs[5:12:2])),
        float(np.sum(obs[4:12:2])),
        float(np.sum(obs[:16])),
    ], dtype=np.float32)


def _router_feature(obs_arr: np.ndarray) -> torch.Tensor:
    x = np.concatenate([
        np.asarray(_FIRST_OBS, dtype=np.float32),
        obs_arr.astype(np.float32, copy=False),
        (_STATS / float(max(1, _PROBE_STEPS))).astype(np.float32, copy=False),
        np.asarray([_PROBE_REWARD_PROXY / 1000.0], dtype=np.float32),
    ])
    return torch.as_tensor(x[None, :], dtype=torch.float32)


@torch.no_grad()
def _p_wall(obs_arr: np.ndarray) -> float:
    logits = _ROUTER(_router_feature(obs_arr))
    return float(torch.softmax(logits, dim=1)[0, 1].item())


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _STEP, _MODE, _STATS, _PROBE_REWARD_PROXY, _SPIN_COUNT, _SPIN_SEEN, _SPIN_LAST_SUM
    _load_once()
    obs_arr = np.asarray(obs, dtype=np.float32)
    if _LAST_RNG_ID != id(rng):
        _reset(obs_arr, rng)

    if _MODE == 'wall':
        _STEP += 1
        return 'L22'
    if _MODE == 'nowall':
        _STEP += 1
        return _NOWALL.act(obs_arr, rng)
    if _MODE == 'nowall_v1':
        _STEP += 1
        return _NOWALL_V1.act(obs_arr, rng)

    if _MODE == 'spin_probe':
        _SPIN_LAST_SUM = float(np.sum(obs_arr[:16]))
        _SPIN_SEEN += int(np.sum(obs_arr[:16]) > 0.0)
        if _SPIN_COUNT >= 20:
            _MODE = 'wall' if _SPIN_LAST_SUM > 0.0 else 'nowall_v1'
            _STEP += 1
            return 'L22' if _MODE == 'wall' else _NOWALL_V1.act(obs_arr, rng)
        _SPIN_COUNT += 1
        _STEP += 1
        return 'L45'

    if _MODE == 'nowall_spin':
        if _SPIN_COUNT >= 20:
            _MODE = 'nowall'
            _STEP += 1
            return _NOWALL.act(obs_arr, rng)
        _SPIN_COUNT += 1
        _STEP += 1
        return 'L45'

    if _MODE == 'long_probe':
        if _STEP < _LONG_PROBE_TARGET:
            _STEP += 1
            return _WALL.act(obs_arr, rng)
        _MODE = 'spin_probe'
        _SPIN_COUNT = 1
        _SPIN_SEEN = int(np.sum(obs_arr[:16]) > 0.0)
        _STEP += 1
        return 'L45'

    if _STEP < _PROBE_STEPS:
        _STATS += _obs_stats(obs_arr)
        _PROBE_REWARD_PROXY -= 1.0
        _STEP += 1
        return _WALL.act(obs_arr, rng)

    wall_prob = _p_wall(obs_arr)
    blind_probe = float(_STATS[11]) <= 0.05
    first_sum = float(np.sum(_FIRST_OBS[:16]))
    first_front = float(np.sum(_FIRST_OBS[4:12]))
    clear_no_wall_front_start = first_front >= 4.0 and wall_prob < 0.1
    conservative_wall_start = first_sum > 0.0 and not clear_no_wall_front_start
    if wall_prob >= _ROUTER_THRESHOLD or conservative_wall_start:
        _MODE = 'wall'
        action = 'L22'
    elif blind_probe:
        _MODE = 'long_probe'
        action = _WALL.act(obs_arr, rng)
    else:
        _MODE = 'nowall_spin'
        _SPIN_COUNT = 1
        action = 'L45'
    _STEP += 1
    return action
