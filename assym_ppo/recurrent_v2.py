from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

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


@dataclass
class RecurrentFeatureConfig:
    max_steps: int = 500
    pose_clip: float = 500.0
    blind_clip: float = 120.0
    stuck_clip: float = 24.0
    contact_clip: float = 24.0
    same_obs_clip: float = 64.0
    wall_hit_clip: float = 24.0
    blind_turn_clip: float = 24.0
    stuck_memory_clip: float = 24.0
    turn_streak_clip: float = 24.0
    forward_streak_clip: float = 24.0
    last_action_hist: int = 6
    heading_bins: int = 8
    use_current_obs: bool = True
    use_delta_obs: bool = True
    use_derived_obs: bool = True
    use_pose_features: bool = True
    use_counter_features: bool = True
    use_action_history: bool = True
    use_same_obs_feature: bool = True
    use_wall_hit_memory: bool = True
    use_last_seen_features: bool = True
    use_sensor_seen_mask: bool = True

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


class RecurrentAsymmetricActorCritic(nn.Module):
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


class RecurrentRolloutBuffer:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        actor_dim: int,
        privileged_dim: int,
        rnn_hidden_dim: int,
        rnn_layers: int,
        action_dim: int,
        device: torch.device,
        use_lstm: bool,
    ) -> None:
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.actor_dim = int(actor_dim)
        self.privileged_dim = int(privileged_dim)
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.rnn_layers = int(rnn_layers)
        self.action_dim = int(action_dim)
        self.device = device
        self.use_lstm = bool(use_lstm)

        self.actor_obs = torch.zeros((self.num_steps, self.num_envs, self.actor_dim), dtype=torch.float32, device=device)
        self.privileged_obs = torch.zeros(
            (self.num_steps, self.num_envs, self.privileged_dim),
            dtype=torch.float32,
            device=device,
        )
        self.starts = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.state_h = torch.zeros(
            (self.num_steps, self.rnn_layers, self.num_envs, self.rnn_hidden_dim),
            dtype=torch.float32,
            device=device,
        )
        self.state_c = None
        if self.use_lstm:
            self.state_c = torch.zeros_like(self.state_h)
        self.actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long, device=device)
        self.teacher_actions = torch.full((self.num_steps, self.num_envs), -1, dtype=torch.long, device=device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.logits = torch.zeros((self.num_steps, self.num_envs, self.action_dim), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)

    def add(
        self,
        step: int,
        *,
        actor_obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        starts: torch.Tensor,
        state: RecurrentState,
        actions: torch.Tensor,
        teacher_actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        self.actor_obs[step].copy_(actor_obs)
        self.privileged_obs[step].copy_(privileged_obs)
        self.starts[step].copy_(starts)
        self.state_h[step].copy_(state[0])
        if self.state_c is not None and state[1] is not None:
            self.state_c[step].copy_(state[1])
        self.actions[step].copy_(actions)
        self.teacher_actions[step].copy_(teacher_actions)
        self.log_probs[step].copy_(log_probs)
        self.rewards[step].copy_(torch.as_tensor(rewards, dtype=torch.float32, device=self.device))
        self.dones[step].copy_(torch.as_tensor(dones, dtype=torch.float32, device=self.device))
        self.values[step].copy_(values)
        self.logits[step].copy_(logits)

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        normalize_advantages: bool,
    ) -> None:
        last_advantage = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        for step in reversed(range(self.num_steps)):
            next_values = last_values if step == self.num_steps - 1 else self.values[step + 1]
            next_nonterminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_nonterminal - self.values[step]
            last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
            self.advantages[step] = last_advantage

        self.returns.copy_(self.advantages + self.values)
        if normalize_advantages:
            flat = self.advantages.reshape(-1)
            flat = (flat - flat.mean()) / (flat.std(unbiased=False) + 1e-8)
            self.advantages.copy_(flat.view_as(self.advantages))

    def sequence_mini_batches(self, minibatch_size: int, num_learning_epochs: int, seq_len: int):
        if self.num_steps % int(seq_len) != 0:
            raise ValueError("rollout_steps must be divisible by seq_len for recurrent minibatches")

        sequences: list[tuple[int, int]] = []
        for env_id in range(self.num_envs):
            for start in range(0, self.num_steps, int(seq_len)):
                sequences.append((start, env_id))

        seqs_per_batch = max(1, int(minibatch_size) // int(seq_len))
        total_sequences = len(sequences)

        for _ in range(int(num_learning_epochs)):
            perm = torch.randperm(total_sequences, device=self.device)
            for start_idx in range(0, total_sequences, seqs_per_batch):
                batch_perm = perm[start_idx : start_idx + seqs_per_batch].tolist()
                batch_items = [sequences[int(i)] for i in batch_perm]
                batch_size = len(batch_items)

                actor_obs = torch.zeros((seq_len, batch_size, self.actor_dim), dtype=torch.float32, device=self.device)
                privileged_obs = torch.zeros(
                    (seq_len, batch_size, self.privileged_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
                starts = torch.zeros((seq_len, batch_size), dtype=torch.float32, device=self.device)
                h0 = torch.zeros(
                    (self.rnn_layers, batch_size, self.rnn_hidden_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
                c0 = None
                if self.use_lstm:
                    c0 = torch.zeros_like(h0)
                actions = torch.zeros((seq_len, batch_size), dtype=torch.long, device=self.device)
                teacher_actions = torch.full((seq_len, batch_size), -1, dtype=torch.long, device=self.device)
                old_log_probs = torch.zeros((seq_len, batch_size), dtype=torch.float32, device=self.device)
                old_values = torch.zeros((seq_len, batch_size), dtype=torch.float32, device=self.device)
                returns = torch.zeros((seq_len, batch_size), dtype=torch.float32, device=self.device)
                advantages = torch.zeros((seq_len, batch_size), dtype=torch.float32, device=self.device)
                old_logits = torch.zeros((seq_len, batch_size, self.action_dim), dtype=torch.float32, device=self.device)

                for b, (t0, env_id) in enumerate(batch_items):
                    sl = slice(t0, t0 + seq_len)
                    actor_obs[:, b] = self.actor_obs[sl, env_id]
                    privileged_obs[:, b] = self.privileged_obs[sl, env_id]
                    starts[:, b] = self.starts[sl, env_id]
                    h0[:, b, :] = self.state_h[t0, :, env_id, :]
                    if self.use_lstm and c0 is not None and self.state_c is not None:
                        c0[:, b, :] = self.state_c[t0, :, env_id, :]
                    actions[:, b] = self.actions[sl, env_id]
                    teacher_actions[:, b] = self.teacher_actions[sl, env_id]
                    old_log_probs[:, b] = self.log_probs[sl, env_id]
                    old_values[:, b] = self.values[sl, env_id]
                    returns[:, b] = self.returns[sl, env_id]
                    advantages[:, b] = self.advantages[sl, env_id]
                    old_logits[:, b] = self.logits[sl, env_id]

                yield (
                    actor_obs,
                    privileged_obs,
                    starts,
                    (h0, c0),
                    actions,
                    teacher_actions,
                    old_log_probs,
                    old_values,
                    returns,
                    advantages,
                    old_logits,
                )


def categorical_kl(old_logits: torch.Tensor, new_logits: torch.Tensor) -> torch.Tensor:
    old_log_probs = F.log_softmax(old_logits, dim=-1)
    new_log_probs = F.log_softmax(new_logits, dim=-1)
    old_probs = old_log_probs.exp()
    return torch.sum(old_probs * (old_log_probs - new_log_probs), dim=-1)


def _unwrapped_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def make_checkpoint(
    model: nn.Module,
    *,
    feature_config: RecurrentFeatureConfig,
    encoder_dims: tuple[int, ...],
    rnn_hidden_dim: int,
    critic_hidden_dims: tuple[int, ...],
    privileged_dim: int,
    rnn_layers: int,
    rnn_dropout: float,
    actor_dropout: float,
    critic_dropout: float,
    feature_dropout: float,
    aux_target_dim: int,
    aux_hidden_dim: int,
    rnn_type: str,
    args,
    best_eval: float,
) -> dict:
    base_model = _unwrapped_model(model)
    checkpoint = {
        "full_state_dict": base_model.state_dict(),
        "actor_encoder_state_dict": base_model.actor_encoder.state_dict(),
        "actor_rnn_state_dict": base_model.actor_rnn.state_dict(),
        "policy_head_state_dict": base_model.policy_head.state_dict(),
        "actions": ACTIONS,
        "feature_config": asdict(feature_config),
        "encoder_dims": [int(h) for h in encoder_dims],
        "rnn_hidden_dim": int(rnn_hidden_dim),
        "critic_hidden_dims": [int(h) for h in critic_hidden_dims],
        "privileged_dim": int(privileged_dim),
        "rnn_layers": int(rnn_layers),
        "rnn_dropout": float(rnn_dropout),
        "actor_dropout": float(actor_dropout),
        "critic_dropout": float(critic_dropout),
        "feature_dropout": float(feature_dropout),
        "aux_target_dim": int(aux_target_dim),
        "aux_hidden_dim": int(aux_hidden_dim),
        "rnn_type": str(rnn_type),
        "best_eval": float(best_eval),
        "config": vars(args),
    }
    return checkpoint


def save_checkpoint(path: str, checkpoint: dict) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    raw = torch.load(path, map_location=device)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {path}")
    return raw
