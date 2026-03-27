from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
OBS_DIM = 18
ACTION_DIM = len(ACTIONS)
FW_ACTION_INDEX = ACTION_TO_INDEX["FW"]
TURN_DELTAS = torch.tensor([45.0, 22.5, 0.0, -22.5, -45.0], dtype=torch.float32)
FORWARD_STEP = 5.0


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def wrap_angle_deg(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + 180.0, 360.0) - 180.0


class RecurrentFeatureConfig:
    def __init__(self, **kwargs) -> None:
        self.max_steps = int(kwargs.get("max_steps", 500))
        self.pose_clip = float(kwargs.get("pose_clip", 500.0))
        self.blind_clip = float(kwargs.get("blind_clip", 120.0))
        self.stuck_clip = float(kwargs.get("stuck_clip", 24.0))
        self.contact_clip = float(kwargs.get("contact_clip", 24.0))
        self.same_obs_clip = float(kwargs.get("same_obs_clip", 64.0))
        self.wall_hit_clip = float(kwargs.get("wall_hit_clip", 24.0))
        self.blind_turn_clip = float(kwargs.get("blind_turn_clip", 24.0))
        self.stuck_memory_clip = float(kwargs.get("stuck_memory_clip", 24.0))
        self.turn_streak_clip = float(kwargs.get("turn_streak_clip", 24.0))
        self.forward_streak_clip = float(kwargs.get("forward_streak_clip", 24.0))
        self.last_action_hist = int(kwargs.get("last_action_hist", 6))
        self.heading_bins = int(kwargs.get("heading_bins", 8))
        self.use_current_obs = bool(kwargs.get("use_current_obs", True))
        self.use_delta_obs = bool(kwargs.get("use_delta_obs", True))
        self.use_derived_obs = bool(kwargs.get("use_derived_obs", True))
        self.use_pose_features = bool(kwargs.get("use_pose_features", True))
        self.use_counter_features = bool(kwargs.get("use_counter_features", True))
        self.use_action_history = bool(kwargs.get("use_action_history", True))
        self.use_same_obs_feature = bool(kwargs.get("use_same_obs_feature", True))
        self.use_wall_hit_memory = bool(kwargs.get("use_wall_hit_memory", True))
        self.use_last_seen_features = bool(kwargs.get("use_last_seen_features", True))
        self.use_sensor_seen_mask = bool(kwargs.get("use_sensor_seen_mask", True))

    @property
    def feature_dim(self) -> int:
        dim = 0
        if self.use_current_obs:
            dim += OBS_DIM
        if self.use_delta_obs:
            dim += OBS_DIM
        if self.use_derived_obs:
            dim += 16
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


class PoseMemoryTracker:
    def __init__(self, config: RecurrentFeatureConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.turn_deltas = TURN_DELTAS.to(device=device)
        self.current_obs = torch.zeros((1, OBS_DIM), dtype=torch.float32, device=device)
        self.prev_obs = torch.zeros((1, OBS_DIM), dtype=torch.float32, device=device)
        self.last_action_hist = torch.zeros((1, self.config.last_action_hist, ACTION_DIM), dtype=torch.float32, device=device)
        self.x_rel = torch.zeros((1,), dtype=torch.float32, device=device)
        self.y_rel = torch.zeros((1,), dtype=torch.float32, device=device)
        self.theta_deg = torch.zeros((1,), dtype=torch.float32, device=device)
        self.blind_steps = torch.zeros((1,), dtype=torch.float32, device=device)
        self.stuck_steps = torch.zeros((1,), dtype=torch.float32, device=device)
        self.contact_steps = torch.zeros((1,), dtype=torch.float32, device=device)
        self.same_obs_count = torch.zeros((1,), dtype=torch.float32, device=device)
        self.wall_hit_count_by_heading_bin = torch.zeros((1, self.config.heading_bins), dtype=torch.float32, device=device)
        self.last_seen_side = torch.zeros((1,), dtype=torch.float32, device=device)
        self.last_seen_front_strength = torch.zeros((1,), dtype=torch.float32, device=device)
        self.sensor_seen = torch.zeros((1, 17), dtype=torch.float32, device=device)
        self.blind_turn_steps = torch.zeros((1,), dtype=torch.float32, device=device)
        self.stuck_memory = torch.zeros((1,), dtype=torch.float32, device=device)
        self.turn_streak = torch.zeros((1,), dtype=torch.float32, device=device)
        self.forward_streak = torch.zeros((1,), dtype=torch.float32, device=device)

    def _obs_to_tensor(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.to(device=self.device, dtype=torch.float32)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        return obs_t

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

    def _update_meta(self, obs_t: torch.Tensor) -> None:
        visible = torch.any(obs_t[:, :17] > 0.5, dim=1)
        stuck = obs_t[:, 17] > 0.5
        front_near = torch.sum(obs_t[:, 5:12:2], dim=1)
        contact_like = (obs_t[:, 16] > 0.5) | (front_near > 0.0)
        left = torch.sum(obs_t[:, :4], dim=1)
        right = torch.sum(obs_t[:, 12:16], dim=1)
        front_strength = self._front_strength(obs_t)

        self.blind_steps = torch.where(
            visible,
            torch.zeros_like(self.blind_steps),
            torch.clamp(self.blind_steps + 1.0, max=float(self.config.blind_clip)),
        )
        self.stuck_steps = torch.where(
            stuck,
            torch.clamp(self.stuck_steps + 1.0, max=float(self.config.stuck_clip)),
            torch.zeros_like(self.stuck_steps),
        )
        self.contact_steps = torch.where(
            contact_like,
            torch.clamp(self.contact_steps + 1.0, max=float(self.config.contact_clip)),
            torch.zeros_like(self.contact_steps),
        )

        side = self.last_seen_side.clone()
        side = torch.where((left > right) & (left > 0.0), torch.full_like(side, -1.0), side)
        side = torch.where((right > left) & (right > 0.0), torch.full_like(side, 1.0), side)
        side = torch.where(front_strength > 0.0, torch.zeros_like(side), side)
        self.last_seen_side = side
        self.last_seen_front_strength = torch.where(
            front_strength > 0.0,
            front_strength,
            torch.clamp(self.last_seen_front_strength - 0.05, min=0.0),
        )
        self.sensor_seen = torch.maximum(self.sensor_seen, (obs_t[:, :17] > 0.5).to(torch.float32))
        self.stuck_memory = torch.where(
            stuck,
            torch.clamp(self.stuck_memory + 1.0, max=float(self.config.stuck_memory_clip)),
            torch.clamp(self.stuck_memory - 1.0, min=0.0),
        )

    def reset_all(self, obs: np.ndarray | torch.Tensor) -> None:
        obs_t = self._obs_to_tensor(obs)
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
        self._update_meta(obs_t)

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
                torch.clamp(self.blind_turn_steps / max(1.0, float(self.config.blind_turn_clip)), 0.0, 1.0),
                torch.clamp(self.stuck_memory / max(1.0, float(self.config.stuck_memory_clip)), 0.0, 1.0),
                torch.clamp(self.turn_streak / max(1.0, float(self.config.turn_streak_clip)), 0.0, 1.0),
                torch.clamp(self.forward_streak / max(1.0, float(self.config.forward_streak_clip)), 0.0, 1.0),
            ],
            dim=1,
        )
        last_actions = self.last_action_hist.reshape(1, -1)
        same_obs = torch.clamp(self.same_obs_count / max(1.0, float(self.config.same_obs_clip)), 0.0, 1.0).unsqueeze(1)
        wall_hits = torch.clamp(self.wall_hit_count_by_heading_bin / max(1.0, float(self.config.wall_hit_clip)), 0.0, 1.0)
        last_seen = torch.stack([self.last_seen_side, torch.clamp(self.last_seen_front_strength, 0.0, 1.0)], dim=1)

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
        return torch.cat(parts, dim=1)

    def post_step(self, action_idx: int, next_obs: np.ndarray | torch.Tensor) -> None:
        next_obs_t = self._obs_to_tensor(next_obs)
        prev_obs = self.current_obs.clone()
        action_t = torch.tensor([int(action_idx)], dtype=torch.long, device=self.device)

        rolled_actions = torch.roll(self.last_action_hist, shifts=-1, dims=1)
        rolled_actions[:, -1] = 0.0
        rolled_actions[:, -1, :] = torch.nn.functional.one_hot(action_t, num_classes=ACTION_DIM).to(torch.float32)
        self.last_action_hist = rolled_actions

        theta_next = wrap_angle_deg(self.theta_deg + self.turn_deltas[action_t])
        self.theta_deg = theta_next

        turning = action_t != FW_ACTION_INDEX
        self.turn_streak = torch.where(
            turning,
            torch.clamp(self.turn_streak + 1.0, max=float(self.config.turn_streak_clip)),
            torch.zeros_like(self.turn_streak),
        )
        fw_mask = action_t == FW_ACTION_INDEX
        self.forward_streak = torch.where(
            fw_mask,
            torch.clamp(self.forward_streak + 1.0, max=float(self.config.forward_streak_clip)),
            torch.zeros_like(self.forward_streak),
        )
        if bool(torch.any(fw_mask)):
            fw_success = next_obs_t[:, 17] <= 0.5
            if bool(torch.any(fw_success)):
                rad = torch.deg2rad(theta_next[fw_success])
                self.x_rel[fw_success] += float(FORWARD_STEP) * torch.cos(rad)
                self.y_rel[fw_success] += float(FORWARD_STEP) * torch.sin(rad)
            if bool(torch.any(~fw_success)):
                hit_bins = self._heading_bins_for(theta_next[~fw_success])
                self.wall_hit_count_by_heading_bin[0, hit_bins] = torch.clamp(
                    self.wall_hit_count_by_heading_bin[0, hit_bins] + 1.0,
                    max=float(self.config.wall_hit_clip),
                )

        blind = ~torch.any(next_obs_t[:, :16] > 0.5, dim=1)
        self.blind_turn_steps = torch.where(
            blind & turning,
            torch.clamp(self.blind_turn_steps + 1.0, max=float(self.config.blind_turn_clip)),
            torch.zeros_like(self.blind_turn_steps),
        )
        same_obs = torch.all((prev_obs[:, :17] > 0.5) == (next_obs_t[:, :17] > 0.5), dim=1)
        self.same_obs_count = torch.where(
            same_obs,
            torch.clamp(self.same_obs_count + 1.0, max=float(self.config.same_obs_clip)),
            torch.zeros_like(self.same_obs_count),
        )

        self.prev_obs = prev_obs
        self.current_obs = next_obs_t
        self._update_meta(next_obs_t)


class RecurrentActor(nn.Module):
    def __init__(
        self,
        actor_dim: int,
        encoder_dims: tuple[int, ...],
        rnn_hidden_dim: int,
        rnn_layers: int,
        rnn_dropout: float,
        actor_dropout: float,
        rnn_type: str,
    ) -> None:
        super().__init__()
        actor_layers: list[nn.Module] = []
        last_dim = actor_dim
        for hidden_dim in encoder_dims:
            actor_layers.append(layer_init(nn.Linear(last_dim, int(hidden_dim))))
            actor_layers.append(nn.Tanh())
            if float(actor_dropout) > 0.0:
                actor_layers.append(nn.Dropout(float(actor_dropout)))
            last_dim = int(hidden_dim)
        self.actor_encoder = nn.Sequential(*actor_layers)
        self.rnn_type = str(rnn_type).lower()
        if self.rnn_type == "lstm":
            self.actor_rnn: nn.Module = nn.LSTM(
                input_size=last_dim,
                hidden_size=int(rnn_hidden_dim),
                num_layers=int(rnn_layers),
                dropout=float(rnn_dropout) if int(rnn_layers) > 1 else 0.0,
            )
        else:
            self.actor_rnn = nn.GRU(
                input_size=last_dim,
                hidden_size=int(rnn_hidden_dim),
                num_layers=int(rnn_layers),
                dropout=float(rnn_dropout) if int(rnn_layers) > 1 else 0.0,
            )
        self.policy_head = layer_init(nn.Linear(int(rnn_hidden_dim), ACTION_DIM), std=0.01)
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.rnn_layers = int(rnn_layers)

    def initial_state(self, device: torch.device):
        h = torch.zeros((self.rnn_layers, 1, self.rnn_hidden_dim), dtype=torch.float32, device=device)
        if self.rnn_type == "lstm":
            c = torch.zeros_like(h)
            return h, c
        return h, None

    def actor_step(self, actor_obs: torch.Tensor, state, starts: torch.Tensor):
        mask = 1.0 - starts.to(dtype=torch.float32, device=actor_obs.device).view(1, -1, 1)
        h = state[0] * mask
        c = state[1] * mask if state[1] is not None else None
        encoded = self.actor_encoder(actor_obs)
        if self.rnn_type == "lstm":
            out, (h, c) = self.actor_rnn(encoded.unsqueeze(0), (h, c))
            next_state = (h, c)
        else:
            out, h = self.actor_rnn(encoded.unsqueeze(0), h)
            next_state = (h, None)
        logits = self.policy_head(out.squeeze(0))
        return logits, next_state


_MODEL: Optional[RecurrentActor] = None
_TRACKER: Optional[PoseMemoryTracker] = None
_STATE = None
_LAST_RNG_ID: Optional[int] = None
_PENDING_ACTION: Optional[int] = None


def _checkpoint_path() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(here, "weights.pth")


def _load_once() -> None:
    global _MODEL, _TRACKER, _STATE
    if _MODEL is not None and _TRACKER is not None and _STATE is not None:
        return

    checkpoint = torch.load(_checkpoint_path(), map_location="cpu")
    feature_config = RecurrentFeatureConfig(**checkpoint.get("feature_config", {}))
    encoder_dims = tuple(int(x) for x in checkpoint.get("encoder_dims", [256, 128]))
    model = RecurrentActor(
        actor_dim=feature_config.feature_dim,
        encoder_dims=encoder_dims,
        rnn_hidden_dim=int(checkpoint.get("rnn_hidden_dim", 192)),
        rnn_layers=int(checkpoint.get("rnn_layers", 1)),
        rnn_dropout=float(checkpoint.get("rnn_dropout", 0.0)),
        actor_dropout=float(checkpoint.get("actor_dropout", 0.0)),
        rnn_type=str(checkpoint.get("rnn_type", "gru")),
    )
    state_dict = checkpoint["full_state_dict"]
    model.load_state_dict(
        {
            k: v
            for k, v in state_dict.items()
            if k.startswith("actor_encoder.") or k.startswith("actor_rnn.") or k.startswith("policy_head.")
        },
        strict=True,
    )
    model.eval()
    _MODEL = model
    _TRACKER = PoseMemoryTracker(config=feature_config, device=torch.device("cpu"))
    _STATE = _MODEL.initial_state(torch.device("cpu"))


def _reset_episode(obs: np.ndarray) -> None:
    global _STATE, _PENDING_ACTION
    _TRACKER.reset_all(obs)
    _STATE = _MODEL.initial_state(torch.device("cpu"))
    _PENDING_ACTION = None


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_RNG_ID, _PENDING_ACTION, _STATE
    _load_once()

    obs_arr = np.asarray(obs, dtype=np.float32)
    rng_id = id(rng)
    if _LAST_RNG_ID != rng_id:
        _reset_episode(obs_arr)
        _LAST_RNG_ID = rng_id
        starts = torch.ones((1,), dtype=torch.float32)
    else:
        if _PENDING_ACTION is not None:
            _TRACKER.post_step(_PENDING_ACTION, obs_arr)
        starts = torch.zeros((1,), dtype=torch.float32)

    features = _TRACKER.features()
    logits, _STATE = _MODEL.actor_step(features, _STATE, starts)
    action_idx = int(torch.argmax(logits, dim=1).item())
    _PENDING_ACTION = action_idx
    return ACTIONS[action_idx]
