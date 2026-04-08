from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
OBS_DIM = 18
ACTION_DIM = len(ACTIONS)
FW_ACTION_INDEX = ACTION_TO_INDEX["FW"]
ENCODED_OBS_DIM = 32
DERIVED_OBS_DIM = 16
TURN_DELTAS = torch.tensor([45.0, 22.5, 0.0, -22.5, -45.0], dtype=torch.float32)
FORWARD_STEP = 5.0

PROBE_STEPS = 80
CONTACT_THRESHOLD = 15
BLIND_THRESHOLD = 40
FRONT_TOTAL_THRESHOLD = 0


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

    derived = torch.cat(
        [
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
        ],
        dim=-1,
    )
    return torch.cat([raw, derived], dim=-1)


def wrap_angle_deg(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class RecurrentFeatureConfig:
    def __init__(
        self,
        max_steps: int = 2000,
        pose_clip: float = 500.0,
        blind_clip: float = 120.0,
        stuck_clip: float = 24.0,
        contact_clip: float = 24.0,
        same_obs_clip: float = 64.0,
        wall_hit_clip: float = 24.0,
        last_action_hist: int = 8,
        heading_bins: int = 8,
    ) -> None:
        self.max_steps = int(max_steps)
        self.pose_clip = float(pose_clip)
        self.blind_clip = float(blind_clip)
        self.stuck_clip = float(stuck_clip)
        self.contact_clip = float(contact_clip)
        self.same_obs_clip = float(same_obs_clip)
        self.wall_hit_clip = float(wall_hit_clip)
        self.last_action_hist = int(last_action_hist)
        self.heading_bins = int(heading_bins)

    @property
    def feature_dim(self) -> int:
        return (
            OBS_DIM
            + OBS_DIM
            + 4
            + 4
            + self.last_action_hist * ACTION_DIM
            + 1
            + self.heading_bins
            + 2
        )


class PoseMemoryTracker:
    def __init__(self, config: RecurrentFeatureConfig) -> None:
        self.config = config
        self.turn_deltas = TURN_DELTAS
        self.current_obs = torch.zeros((1, OBS_DIM), dtype=torch.float32)
        self.prev_obs = torch.zeros((1, OBS_DIM), dtype=torch.float32)
        self.last_action_hist = torch.zeros((1, self.config.last_action_hist, ACTION_DIM), dtype=torch.float32)
        self.x_rel = torch.zeros((1,), dtype=torch.float32)
        self.y_rel = torch.zeros((1,), dtype=torch.float32)
        self.theta_deg = torch.zeros((1,), dtype=torch.float32)
        self.blind_steps = torch.zeros((1,), dtype=torch.float32)
        self.stuck_steps = torch.zeros((1,), dtype=torch.float32)
        self.contact_steps = torch.zeros((1,), dtype=torch.float32)
        self.same_obs_count = torch.zeros((1,), dtype=torch.float32)
        self.wall_hit_count_by_heading_bin = torch.zeros((1, self.config.heading_bins), dtype=torch.float32)
        self.last_seen_side = torch.zeros((1,), dtype=torch.float32)
        self.last_seen_front_strength = torch.zeros((1,), dtype=torch.float32)

    def _obs_to_tensor(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            obs_t = obs.to(dtype=torch.float32)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
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
        return torch.clamp((front_far + 2.0 * front_near) / 12.0, 0.0, 1.0)

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

        side = self.last_seen_side
        side = torch.where((left > right) & (left > 0.0), torch.full_like(side, -1.0), side)
        side = torch.where((right > left) & (right > 0.0), torch.full_like(side, 1.0), side)
        side = torch.where(front_strength > 0.0, torch.zeros_like(side), side)
        self.last_seen_side = side
        self.last_seen_front_strength = torch.where(
            front_strength > 0.0,
            front_strength,
            torch.clamp(self.last_seen_front_strength - 0.05, min=0.0),
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
                torch.clamp(self.last_seen_front_strength, 0.0, 1.0),
            ],
            dim=1,
        )
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
        return torch.cat(
            [
                self.current_obs,
                delta_obs,
                pose,
                counters,
                self.last_action_hist.reshape(1, -1),
                same_obs,
                wall_hits,
                self.last_seen_side.unsqueeze(1),
                self.last_seen_front_strength.unsqueeze(1),
            ],
            dim=1,
        )

    def post_step(self, action_idx: int, next_obs: np.ndarray | torch.Tensor) -> None:
        next_obs_t = self._obs_to_tensor(next_obs)
        prev_obs = self.current_obs.clone()
        action_t = torch.tensor([int(action_idx)], dtype=torch.long)

        rolled_actions = torch.roll(self.last_action_hist, shifts=-1, dims=1)
        rolled_actions[:, -1] = 0.0
        rolled_actions[:, -1, :] = F.one_hot(action_t, num_classes=ACTION_DIM).to(torch.float32)
        self.last_action_hist = rolled_actions

        theta_next = wrap_angle_deg(self.theta_deg + self.turn_deltas[action_t])
        self.theta_deg = theta_next
        if int(action_idx) == FW_ACTION_INDEX:
            if next_obs_t[0, 17] > 0.5:
                hit_bin = self._heading_bins_for(theta_next)
                self.wall_hit_count_by_heading_bin[0, hit_bin] = torch.clamp(
                    self.wall_hit_count_by_heading_bin[0, hit_bin] + 1.0,
                    max=float(self.config.wall_hit_clip),
                )
            else:
                rad = torch.deg2rad(theta_next)
                self.x_rel += float(FORWARD_STEP) * torch.cos(rad)
                self.y_rel += float(FORWARD_STEP) * torch.sin(rad)

        same_obs = torch.all(prev_obs == next_obs_t, dim=1)
        self.same_obs_count = torch.where(
            same_obs,
            torch.clamp(self.same_obs_count + 1.0, max=float(self.config.same_obs_clip)),
            torch.zeros_like(self.same_obs_count),
        )
        self.prev_obs.copy_(prev_obs)
        self.current_obs.copy_(next_obs_t)
        self._update_meta(next_obs_t)


class RecurrentWallActor(nn.Module):
    def __init__(self, actor_dim: int, encoder_dims: tuple[int, ...], gru_hidden_dim: int, gru_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(actor_dim)
        for hidden_dim in encoder_dims:
            layers.append(_layer_init(nn.Linear(last_dim, int(hidden_dim))))
            layers.append(nn.Tanh())
            last_dim = int(hidden_dim)
        self.actor_encoder = nn.Sequential(*layers)
        self.actor_rnn = nn.GRU(input_size=last_dim, hidden_size=int(gru_hidden_dim), num_layers=int(gru_layers))
        self.policy_head = _layer_init(nn.Linear(int(gru_hidden_dim), ACTION_DIM), std=0.01)
        self.gru_layers = int(gru_layers)
        self.gru_hidden_dim = int(gru_hidden_dim)

    def initial_state(self) -> torch.Tensor:
        return torch.zeros((self.gru_layers, 1, self.gru_hidden_dim), dtype=torch.float32)

    def actor_step(self, actor_obs: torch.Tensor, hidden: torch.Tensor, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts_t = starts.to(dtype=torch.float32).view(1, -1, 1)
        hidden = hidden * (1.0 - starts_t)
        encoded = self.actor_encoder(actor_obs)
        out, next_hidden = self.actor_rnn(encoded.unsqueeze(0), hidden)
        return self.policy_head(out.squeeze(0)), next_hidden


class NoWallActorCritic(nn.Module):
    def __init__(self, hidden_dims: tuple[int, ...], use_rec_encoder: bool) -> None:
        super().__init__()
        self.use_rec_encoder = bool(use_rec_encoder)
        layers: list[nn.Module] = []
        last_dim = ENCODED_OBS_DIM if self.use_rec_encoder else OBS_DIM
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


def _infer_nowall_hidden_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    keys = []
    for key, value in state_dict.items():
        if key.startswith("backbone.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                keys.append((int(parts[1]), int(value.shape[0])))
    keys.sort(key=lambda x: x[0])
    return tuple(dim for _, dim in keys) if keys else (128, 64)


def _infer_nowall_encoder(state_dict: dict[str, torch.Tensor]) -> bool:
    first = state_dict.get("backbone.0.weight")
    if first is None:
        return False
    return int(first.shape[1]) == ENCODED_OBS_DIM


class WallStudentPolicy:
    def __init__(self, checkpoint: dict) -> None:
        self.config = RecurrentFeatureConfig(**checkpoint["feature_config"])
        self.model = RecurrentWallActor(
            actor_dim=self.config.feature_dim,
            encoder_dims=tuple(int(x) for x in checkpoint["encoder_dims"]),
            gru_hidden_dim=int(checkpoint["gru_hidden_dim"]),
            gru_layers=int(checkpoint["gru_layers"]),
        )
        self.model.actor_encoder.load_state_dict(checkpoint["actor_encoder_state_dict"], strict=True)
        self.model.actor_rnn.load_state_dict(checkpoint["actor_rnn_state_dict"], strict=True)
        self.model.policy_head.load_state_dict(checkpoint["policy_head_state_dict"], strict=True)
        self.model.eval()
        self.tracker = PoseMemoryTracker(self.config)
        self.hidden = self.model.initial_state()
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
            self.hidden = self.model.initial_state()
            starts = torch.ones((1,), dtype=torch.float32)
        else:
            if self.pending_action is not None:
                self.tracker.post_step(self.pending_action, obs_arr[None, :])
            starts = torch.zeros((1,), dtype=torch.float32)
        logits, self.hidden = self.model.actor_step(self.tracker.features(), self.hidden, starts)
        action_idx = int(torch.argmax(logits, dim=1).item())
        self.pending_action = action_idx
        return ACTIONS[action_idx]


class NoWallPolicy:
    def __init__(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        hidden_dims = tuple(int(x) for x in checkpoint.get("hidden_dims", _infer_nowall_hidden_dims(state_dict)))
        use_rec_encoder = bool(checkpoint.get("use_rec_encoder", _infer_nowall_encoder(state_dict)))
        self.model = NoWallActorCritic(hidden_dims=hidden_dims, use_rec_encoder=use_rec_encoder)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> str:
        del rng
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None, :], dtype=torch.float32)
        logits = self.model(obs_t)
        return ACTIONS[int(torch.argmax(logits, dim=1).item())]


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.Tanh())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RouterPolicy:
    def __init__(self, checkpoint: dict) -> None:
        self.model = RouterMLP(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims=tuple(int(x) for x in checkpoint.get("hidden_dims", [64, 32])),
        )
        state_dict = checkpoint["state_dict"]
        if state_dict and not next(iter(state_dict)).startswith("net."):
            state_dict = {f"net.{key}": value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @staticmethod
    def features(first_obs: np.ndarray, obs: np.ndarray, step: int) -> torch.Tensor:
        first = np.asarray(first_obs, dtype=np.float32)
        cur = np.asarray(obs, dtype=np.float32)
        step_oh = np.zeros(6, dtype=np.float32)
        step_oh[min(int(step), 5)] = 1.0
        counts = np.asarray(
            [
                np.sum(first[:16]),
                np.sum(first[:4]),
                np.sum(first[4:12]),
                np.sum(first[12:16]),
                np.sum(cur[:16]),
                np.sum(cur[:4]),
                np.sum(cur[4:12]),
                np.sum(cur[12:16]),
                float(step) / 10.0,
            ],
            dtype=np.float32,
        )
        x = np.concatenate([first, cur, step_oh, counts]).astype(np.float32, copy=False)
        return torch.as_tensor(x[None, :], dtype=torch.float32)

    @torch.no_grad()
    def decide(self, first_obs: np.ndarray, obs: np.ndarray, step: int) -> int:
        logits = self.model(self.features(first_obs, obs, step))
        return int(torch.argmax(logits, dim=1).item())


_WALL_POLICY: Optional[WallStudentPolicy] = None
_NOWALL_POLICY: Optional[WallStudentPolicy] = None
_ROUTER_POLICY: Optional[RouterPolicy] = None
_LAST_RNG_ID: Optional[int] = None
_STEP_COUNT = 0
_MODE: Optional[str] = None
_FIRST_OBS: Optional[np.ndarray] = None


def _weights_path() -> str:
    return os.path.join(os.path.dirname(__file__), "weights.pth")


def _load_once() -> tuple[WallStudentPolicy, WallStudentPolicy, RouterPolicy]:
    global _WALL_POLICY, _NOWALL_POLICY, _ROUTER_POLICY
    if _WALL_POLICY is None or _NOWALL_POLICY is None or _ROUTER_POLICY is None:
        bundle = torch.load(_weights_path(), map_location="cpu")
        _WALL_POLICY = WallStudentPolicy(bundle["wall_student"])
        _NOWALL_POLICY = WallStudentPolicy(bundle["nowall_student"])
        _ROUTER_POLICY = RouterPolicy(bundle["router"])
    return _WALL_POLICY, _NOWALL_POLICY, _ROUTER_POLICY


def _reset_episode(rng: np.random.Generator, obs_arr: np.ndarray) -> None:
    global _LAST_RNG_ID, _STEP_COUNT, _MODE, _FIRST_OBS
    _LAST_RNG_ID = id(rng)
    _STEP_COUNT = 0
    _MODE = None
    _FIRST_OBS = np.asarray(obs_arr, dtype=np.float32).copy()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _STEP_COUNT, _MODE
    wall_policy, nowall_policy, router_policy = _load_once()
    obs_arr = np.asarray(obs, dtype=np.float32)
    if _LAST_RNG_ID != id(rng):
        _reset_episode(rng, obs_arr)

    if _MODE == "wall":
        action = wall_policy.act(obs_arr, rng)
    elif _MODE == "nowall":
        action = nowall_policy.act(obs_arr, rng)
    else:
        decision = router_policy.decide(_FIRST_OBS, obs_arr, _STEP_COUNT)
        if decision == 0:
            action = wall_policy.act(obs_arr, rng)
        elif decision == 1:
            _MODE = "wall"
            action = wall_policy.act(obs_arr, rng)
        elif decision == 2:
            _MODE = "nowall"
            action = nowall_policy.act(obs_arr, rng)
        else:
            _MODE = "nowall"
            action = "L45"

    _STEP_COUNT += 1
    return action
