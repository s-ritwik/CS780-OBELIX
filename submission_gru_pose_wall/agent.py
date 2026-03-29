from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18
ACTION_DIM = len(ACTIONS)
FW_ACTION_INDEX = ACTIONS.index("FW")
TURN_DELTAS = torch.tensor([45.0, 22.5, 0.0, -22.5, -45.0], dtype=torch.float32)
FORWARD_STEP = 5.0
STOCHASTIC_POLICY = True
DEFAULT_WEIGHT_NAME = "weights.pth"


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def wrap_angle_deg(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + 180.0, 360.0) - 180.0


class FeatureConfig:
    def __init__(
        self,
        max_steps: int = 500,
        pose_clip: float = 500.0,
        blind_clip: float = 100.0,
        stuck_clip: float = 20.0,
        contact_clip: float = 20.0,
        same_obs_clip: float = 50.0,
        wall_hit_clip: float = 20.0,
        last_action_hist: int = 5,
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
            + 3
            + self.last_action_hist * ACTION_DIM
            + 1
            + self.heading_bins
            + 1
            + 1
        )


class PoseGRUFeatureTracker:
    def __init__(self, num_envs: int, config: FeatureConfig, device: torch.device) -> None:
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
        front_near = torch.sum(obs_t[:, 4:12:2], dim=1)
        front_far = torch.sum(obs_t[:, 5:12:2], dim=1)
        strength = 2.0 * front_near + front_far
        return torch.clamp(strength / 12.0, 0.0, 1.0)

    def _update_meta(self, env_indices: torch.Tensor, obs_t: torch.Tensor) -> None:
        if env_indices.numel() == 0:
            return

        visible = torch.any(obs_t[:, :17] > 0.5, dim=1)
        stuck = obs_t[:, 17] > 0.5
        front_near = torch.sum(obs_t[:, 4:12:2], dim=1)
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
        side = self.last_seen_side.unsqueeze(1)
        front = self.last_seen_front_strength.unsqueeze(1)

        return torch.cat(
            [
                self.current_obs,
                delta_obs,
                pose,
                counters,
                last_actions,
                same_obs,
                wall_hits,
                side,
                front,
            ],
            dim=1,
        )

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

            fw_mask = act == FW_ACTION_INDEX
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

            same_obs = torch.all(prev_obs == obs_after, dim=1)
            self.same_obs_count[active_idx] = torch.where(
                same_obs,
                torch.clamp(self.same_obs_count[active_idx] + 1.0, max=float(self.config.same_obs_clip)),
                torch.zeros_like(self.same_obs_count[active_idx]),
            )

            self.prev_obs[active_idx] = prev_obs
            self.current_obs[active_idx] = obs_after
            self._update_meta(active_idx, obs_after)

        if bool(torch.any(done_t)):
            done_idx = torch.nonzero(done_t, as_tuple=False).squeeze(1)
            self.reset_indices(done_idx, next_obs_t[done_idx])


class GRUPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dims: tuple[int, ...] = (128,),
        gru_hidden_dim: int = 64,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in encoder_dims:
            h = int(hidden_dim)
            layers.append(layer_init(nn.Linear(last_dim, h)))
            layers.append(nn.Tanh())
            last_dim = h
        self.encoder = nn.Sequential(*layers)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.gru = nn.GRU(last_dim, self.gru_hidden_dim)
        self.policy_head = layer_init(nn.Linear(self.gru_hidden_dim, int(action_dim)), std=0.01)
        self.value_head = layer_init(nn.Linear(self.gru_hidden_dim, 1), std=1.0)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((1, int(batch_size), self.gru_hidden_dim), dtype=torch.float32, device=device)

    def _mask_hidden(self, hidden: torch.Tensor, starts: torch.Tensor | None) -> torch.Tensor:
        if starts is None:
            return hidden
        starts_t = starts.to(dtype=torch.float32, device=hidden.device).view(1, -1, 1)
        return hidden * (1.0 - starts_t)

    def forward_step(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
        starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        hidden = self._mask_hidden(hidden, starts)
        encoded = self.encoder(features)
        out, next_hidden = self.gru(encoded.unsqueeze(0), hidden)
        feat = out.squeeze(0)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value, next_hidden


def load_checkpoint(path: str, device: torch.device) -> dict:
    raw = torch.load(path, map_location=device)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {path}")
    if "state_dict" not in raw:
        raw = {"state_dict": raw}
    return raw


_MODEL: Optional[GRUPolicy] = None
_TRACKER: Optional[PoseGRUFeatureTracker] = None
_HIDDEN: Optional[torch.Tensor] = None
_LAST_EPISODE_KEY: Optional[int] = None
_PENDING_ACTION: Optional[int] = None


def _checkpoint_path() -> str:
    here = os.path.dirname(__file__)
    override = os.environ.get("OBELIX_WEIGHTS")
    if override:
        return override if os.path.isabs(override) else os.path.join(here, override)
    return os.path.join(here, DEFAULT_WEIGHT_NAME)


def _load_once() -> None:
    global _MODEL, _TRACKER, _HIDDEN
    if _MODEL is not None and _TRACKER is not None and _HIDDEN is not None:
        return

    checkpoint = load_checkpoint(_checkpoint_path(), device=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]
    encoder_dims = tuple(int(x) for x in checkpoint.get("encoder_dims", [128]))
    gru_hidden_dim = int(checkpoint.get("gru_hidden_dim", 64))
    feature_payload = checkpoint.get("feature_config", {})
    feature_config = FeatureConfig(
        max_steps=int(feature_payload.get("max_steps", 500)),
        pose_clip=float(feature_payload.get("pose_clip", 500.0)),
        blind_clip=float(feature_payload.get("blind_clip", 100.0)),
        stuck_clip=float(feature_payload.get("stuck_clip", 20.0)),
        contact_clip=float(feature_payload.get("contact_clip", 20.0)),
        same_obs_clip=float(feature_payload.get("same_obs_clip", 50.0)),
        wall_hit_clip=float(feature_payload.get("wall_hit_clip", 20.0)),
        last_action_hist=int(feature_payload.get("last_action_hist", 5)),
        heading_bins=int(feature_payload.get("heading_bins", 8)),
    )

    model = GRUPolicy(
        input_dim=feature_config.feature_dim,
        encoder_dims=encoder_dims,
        gru_hidden_dim=gru_hidden_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _MODEL = model
    _TRACKER = PoseGRUFeatureTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
    _HIDDEN = _MODEL.initial_state(1, torch.device("cpu"))


def _reset_episode(obs: np.ndarray) -> None:
    global _TRACKER, _HIDDEN, _PENDING_ACTION
    _TRACKER.reset_all(obs[None, :])
    _HIDDEN = _MODEL.initial_state(1, torch.device("cpu"))
    _PENDING_ACTION = None


def _episode_key(rng: np.random.Generator) -> int:
    seed_seq = getattr(rng.bit_generator, "_seed_seq", None)
    if seed_seq is not None and hasattr(seed_seq, "entropy"):
        return int(seed_seq.entropy)
    return id(rng)


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_EPISODE_KEY, _PENDING_ACTION, _HIDDEN
    _load_once()

    obs_arr = np.asarray(obs, dtype=np.float32)
    episode_key = _episode_key(rng)
    if _LAST_EPISODE_KEY != episode_key:
        _reset_episode(obs_arr)
        _LAST_EPISODE_KEY = episode_key
        starts = torch.ones((1,), dtype=torch.float32)
    else:
        if _PENDING_ACTION is not None:
            _TRACKER.post_step(
                actions=torch.tensor([_PENDING_ACTION], dtype=torch.long),
                next_obs=obs_arr[None, :],
                dones=np.asarray([False]),
            )
        starts = torch.zeros((1,), dtype=torch.float32)

    features = _TRACKER.features()
    logits, _, _HIDDEN = _MODEL.forward_step(features, _HIDDEN, starts)
    if STOCHASTIC_POLICY:
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64, copy=False)
        probs /= probs.sum()
        action_idx = int(rng.choice(len(ACTIONS), p=probs))
    else:
        action_idx = int(torch.argmax(logits, dim=1).item())

    _PENDING_ACTION = action_idx
    return ACTIONS[action_idx]
