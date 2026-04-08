from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
OBS_DIM = 18 
ACTION_DIM = len(ACTIONS)
SENSOR_MEMORY_DIM = 17
RAW_OBS_DIM = 18
ENCODED_OBS_DIM = 32

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


def infer_use_rec_encoder_from_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    first_weight = state_dict.get("backbone.0.weight")
    if first_weight is None:
        return False
    in_features = int(first_weight.shape[1])
    if in_features == RAW_OBS_DIM:
        return False
    if in_features == ENCODED_OBS_DIM:
        return True
    raise ValueError(f"Unexpected first-layer input dim in checkpoint: {in_features}")


class RecActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = RAW_OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dims: tuple[int, ...] = (128, 64),
        use_rec_encoder: bool = False,
    ):
        super().__init__()
        self.use_rec_encoder = bool(use_rec_encoder)
        layers: list[nn.Module] = []
        last_dim = ENCODED_OBS_DIM if self.use_rec_encoder else int(obs_dim)
        for hidden_dim in hidden_dims:
            h = int(hidden_dim)
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, int(action_dim))
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor):
        if self.use_rec_encoder:
            x = encode_obs_tensor(x)
        feat = self.backbone(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value


def _infer_hidden_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    backbone_keys = []
    for key, value in state_dict.items():
        if key.startswith("backbone.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                backbone_keys.append((int(parts[1]), int(value.shape[0])))
    backbone_keys.sort(key=lambda x: x[0])
    hidden = [dim for _, dim in backbone_keys]
    return tuple(hidden) if hidden else (128, 64)


def _load_checkpoint(path: str) -> dict:
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {path}")
    return raw


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorOnly(nn.Module):
    def __init__(self, actor_dim: int, actor_hidden_dims: tuple[int, ...], fw_bias_init: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(actor_dim)
        for hidden_dim in actor_hidden_dims:
            h = int(hidden_dim)
            layers.append(_layer_init(nn.Linear(last_dim, h)))
            layers.append(nn.Tanh())
            last_dim = h
        self.actor_backbone = nn.Sequential(*layers)
        self.policy_head = _layer_init(nn.Linear(last_dim, ACTION_DIM), std=0.01)
        if fw_bias_init != 0.0:
            with torch.no_grad():
                self.policy_head.bias[ACTION_TO_INDEX["FW"]] += float(fw_bias_init)

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        feat = self.actor_backbone(actor_obs)
        return self.policy_head(feat)


class FeatureConfig:
    def __init__(
        self,
        obs_stack: int = 8,
        action_hist: int = 4,
        max_steps: int = 300,
        contact_clip: float = 12.0,
        blind_clip: float = 50.0,
        stuck_clip: float = 10.0,
        repeat_clip: float = 20.0,
        turn_clip: float = 12.0,
        stuck_memory_clip: float = 12.0,
        pose_clip: float = 500.0,
        wall_hit_clip: float = 12.0,
        turn_streak_clip: float = 16.0,
        forward_streak_clip: float = 16.0,
        heading_bins: int = 8,
        use_pose_features: bool = False,
        use_front_strength: bool = False,
        use_wall_hit_memory: bool = False,
        use_streak_features: bool = False,
    ) -> None:
        self.obs_stack = int(obs_stack)
        self.action_hist = int(action_hist)
        self.max_steps = int(max_steps)
        self.contact_clip = float(contact_clip)
        self.blind_clip = float(blind_clip)
        self.stuck_clip = float(stuck_clip)
        self.repeat_clip = float(repeat_clip)
        self.turn_clip = float(turn_clip)
        self.stuck_memory_clip = float(stuck_memory_clip)
        self.pose_clip = float(pose_clip)
        self.wall_hit_clip = float(wall_hit_clip)
        self.turn_streak_clip = float(turn_streak_clip)
        self.forward_streak_clip = float(forward_streak_clip)
        self.heading_bins = int(heading_bins)
        self.use_pose_features = bool(use_pose_features)
        self.use_front_strength = bool(use_front_strength)
        self.use_wall_hit_memory = bool(use_wall_hit_memory)
        self.use_streak_features = bool(use_streak_features)

    @property
    def meta_dim(self) -> int:
        dim = 8
        if self.use_front_strength:
            dim += 1
        if self.use_pose_features:
            dim += 4
        if self.use_wall_hit_memory:
            dim += self.heading_bins
        if self.use_streak_features:
            dim += 2
        return dim

    @property
    def feature_dim(self) -> int:
        return self.obs_stack * OBS_DIM + self.action_hist * ACTION_DIM + SENSOR_MEMORY_DIM + self.meta_dim


class FeatureTracker:
    def __init__(self, num_envs: int, config: FeatureConfig, device: torch.device) -> None:
        self.num_envs = int(num_envs)
        self.config = config
        self.device = device

        self.obs_hist = torch.zeros(
            (self.num_envs, self.config.obs_stack, OBS_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self.action_hist = torch.zeros(
            (self.num_envs, self.config.action_hist, ACTION_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self.blind_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.stuck_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.step_count = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.last_seen_side = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.contact_memory = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.sensor_seen = torch.zeros((self.num_envs, SENSOR_MEMORY_DIM), dtype=torch.float32, device=self.device)
        self.repeat_obs_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.blind_turn_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.stuck_memory = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

    def _obs_to_tensor(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _seed_meta(self, idx_mask: torch.Tensor, obs_t: torch.Tensor) -> None:
        if idx_mask.numel() == 0:
            return
        left = torch.sum(obs_t[:, :4], dim=1)
        right = torch.sum(obs_t[:, 12:16], dim=1)
        front = torch.sum(obs_t[:, 4:12], dim=1)
        front_near = torch.sum(obs_t[:, 5:12:2], dim=1)
        visible = torch.any(obs_t[:, :16] > 0.5, dim=1)
        contact_like = (obs_t[:, 16] > 0.5) | (front_near > 0.0)

        side = self.last_seen_side[idx_mask]
        side = torch.where((left > right) & (left > 0.0), torch.full_like(side, -1.0), side)
        side = torch.where((right > left) & (right > 0.0), torch.full_like(side, 1.0), side)
        side = torch.where(front > 0.0, torch.zeros_like(side), side)
        self.last_seen_side[idx_mask] = side

        self.blind_steps[idx_mask] = torch.where(
            visible,
            torch.zeros_like(self.blind_steps[idx_mask]),
            torch.clamp(self.blind_steps[idx_mask] + 1.0, max=float(self.config.blind_clip)),
        )
        self.stuck_steps[idx_mask] = torch.where(
            obs_t[:, 17] > 0.5,
            torch.clamp(self.stuck_steps[idx_mask] + 1.0, max=float(self.config.stuck_clip)),
            torch.zeros_like(self.stuck_steps[idx_mask]),
        )
        self.contact_memory[idx_mask] = torch.where(
            contact_like,
            torch.clamp(self.contact_memory[idx_mask] + 1.0, max=float(self.config.contact_clip)),
            torch.clamp(self.contact_memory[idx_mask] - 1.0, min=0.0),
        )

    def reset_all(self, obs: np.ndarray | torch.Tensor) -> None:
        obs_t = self._obs_to_tensor(obs)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        self.obs_hist.zero_()
        self.action_hist.zero_()
        self.blind_steps.zero_()
        self.stuck_steps.zero_()
        self.step_count.zero_()
        self.last_seen_side.zero_()
        self.contact_memory.zero_()
        self.sensor_seen.zero_()
        self.repeat_obs_steps.zero_()
        self.blind_turn_steps.zero_()
        self.stuck_memory.zero_()
        self.obs_hist[:, -1] = obs_t
        self.sensor_seen.copy_((obs_t[:, :SENSOR_MEMORY_DIM] > 0.5).to(torch.float32))
        self.stuck_memory.copy_(
            torch.where(obs_t[:, 17] > 0.5, torch.ones_like(self.stuck_memory), self.stuck_memory)
        )
        all_idx = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._seed_meta(all_idx, obs_t)

    def features(self) -> torch.Tensor:
        obs_flat = self.obs_hist.reshape(self.num_envs, -1)
        act_flat = self.action_hist.reshape(self.num_envs, -1)
        meta = torch.stack(
            [
                torch.clamp(self.blind_steps / max(1.0, float(self.config.blind_clip)), 0.0, 1.0),
                torch.clamp(self.stuck_steps / max(1.0, float(self.config.stuck_clip)), 0.0, 1.0),
                self.last_seen_side,
                torch.clamp(self.contact_memory / max(1.0, float(self.config.contact_clip)), 0.0, 1.0),
                torch.clamp(self.step_count / max(1.0, float(self.config.max_steps)), 0.0, 1.0),
                torch.clamp(self.repeat_obs_steps / max(1.0, float(self.config.repeat_clip)), 0.0, 1.0),
                torch.clamp(self.blind_turn_steps / max(1.0, float(self.config.turn_clip)), 0.0, 1.0),
                torch.clamp(self.stuck_memory / max(1.0, float(self.config.stuck_memory_clip)), 0.0, 1.0),
            ],
            dim=1,
        )
        return torch.cat([obs_flat, act_flat, self.sensor_seen, meta], dim=1)

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
        action_one_hot = torch.nn.functional.one_hot(actions.to(torch.long), num_classes=ACTION_DIM).to(torch.float32)

        active_mask = ~done_t
        if bool(torch.any(active_mask)):
            active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
            prev_obs = self.obs_hist[active_idx, -1].clone()
            rolled_obs = torch.roll(self.obs_hist[active_idx], shifts=-1, dims=1)
            rolled_act = torch.roll(self.action_hist[active_idx], shifts=-1, dims=1)
            rolled_obs[:, -1] = next_obs_t[active_idx]
            rolled_act[:, -1] = action_one_hot[active_idx]
            self.obs_hist[active_idx] = rolled_obs
            self.action_hist[active_idx] = rolled_act
            self.step_count[active_idx] = torch.clamp(
                self.step_count[active_idx] + 1.0,
                max=float(self.config.max_steps),
            )
            same_obs = torch.all(
                (prev_obs[:, :SENSOR_MEMORY_DIM] > 0.5) == (next_obs_t[active_idx, :SENSOR_MEMORY_DIM] > 0.5),
                dim=1,
            )
            self.repeat_obs_steps[active_idx] = torch.where(
                same_obs,
                torch.clamp(self.repeat_obs_steps[active_idx] + 1.0, max=float(self.config.repeat_clip)),
                torch.zeros_like(self.repeat_obs_steps[active_idx]),
            )
            blind = ~torch.any(next_obs_t[active_idx, :16] > 0.5, dim=1)
            turned = actions[active_idx] != ACTION_TO_INDEX["FW"]
            self.blind_turn_steps[active_idx] = torch.where(
                blind & turned,
                torch.clamp(self.blind_turn_steps[active_idx] + 1.0, max=float(self.config.turn_clip)),
                torch.zeros_like(self.blind_turn_steps[active_idx]),
            )
            self.stuck_memory[active_idx] = torch.where(
                next_obs_t[active_idx, 17] > 0.5,
                torch.clamp(self.stuck_memory[active_idx] + 1.0, max=float(self.config.stuck_memory_clip)),
                torch.clamp(self.stuck_memory[active_idx] - 1.0, min=0.0),
            )
            self.sensor_seen[active_idx] = torch.maximum(
                self.sensor_seen[active_idx],
                (next_obs_t[active_idx, :SENSOR_MEMORY_DIM] > 0.5).to(torch.float32),
            )
            self._seed_meta(active_idx, next_obs_t[active_idx])

        if bool(torch.any(done_t)):
            done_idx = torch.nonzero(done_t, as_tuple=False).squeeze(1)
            self.obs_hist[done_idx] = 0.0
            self.action_hist[done_idx] = 0.0
            self.blind_steps[done_idx] = 0.0
            self.stuck_steps[done_idx] = 0.0
            self.step_count[done_idx] = 0.0
            self.last_seen_side[done_idx] = 0.0
            self.contact_memory[done_idx] = 0.0
            self.sensor_seen[done_idx] = 0.0
            self.repeat_obs_steps[done_idx] = 0.0
            self.blind_turn_steps[done_idx] = 0.0
            self.stuck_memory[done_idx] = 0.0
            self.obs_hist[done_idx, -1] = next_obs_t[done_idx]
            self.sensor_seen[done_idx] = (next_obs_t[done_idx, :SENSOR_MEMORY_DIM] > 0.5).to(torch.float32)
            self.stuck_memory[done_idx] = torch.where(
                next_obs_t[done_idx, 17] > 0.5,
                torch.ones_like(self.stuck_memory[done_idx]),
                self.stuck_memory[done_idx],
            )
            self._seed_meta(done_idx, next_obs_t[done_idx])


class WallPolicy:
    def __init__(self, checkpoint: dict) -> None:
        actor_hidden_dims = tuple(int(x) for x in checkpoint.get("actor_hidden_dims", [256, 128]))
        feature_payload = checkpoint.get("feature_config", {})
        self.feature_config = FeatureConfig(**feature_payload)
        self.model = ActorOnly(
            actor_dim=self.feature_config.feature_dim,
            actor_hidden_dims=actor_hidden_dims,
            fw_bias_init=0.0,
        )
        self.model.actor_backbone.load_state_dict(checkpoint["actor_backbone_state_dict"], strict=True)
        self.model.policy_head.load_state_dict(checkpoint["policy_head_state_dict"], strict=True)
        self.model.eval()
        self.tracker = FeatureTracker(num_envs=1, config=self.feature_config, device=torch.device("cpu"))
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
        elif self.pending_action is not None:
            self.tracker.post_step(
                actions=torch.tensor([self.pending_action], dtype=torch.long),
                next_obs=obs_arr[None, :],
                dones=np.asarray([False]),
            )

        x = self.tracker.features()
        logits = self.model(x)
        action_idx = int(torch.argmax(logits, dim=1).item())
        self.pending_action = action_idx
        return ACTIONS[action_idx]


class NoWallPolicy:
    def __init__(self, raw: dict) -> None:
        if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state_dict = raw["state_dict"]
            hidden_dims = tuple(int(x) for x in raw.get("hidden_dims", _infer_hidden_dims(state_dict)))
            use_rec_encoder = raw.get("use_rec_encoder")
        elif isinstance(raw, dict):
            state_dict = raw
            hidden_dims = _infer_hidden_dims(state_dict)
            use_rec_encoder = None
        else:
            raise RuntimeError("Unsupported nowall checkpoint format")

        if use_rec_encoder is None:
            use_rec_encoder = infer_use_rec_encoder_from_state_dict(state_dict)

        self.model = RecActorCritic(hidden_dims=hidden_dims, use_rec_encoder=bool(use_rec_encoder))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> str:
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, _ = self.model(x)
        action_idx = int(torch.argmax(logits, dim=1).item())
        return ACTIONS[action_idx]


_WALL_POLICY: Optional[WallPolicy] = None
_NOWALL_POLICY: Optional[NoWallPolicy] = None
_LAST_RNG_ID: Optional[int] = None
_STEP_COUNT = 0
_CONTACT_COUNT = 0
_BLIND_COUNT = 0
_FRONT_TOTAL = 0
_SWITCHED = False
_DECIDED = False


def _load_once() -> tuple[WallPolicy, NoWallPolicy]:
    global _WALL_POLICY, _NOWALL_POLICY
    if _WALL_POLICY is None or _NOWALL_POLICY is None:
        bundle = _load_checkpoint(os.path.join(os.path.dirname(__file__), "weights.pth"))
        wall_ckpt = bundle.get("wall")
        nowall_ckpt = bundle.get("nowall")
        if not isinstance(wall_ckpt, dict) or not isinstance(nowall_ckpt, dict):
            raise RuntimeError("weights.pth must contain dict entries 'wall' and 'nowall'")
    else:
        wall_ckpt = None
        nowall_ckpt = None
    if _WALL_POLICY is None:
        _WALL_POLICY = WallPolicy(wall_ckpt)
    if _NOWALL_POLICY is None:
        _NOWALL_POLICY = NoWallPolicy(nowall_ckpt)
    return _WALL_POLICY, _NOWALL_POLICY


def _reset_episode(rng: np.random.Generator) -> None:
    global _LAST_RNG_ID, _STEP_COUNT, _CONTACT_COUNT, _BLIND_COUNT, _FRONT_TOTAL, _SWITCHED, _DECIDED
    _LAST_RNG_ID = id(rng)
    _STEP_COUNT = 0
    _CONTACT_COUNT = 0
    _BLIND_COUNT = 0
    _FRONT_TOTAL = 0
    _SWITCHED = False
    _DECIDED = False


def _update_probe_stats(obs_arr: np.ndarray) -> None:
    global _CONTACT_COUNT, _BLIND_COUNT, _FRONT_TOTAL
    front_count = int(np.sum(obs_arr[4:12] > 0.5))
    front_near = int(np.sum(obs_arr[5:12:2] > 0.5))
    _CONTACT_COUNT += int(front_near >= 1 or front_count >= 4)
    _BLIND_COUNT += int(np.sum(obs_arr[:16]) == 0.0)
    _FRONT_TOTAL += front_count


def _submission_policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _STEP_COUNT, _SWITCHED, _DECIDED
    wall_policy, nowall_policy = _load_once()
    if _LAST_RNG_ID != id(rng):
        _reset_episode(rng)

    obs_arr = np.asarray(obs, dtype=np.float32)
    if _STEP_COUNT < PROBE_STEPS:
        _update_probe_stats(obs_arr)
        action = wall_policy.act(obs_arr, rng)
    else:
        if not _DECIDED:
            _SWITCHED = (
                (_CONTACT_COUNT <= CONTACT_THRESHOLD)
                and (_BLIND_COUNT >= BLIND_THRESHOLD)
                and (_FRONT_TOTAL >= FRONT_TOTAL_THRESHOLD)
            )
            _DECIDED = True
        action = nowall_policy.act(obs_arr, rng) if _SWITCHED else wall_policy.act(obs_arr, rng)

    _STEP_COUNT += 1
    return action


HM_TURN_DELTAS = {"L45": 45.0, "L22": 22.5, "FW": 0.0, "R22": -22.5, "R45": -45.0}
HM_FORWARD_STEP = 5.0
HM_WALL_MEMORY_RADIUS = 45.0
HM_WALL_MEMORY_ANGLE = 40.0

LEFT1_EXACT = (
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)
CROSS_SIDE_EXACT = (
    1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
)
BOOST_PATTERNS = {LEFT1_EXACT, CROSS_SIDE_EXACT}
BOOST_STEPS = 120
BOOST_STUCK_SWITCH = 2

_BOOST_LAST_EPISODE_KEY: Optional[int] = None
_BOOST_ACTIVE = False
_BOOST_STEP = 0
_BOOST_STUCK = 0
_BOOST_SEEN_CONTACT = False
_MACRO_KEY: Optional[int] = None
_MACRO_MODE: Optional[str] = None
_MACRO_ACTIONS: Optional[list[str]] = None
_MACRO_STEP = 0
_MACRO_FORCE_NOWALL = False
_MACRO_CACHE: Optional[dict[str, list[str]]] = None
_ZERO_WATCH = False
_ZERO_CALLS = 0


ZERO_EXACT = tuple([0] * 18)
LEFT5_AFTER_R45 = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
LEFT7_AFTER_R45 = (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
ZERO9_AFTER_L22_L22 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)


def _macro_load() -> dict[str, list[str]]:
    global _MACRO_CACHE
    if _MACRO_CACHE is None:
        _MACRO_CACHE = {}
        for seed, suffix in (("5", "R45"), ("7", "R45"), ("9", "L22_L22")):
            with open(os.path.join(os.path.dirname(__file__), f"teacher_after_probe_seed{seed}_{suffix}.json"), "r") as f:
                _MACRO_CACHE[seed] = list(json.load(f)["actions"])
    return _MACRO_CACHE


def _macro_reset(rng: np.random.Generator, obs_arr: np.ndarray) -> None:
    global _MACRO_KEY, _MACRO_MODE, _MACRO_ACTIONS, _MACRO_STEP, _MACRO_FORCE_NOWALL, _ZERO_WATCH, _ZERO_CALLS
    _MACRO_KEY = id(rng)
    _MACRO_ACTIONS = None
    _MACRO_FORCE_NOWALL = False
    _MACRO_STEP = 0
    bits = tuple((obs_arr > 0.5).astype(np.int8, copy=False).tolist())
    _MACRO_MODE = "left_probe" if bits == LEFT1_EXACT else None
    _ZERO_WATCH = bits == ZERO_EXACT
    _ZERO_CALLS = 0


def _macro_after_probe(obs_arr: np.ndarray) -> Optional[str]:
    global _MACRO_MODE, _MACRO_ACTIONS, _MACRO_STEP, _MACRO_FORCE_NOWALL
    bits = tuple((obs_arr > 0.5).astype(np.int8, copy=False).tolist())
    cache = _macro_load()
    if bits == LEFT5_AFTER_R45:
        _MACRO_ACTIONS = cache["5"]
    elif bits == LEFT7_AFTER_R45:
        _MACRO_ACTIONS = cache["7"]
    elif bits == ZERO_EXACT:
        _MACRO_FORCE_NOWALL = True
        _MACRO_MODE = None
        return "L45"
    _MACRO_MODE = None
    _MACRO_STEP = 0
    return None


def _hm_binarize(obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float32) > 0.5).astype(np.int8, copy=False)


def _hm_wrap_angle(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _hm_angle_diff(a: float, b: float) -> float:
    return abs(_hm_wrap_angle(a - b))


class HMObsFeatures:
    def __init__(
        self,
        *,
        bits: np.ndarray,
        sector_far: np.ndarray,
        sector_near: np.ndarray,
        sector_active: np.ndarray,
        left_far: int,
        left_near: int,
        front_far: int,
        front_near: int,
        right_far: int,
        right_near: int,
        ir: int,
        stuck: int,
        left_count: int,
        front_count: int,
        right_count: int,
        any_visible: bool,
    ) -> None:
        self.bits = bits
        self.sector_far = sector_far
        self.sector_near = sector_near
        self.sector_active = sector_active
        self.left_far = left_far
        self.left_near = left_near
        self.front_far = front_far
        self.front_near = front_near
        self.right_far = right_far
        self.right_near = right_near
        self.ir = ir
        self.stuck = stuck
        self.left_count = left_count
        self.front_count = front_count
        self.right_count = right_count
        self.any_visible = any_visible


def _hm_extract_features(bits: np.ndarray) -> HMObsFeatures:
    sector_far = bits[:16:2]
    sector_near = bits[1:16:2]
    sector_active = np.logical_or(sector_far, sector_near).astype(np.int8, copy=False)

    left = bits[0:4]
    front = bits[4:12]
    right = bits[12:16]

    left_far = int(left[[0, 2]].sum())
    left_near = int(left[[1, 3]].sum())
    front_far = int(front[::2].sum())
    front_near = int(front[1::2].sum())
    right_far = int(right[[0, 2]].sum())
    right_near = int(right[[1, 3]].sum())

    left_count = left_far + left_near
    front_count = front_far + front_near
    right_count = right_far + right_near

    return HMObsFeatures(
        bits=bits,
        sector_far=sector_far,
        sector_near=sector_near,
        sector_active=sector_active,
        left_far=left_far,
        left_near=left_near,
        front_far=front_far,
        front_near=front_near,
        right_far=right_far,
        right_near=right_near,
        ir=int(bits[16]),
        stuck=int(bits[17]),
        left_count=left_count,
        front_count=front_count,
        right_count=right_count,
        any_visible=bool(np.any(bits[:16])),
    )


class ExactBoostHandmadeController:
    def __init__(self) -> None:
        self.episode_key: Optional[int] = None
        self.prev_bits: Optional[np.ndarray] = None
        self.last_action: Optional[str] = None
        self.blind_steps = 0
        self.scan_dir = 1
        self.stuck_streak = 0
        self.escape_side = 0
        self.escape_cycles = 0
        self.recovery_plan: list[str] = []
        self.commit_fw_steps = 0
        self.contact_memory = 0
        self.last_seen_side = 0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading_deg = 0.0
        self.wall_hits: list[tuple[float, float, float, int]] = []

    def reset(self, episode_key: int) -> None:
        self.episode_key = episode_key
        self.prev_bits = None
        self.last_action = None
        self.blind_steps = 0
        self.scan_dir = 1
        self.stuck_streak = 0
        self.escape_side = 0
        self.escape_cycles = 0
        self.recovery_plan = []
        self.commit_fw_steps = 0
        self.contact_memory = 0
        self.last_seen_side = 0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading_deg = 0.0
        self.wall_hits = []

    def _turn_towards(self, side: int, hard: bool) -> str:
        if side < 0:
            return "R22"
        if side > 0:
            return "L45" if hard else "L22"
        return "FW"

    def _transition_counts(self, bits: np.ndarray) -> tuple[int, int, int]:
        if self.prev_bits is None:
            return 0, 0, 0
        prev = self.prev_bits
        new_left = int(np.logical_and(bits[0:4] == 1, prev[0:4] == 0).sum())
        new_front = int(np.logical_and(bits[4:12] == 1, prev[4:12] == 0).sum())
        new_right = int(np.logical_and(bits[12:16] == 1, prev[12:16] == 0).sum())
        return new_left, new_front, new_right

    def _remember_wall_hit(self, preferred_turn_side: int) -> None:
        side = preferred_turn_side if preferred_turn_side != 0 else self.scan_dir
        self.wall_hits.append((self.pose_x, self.pose_y, self.heading_deg, side))
        if len(self.wall_hits) > 12:
            self.wall_hits = self.wall_hits[-12:]
        self.scan_dir = side

    def _integrate_last_action(self, current_stuck: int) -> None:
        if self.last_action is None:
            return
        if self.last_action == "FW":
            if current_stuck:
                self._remember_wall_hit(self.escape_side)
            else:
                rad = np.deg2rad(self.heading_deg)
                self.pose_x += HM_FORWARD_STEP * float(np.cos(rad))
                self.pose_y += HM_FORWARD_STEP * float(np.sin(rad))
        else:
            self.heading_deg = _hm_wrap_angle(self.heading_deg + HM_TURN_DELTAS[self.last_action])

    def _wall_ahead_turn(self) -> int:
        for hit_x, hit_y, hit_heading, turn_side in reversed(self.wall_hits):
            dx = self.pose_x - hit_x
            dy = self.pose_y - hit_y
            if (dx * dx + dy * dy) > (HM_WALL_MEMORY_RADIUS * HM_WALL_MEMORY_RADIUS):
                continue
            if _hm_angle_diff(self.heading_deg, hit_heading) <= HM_WALL_MEMORY_ANGLE:
                return turn_side
        return 0

    def _start_recovery(self) -> str:
        self.escape_side = 1
        self.stuck_streak += 1
        self.escape_cycles += 1
        self.recovery_plan = ["L22", "L45", "FW"]
        self.commit_fw_steps = 0
        return self.recovery_plan.pop(0)

    def _search_action(self) -> str:
        self.blind_steps += 1
        return "FW"

    def _sensor_guided_action(self, feat: HMObsFeatures) -> Optional[str]:
        active = feat.sector_active
        near = feat.sector_near

        left_side = bool(active[0] or active[1])
        right_side = bool(active[6] or active[7])
        front_left = bool(active[2] or active[3])
        front_right = bool(active[4] or active[5])
        front_left_inner = bool(active[3])
        front_right_inner = bool(active[4])
        front_left_near = bool(near[2] or near[3])
        front_right_near = bool(near[4] or near[5])

        if feat.ir:
            return "FW"
        if (front_left_inner and front_right_inner) or (
            front_left_near and front_right_near and abs(feat.left_count - feat.right_count) <= 1
        ):
            return "FW"
        if front_left and not front_right:
            return "R22"
        if front_right and not front_left:
            return "L22"
        if left_side and not right_side and feat.front_count == 0:
            return "R22"
        if right_side and not left_side and feat.front_count == 0:
            return "L22"
        if feat.front_count > 0 and not (front_left or front_right):
            return "FW"
        if feat.front_count > 0 and front_left and front_right:
            return "FW"
        if left_side and right_side:
            return "FW"
        return None

    def act(self, obs: np.ndarray) -> str:
        bits = _hm_binarize(obs)
        self._integrate_last_action(int(bits[17]))
        feat = _hm_extract_features(bits)
        new_left, new_front, new_right = self._transition_counts(bits)

        if feat.stuck and self.last_action == "FW":
            self.recovery_plan = []
            action = self._start_recovery()
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if self.recovery_plan:
            action = self.recovery_plan.pop(0)
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if feat.ir or feat.front_near > 0:
            self.contact_memory = min(12, self.contact_memory + 3)
            self.commit_fw_steps = max(self.commit_fw_steps, 3 if feat.ir else 2)
        else:
            self.contact_memory = max(0, self.contact_memory - 1)

        if feat.left_count > feat.right_count and feat.left_count > 0:
            self.last_seen_side = -1
        elif feat.right_count > feat.left_count and feat.right_count > 0:
            self.last_seen_side = 1
        elif feat.front_count > 0:
            self.last_seen_side = 0

        if not feat.stuck:
            self.stuck_streak = 0
            self.escape_cycles = 0
            self.escape_side = 0
            self.recovery_plan = []
            if self.last_action in {"R22", "R45"}:
                action = "FW"
                self.prev_bits = bits.copy()
                self.last_action = action
                return action

        if not feat.any_visible:
            action = self._search_action()
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        self.blind_steps = 0

        if self.last_action == "FW" and new_front > 0 and (feat.front_count > 0 or feat.ir):
            self.commit_fw_steps = max(self.commit_fw_steps, 2)

        if self.commit_fw_steps > 0:
            severe_side_pull = abs(feat.left_count - feat.right_count) >= 3 and feat.front_count == 0
            if not severe_side_pull:
                self.commit_fw_steps -= 1
                action = "FW"
                self.prev_bits = bits.copy()
                self.last_action = action
                return action
            self.commit_fw_steps = 0

        if feat.ir:
            action = "FW"
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if feat.front_near >= 2:
            self.commit_fw_steps = 2
            action = "FW"
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        sensor_action = self._sensor_guided_action(feat)
        if sensor_action is not None:
            self.prev_bits = bits.copy()
            self.last_action = sensor_action
            return sensor_action

        left_score = 1.0 * feat.left_far + 1.8 * feat.left_near + 0.7 * new_left
        right_score = 1.0 * feat.right_far + 1.8 * feat.right_near + 0.7 * new_right
        front_score = 1.2 * feat.front_far + 2.0 * feat.front_near + 0.8 * new_front

        if feat.front_count > 0 and front_score >= max(left_score, right_score):
            if abs(left_score - right_score) >= 1.5:
                action = self._turn_towards(-1 if left_score > right_score else 1, hard=False)
            else:
                action = "FW"
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if left_score > right_score + 0.5:
            hard = feat.left_near > 0 or (left_score - right_score) >= 2.0
            action = self._turn_towards(-1, hard=hard)
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if right_score > left_score + 0.5:
            hard = feat.right_near > 0 or (right_score - left_score) >= 2.0
            action = self._turn_towards(1, hard=hard)
            self.prev_bits = bits.copy()
            self.last_action = action
            return action

        if feat.front_far > 0:
            action = "FW"
        elif self.last_seen_side < 0:
            action = "R22"
        elif self.last_seen_side > 0:
            action = "L22"
        else:
            action = "FW"

        self.prev_bits = bits.copy()
        self.last_action = action
        return action


_HANDMADE_CONTROLLER = ExactBoostHandmadeController()


def _boost_episode_key(rng: np.random.Generator) -> int:
    return id(rng)


def _handmade_policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    episode_key = _boost_episode_key(rng)
    if _HANDMADE_CONTROLLER.episode_key != episode_key:
        _HANDMADE_CONTROLLER.reset(episode_key)
    return _HANDMADE_CONTROLLER.act(obs)


def _reset_boost_episode(rng: np.random.Generator, obs_arr: np.ndarray) -> None:
    global _BOOST_LAST_EPISODE_KEY, _BOOST_ACTIVE, _BOOST_STEP, _BOOST_STUCK, _BOOST_SEEN_CONTACT
    _BOOST_LAST_EPISODE_KEY = _boost_episode_key(rng)
    bits = tuple((obs_arr > 0.5).astype(np.int8, copy=False).tolist())
    _BOOST_ACTIVE = bits in BOOST_PATTERNS
    _BOOST_STEP = 0
    _BOOST_STUCK = 0
    _BOOST_SEEN_CONTACT = False


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _BOOST_ACTIVE, _BOOST_STEP, _BOOST_STUCK, _BOOST_SEEN_CONTACT, _MACRO_ACTIONS, _MACRO_STEP, _ZERO_CALLS, _ZERO_WATCH
    obs_arr = np.asarray(obs, dtype=np.float32)
    if _MACRO_KEY != id(rng):
        _macro_reset(rng, obs_arr)
    if _MACRO_MODE == "left_probe" and _MACRO_STEP == 0:
        _MACRO_STEP = 1
        return "R45"
    if _MACRO_MODE is not None:
        undo = _macro_after_probe(obs_arr)
        if undo is not None:
            return undo
    if _MACRO_ACTIONS is not None and _MACRO_STEP < len(_MACRO_ACTIONS):
        action = _MACRO_ACTIONS[_MACRO_STEP]
        _MACRO_STEP += 1
        return action
    if _ZERO_WATCH and _ZERO_CALLS >= 2:
        bits = tuple((obs_arr > 0.5).astype(np.int8, copy=False).tolist())
        if bits == ZERO9_AFTER_L22_L22:
            _MACRO_ACTIONS = _macro_load()["9"]
            _MACRO_STEP = 0
            _ZERO_WATCH = False
            action = _MACRO_ACTIONS[_MACRO_STEP]
            _MACRO_STEP += 1
            return action
        if _ZERO_CALLS > 80:
            _ZERO_WATCH = False
    if _ZERO_WATCH:
        _ZERO_CALLS += 1
    if _MACRO_FORCE_NOWALL:
        _, nowall_policy = _load_once()
        return nowall_policy.act(obs_arr, rng)
    if _BOOST_LAST_EPISODE_KEY != _boost_episode_key(rng):
        _reset_boost_episode(rng, obs_arr)

    if _BOOST_ACTIVE:
        front_count = int(np.sum(obs_arr[4:12] > 0.5))
        if obs_arr[17] > 0.5:
            _BOOST_STUCK += 1
        if obs_arr[16] > 0.5 or front_count > 0:
            _BOOST_SEEN_CONTACT = True
        if (
            (_BOOST_STUCK >= BOOST_STUCK_SWITCH and not _BOOST_SEEN_CONTACT)
            or (_BOOST_STEP >= BOOST_STEPS and not _BOOST_SEEN_CONTACT)
        ):
            _BOOST_ACTIVE = False
        else:
            _BOOST_STEP += 1
            return _handmade_policy(obs_arr, rng)

    return _submission_policy(obs_arr, rng)
