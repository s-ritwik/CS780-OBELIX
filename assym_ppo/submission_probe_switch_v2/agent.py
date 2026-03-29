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
SENSOR_MEMORY_DIM = 17
RAW_OBS_DIM = 18
ENCODED_OBS_DIM = 32

PROBE_STEPS = 80
CONTACT_THRESHOLD = 15
BLIND_THRESHOLD = 45
FRONT_TOTAL_THRESHOLD = 20


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
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64, copy=False)
        probs /= probs.sum()
        action_idx = int(rng.choice(len(ACTIONS), p=probs))
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


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
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
