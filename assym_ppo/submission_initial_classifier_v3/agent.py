from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_DIM = len(ACTIONS)
OBS_DIM = 18
FW_ACTION_INDEX = ACTIONS.index("FW")
DEFAULT_WEIGHT_NAME = "weights.pth"

GRU_TURN_DELTAS = torch.tensor([45.0, 22.5, 0.0, -22.5, -45.0], dtype=torch.float32)
HANDMADE_TURN_DELTAS = {"L45": 45.0, "L22": 22.5, "FW": 0.0, "R22": -22.5, "R45": -45.0}
FORWARD_STEP = 5.0
WALL_MEMORY_RADIUS = 45.0
WALL_MEMORY_ANGLE = 40.0

BLIND_STUCK_SWITCH_STEP = 12
BLIND_VISIBLE_SWITCH_STEP = 60


def _episode_key(rng: np.random.Generator) -> int:
    seed_seq = getattr(rng.bit_generator, "_seed_seq", None)
    if seed_seq is not None and hasattr(seed_seq, "entropy"):
        return int(seed_seq.entropy)
    return id(rng)


def _checkpoint_path() -> str:
    here = os.path.dirname(__file__)
    override = os.environ.get("OBELIX_WEIGHTS")
    if override:
        return override if os.path.isabs(override) else os.path.join(here, override)
    return os.path.join(here, DEFAULT_WEIGHT_NAME)


def _binarize(obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float32) > 0.5).astype(np.int8, copy=False)


def _wrap_angle(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _angle_diff(a: float, b: float) -> float:
    return abs(_wrap_angle(a - b))


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def wrap_angle_deg(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + 180.0, 360.0) - 180.0


class ObsFeatures:
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


def _extract_features(bits: np.ndarray) -> ObsFeatures:
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

    return ObsFeatures(
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


class HandmadeController:
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
                self.pose_x += FORWARD_STEP * float(np.cos(rad))
                self.pose_y += FORWARD_STEP * float(np.sin(rad))
        else:
            self.heading_deg = _wrap_angle(self.heading_deg + HANDMADE_TURN_DELTAS[self.last_action])

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

    def _sensor_guided_action(self, feat: ObsFeatures) -> Optional[str]:
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
        if feat.front_count > 0:
            return "FW"
        if left_side and right_side:
            return "FW"
        return None

    def act(self, obs: np.ndarray, episode_key: int) -> str:
        if self.episode_key != episode_key:
            self.reset(episode_key)

        bits = _binarize(obs)
        self._integrate_last_action(int(bits[17]))
        feat = _extract_features(bits)
        new_left, new_front, new_right = self._transition_counts(bits)

        if feat.stuck and self.last_action == "FW":
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
        return OBS_DIM + OBS_DIM + 4 + 3 + self.last_action_hist * ACTION_DIM + 1 + self.heading_bins + 1 + 1


class PoseGRUFeatureTracker:
    def __init__(self, num_envs: int, config: FeatureConfig, device: torch.device) -> None:
        self.num_envs = int(num_envs)
        self.config = config
        self.device = device
        self.turn_deltas = GRU_TURN_DELTAS.to(device=self.device)

        self.current_obs = torch.zeros((self.num_envs, OBS_DIM), dtype=torch.float32, device=self.device)
        self.prev_obs = torch.zeros((self.num_envs, OBS_DIM), dtype=torch.float32, device=self.device)
        self.last_action_hist = torch.zeros(
            (self.num_envs, self.config.last_action_hist, ACTION_DIM), dtype=torch.float32, device=self.device
        )
        self.x_rel = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.y_rel = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.theta_deg = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.blind_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.stuck_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.contact_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.same_obs_count = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.wall_hit_count_by_heading_bin = torch.zeros(
            (self.num_envs, self.config.heading_bins), dtype=torch.float32, device=self.device
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
        same_obs = torch.clamp(self.same_obs_count / max(1.0, float(self.config.same_obs_clip)), 0.0, 1.0).unsqueeze(1)
        wall_hits = torch.clamp(
            self.wall_hit_count_by_heading_bin / max(1.0, float(self.config.wall_hit_clip)), 0.0, 1.0
        )
        side = self.last_seen_side.unsqueeze(1)
        front = self.last_seen_front_strength.unsqueeze(1)
        return torch.cat([self.current_obs, delta_obs, pose, counters, last_actions, same_obs, wall_hits, side, front], dim=1)

    def post_step(self, actions: torch.Tensor, next_obs: np.ndarray | torch.Tensor, dones: np.ndarray | torch.Tensor) -> None:
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
            layers.append(layer_init(nn.Linear(last_dim, int(hidden_dim))))
            layers.append(nn.Tanh())
            last_dim = int(hidden_dim)
        self.encoder = nn.Sequential(*layers)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.gru = nn.GRU(last_dim, self.gru_hidden_dim)
        self.policy_head = layer_init(nn.Linear(self.gru_hidden_dim, int(action_dim)), std=0.01)
        self.value_head = layer_init(nn.Linear(self.gru_hidden_dim, 1), std=1.0)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((1, int(batch_size), self.gru_hidden_dim), dtype=torch.float32, device=device)

    def forward_step(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
        starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if starts is not None:
            starts_t = starts.to(dtype=torch.float32, device=hidden.device).view(1, -1, 1)
            hidden = hidden * (1.0 - starts_t)
        encoded = self.encoder(features)
        out, next_hidden = self.gru(encoded.unsqueeze(0), hidden)
        logits = self.policy_head(out.squeeze(0))
        return logits, next_hidden


class GRUWallPolicy:
    def __init__(self) -> None:
        self.model: Optional[GRUPolicy] = None
        self.tracker: Optional[PoseGRUFeatureTracker] = None
        self.hidden: Optional[torch.Tensor] = None
        self.last_episode_key: Optional[int] = None
        self.pending_action: Optional[int] = None
        self.open_seq: list[str] = []
        self.step = 0

    def _load_once(self) -> None:
        if self.model is not None and self.tracker is not None and self.hidden is not None:
            return

        checkpoint = torch.load(_checkpoint_path(), map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Unsupported checkpoint format")
        if "state_dict" not in checkpoint:
            checkpoint = {"state_dict": checkpoint}

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

        self.model = GRUPolicy(
            input_dim=feature_config.feature_dim,
            encoder_dims=encoder_dims,
            gru_hidden_dim=gru_hidden_dim,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.tracker = PoseGRUFeatureTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
        self.hidden = self.model.initial_state(1, torch.device("cpu"))

    def _reset_episode(self, obs_arr: np.ndarray, episode_key: int) -> None:
        assert self.tracker is not None and self.model is not None
        self.tracker.reset_all(obs_arr[None, :])
        self.hidden = self.model.initial_state(1, torch.device("cpu"))
        self.pending_action = None
        self.last_episode_key = episode_key
        self.step = 0
        front = int(np.sum(obs_arr[4:12] > 0.5))
        if front >= 6:
            self.open_seq = ["FW"] * 6
        elif front > 0:
            self.open_seq = ["FW"] * 4
        else:
            self.open_seq = []

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator, episode_key: int) -> str:
        self._load_once()
        assert self.model is not None and self.tracker is not None and self.hidden is not None

        obs_arr = np.asarray(obs, dtype=np.float32)
        if self.last_episode_key != episode_key:
            self._reset_episode(obs_arr, episode_key)
            starts = torch.ones((1,), dtype=torch.float32)
        else:
            if self.pending_action is not None:
                self.tracker.post_step(
                    actions=torch.tensor([self.pending_action], dtype=torch.long),
                    next_obs=obs_arr[None, :],
                    dones=np.asarray([False]),
                )
            starts = torch.zeros((1,), dtype=torch.float32)

        features = self.tracker.features()
        logits, self.hidden = self.model.forward_step(features, self.hidden, starts)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)
        probs /= probs.sum()
        action_idx = int(rng.choice(len(ACTIONS), p=probs))
        action = ACTIONS[action_idx]

        if self.step < len(self.open_seq):
            action = self.open_seq[self.step]
            action_idx = ACTIONS.index(action)

        self.pending_action = action_idx
        self.step += 1
        return action


class HybridAgent:
    def __init__(self) -> None:
        self.handmade = HandmadeController()
        self.wall = GRUWallPolicy()
        self.last_episode_key: Optional[int] = None
        self.mode = "handmade"
        self.initial_classified = False
        self.blind_start = False
        self.seen_visible = False
        self.step = 0
        self.blind_stuck_before_visible = False

    def _reset_episode(self, episode_key: int) -> None:
        self.last_episode_key = episode_key
        self.mode = "handmade"
        self.initial_classified = False
        self.blind_start = False
        self.seen_visible = False
        self.step = 0
        self.blind_stuck_before_visible = False

    def _classify_initial(self, obs_arr: np.ndarray) -> None:
        left_count = int(np.sum(obs_arr[:4] > 0.5))
        front_count = int(np.sum(obs_arr[4:12] > 0.5))
        right_count = int(np.sum(obs_arr[12:16] > 0.5))

        if front_count > 0 or right_count > 0 or left_count >= 4 or (left_count > 0 and right_count > 0):
            self.mode = "wall"
        else:
            self.mode = "handmade"
        self.blind_start = (left_count + front_count + right_count) == 0
        self.initial_classified = True

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> str:
        episode_key = _episode_key(rng)
        if self.last_episode_key != episode_key:
            self._reset_episode(episode_key)

        obs_arr = np.asarray(obs, dtype=np.float32)
        if not self.initial_classified:
            self._classify_initial(obs_arr)

        currently_visible = bool(np.any(obs_arr[:16] > 0.5))
        if currently_visible and self.blind_start and self.blind_stuck_before_visible and self.step <= BLIND_VISIBLE_SWITCH_STEP:
            self.mode = "wall"
        if currently_visible:
            self.seen_visible = True

        if (
            self.mode == "handmade"
            and self.blind_start
            and not self.seen_visible
            and obs_arr[17] > 0.5
            and self.step <= BLIND_STUCK_SWITCH_STEP
        ):
            self.mode = "wall"
        elif self.mode == "handmade" and self.blind_start and not self.seen_visible and obs_arr[17] > 0.5:
            self.blind_stuck_before_visible = True

        if self.mode == "wall":
            action = self.wall.act(obs_arr, rng, episode_key)
        else:
            action = self.handmade.act(obs_arr, episode_key)

        self.step += 1
        return action


_AGENT = HybridAgent()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    return _AGENT.act(obs, rng)
