from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
OBS_DIM = 18
ACTION_DIM = len(ACTIONS)
SENSOR_MEMORY_DIM = 17
META_DIM = 8


@dataclass
class FeatureConfig:
    obs_stack: int = 8
    action_hist: int = 4
    max_steps: int = 300
    contact_clip: float = 12.0
    blind_clip: float = 50.0
    stuck_clip: float = 10.0
    repeat_clip: float = 20.0
    turn_clip: float = 12.0
    stuck_memory_clip: float = 12.0

    @property
    def feature_dim(self) -> int:
        return self.obs_stack * OBS_DIM + self.action_hist * ACTION_DIM + SENSOR_MEMORY_DIM + META_DIM


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

    def reset_indices(self, env_indices: torch.Tensor, obs: np.ndarray | torch.Tensor) -> None:
        if env_indices.numel() == 0:
            return
        obs_t = self._obs_to_tensor(obs)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        self.obs_hist[env_indices] = 0.0
        self.action_hist[env_indices] = 0.0
        self.blind_steps[env_indices] = 0.0
        self.stuck_steps[env_indices] = 0.0
        self.step_count[env_indices] = 0.0
        self.last_seen_side[env_indices] = 0.0
        self.contact_memory[env_indices] = 0.0
        self.sensor_seen[env_indices] = 0.0
        self.repeat_obs_steps[env_indices] = 0.0
        self.blind_turn_steps[env_indices] = 0.0
        self.stuck_memory[env_indices] = 0.0
        self.obs_hist[env_indices, -1] = obs_t
        self.sensor_seen[env_indices] = (obs_t[:, :SENSOR_MEMORY_DIM] > 0.5).to(torch.float32)
        self.stuck_memory[env_indices] = torch.where(
            obs_t[:, 17] > 0.5,
            torch.ones_like(self.stuck_memory[env_indices]),
            self.stuck_memory[env_indices],
        )
        self._seed_meta(env_indices, obs_t)

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
        action_one_hot = F.one_hot(actions.to(torch.long), num_classes=ACTION_DIM).to(torch.float32)

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
            self.reset_indices(done_idx, next_obs_t[done_idx])
