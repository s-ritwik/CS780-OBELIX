from __future__ import annotations

from dataclasses import asdict, dataclass

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
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def wrap_angle_deg(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + 180.0, 360.0) - 180.0


@dataclass
class FeatureConfig:
    max_steps: int = 500
    pose_clip: float = 500.0
    blind_clip: float = 100.0
    stuck_clip: float = 20.0
    contact_clip: float = 20.0
    same_obs_clip: float = 50.0
    wall_hit_clip: float = 20.0
    last_action_hist: int = 5
    heading_bins: int = 8

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
        fw_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in encoder_dims:
            h = int(hidden_dim)
            if h <= 0:
                raise ValueError("encoder hidden sizes must be positive")
            layers.append(layer_init(nn.Linear(last_dim, h)))
            layers.append(nn.Tanh())
            last_dim = h
        self.encoder = nn.Sequential(*layers)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.gru = nn.GRU(last_dim, self.gru_hidden_dim)
        self.policy_head = layer_init(nn.Linear(self.gru_hidden_dim, int(action_dim)), std=0.01)
        self.value_head = layer_init(nn.Linear(self.gru_hidden_dim, 1), std=1.0)
        apply_forward_bias(self.policy_head, fw_bias_init)

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

    def act(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
        starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, next_hidden = self.forward_step(features, hidden, starts)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, logits, next_hidden

    def evaluate_sequence(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        hidden0: torch.Tensor,
        starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len, batch_size, _ = features.shape
        hidden = hidden0
        logits_list = []
        values_list = []
        log_prob_list = []
        entropy_list = []

        for t in range(seq_len):
            logits_t, value_t, hidden = self.forward_step(features[t], hidden, starts[t])
            dist_t = Categorical(logits=logits_t)
            action_t = actions[t]
            logits_list.append(logits_t)
            values_list.append(value_t)
            log_prob_list.append(dist_t.log_prob(action_t))
            entropy_list.append(dist_t.entropy())

        logits = torch.stack(logits_list, dim=0)
        values = torch.stack(values_list, dim=0)
        log_probs = torch.stack(log_prob_list, dim=0)
        entropy = torch.stack(entropy_list, dim=0)
        return log_probs, entropy, values, logits, hidden


class RecurrentRolloutBuffer:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        feat_dim: int,
        hidden_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.device = device

        self.features = torch.zeros((self.num_steps, self.num_envs, self.feat_dim), dtype=torch.float32, device=device)
        self.starts = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.hidden = torch.zeros((self.num_steps, self.num_envs, self.hidden_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long, device=device)
        self.teacher_actions = torch.full(
            (self.num_steps, self.num_envs),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
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
        features: torch.Tensor,
        starts: torch.Tensor,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        teacher_actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        self.features[step].copy_(features)
        self.starts[step].copy_(starts)
        self.hidden[step].copy_(hidden.squeeze(0))
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
            if step == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
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
                feat = torch.zeros((seq_len, batch_size, self.feat_dim), dtype=torch.float32, device=self.device)
                starts = torch.zeros((seq_len, batch_size), dtype=torch.float32, device=self.device)
                h0 = torch.zeros((1, batch_size, self.hidden_dim), dtype=torch.float32, device=self.device)
                actions = torch.zeros((seq_len, batch_size), dtype=torch.long, device=self.device)
                teacher_actions = torch.full((seq_len, batch_size), -1, dtype=torch.long, device=self.device)
                old_log_probs = torch.zeros((seq_len, batch_size), dtype=torch.float32, device=self.device)
                old_values = torch.zeros((seq_len, batch_size), dtype=torch.float32, device=self.device)
                returns = torch.zeros((seq_len, batch_size), dtype=torch.float32, device=self.device)
                advantages = torch.zeros((seq_len, batch_size), dtype=torch.float32, device=self.device)
                old_logits = torch.zeros((seq_len, batch_size, self.action_dim), dtype=torch.float32, device=self.device)

                for b, (t0, env_id) in enumerate(batch_items):
                    sl = slice(t0, t0 + seq_len)
                    feat[:, b] = self.features[sl, env_id]
                    starts[:, b] = self.starts[sl, env_id]
                    h0[:, b] = self.hidden[t0, env_id].unsqueeze(0)
                    actions[:, b] = self.actions[sl, env_id]
                    teacher_actions[:, b] = self.teacher_actions[sl, env_id]
                    old_log_probs[:, b] = self.log_probs[sl, env_id]
                    old_values[:, b] = self.values[sl, env_id]
                    returns[:, b] = self.returns[sl, env_id]
                    advantages[:, b] = self.advantages[sl, env_id]
                    old_logits[:, b] = self.logits[sl, env_id]

                yield feat, starts, h0, actions, teacher_actions, old_log_probs, old_values, returns, advantages, old_logits


def categorical_kl(old_logits: torch.Tensor, new_logits: torch.Tensor) -> torch.Tensor:
    old_log_probs = F.log_softmax(old_logits, dim=-1)
    new_log_probs = F.log_softmax(new_logits, dim=-1)
    old_probs = old_log_probs.exp()
    return torch.sum(old_probs * (old_log_probs - new_log_probs), dim=-1)


def build_policy_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def make_checkpoint(
    model: nn.Module,
    feature_config: FeatureConfig,
    encoder_dims: tuple[int, ...],
    gru_hidden_dim: int,
    args,
    best_eval: float,
) -> dict:
    return {
        "state_dict": build_policy_state_dict(model),
        "actions": ACTIONS,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "encoder_dims": [int(h) for h in encoder_dims],
        "gru_hidden_dim": int(gru_hidden_dim),
        "feature_config": asdict(feature_config),
        "best_eval": float(best_eval),
        "config": vars(args),
    }


def save_checkpoint(path: str, checkpoint: dict) -> None:
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    raw = torch.load(path, map_location=device)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {path}")
    if "state_dict" not in raw:
        raw = {"state_dict": raw}
    return raw
