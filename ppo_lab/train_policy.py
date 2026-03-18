from __future__ import annotations

import argparse
import importlib.util
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
OBS_DIM = 18
ACTION_DIM = len(ACTIONS)
SENSOR_MEMORY_DIM = 17
META_DIM = 8


def import_symbol(py_file: str, symbol: str):
    spec = importlib.util.spec_from_file_location("obelix_module", py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {symbol} from {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise AttributeError(f"{symbol} not found in {py_file}")
    return getattr(module, symbol)


def import_module(py_file: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def apply_forward_bias(layer: nn.Linear, fw_bias_init: float) -> None:
    if fw_bias_init == 0.0:
        return
    with torch.no_grad():
        layer.bias[ACTION_TO_INDEX["FW"]] += float(fw_bias_init)


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
    def __init__(
        self,
        num_envs: int,
        config: FeatureConfig,
        device: torch.device,
    ) -> None:
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
        self.stuck_memory.copy_(torch.where(obs_t[:, 17] > 0.5, torch.ones_like(self.stuck_memory), self.stuck_memory))
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
            same_obs = torch.all((prev_obs[:, :SENSOR_MEMORY_DIM] > 0.5) == (next_obs_t[active_idx, :SENSOR_MEMORY_DIM] > 0.5), dim=1)
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


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], fw_bias_init: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            h = int(hidden_dim)
            if h <= 0:
                raise ValueError("hidden layer sizes must be positive")
            layers.append(layer_init(nn.Linear(last_dim, h)))
            layers.append(nn.Tanh())
            last_dim = h

        self.backbone = nn.Sequential(*layers)
        self.policy_head = layer_init(nn.Linear(last_dim, ACTION_DIM), std=0.01)
        self.value_head = layer_init(nn.Linear(last_dim, 1), std=1.0)
        apply_forward_bias(self.policy_head, fw_bias_init)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value

    def act(self, x: torch.Tensor):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, logits

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value, logits


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, feat_dim: int, device: torch.device) -> None:
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.feat_dim = int(feat_dim)
        self.device = device

        self.features = torch.zeros(
            (self.num_steps, self.num_envs, self.feat_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.logits = torch.zeros(
            (self.num_steps, self.num_envs, ACTION_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self.advantages = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)

    def add(
        self,
        step: int,
        features: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        self.features[step].copy_(features)
        self.actions[step].copy_(actions)
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
            flat_adv = self.advantages.reshape(-1)
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)
            self.advantages.copy_(flat_adv.view_as(self.advantages))

    def mini_batch_generator(self, minibatch_size: int, num_learning_epochs: int):
        batch_size = self.num_steps * self.num_envs
        mb_size = min(int(minibatch_size), batch_size)

        features = self.features.reshape(batch_size, self.feat_dim)
        actions = self.actions.reshape(batch_size)
        old_log_probs = self.log_probs.reshape(batch_size)
        old_values = self.values.reshape(batch_size)
        returns = self.returns.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        old_logits = self.logits.reshape(batch_size, ACTION_DIM)

        for _ in range(int(num_learning_epochs)):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mb_size):
                mb_idx = indices[start : start + mb_size]
                yield (
                    features[mb_idx],
                    actions[mb_idx],
                    old_log_probs[mb_idx],
                    old_values[mb_idx],
                    returns[mb_idx],
                    advantages[mb_idx],
                    old_logits[mb_idx],
                )


def categorical_kl(old_logits: torch.Tensor, new_logits: torch.Tensor) -> torch.Tensor:
    old_log_probs = F.log_softmax(old_logits, dim=-1)
    new_log_probs = F.log_softmax(new_logits, dim=-1)
    old_probs = old_log_probs.exp()
    return torch.sum(old_probs * (old_log_probs - new_log_probs), dim=-1)


def build_policy_state_dict(model: nn.Module) -> dict:
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def make_checkpoint(
    model: nn.Module,
    feature_config: FeatureConfig,
    hidden_dims: tuple[int, ...],
    args: argparse.Namespace,
    best_eval: float,
) -> dict:
    return {
        "state_dict": build_policy_state_dict(model),
        "actions": ACTIONS,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "hidden_dims": [int(h) for h in hidden_dims],
        "feature_config": asdict(feature_config),
        "best_eval": float(best_eval),
        "config": vars(args),
    }


def save_checkpoint(path: str, checkpoint: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    raw = torch.load(path, map_location=device)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {path}")
    if "state_dict" not in raw:
        raw = {"state_dict": raw}
    return raw


class GreedyRunner:
    def __init__(
        self,
        model: ActorCritic,
        feature_config: FeatureConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.device = device
        self.tracker = FeatureTracker(num_envs=1, config=feature_config, device=device)
        self.pending_action: Optional[int] = None
        self.started = False

    def reset(self, obs: np.ndarray) -> None:
        self.tracker.reset_all(np.asarray(obs, dtype=np.float32)[None, :])
        self.pending_action = None
        self.started = True

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if not self.started:
            self.reset(obs_arr)
        elif self.pending_action is not None:
            self.tracker.post_step(
                actions=torch.tensor([self.pending_action], dtype=torch.long, device=self.device),
                next_obs=obs_arr[None, :],
                dones=np.asarray([False]),
            )

        features = self.tracker.features()
        logits, _ = self.model(features)
        action_idx = int(torch.argmax(logits, dim=1).item())
        self.pending_action = action_idx
        return action_idx


def collect_expert_demos(
    *,
    expert: str,
    episodes: int,
    seed: int,
    obelix_py: str,
    env_kwargs: dict,
    feature_config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if episodes <= 0 or expert == "none":
        return np.empty((0, feature_config.feature_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)

    OBELIX = import_symbol(obelix_py, "OBELIX")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    handmade_path = os.path.join(base_dir, "handmade", "agent.py")
    paper_path = os.path.join(base_dir, "paper_algo", "agent.py")

    expert_name = expert
    expert_policy = None
    handmade_cls = None
    if expert == "handmade":
        handmade_mod = import_module(handmade_path, "handmade_agent")
        handmade_cls = getattr(handmade_mod, "HandmadeController")
    elif expert == "paper":
        paper_mod = import_module(paper_path, "paper_agent")
        expert_policy = getattr(paper_mod, "policy")
    else:
        expert_path = os.path.abspath(expert)
        expert_mod = import_module(expert_path, "expert_agent")
        expert_policy = getattr(expert_mod, "policy")
        expert_name = os.path.basename(expert_path)

    tracker = FeatureTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
    features: list[np.ndarray] = []
    actions: list[int] = []
    returns = deque(maxlen=50)

    print(f"[warm_start] collecting {episodes} expert episodes from {expert_name}")
    for episode in range(int(episodes)):
        episode_seed = seed + 500_000 + episode
        env = OBELIX(seed=episode_seed, **env_kwargs)
        obs = env.reset(seed=episode_seed)
        tracker.reset_all(np.asarray(obs, dtype=np.float32)[None, :])

        rng = np.random.default_rng(episode_seed)
        handmade = handmade_cls() if handmade_cls is not None else None
        if handmade is not None:
            handmade.reset(episode_seed)

        total_reward = 0.0
        done = False
        while not done:
            feat = tracker.features().cpu().numpy()[0]
            if handmade is not None:
                action_name = handmade.act(obs)
            else:
                action_name = expert_policy(obs, rng)
            action_idx = ACTION_TO_INDEX[action_name]
            features.append(feat.astype(np.float32, copy=True))
            actions.append(action_idx)

            obs, reward, done = env.step(action_name, render=False)
            total_reward += float(reward)
            if not done:
                tracker.post_step(
                    actions=torch.tensor([action_idx], dtype=torch.long),
                    next_obs=np.asarray(obs, dtype=np.float32)[None, :],
                    dones=np.asarray([False]),
                )

        returns.append(total_reward)
        if (episode + 1) % max(1, min(20, episodes)) == 0:
            print(
                f"[warm_start] demos={episode + 1}/{episodes} "
                f"recent_return={float(np.mean(returns)):.1f}"
            )

    feats = np.asarray(features, dtype=np.float32)
    acts = np.asarray(actions, dtype=np.int64)
    print(f"[warm_start] collected {feats.shape[0]} state-action pairs")
    return feats, acts


def warm_start_policy(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    feats: np.ndarray,
    acts: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    grad_clip: float,
) -> None:
    if feats.size == 0 or acts.size == 0 or epochs <= 0:
        return

    x = torch.as_tensor(feats, dtype=torch.float32, device=device)
    y = torch.as_tensor(acts, dtype=torch.long, device=device)
    total = int(x.shape[0])
    batch = min(int(batch_size), total)

    print(f"[warm_start] behavior cloning on {total} samples")
    for epoch in range(int(epochs)):
        perm = torch.randperm(total, device=device)
        total_loss = 0.0
        total_correct = 0
        seen = 0

        for start in range(0, total, batch):
            idx = perm[start : start + batch]
            logits, _ = model(x[idx])
            loss = F.cross_entropy(logits, y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += float(loss.item()) * idx.numel()
            total_correct += int((torch.argmax(logits, dim=1) == y[idx]).sum().item())
            seen += int(idx.numel())

        print(
            f"[warm_start] epoch={epoch + 1}/{epochs} "
            f"loss={total_loss / max(1, seen):.4f} acc={total_correct / max(1, seen):.3f}"
        )


@torch.no_grad()
def evaluate_model(
    model: ActorCritic,
    *,
    feature_config: FeatureConfig,
    obelix_py: str,
    runs: int,
    seed: int,
    env_kwargs: dict,
    device: torch.device,
) -> dict[str, float]:
    OBELIX = import_symbol(obelix_py, "OBELIX")
    runner = GreedyRunner(model=model, feature_config=feature_config, device=device)
    scores: list[float] = []
    successes = 0
    lengths: list[int] = []

    model.eval()
    for run_idx in range(int(runs)):
        episode_seed = int(seed + run_idx)
        env = OBELIX(seed=episode_seed, **env_kwargs)
        obs = env.reset(seed=episode_seed)
        runner.reset(obs)

        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action_idx = runner.act(obs)
            obs, reward, done = env.step(ACTIONS[action_idx], render=False)
            total_reward += float(reward)
            steps += 1

        scores.append(total_reward)
        lengths.append(steps)
        if total_reward >= 1000.0:
            successes += 1

    return {
        "mean_reward": float(np.mean(scores)),
        "std_reward": float(np.std(scores)),
        "success_rate": float(successes) / float(max(1, runs)),
        "mean_length": float(np.mean(lengths)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stacked PPO trainer for OBELIX")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_dir)

    parser.add_argument("--obelix_py", type=str, default=os.path.join(repo_dir, "obelix.py"))
    parser.add_argument("--obelix_torch_py", type=str, default=os.path.join(repo_dir, "obelix_torch.py"))
    parser.add_argument("--out", type=str, default=os.path.join(base_dir, "weights_best.pth"))
    parser.add_argument("--load", type=str, default=None)

    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--total_env_steps", type=int, default=4_000_000)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--env_device", type=str, default="auto")

    parser.add_argument("--obs_stack", type=int, default=8)
    parser.add_argument("--action_hist", type=int, default=4)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument(
        "--fw_bias_init",
        type=float,
        default=1.0,
        help="Initial logit bias added to the forward action.",
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--minibatch_size", type=int, default=16384)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--normalize_advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--reward_clip", type=float, default=0.0)
    parser.add_argument("--schedule", type=str, choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--desired_kl", type=float, default=0.01)
    parser.add_argument("--use_clipped_value_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--warm_start_expert", type=str, default="none")
    parser.add_argument("--warm_start_episodes", type=int, default=0)
    parser.add_argument("--warm_start_epochs", type=int, default=5)
    parser.add_argument("--warm_start_batch_size", type=int, default=8192)

    parser.add_argument("--eval_runs", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=500_000)
    parser.add_argument("--log_interval", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    hidden_dims = tuple(int(h) for h in args.hidden_dims)
    feature_config = FeatureConfig(
        obs_stack=int(args.obs_stack),
        action_hist=int(args.action_hist),
        max_steps=int(args.max_steps),
    )

    model = ActorCritic(
        input_dim=feature_config.feature_dim,
        hidden_dims=hidden_dims,
        fw_bias_init=args.fw_bias_init,
    ).to(device)
    if args.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.load:
        checkpoint = load_checkpoint(args.load, device=device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"[setup] loaded checkpoint {args.load}")

    env_device = str(device) if args.env_device == "auto" else args.env_device
    VecEnvCls = import_symbol(args.obelix_torch_py, "OBELIXVectorized")
    vec_env = VecEnvCls(
        num_envs=args.num_envs,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed * 10_000,
        device=env_device,
    )

    env_kwargs = {
        "scaling_factor": args.scaling_factor,
        "arena_size": args.arena_size,
        "max_steps": args.max_steps,
        "wall_obstacles": args.wall_obstacles,
        "difficulty": args.difficulty,
        "box_speed": args.box_speed,
    }

    print(
        f"[setup] device={device} env_device={env_device} num_envs={args.num_envs} "
        f"wall_obstacles={args.wall_obstacles} difficulty={args.difficulty}"
    )
    print(
        f"[setup] feature_dim={feature_config.feature_dim} obs_stack={feature_config.obs_stack} "
        f"action_hist={feature_config.action_hist} hidden_dims={hidden_dims}"
    )

    if args.warm_start_expert != "none" and args.warm_start_episodes > 0:
        feats, acts = collect_expert_demos(
            expert=args.warm_start_expert,
            episodes=args.warm_start_episodes,
            seed=args.seed,
            obelix_py=args.obelix_py,
            env_kwargs=env_kwargs,
            feature_config=feature_config,
        )
        warm_start_policy(
            model=model,
            optimizer=optimizer,
            feats=feats,
            acts=acts,
            device=device,
            epochs=args.warm_start_epochs,
            batch_size=args.warm_start_batch_size,
            grad_clip=args.grad_clip,
        )

    obs = vec_env.reset_all(seed=args.seed * 10_000)
    tracker = FeatureTracker(num_envs=args.num_envs, config=feature_config, device=device)
    tracker.reset_all(obs)

    episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.num_envs,), dtype=np.int32)
    recent_returns = deque(maxlen=200)
    recent_lengths = deque(maxlen=200)
    recent_successes = deque(maxlen=200)

    env_steps = 0
    update_idx = 0
    last_log_env_step = 0
    last_eval_env_step = 0
    start_time = time.time()
    best_eval = -float("inf")
    current_lr = float(args.lr)

    try:
        while env_steps < args.total_env_steps:
            buffer = RolloutBuffer(
                num_steps=args.rollout_steps,
                num_envs=args.num_envs,
                feat_dim=feature_config.feature_dim,
                device=device,
            )
            rollout_action_counts = np.zeros((ACTION_DIM,), dtype=np.int64)

            model.eval()
            for step in range(args.rollout_steps):
                features = tracker.features()
                with torch.no_grad():
                    actions_t, log_probs_t, _, values_t, logits_t = model.act(features)

                action_idx = actions_t.detach().cpu().numpy()
                rollout_action_counts += np.bincount(action_idx, minlength=ACTION_DIM)
                next_obs, rewards, dones = vec_env.step(action_idx)
                train_rewards = rewards.astype(np.float32, copy=True)
                if args.reward_scale != 1.0:
                    train_rewards /= float(args.reward_scale)
                if args.reward_clip > 0.0:
                    np.clip(train_rewards, -float(args.reward_clip), float(args.reward_clip), out=train_rewards)

                done_idx = np.flatnonzero(dones)
                if done_idx.size > 0:
                    for idx in done_idx:
                        terminal_reward = float(episode_returns[idx] + rewards[idx])
                        terminal_length = int(episode_lengths[idx] + 1)
                        recent_returns.append(terminal_reward)
                        recent_lengths.append(terminal_length)
                        recent_successes.append(int(terminal_reward >= 1000.0))

                    reset_map = vec_env.reset(
                        env_indices=done_idx.tolist(),
                        seed=args.seed * 10_000 + env_steps + step * args.num_envs,
                    )
                    for idx, reset_obs in reset_map.items():
                        next_obs[idx] = reset_obs

                buffer.add(
                    step=step,
                    features=features,
                    actions=actions_t,
                    log_probs=log_probs_t,
                    rewards=train_rewards,
                    dones=dones.astype(np.float32, copy=False),
                    values=values_t,
                    logits=logits_t,
                )

                episode_returns += rewards
                episode_lengths += 1
                if done_idx.size > 0:
                    episode_returns[done_idx] = 0.0
                    episode_lengths[done_idx] = 0

                tracker.post_step(actions=actions_t, next_obs=next_obs, dones=dones)
                env_steps += args.num_envs

            with torch.no_grad():
                next_features = tracker.features()
                _, last_values = model(next_features)

            buffer.compute_returns(
                last_values=last_values,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                normalize_advantages=args.normalize_advantages,
            )

            model.train()
            mean_policy_loss = 0.0
            mean_value_loss = 0.0
            mean_entropy = 0.0
            mean_kl = 0.0
            num_minibatches = 0

            for (
                feature_batch,
                actions_batch,
                old_log_probs_batch,
                old_values_batch,
                returns_batch,
                advantages_batch,
                old_logits_batch,
            ) in buffer.mini_batch_generator(args.minibatch_size, args.update_epochs):
                new_log_probs, entropy, values, new_logits = model.evaluate_actions(feature_batch, actions_batch)

                if args.schedule == "adaptive" and args.desired_kl > 0.0:
                    with torch.no_grad():
                        kl_mean = float(categorical_kl(old_logits_batch, new_logits).mean().item())
                        if kl_mean > args.desired_kl * 2.0:
                            current_lr = max(1e-5, current_lr / 1.5)
                        elif 0.0 < kl_mean < args.desired_kl / 2.0:
                            current_lr = min(1e-2, current_lr * 1.5)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = current_lr
                else:
                    with torch.no_grad():
                        kl_mean = float(categorical_kl(old_logits_batch, new_logits).mean().item())

                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surrogate = -advantages_batch * ratio
                surrogate_clipped = -advantages_batch * torch.clamp(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + args.clip_coef,
                )
                policy_loss = torch.max(surrogate, surrogate_clipped).mean()

                if args.use_clipped_value_loss:
                    value_clipped = old_values_batch + (values - old_values_batch).clamp(
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    value_losses = (values - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - values).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                mean_policy_loss += float(policy_loss.item())
                mean_value_loss += float(value_loss.item())
                mean_entropy += float(entropy_loss.item())
                mean_kl += kl_mean
                num_minibatches += 1

            update_idx += 1
            if num_minibatches > 0:
                mean_policy_loss /= num_minibatches
                mean_value_loss /= num_minibatches
                mean_entropy /= num_minibatches
                mean_kl /= num_minibatches

            if env_steps - last_log_env_step >= args.log_interval:
                elapsed = max(1e-6, time.time() - start_time)
                sps = env_steps / elapsed
                recent_mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
                recent_mean_length = float(np.mean(recent_lengths)) if recent_lengths else float("nan")
                recent_success = float(np.mean(recent_successes)) if recent_successes else float("nan")
                rollout_total_actions = max(1, int(np.sum(rollout_action_counts)))
                action_mix = " ".join(
                    f"{name}:{(count / rollout_total_actions):.2f}"
                    for name, count in zip(ACTIONS, rollout_action_counts.tolist())
                )
                print(
                    f"[train] update={update_idx} env_steps={env_steps} "
                    f"policy_loss={mean_policy_loss:.4f} value_loss={mean_value_loss:.4f} "
                    f"entropy={mean_entropy:.4f} kl={mean_kl:.5f} lr={current_lr:.6f} "
                    f"recent_return={recent_mean_return:.1f} recent_len={recent_mean_length:.1f} "
                    f"recent_success={recent_success:.3f} actions=[{action_mix}] "
                    f"sps={sps:.1f} elapsed={format_hms(elapsed)}"
                )
                last_log_env_step = env_steps

            if env_steps - last_eval_env_step >= args.eval_interval:
                eval_stats = evaluate_model(
                    model=model,
                    feature_config=feature_config,
                    obelix_py=args.obelix_py,
                    runs=args.eval_runs,
                    seed=args.seed + 100_000,
                    env_kwargs=env_kwargs,
                    device=device,
                )
                print(
                    f"[eval] env_steps={env_steps} mean={eval_stats['mean_reward']:.1f} "
                    f"std={eval_stats['std_reward']:.1f} success={eval_stats['success_rate']:.3f} "
                    f"mean_len={eval_stats['mean_length']:.1f}"
                )
                if eval_stats["mean_reward"] > best_eval:
                    best_eval = eval_stats["mean_reward"]
                    checkpoint = make_checkpoint(
                        model=model,
                        feature_config=feature_config,
                        hidden_dims=hidden_dims,
                        args=args,
                        best_eval=best_eval,
                    )
                    save_checkpoint(args.out, checkpoint)
                    print(f"[eval] new best -> {args.out} ({best_eval:.1f})")
                last_eval_env_step = env_steps

    finally:
        vec_env.close()

    total_elapsed = max(0.0, time.time() - start_time)
    final_checkpoint = make_checkpoint(
        model=model,
        feature_config=feature_config,
        hidden_dims=hidden_dims,
        args=args,
        best_eval=best_eval,
    )
    if best_eval == -float("inf"):
        save_checkpoint(args.out, final_checkpoint)
        print(f"[done] no eval checkpoint was saved during training, wrote final -> {args.out}")
    final_path = os.path.splitext(args.out)[0] + "_final.pth"
    save_checkpoint(final_path, final_checkpoint)
    print(f"[done] total_train_time={format_hms(total_elapsed)} ({total_elapsed / 60.0:.2f} min)")
    print(f"[done] final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
