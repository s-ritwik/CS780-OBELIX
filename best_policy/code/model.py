from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from features import ACTIONS, ACTION_DIM, FeatureConfig


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def apply_forward_bias(layer: nn.Linear, fw_bias_init: float) -> None:
    if fw_bias_init == 0.0:
        return
    with torch.no_grad():
        layer.bias[ACTIONS.index("FW")] += float(fw_bias_init)


class ActorOnly(nn.Module):
    def __init__(self, actor_dim: int, actor_hidden_dims: tuple[int, ...], fw_bias_init: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(actor_dim)
        for hidden_dim in actor_hidden_dims:
            h = int(hidden_dim)
            layers.append(layer_init(nn.Linear(last_dim, h)))
            layers.append(nn.Tanh())
            last_dim = h

        self.actor_backbone = nn.Sequential(*layers)
        self.policy_head = layer_init(nn.Linear(last_dim, ACTION_DIM), std=0.01)
        apply_forward_bias(self.policy_head, fw_bias_init)

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        feat = self.actor_backbone(actor_obs)
        return self.policy_head(feat)


class AsymmetricActorCritic(nn.Module):
    def __init__(
        self,
        actor_dim: int,
        privileged_dim: int,
        actor_hidden_dims: tuple[int, ...],
        critic_hidden_dims: tuple[int, ...],
        fw_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.actor_dim = int(actor_dim)
        self.privileged_dim = int(privileged_dim)
        self.actor_hidden_dims = tuple(int(x) for x in actor_hidden_dims)
        self.critic_hidden_dims = tuple(int(x) for x in critic_hidden_dims)
        self.actor = ActorOnly(
            actor_dim=self.actor_dim,
            actor_hidden_dims=self.actor_hidden_dims,
            fw_bias_init=fw_bias_init,
        )

        critic_layers: list[nn.Module] = []
        critic_last_dim = self.actor_dim + self.privileged_dim
        for hidden_dim in self.critic_hidden_dims:
            h = int(hidden_dim)
            critic_layers.append(layer_init(nn.Linear(critic_last_dim, h)))
            critic_layers.append(nn.Tanh())
            critic_last_dim = h
        self.critic_backbone = nn.Sequential(*critic_layers)
        self.value_head = layer_init(nn.Linear(critic_last_dim, 1), std=1.0)

    def value(self, actor_obs: torch.Tensor, privileged_obs: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat([actor_obs, privileged_obs], dim=1)
        critic_feat = self.critic_backbone(critic_input)
        return self.value_head(critic_feat).squeeze(-1)

    def act(self, actor_obs: torch.Tensor, privileged_obs: torch.Tensor):
        logits = self.actor(actor_obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value(actor_obs, privileged_obs)
        return action, log_prob, entropy, value, logits

    def evaluate_actions(self, actor_obs: torch.Tensor, privileged_obs: torch.Tensor, actions: torch.Tensor):
        logits = self.actor(actor_obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value(actor_obs, privileged_obs)
        return log_prob, entropy, value, logits


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, actor_dim: int, privileged_dim: int, device: torch.device):
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.actor_dim = int(actor_dim)
        self.privileged_dim = int(privileged_dim)
        self.device = device

        self.actor_obs = torch.zeros(
            (self.num_steps, self.num_envs, self.actor_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.privileged_obs = torch.zeros(
            (self.num_steps, self.num_envs, self.privileged_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.logits = torch.zeros((self.num_steps, self.num_envs, ACTION_DIM), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)

    def add(
        self,
        *,
        step: int,
        actor_obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        self.actor_obs[step].copy_(actor_obs)
        self.privileged_obs[step].copy_(privileged_obs)
        self.actions[step].copy_(actions)
        self.log_probs[step].copy_(log_probs)
        self.rewards[step].copy_(torch.as_tensor(rewards, dtype=torch.float32, device=self.device))
        self.dones[step].copy_(torch.as_tensor(dones, dtype=torch.float32, device=self.device))
        self.values[step].copy_(values)
        self.logits[step].copy_(logits)

    def compute_returns(
        self,
        *,
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
            flat_adv = self.advantages.reshape(-1)
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)
            self.advantages.copy_(flat_adv.view_as(self.advantages))

    def mini_batch_generator(self, minibatch_size: int, num_learning_epochs: int):
        batch_size = self.num_steps * self.num_envs
        mb_size = min(int(minibatch_size), batch_size)

        actor_obs = self.actor_obs.reshape(batch_size, self.actor_dim)
        privileged_obs = self.privileged_obs.reshape(batch_size, self.privileged_dim)
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
                    actor_obs[mb_idx],
                    privileged_obs[mb_idx],
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


def _unwrapped_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def make_checkpoint(
    model: nn.Module,
    *,
    feature_config: FeatureConfig,
    actor_hidden_dims: tuple[int, ...],
    critic_hidden_dims: tuple[int, ...],
    privileged_dim: int,
    args,
    best_eval: float,
) -> dict:
    base_model = _unwrapped_model(model)
    return {
        "full_state_dict": base_model.state_dict(),
        "actor_backbone_state_dict": base_model.actor.actor_backbone.state_dict(),
        "policy_head_state_dict": base_model.actor.policy_head.state_dict(),
        "actions": ACTIONS,
        "feature_config": asdict(feature_config),
        "actor_hidden_dims": [int(h) for h in actor_hidden_dims],
        "critic_hidden_dims": [int(h) for h in critic_hidden_dims],
        "privileged_dim": int(privileged_dim),
        "best_eval": float(best_eval),
        "config": vars(args),
    }


def save_checkpoint(path: str, checkpoint: dict) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    raw = torch.load(path, map_location=device)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {path}")
    return raw
