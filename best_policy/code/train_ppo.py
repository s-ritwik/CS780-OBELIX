"""Improved PPO trainer for OBELIX.

This version is rollout-based and closer in spirit to rsl_rl PPO:
- vectorized env collection
- GAE over fixed rollout horizons
- clipped surrogate loss
- clipped value loss
- optional adaptive KL learning rate schedule
- multi-epoch mini-batch updates

Codabench still evaluates only the submitted policy(obs, rng) on CPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_POLICY_DIR = os.path.dirname(CODE_DIR)

from obs_encoder import ENCODED_OBS_DIM, RAW_OBS_DIM, encode_obs_tensor, infer_use_rec_encoder_from_state_dict


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
FW_ACTION_INDEX = ACTIONS.index("FW")


def import_symbol(py_file: str, symbol: str):
    spec = importlib.util.spec_from_file_location("obelix_module", py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {symbol} from {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise AttributeError(f"{symbol} not found in {py_file}")
    return getattr(module, symbol)


def import_parallel_obelix(parallel_dir: str):
    if parallel_dir not in sys.path:
        sys.path.insert(0, parallel_dir)
    from parallel_env import ParallelOBELIX

    return ParallelOBELIX


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def apply_forward_bias(layer: nn.Linear, fw_bias_init: float) -> None:
    if fw_bias_init == 0.0:
        return
    with torch.no_grad():
        layer.bias[FW_ACTION_INDEX] += float(fw_bias_init)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = RAW_OBS_DIM,
        action_dim: int = 5,
        hidden_dims: tuple[int, ...] = (128, 64),
        fw_bias_init: float = 0.0,
        use_rec_encoder: bool = False,
    ):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers: list[nn.Module] = []
        self.use_rec_encoder = bool(use_rec_encoder)
        last_dim = ENCODED_OBS_DIM if self.use_rec_encoder else int(obs_dim)
        for h in hidden_dims:
            h_i = int(h)
            if h_i <= 0:
                raise ValueError("All hidden_dims values must be > 0")
            layers.append(layer_init(nn.Linear(last_dim, h_i)))
            layers.append(nn.Tanh())
            last_dim = h_i

        self.backbone = nn.Sequential(*layers)
        self.policy_head = layer_init(nn.Linear(last_dim, int(action_dim)), std=0.01)
        self.value_head = layer_init(nn.Linear(last_dim, 1), std=1.0)
        apply_forward_bias(self.policy_head, fw_bias_init)

    def forward(self, x: torch.Tensor):
        if self.use_rec_encoder:
            x = encode_obs_tensor(x)
        feat = self.backbone(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, logits

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value, logits


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device

        self.obs = torch.zeros((self.num_steps, self.num_envs, self.obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.logits = torch.zeros(
            (self.num_steps, self.num_envs, self.action_dim), dtype=torch.float32, device=device
        )
        self.advantages = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=device)

    def add(
        self,
        step: int,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        self.obs[step].copy_(obs)
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
        normalize_advantages_per_minibatch: bool,
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

        if normalize_advantages and not normalize_advantages_per_minibatch:
            flat_adv = self.advantages.reshape(-1)
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)
            self.advantages.copy_(flat_adv.view_as(self.advantages))

    def mini_batch_generator(self, minibatch_size: int, num_learning_epochs: int):
        batch_size = self.num_steps * self.num_envs
        mb_size = min(int(minibatch_size), batch_size)

        obs = self.obs.reshape(batch_size, self.obs_dim)
        actions = self.actions.reshape(batch_size)
        old_log_probs = self.log_probs.reshape(batch_size)
        old_values = self.values.reshape(batch_size)
        returns = self.returns.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        old_logits = self.logits.reshape(batch_size, self.action_dim)

        for _ in range(int(num_learning_epochs)):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mb_size):
                mb_idx = indices[start : start + mb_size]
                yield (
                    obs[mb_idx],
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


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_checkpoint_state(checkpoint_path: str):
    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = raw["state_dict"]
        metadata = raw
    elif isinstance(raw, dict):
        state_dict = raw
        metadata = {}
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    return state_dict, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Improved PPO trainer for OBELIX")

    base_dir = CODE_DIR
    repo_dir = BEST_POLICY_DIR
    default_obelix_np = os.path.join(repo_dir, "obelix.py")
    default_obelix_torch = os.path.join(repo_dir, "obelix_torch.py")

    parser.add_argument(
        "--env_backend",
        type=str,
        choices=["numpy", "torch", "torch_vec"],
        default="numpy",
        help=(
            "Environment backend: "
            "numpy=obelix.py via subprocess workers, "
            "torch=obelix_torch.py via subprocess workers, "
            "torch_vec=single-process batched OBELIXVectorized."
        ),
    )
    parser.add_argument(
        "--obelix_py",
        type=str,
        default=None,
        help=(
            "Optional explicit OBELIX implementation path. "
            f"Defaults to {default_obelix_np} for numpy backend, "
            f"or {default_obelix_torch} for torch and torch_vec backends."
        ),
    )
    parser.add_argument(
        "--env_device",
        type=str,
        default="cpu",
        help="Device passed to torch backend envs (cpu|cuda|auto). Ignored for numpy backend.",
    )
    parser.add_argument("--out", type=str, default="weights.pth")

    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--total_env_steps", type=int, default=2_000_000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--mp_start_method", type=str, default="spawn")

    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden layer sizes, e.g. --hidden_dims 128 64",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--minibatch_size", type=int, default=4096)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--normalize_advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--normalize_advantages_per_minibatch", action="store_true")
    parser.add_argument("--schedule", type=str, choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--desired_kl", type=float, default=0.01)
    parser.add_argument(
        "--use_clipped_value_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--fw_bias_init",
        type=float,
        default=1.0,
        help="Initial logit bias added to the FW action in the actor head.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path used to warm-start model weights before training.",
    )
    parser.add_argument(
        "--rec_enco",
        action="store_true",
        help="Use the recommended structured observation encoder before the policy/value network.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100_000_0)
    parser.add_argument(
        "--success_reward_threshold",
        type=float,
        default=1000.0,
        help=(
            "Heuristic threshold on terminal-step reward used to count likely successes in logs. "
            "This affects logging only."
        ),
    )
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

    base_dir = CODE_DIR
    repo_dir = BEST_POLICY_DIR
    parallel_env_dir = base_dir
    default_impl = {
        "numpy": os.path.join(repo_dir, "obelix.py"),
        "torch": os.path.join(repo_dir, "obelix_torch.py"),
        "torch_vec": os.path.join(repo_dir, "obelix_torch.py"),
    }
    obelix_py = args.obelix_py if args.obelix_py is not None else default_impl[args.env_backend]
    if not os.path.exists(obelix_py):
        raise FileNotFoundError(f"Environment file not found: {obelix_py}")

    print(f"[setup] device={device} env_backend={args.env_backend} num_envs={args.num_envs}")
    print(f"[setup] obelix_py={obelix_py}")
    if args.env_backend in ("torch", "torch_vec"):
        print(f"[setup] env_device={args.env_device}")

    hidden_dims = tuple(int(h) for h in args.hidden_dims)
    model = ActorCritic(
        hidden_dims=hidden_dims,
        fw_bias_init=args.fw_bias_init,
        use_rec_encoder=args.rec_enco,
    ).to(device)
    if args.rec_enco:
        print(f"[setup] recommended encoder enabled (input_dim={ENCODED_OBS_DIM})")
    if args.init_checkpoint is not None:
        state_dict, metadata = load_checkpoint_state(args.init_checkpoint)
        ckpt_hidden_dims = metadata.get("hidden_dims")
        if ckpt_hidden_dims is not None and tuple(int(h) for h in ckpt_hidden_dims) != hidden_dims:
            raise ValueError(
                "Checkpoint hidden_dims do not match requested model shape: "
                f"checkpoint={tuple(int(h) for h in ckpt_hidden_dims)} requested={hidden_dims}"
            )
        ckpt_rec_enco = metadata.get("use_rec_encoder")
        if ckpt_rec_enco is None:
            ckpt_rec_enco = infer_use_rec_encoder_from_state_dict(state_dict)
        if bool(ckpt_rec_enco) != bool(args.rec_enco):
            raise ValueError(
                "Checkpoint encoder setting does not match this run: "
                f"checkpoint_use_rec_encoder={bool(ckpt_rec_enco)} requested={bool(args.rec_enco)}"
            )
        model.load_state_dict(state_dict, strict=True)
        print(f"[setup] warm-started model weights from {args.init_checkpoint}")
    if args.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    def online_state_dict():
        if hasattr(model, "_orig_mod"):
            return model._orig_mod.state_dict()
        return model.state_dict()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    current_lr = float(args.lr)

    env_kwargs = dict(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )
    if args.env_backend == "torch":
        if args.env_device != "auto":
            env_kwargs["device"] = args.env_device
    elif args.env_backend == "torch_vec":
        env_kwargs["device"] = str(device) if args.env_device == "auto" else args.env_device

    if args.env_backend == "torch_vec":
        VecEnvCls = import_symbol(obelix_py, "OBELIXVectorized")
        vec_env = VecEnvCls(
            num_envs=args.num_envs,
            seed=args.seed * 10_000,
            **env_kwargs,
        )
    else:
        ParallelOBELIX = import_parallel_obelix(parallel_env_dir)
        vec_env = ParallelOBELIX(
            obelix_py=obelix_py,
            num_envs=args.num_envs,
            base_seed=args.seed * 10_000,
            env_kwargs=env_kwargs,
            mp_start_method=args.mp_start_method,
        )

    obs = vec_env.reset_all(seed=args.seed * 10_000)
    episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.num_envs,), dtype=np.int32)
    recent_returns = deque(maxlen=200)
    recent_lengths = deque(maxlen=200)
    recent_successes = deque(maxlen=200)
    recent_timeouts = deque(maxlen=200)
    recent_terminal_rewards = deque(maxlen=200)

    env_steps = 0
    update_idx = 0
    last_log_env_step = 0
    start_time = time.time()
    total_completed_eps = 0
    total_successes = 0
    total_timeouts = 0

    try:
        while env_steps < args.total_env_steps:
            buffer = RolloutBuffer(
                num_steps=args.rollout_steps,
                num_envs=args.num_envs,
                obs_dim=RAW_OBS_DIM,
                action_dim=len(ACTIONS),
                device=device,
            )

            model.eval()
            rollout_action_counts = np.zeros((len(ACTIONS),), dtype=np.int64)
            for step in range(args.rollout_steps):
                obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    actions_t, log_probs_t, _, values_t, logits_t = model.act(obs_t)

                action_idx = actions_t.cpu().numpy()
                rollout_action_counts += np.bincount(action_idx, minlength=len(ACTIONS))
                if args.env_backend == "torch_vec":
                    next_obs, rewards, dones = vec_env.step(action_idx)
                else:
                    actions = [ACTIONS[int(a)] for a in action_idx]
                    next_obs, rewards, dones = vec_env.step(actions)

                buffer.add(
                    step=step,
                    obs=obs_t,
                    actions=actions_t,
                    log_probs=log_probs_t,
                    rewards=rewards.astype(np.float32, copy=False),
                    dones=dones.astype(np.float32, copy=False),
                    values=values_t,
                    logits=logits_t,
                )

                episode_returns += rewards
                episode_lengths += 1

                done_idx = np.nonzero(dones)[0]
                if done_idx.size > 0:
                    for idx in done_idx:
                        recent_returns.append(float(episode_returns[idx]))
                        recent_lengths.append(int(episode_lengths[idx]))
                        terminal_reward = float(rewards[idx])
                        timeout_flag = int(episode_lengths[idx] >= args.max_steps)
                        success_flag = int(terminal_reward >= args.success_reward_threshold)
                        recent_successes.append(success_flag)
                        recent_timeouts.append(timeout_flag)
                        recent_terminal_rewards.append(terminal_reward)
                        total_completed_eps += 1
                        total_successes += success_flag
                        total_timeouts += timeout_flag
                    episode_returns[done_idx] = 0.0
                    episode_lengths[done_idx] = 0

                    reset_map = vec_env.reset(
                        env_indices=done_idx.tolist(),
                        seed=args.seed * 10_000 + env_steps + step * args.num_envs,
                    )
                    for idx, reset_obs in reset_map.items():
                        next_obs[idx] = reset_obs

                obs = next_obs
                env_steps += args.num_envs

            with torch.no_grad():
                next_obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
                _, last_values = model(next_obs_t)

            buffer.compute_returns(
                last_values=last_values,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                normalize_advantages=args.normalize_advantages,
                normalize_advantages_per_minibatch=args.normalize_advantages_per_minibatch,
            )

            model.train()
            mean_policy_loss = 0.0
            mean_value_loss = 0.0
            mean_entropy = 0.0
            mean_kl = 0.0
            num_minibatches = 0

            for (
                obs_batch,
                actions_batch,
                old_log_probs_batch,
                old_values_batch,
                returns_batch,
                advantages_batch,
                old_logits_batch,
            ) in buffer.mini_batch_generator(args.minibatch_size, args.update_epochs):
                if args.normalize_advantages_per_minibatch:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std(unbiased=False) + 1e-8
                    )

                new_log_probs, entropy, values, new_logits = model.evaluate_actions(obs_batch, actions_batch)

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
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                policy_loss = torch.max(surrogate, surrogate_clipped).mean()

                if args.use_clipped_value_loss:
                    value_clipped = old_values_batch + (values - old_values_batch).clamp(
                        -args.clip_coef, args.clip_coef
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
                mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
                mean_len = float(np.mean(recent_lengths)) if recent_lengths else float("nan")
                recent_success_rate = (
                    float(np.mean(recent_successes)) if recent_successes else float("nan")
                )
                recent_timeout_rate = (
                    float(np.mean(recent_timeouts)) if recent_timeouts else float("nan")
                )
                mean_terminal_reward = (
                    float(np.mean(recent_terminal_rewards)) if recent_terminal_rewards else float("nan")
                )
                total_success_rate = (
                    float(total_successes) / float(total_completed_eps)
                    if total_completed_eps > 0
                    else float("nan")
                )
                total_timeout_rate = (
                    float(total_timeouts) / float(total_completed_eps)
                    if total_completed_eps > 0
                    else float("nan")
                )
                rollout_total_actions = max(1, int(np.sum(rollout_action_counts)))
                action_mix = " ".join(
                    f"{name}:{(count / rollout_total_actions):.2f}"
                    for name, count in zip(ACTIONS, rollout_action_counts.tolist())
                )
                print(
                    f"[train] update={update_idx} env_steps={env_steps} "
                    f"policy_loss={mean_policy_loss:.4f} value_loss={mean_value_loss:.4f} "
                    f"entropy={mean_entropy:.4f} kl={mean_kl:.5f} lr={current_lr:.6f} "
                    f"sps={sps:.1f} recent_return={mean_ret:.1f} recent_len={mean_len:.1f} "
                    f"recent_success={recent_success_rate:.3f} recent_timeout={recent_timeout_rate:.3f} "
                    f"total_success={total_success_rate:.3f} total_timeout={total_timeout_rate:.3f} "
                    f"terminal_reward={mean_terminal_reward:.1f} completed_eps={total_completed_eps} "
                    f"actions=[{action_mix}] "
                    f"elapsed={format_hms(elapsed)}"
                )
                last_log_env_step = env_steps

    finally:
        vec_env.close()

    total_elapsed = max(0.0, time.time() - start_time)
    checkpoint = {
        "state_dict": online_state_dict(),
        "obs_dim": RAW_OBS_DIM,
        "model_input_dim": ENCODED_OBS_DIM if args.rec_enco else RAW_OBS_DIM,
        "action_dim": len(ACTIONS),
        "hidden_dims": [int(h) for h in hidden_dims],
        "actions": ACTIONS,
        "use_rec_encoder": bool(args.rec_enco),
        "config": vars(args),
    }
    torch.save(checkpoint, args.out)
    print(f"[done] total_train_time={format_hms(total_elapsed)} ({total_elapsed / 60.0:.2f} min)")
    print(f"[done] saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
