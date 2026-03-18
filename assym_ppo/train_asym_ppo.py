from __future__ import annotations

import argparse
import importlib.util
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from features import ACTIONS, ACTION_DIM, FeatureConfig, FeatureTracker
from model import AsymmetricActorCritic, RolloutBuffer, categorical_kl, load_checkpoint, make_checkpoint, save_checkpoint
from privileged import extract_privileged_obs, extract_shaping_metrics, privileged_obs_dim


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
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def resolve_expert_path(expert: str, repo_dir: str) -> str:
    mapping = {
        "wall_submission": os.path.join(repo_dir, "ppo_lab", "agent_wall_submission.py"),
        "nowall": os.path.join(repo_dir, "ppo_lab", "agent_nowall.py"),
        "eval300": os.path.join(repo_dir, "ppo_lab", "agent_eval300.py"),
        "handmade": os.path.join(repo_dir, "handmade", "agent.py"),
    }
    if expert in mapping:
        return mapping[expert]
    if os.path.isfile(expert):
        return os.path.abspath(expert)
    candidate = os.path.join(repo_dir, expert)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Unknown expert '{expert}'")


def collect_expert_demos(
    *,
    expert: str,
    episodes: int,
    seed: int,
    obelix_py: str,
    env_kwargs: dict,
    feature_config: FeatureConfig,
    repo_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    if expert == "none" or episodes <= 0:
        return (
            np.empty((0, feature_config.feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    obelix_cls = import_symbol(obelix_py, "OBELIX")
    expert_path = resolve_expert_path(expert, repo_dir)
    expert_mod = import_module(expert_path, f"expert_{abs(hash(expert_path))}")
    if not hasattr(expert_mod, "policy"):
        raise AttributeError(f"Expert module does not define policy(): {expert_path}")
    expert_policy = getattr(expert_mod, "policy")

    tracker = FeatureTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
    feature_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    returns = deque(maxlen=50)

    print(f"[warm_start] collecting {episodes} expert episodes from {os.path.basename(expert_path)}")
    for episode in range(int(episodes)):
        episode_seed = int(seed + 500_000 + episode)
        env = obelix_cls(seed=episode_seed, **env_kwargs)
        obs = env.reset(seed=episode_seed)
        tracker.reset_all(np.asarray(obs, dtype=np.float32)[None, :])
        rng = np.random.default_rng(episode_seed)

        total_reward = 0.0
        done = False
        while not done:
            feature_rows.append(tracker.features().cpu().numpy()[0].astype(np.float32, copy=True))
            action_name = expert_policy(obs, rng)
            action_idx = ACTIONS.index(action_name)
            action_rows.append(action_idx)

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

    feats = np.asarray(feature_rows, dtype=np.float32)
    acts = np.asarray(action_rows, dtype=np.int64)
    print(f"[warm_start] collected {feats.shape[0]} state-action pairs")
    return feats, acts


def warm_start_policy(
    model: AsymmetricActorCritic,
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
            logits = model.actor(x[idx])
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


def shape_rewards(
    *,
    args: argparse.Namespace,
    env_rewards: np.ndarray,
    next_obs: np.ndarray,
    actions: torch.Tensor,
    prev_metrics: dict[str, torch.Tensor],
    next_metrics: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    rewards_t = torch.as_tensor(env_rewards, dtype=torch.float32, device=device)

    if args.visible_bonus != 0.0:
        rewards_t = rewards_t + float(args.visible_bonus) * next_metrics["box_visible"]

    if args.ir_bonus != 0.0:
        ir_seen = torch.as_tensor(next_obs[:, 16] > 0.5, dtype=torch.float32, device=device)
        rewards_t = rewards_t + float(args.ir_bonus) * ir_seen

    if args.blind_turn_penalty != 0.0:
        blind = torch.as_tensor(np.sum(next_obs[:, :16], axis=1) == 0.0, dtype=torch.bool, device=device)
        turn_mask = actions != ACTIONS.index("FW")
        rewards_t = rewards_t - float(args.blind_turn_penalty) * (blind & turn_mask).to(torch.float32)

    if args.stuck_extra_penalty != 0.0:
        rewards_t = rewards_t - float(args.stuck_extra_penalty) * next_metrics["stuck"]

    if args.approach_progress_bonus != 0.0:
        nonpush = 1.0 - prev_metrics["push_active"]
        progress = torch.clamp(
            prev_metrics["bot_box_distance"] - next_metrics["bot_box_distance"],
            min=-0.05,
            max=0.05,
        )
        rewards_t = rewards_t + float(args.approach_progress_bonus) * progress * nonpush

    if args.alignment_bonus != 0.0:
        nonpush = 1.0 - prev_metrics["push_active"]
        align_gain = torch.clamp(
            next_metrics["heading_alignment"] - prev_metrics["heading_alignment"],
            min=-0.5,
            max=0.5,
        )
        rewards_t = rewards_t + float(args.alignment_bonus) * align_gain * nonpush

    if args.push_progress_bonus != 0.0:
        push_mask = torch.maximum(prev_metrics["push_active"], next_metrics["push_active"])
        goal_progress = torch.clamp(
            prev_metrics["goal_distance"] - next_metrics["goal_distance"],
            min=-0.05,
            max=0.05,
        )
        rewards_t = rewards_t + float(args.push_progress_bonus) * goal_progress * push_mask

    if args.reward_scale != 1.0:
        rewards_t = rewards_t / float(args.reward_scale)
    if args.reward_clip > 0.0:
        rewards_t = torch.clamp(rewards_t, min=-float(args.reward_clip), max=float(args.reward_clip))
    return rewards_t.detach().cpu().numpy().astype(np.float32, copy=False)


class GreedyRunner:
    def __init__(self, model: AsymmetricActorCritic, feature_config: FeatureConfig, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.tracker = FeatureTracker(num_envs=1, config=feature_config, device=device)
        self.pending_action: int | None = None
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

        actor_obs = self.tracker.features()
        logits = self.model.actor(actor_obs)
        action_idx = int(torch.argmax(logits, dim=1).item())
        self.pending_action = action_idx
        return action_idx


@torch.no_grad()
def evaluate_actor(
    model: AsymmetricActorCritic,
    *,
    feature_config: FeatureConfig,
    obelix_py: str,
    env_kwargs: dict,
    runs: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    obelix_cls = import_symbol(obelix_py, "OBELIX")
    runner = GreedyRunner(model=model, feature_config=feature_config, device=device)
    scores: list[float] = []
    lengths: list[int] = []
    successes = 0

    model.eval()
    for run_idx in range(int(runs)):
        episode_seed = int(seed + run_idx)
        env = obelix_cls(seed=episode_seed, **env_kwargs)
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
    parser = argparse.ArgumentParser(description="Asymmetric PPO trainer for OBELIX")
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
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch_compile", action="store_true")

    parser.add_argument("--obs_stack", type=int, default=12)
    parser.add_argument("--action_hist", type=int, default=6)
    parser.add_argument("--actor_hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[1024, 512, 256])
    parser.add_argument("--fw_bias_init", type=float, default=1.0)

    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--minibatch_size", type=int, default=16384)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--normalize_advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--reward_clip", type=float, default=0.0)
    parser.add_argument("--visible_bonus", type=float, default=0.0)
    parser.add_argument("--ir_bonus", type=float, default=0.0)
    parser.add_argument("--blind_turn_penalty", type=float, default=0.0)
    parser.add_argument("--stuck_extra_penalty", type=float, default=0.0)
    parser.add_argument("--approach_progress_bonus", type=float, default=0.0)
    parser.add_argument("--alignment_bonus", type=float, default=0.0)
    parser.add_argument("--push_progress_bonus", type=float, default=0.0)
    parser.add_argument("--schedule", type=str, choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--desired_kl", type=float, default=0.01)
    parser.add_argument("--use_clipped_value_loss", action=argparse.BooleanOptionalAction, default=True)

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_dir)

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

    actor_hidden_dims = tuple(int(x) for x in args.actor_hidden_dims)
    critic_hidden_dims = tuple(int(x) for x in args.critic_hidden_dims)
    feature_config = FeatureConfig(
        obs_stack=int(args.obs_stack),
        action_hist=int(args.action_hist),
        max_steps=int(args.max_steps),
    )
    privileged_dim = privileged_obs_dim()

    model = AsymmetricActorCritic(
        actor_dim=feature_config.feature_dim,
        privileged_dim=privileged_dim,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        fw_bias_init=args.fw_bias_init,
    ).to(device)
    if args.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_eval = -float("inf")

    if args.load:
        checkpoint = load_checkpoint(args.load, device=device)
        if "full_state_dict" not in checkpoint:
            raise RuntimeError("Asymmetric PPO checkpoints must contain full_state_dict for resume")
        model.load_state_dict(checkpoint["full_state_dict"], strict=True)
        best_eval = float(checkpoint.get("best_eval", best_eval))
        print(f"[setup] loaded checkpoint {args.load}")

    env_kwargs = {
        "scaling_factor": args.scaling_factor,
        "arena_size": args.arena_size,
        "max_steps": args.max_steps,
        "wall_obstacles": args.wall_obstacles,
        "difficulty": args.difficulty,
        "box_speed": args.box_speed,
    }

    if args.warm_start_expert != "none" and args.warm_start_episodes > 0:
        feats, acts = collect_expert_demos(
            expert=args.warm_start_expert,
            episodes=args.warm_start_episodes,
            seed=args.seed,
            obelix_py=args.obelix_py,
            env_kwargs=env_kwargs,
            feature_config=feature_config,
            repo_dir=repo_dir,
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

    env_device = str(device) if args.env_device == "auto" else args.env_device
    vec_env_cls = import_symbol(args.obelix_torch_py, "OBELIXVectorized")
    vec_env = vec_env_cls(
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

    print(
        f"[setup] device={device} env_device={env_device} num_envs={args.num_envs} "
        f"wall_obstacles={args.wall_obstacles} difficulty={args.difficulty}"
    )
    print(
        f"[setup] actor_dim={feature_config.feature_dim} privileged_dim={privileged_dim} "
        f"actor_hidden={actor_hidden_dims} critic_hidden={critic_hidden_dims}"
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
    current_lr = float(args.lr)

    try:
        while env_steps < args.total_env_steps:
            buffer = RolloutBuffer(
                num_steps=args.rollout_steps,
                num_envs=args.num_envs,
                actor_dim=feature_config.feature_dim,
                privileged_dim=privileged_dim,
                device=device,
            )
            rollout_action_counts = np.zeros((ACTION_DIM,), dtype=np.int64)

            model.eval()
            for step in range(args.rollout_steps):
                actor_obs = tracker.features()
                priv_obs = extract_privileged_obs(vec_env, target_device=device)
                prev_metrics = extract_shaping_metrics(vec_env, target_device=device)
                with torch.no_grad():
                    actions_t, log_probs_t, _, values_t, logits_t = model.act(actor_obs, priv_obs)

                action_idx = actions_t.detach().cpu().numpy()
                rollout_action_counts += np.bincount(action_idx, minlength=ACTION_DIM)
                next_obs, rewards, dones = vec_env.step(action_idx)
                next_metrics = extract_shaping_metrics(vec_env, target_device=device)
                train_rewards = shape_rewards(
                    args=args,
                    env_rewards=rewards,
                    next_obs=next_obs,
                    actions=actions_t,
                    prev_metrics=prev_metrics,
                    next_metrics=next_metrics,
                    device=device,
                )

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
                    actor_obs=actor_obs,
                    privileged_obs=priv_obs,
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
                next_actor_obs = tracker.features()
                next_priv_obs = extract_privileged_obs(vec_env, target_device=device)
                last_values = model.value(next_actor_obs, next_priv_obs)

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
                actor_batch,
                priv_batch,
                actions_batch,
                old_log_probs_batch,
                old_values_batch,
                returns_batch,
                advantages_batch,
                old_logits_batch,
            ) in buffer.mini_batch_generator(args.minibatch_size, args.update_epochs):
                new_log_probs, entropy, values, new_logits = model.evaluate_actions(
                    actor_batch,
                    priv_batch,
                    actions_batch,
                )

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
                eval_stats = evaluate_actor(
                    model=model,
                    feature_config=feature_config,
                    obelix_py=args.obelix_py,
                    env_kwargs=env_kwargs,
                    runs=args.eval_runs,
                    seed=args.seed + 100_000,
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
                        actor_hidden_dims=actor_hidden_dims,
                        critic_hidden_dims=critic_hidden_dims,
                        privileged_dim=privileged_dim,
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
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        privileged_dim=privileged_dim,
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
