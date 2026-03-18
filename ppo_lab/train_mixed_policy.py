from __future__ import annotations

import argparse
import importlib.util
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from train_policy import (
    ACTION_DIM,
    ACTIONS,
    ActorCritic,
    FeatureConfig,
    FeatureTracker,
    RolloutBuffer,
    categorical_kl,
    collect_expert_demos,
    format_hms,
    load_checkpoint,
    make_checkpoint,
    save_checkpoint,
    warm_start_policy,
)


@dataclass(frozen=True)
class ScenarioSpec:
    tag: str
    difficulty: int
    wall_obstacles: bool
    box_speed: int = 2


SCENARIO_LIBRARY: dict[str, ScenarioSpec] = {
    "static_nowall": ScenarioSpec("static_nowall", difficulty=0, wall_obstacles=False, box_speed=2),
    "static_wall": ScenarioSpec("static_wall", difficulty=0, wall_obstacles=True, box_speed=2),
    "blink_nowall": ScenarioSpec("blink_nowall", difficulty=2, wall_obstacles=False, box_speed=2),
    "blink_wall": ScenarioSpec("blink_wall", difficulty=2, wall_obstacles=True, box_speed=2),
    "move_nowall": ScenarioSpec("move_nowall", difficulty=3, wall_obstacles=False, box_speed=2),
    "move_wall": ScenarioSpec("move_wall", difficulty=3, wall_obstacles=True, box_speed=2),
}


def import_symbol(py_file: str, symbol: str):
    spec = importlib.util.spec_from_file_location("obelix_module", py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {symbol} from {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise AttributeError(f"{symbol} not found in {py_file}")
    return getattr(module, symbol)


class MixedVecEnv:
    def __init__(
        self,
        *,
        obelix_torch_py: str,
        num_envs: int,
        scenario_specs: list[ScenarioSpec],
        scaling_factor: int,
        arena_size: int,
        max_steps: int,
        seed: int,
        env_device: str,
    ) -> None:
        if num_envs < len(scenario_specs):
            raise ValueError("num_envs must be at least the number of scenarios")

        VecEnvCls = import_symbol(obelix_torch_py, "OBELIXVectorized")
        self.num_envs = int(num_envs)
        self.scenario_specs = list(scenario_specs)
        self.envs = []
        self.counts: list[int] = []
        self.offsets: list[int] = []

        base = self.num_envs // len(self.scenario_specs)
        rem = self.num_envs % len(self.scenario_specs)

        offset = 0
        for idx, scenario in enumerate(self.scenario_specs):
            count = base + (1 if idx < rem else 0)
            self.counts.append(count)
            self.offsets.append(offset)
            env = VecEnvCls(
                num_envs=count,
                scaling_factor=scaling_factor,
                arena_size=arena_size,
                max_steps=max_steps,
                wall_obstacles=scenario.wall_obstacles,
                difficulty=scenario.difficulty,
                box_speed=scenario.box_speed,
                seed=seed + offset * 1000,
                device=env_device,
            )
            self.envs.append(env)
            offset += count

    def reset_all(self, seed: int | None = None) -> np.ndarray:
        batches: list[np.ndarray] = []
        for env_idx, env in enumerate(self.envs):
            env_seed = None if seed is None else int(seed + self.offsets[env_idx])
            batches.append(env.reset_all(seed=env_seed))
        return np.concatenate(batches, axis=0)

    def reset(self, env_indices: list[int], seed: int | None = None) -> dict[int, np.ndarray]:
        obs_map: dict[int, np.ndarray] = {}
        if not env_indices:
            return obs_map

        for env_idx, env in enumerate(self.envs):
            offset = self.offsets[env_idx]
            count = self.counts[env_idx]
            local = [int(idx - offset) for idx in env_indices if offset <= idx < offset + count]
            if not local:
                continue
            env_seed = None if seed is None else int(seed + offset)
            local_map = env.reset(env_indices=local, seed=env_seed)
            for local_idx, obs in local_map.items():
                obs_map[offset + int(local_idx)] = obs
        return obs_map

    def step(self, actions: np.ndarray | torch.Tensor | list[int]):
        if isinstance(actions, torch.Tensor):
            action_arr = actions.detach().cpu().numpy()
        else:
            action_arr = np.asarray(actions)

        obs_batches: list[np.ndarray] = []
        reward_batches: list[np.ndarray] = []
        done_batches: list[np.ndarray] = []

        for env_idx, env in enumerate(self.envs):
            offset = self.offsets[env_idx]
            count = self.counts[env_idx]
            local_actions = action_arr[offset : offset + count]
            obs, rewards, dones = env.step(local_actions)
            obs_batches.append(obs)
            reward_batches.append(rewards)
            done_batches.append(dones)

        return (
            np.concatenate(obs_batches, axis=0),
            np.concatenate(reward_batches, axis=0),
            np.concatenate(done_batches, axis=0),
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


class PolicyRunner:
    def __init__(
        self,
        *,
        model: ActorCritic,
        feature_config: FeatureConfig,
        device: torch.device,
        stochastic: bool,
    ) -> None:
        self.model = model
        self.device = device
        self.tracker = FeatureTracker(num_envs=1, config=feature_config, device=device)
        self.pending_action: int | None = None
        self.started = False
        self.stochastic = bool(stochastic)

    def reset(self, obs: np.ndarray) -> None:
        self.tracker.reset_all(np.asarray(obs, dtype=np.float32)[None, :])
        self.pending_action = None
        self.started = True

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
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
        if self.stochastic:
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)
            probs /= probs.sum()
            action_idx = int(rng.choice(len(ACTIONS), p=probs))
        else:
            action_idx = int(torch.argmax(logits, dim=1).item())
        self.pending_action = action_idx
        return action_idx


@torch.no_grad()
def evaluate_scenarios(
    *,
    model: ActorCritic,
    feature_config: FeatureConfig,
    obelix_py: str,
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    scenarios: list[ScenarioSpec],
    runs_per_scenario: int,
    seed: int,
    device: torch.device,
    stochastic: bool,
) -> dict:
    OBELIX = import_symbol(obelix_py, "OBELIX")
    runner = PolicyRunner(model=model, feature_config=feature_config, device=device, stochastic=stochastic)

    results: dict[str, dict[str, float]] = {}
    all_scores: list[float] = []
    total_successes = 0
    total_episodes = 0

    for scenario_idx, scenario in enumerate(scenarios):
        scenario_scores: list[float] = []
        scenario_lengths: list[int] = []
        scenario_successes = 0

        for run_idx in range(int(runs_per_scenario)):
            episode_seed = int(seed + scenario_idx * 10_000 + run_idx)
            env = OBELIX(
                scaling_factor=scaling_factor,
                arena_size=arena_size,
                max_steps=max_steps,
                wall_obstacles=scenario.wall_obstacles,
                difficulty=scenario.difficulty,
                box_speed=scenario.box_speed,
                seed=episode_seed,
            )
            obs = env.reset(seed=episode_seed)
            runner.reset(obs)
            rng = np.random.default_rng(episode_seed)

            total_reward = 0.0
            steps = 0
            done = False
            while not done:
                action_idx = runner.act(obs, rng)
                obs, reward, done = env.step(ACTIONS[action_idx], render=False)
                total_reward += float(reward)
                steps += 1

            scenario_scores.append(total_reward)
            scenario_lengths.append(steps)
            all_scores.append(total_reward)
            success = int(total_reward >= 1000.0)
            scenario_successes += success
            total_successes += success
            total_episodes += 1

        results[scenario.tag] = {
            "mean_reward": float(np.mean(scenario_scores)),
            "std_reward": float(np.std(scenario_scores)),
            "mean_length": float(np.mean(scenario_lengths)),
            "success_rate": float(scenario_successes) / float(max(1, runs_per_scenario)),
        }

    results["overall"] = {
        "mean_reward": float(np.mean(all_scores)),
        "std_reward": float(np.std(all_scores)),
        "success_rate": float(total_successes) / float(max(1, total_episodes)),
    }
    return results


def parse_scenarios(tags: list[str]) -> list[ScenarioSpec]:
    specs = []
    for tag in tags:
        if tag not in SCENARIO_LIBRARY:
            raise ValueError(
                f"Unknown scenario tag '{tag}'. Valid tags: {', '.join(sorted(SCENARIO_LIBRARY))}"
            )
        specs.append(SCENARIO_LIBRARY[tag])
    return specs


def collect_mixed_expert_demos(
    *,
    expert: str,
    episodes_per_scenario: int,
    scenarios: list[ScenarioSpec],
    seed: int,
    obelix_py: str,
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    feature_config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if expert == "none" or episodes_per_scenario <= 0 or not scenarios:
        return (
            np.empty((0, feature_config.feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    all_features: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    total_episodes = int(episodes_per_scenario) * len(scenarios)
    print(
        f"[warm_start] collecting mixed demos from {expert} "
        f"across {len(scenarios)} scenarios ({total_episodes} episodes total)"
    )

    local_agent_dir = os.path.dirname(os.path.abspath(__file__))

    for scenario_idx, scenario in enumerate(scenarios):
        print(f"[warm_start] scenario={scenario.tag} episodes={episodes_per_scenario}")
        scenario_expert = expert
        if expert == "scenario_specialist":
            scenario_expert = (
                os.path.join(local_agent_dir, "agent_wall_submission.py")
                if scenario.wall_obstacles
                else os.path.join(local_agent_dir, "agent_nowall.py")
            )
        feats, acts = collect_expert_demos(
            expert=scenario_expert,
            episodes=int(episodes_per_scenario),
            seed=int(seed + scenario_idx * 10_000),
            obelix_py=obelix_py,
            env_kwargs={
                "scaling_factor": int(scaling_factor),
                "arena_size": int(arena_size),
                "max_steps": int(max_steps),
                "wall_obstacles": bool(scenario.wall_obstacles),
                "difficulty": int(scenario.difficulty),
                "box_speed": int(scenario.box_speed),
            },
            feature_config=feature_config,
        )
        if feats.size > 0:
            all_features.append(feats)
            all_actions.append(acts)

    if not all_features:
        return (
            np.empty((0, feature_config.feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.concatenate(all_features, axis=0), np.concatenate(all_actions, axis=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mixed-scenario stacked PPO trainer for OBELIX")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_dir)

    parser.add_argument("--obelix_py", type=str, default=os.path.join(repo_dir, "obelix.py"))
    parser.add_argument("--obelix_torch_py", type=str, default=os.path.join(repo_dir, "obelix_torch.py"))
    parser.add_argument("--out", type=str, default=os.path.join(base_dir, "weights_mixed_best.pth"))
    parser.add_argument("--load", type=str, default=None)

    parser.add_argument("--num_envs", type=int, default=384)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--total_env_steps", type=int, default=6_000_000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--env_device", type=str, default="auto")

    parser.add_argument(
        "--train_scenarios",
        nargs="+",
        default=[
            "static_nowall",
            "static_wall",
            "blink_nowall",
            "blink_wall",
            "move_nowall",
            "move_wall",
        ],
    )
    parser.add_argument("--eval_scenarios", nargs="+", default=None)

    parser.add_argument("--obs_stack", type=int, default=8)
    parser.add_argument("--action_hist", type=int, default=4)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--fw_bias_init", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
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
    parser.add_argument("--schedule", type=str, choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--desired_kl", type=float, default=0.01)
    parser.add_argument("--use_clipped_value_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--eval_runs", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=750_000)
    parser.add_argument("--eval_mode", choices=["greedy", "stochastic"], default="stochastic")
    parser.add_argument("--log_interval", type=int, default=500_000)
    parser.add_argument("--warm_start_expert", type=str, default="none")
    parser.add_argument("--warm_start_episodes_per_scenario", type=int, default=0)
    parser.add_argument("--warm_start_epochs", type=int, default=5)
    parser.add_argument("--warm_start_batch_size", type=int, default=8192)
    parser.add_argument("--warm_start_scenarios", nargs="+", default=None)
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
    train_scenarios = parse_scenarios(list(args.train_scenarios))
    eval_scenarios = parse_scenarios(list(args.eval_scenarios) if args.eval_scenarios else list(args.train_scenarios))
    warm_start_scenarios = parse_scenarios(
        list(args.warm_start_scenarios) if args.warm_start_scenarios else list(args.train_scenarios)
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

    if args.warm_start_expert != "none" and args.warm_start_episodes_per_scenario > 0:
        feats, acts = collect_mixed_expert_demos(
            expert=args.warm_start_expert,
            episodes_per_scenario=args.warm_start_episodes_per_scenario,
            scenarios=warm_start_scenarios,
            seed=args.seed,
            obelix_py=args.obelix_py,
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
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

    env_device = str(device) if args.env_device == "auto" else args.env_device
    vec_env = MixedVecEnv(
        obelix_torch_py=args.obelix_torch_py,
        num_envs=args.num_envs,
        scenario_specs=train_scenarios,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        seed=args.seed * 10_000,
        env_device=env_device,
    )

    print(
        f"[setup] device={device} env_device={env_device} num_envs={args.num_envs} "
        f"scenarios={','.join(spec.tag for spec in train_scenarios)}"
    )
    print(
        f"[setup] feature_dim={feature_config.feature_dim} obs_stack={feature_config.obs_stack} "
        f"action_hist={feature_config.action_hist} hidden_dims={hidden_dims}"
    )

    obs = vec_env.reset_all(seed=args.seed * 10_000)
    tracker = FeatureTracker(num_envs=args.num_envs, config=feature_config, device=device)
    tracker.reset_all(obs)

    episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.num_envs,), dtype=np.int32)
    recent_returns = deque(maxlen=500)
    recent_lengths = deque(maxlen=500)
    recent_successes = deque(maxlen=500)

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
                if args.visible_bonus != 0.0:
                    train_rewards += float(args.visible_bonus) * np.any(
                        next_obs[:, :16] > 0.5,
                        axis=1,
                    ).astype(np.float32, copy=False)
                if args.ir_bonus != 0.0:
                    train_rewards += float(args.ir_bonus) * (next_obs[:, 16] > 0.5).astype(
                        np.float32,
                        copy=False,
                    )
                if args.blind_turn_penalty != 0.0:
                    blind_turn = (np.sum(obs[:, :16], axis=1) <= 0.0) & (action_idx != ACTIONS.index("FW"))
                    train_rewards -= float(args.blind_turn_penalty) * blind_turn.astype(
                        np.float32,
                        copy=False,
                    )
                if args.stuck_extra_penalty != 0.0:
                    train_rewards -= float(args.stuck_extra_penalty) * (next_obs[:, 17] > 0.5).astype(
                        np.float32,
                        copy=False,
                    )
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
                eval_stats = evaluate_scenarios(
                    model=model,
                    feature_config=feature_config,
                    obelix_py=args.obelix_py,
                    scaling_factor=args.scaling_factor,
                    arena_size=args.arena_size,
                    max_steps=args.max_steps,
                    scenarios=eval_scenarios,
                    runs_per_scenario=args.eval_runs,
                    seed=args.seed + 100_000,
                    device=device,
                    stochastic=(args.eval_mode == "stochastic"),
                )
                overall = eval_stats["overall"]
                print(
                    f"[eval] env_steps={env_steps} overall_mean={overall['mean_reward']:.1f} "
                    f"overall_std={overall['std_reward']:.1f} overall_success={overall['success_rate']:.3f}"
                )
                for scenario in eval_scenarios:
                    stats = eval_stats[scenario.tag]
                    print(
                        f"[eval] {scenario.tag} mean={stats['mean_reward']:.1f} "
                        f"std={stats['std_reward']:.1f} success={stats['success_rate']:.3f} "
                        f"len={stats['mean_length']:.1f}"
                    )

                if overall["mean_reward"] > best_eval:
                    best_eval = overall["mean_reward"]
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
