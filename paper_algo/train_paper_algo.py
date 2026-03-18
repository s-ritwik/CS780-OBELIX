from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from controller import (
    ACTIONS,
    BASE_OBS_DIM,
    BEHAVIORS,
    IntrinsicRewardConfig,
    L45_ACTION,
    PaperSubsumptionController,
    R45_ACTION,
    STUCK_BIT,
    behavior_reward,
    binarize_observation,
)


def import_symbol(py_file: str, symbol: str):
    spec = importlib.util.spec_from_file_location("obelix_module", py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {symbol} from {py_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise AttributeError(f"{symbol} not found in {py_file}")
    return getattr(module, symbol)


class MixedWallVecEnv:
    def __init__(
        self,
        *,
        vec_env_cls,
        num_envs: int,
        scaling_factor: int,
        arena_size: int,
        max_steps: int,
        difficulty: int,
        box_speed: int,
        seed: int,
        device: str,
    ) -> None:
        if int(num_envs) < 2:
            raise ValueError("wall_mix requires num_envs >= 2")

        self.num_envs = int(num_envs)
        self.counts = [self.num_envs // 2, self.num_envs - (self.num_envs // 2)]
        self.offsets = [0, self.counts[0]]
        wall_flags = [False, True]
        self.envs = []

        for idx, wall_obstacles in enumerate(wall_flags):
            env = vec_env_cls(
                num_envs=self.counts[idx],
                scaling_factor=scaling_factor,
                arena_size=arena_size,
                max_steps=max_steps,
                wall_obstacles=wall_obstacles,
                difficulty=difficulty,
                box_speed=box_speed,
                seed=seed + idx * 100_000,
                device=device,
            )
            self.envs.append(env)

    def reset_all(self, seed: int | None = None) -> np.ndarray:
        batches: list[np.ndarray] = []
        for idx, env in enumerate(self.envs):
            env_seed = None if seed is None else int(seed + self.offsets[idx])
            batches.append(env.reset_all(seed=env_seed))
        return np.concatenate(batches, axis=0)

    def reset(self, env_indices: list[int], seed: int | None = None) -> dict[int, np.ndarray]:
        obs_map: dict[int, np.ndarray] = {}
        if not env_indices:
            return obs_map

        for env_idx, env in enumerate(self.envs):
            offset = self.offsets[env_idx]
            count = self.counts[env_idx]
            local_indices = [int(idx - offset) for idx in env_indices if offset <= idx < offset + count]
            if not local_indices:
                continue
            env_seed = None if seed is None else int(seed + offset)
            local_obs = env.reset(env_indices=local_indices, seed=env_seed)
            for local_idx, obs in local_obs.items():
                obs_map[offset + int(local_idx)] = obs
        return obs_map

    def step(self, actions: np.ndarray):
        action_arr = np.asarray(actions, dtype=np.int64)
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


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def resolve_env_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested env device '{requested_device}' but CUDA is not available in this Python env."
        )
    return requested_device


def mix_env_reward(env_reward: float, env_reward_scale: float, env_reward_clip: float) -> float:
    if env_reward_scale == 0.0:
        return 0.0
    return float(env_reward_scale) * float(np.clip(env_reward, -env_reward_clip, env_reward_clip))


def last_checkpoint_path(out_path: str | Path) -> Path:
    out = Path(out_path)
    if out.suffix:
        return out.with_name(f"{out.stem}.last{out.suffix}")
    return out.with_name(f"{out.name}.last")


def choose_unwedge_turn_action(obs_bits: np.ndarray, last_turn_sign: int) -> tuple[int, int]:
    left_score = int(np.sum(obs_bits[:4]))
    right_score = int(np.sum(obs_bits[12:16]))
    if left_score > right_score:
        return R45_ACTION, -1
    if right_score > left_score:
        return L45_ACTION, 1
    if last_turn_sign >= 0:
        return R45_ACTION, -1
    return L45_ACTION, 1


def set_controller_epsilon(controller: PaperSubsumptionController, epsilon: float) -> None:
    epsilon_value = float(epsilon)
    for module in controller.modules.values():
        module.epsilon = epsilon_value


def reconfigure_loaded_controller(controller: PaperSubsumptionController, args: argparse.Namespace) -> None:
    controller.push_persistence_steps = int(args.push_persistence_steps)
    controller.unwedge_persistence_steps = int(args.unwedge_persistence_steps)
    for module in controller.modules.values():
        module.alpha = float(args.alpha)
        module.gamma = float(args.gamma)
        module.epsilon = float(args.epsilon)
        module.match_threshold = float(args.match_threshold)
        module.q_match_delta = float(args.q_match_delta)
        module.cluster_distance_threshold = float(args.cluster_distance_threshold)
        module.max_clusters_per_action = int(args.max_clusters_per_action)
        for action_model in module.actions:
            action_model.match_threshold = float(args.match_threshold)
            action_model.q_match_delta = float(args.q_match_delta)
            action_model.cluster_distance_threshold = float(args.cluster_distance_threshold)
            action_model.max_clusters = int(args.max_clusters_per_action)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-style clustered Q-learning trainer for OBELIX",
    )

    base_dir = Path(__file__).resolve().parent
    repo_dir = base_dir.parent
    default_env = repo_dir / "obelix_torch.py"

    parser.add_argument("--obelix_torch_py", type=str, default=str(default_env))
    parser.add_argument("--out", type=str, default=str(base_dir / "paper_policy.json"))

    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--total_env_steps", type=int, default=500_000)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument(
        "--wall_mix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train on a 50/50 mix of wall/no-wall environments.",
    )
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument(
        "--env_device",
        type=str,
        default="auto",
        help="OBELIXVectorized device: auto|cpu|cuda|cuda:N",
    )

    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument(
        "--epsilon_end",
        type=float,
        default=0.03,
        help="Final epsilon for linear exploration decay. Defaults to --epsilon if unset.",
    )
    parser.add_argument("--match_threshold", type=float, default=1e-4)
    parser.add_argument("--q_match_delta", type=float, default=1.5)
    parser.add_argument("--cluster_distance_threshold", type=float, default=1.0)
    parser.add_argument("--max_clusters_per_action", type=int, default=256)
    parser.add_argument("--push_persistence_steps", type=int, default=5)
    parser.add_argument("--unwedge_persistence_steps", type=int, default=5)
    parser.add_argument(
        "--init_policy",
        type=str,
        default=None,
        help="Optional existing paper_policy.json to continue training from.",
    )
    parser.add_argument(
        "--env_reward_scale",
        type=float,
        default=0.08,
        help="Weight for clipped OBELIX env reward added to intrinsic module reward.",
    )
    parser.add_argument(
        "--env_reward_clip",
        type=float,
        default=100.0,
        help="Absolute clip applied to env reward before mixing into module reward.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20_000)
    parser.add_argument(
        "--save_best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the best checkpoint at --out and save the final weights separately as *.last.json.",
    )
    parser.add_argument(
        "--best_checkpoint_min_episodes",
        type=int,
        default=20,
        help="Minimum completed episodes before best-checkpoint tracking becomes active.",
    )
    parser.add_argument(
        "--success_reward_threshold",
        type=float,
        default=1000.0,
        help="Logging-only threshold for counting likely task successes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    resolved_env_device = resolve_env_device(args.env_device)

    if resolved_env_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    vec_env_cls = import_symbol(args.obelix_torch_py, "OBELIXVectorized")
    if args.wall_mix:
        vec_env = MixedWallVecEnv(
            vec_env_cls=vec_env_cls,
            num_envs=args.num_envs,
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed * 10_000,
            device=resolved_env_device,
        )
    else:
        vec_env = vec_env_cls(
            num_envs=args.num_envs,
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed * 10_000,
            device=resolved_env_device,
        )

    if args.init_policy:
        controller = PaperSubsumptionController.load(args.init_policy)
        reconfigure_loaded_controller(controller, args)
    else:
        controller = PaperSubsumptionController(
            obs_dim=BASE_OBS_DIM,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            match_threshold=args.match_threshold,
            q_match_delta=args.q_match_delta,
            cluster_distance_threshold=args.cluster_distance_threshold,
            max_clusters_per_action=args.max_clusters_per_action,
            push_persistence_steps=args.push_persistence_steps,
            unwedge_persistence_steps=args.unwedge_persistence_steps,
        )
    reward_config = IntrinsicRewardConfig()

    rng = np.random.default_rng(args.seed)
    obs = vec_env.reset_all(seed=args.seed * 10_000)
    prev_stuck_memory = np.zeros((args.num_envs,), dtype=np.int8)
    unwedge_turn_action = np.full((args.num_envs,), -1, dtype=np.int64)
    unwedge_probe_forward = np.zeros((args.num_envs,), dtype=bool)
    last_unwedge_turn_sign = np.zeros((args.num_envs,), dtype=np.int8)
    push_timers = np.zeros((args.num_envs,), dtype=np.int32)
    unwedge_timers = np.zeros((args.num_envs,), dtype=np.int32)
    env_returns = np.zeros((args.num_envs,), dtype=np.float32)
    env_lengths = np.zeros((args.num_envs,), dtype=np.int32)

    recent_returns = deque(maxlen=200)
    recent_lengths = deque(maxlen=200)
    recent_successes = deque(maxlen=200)
    train_reward_history = {behavior: deque(maxlen=1000) for behavior in BEHAVIORS}
    intrinsic_reward_history = {behavior: deque(maxlen=1000) for behavior in BEHAVIORS}
    behavior_counts_since_log = {behavior: 0 for behavior in BEHAVIORS}

    total_completed_eps = 0
    env_steps = 0
    last_log_step = 0
    start_time = time.time()
    best_score: tuple[float, float] | None = None
    best_step = 0
    best_success = float("nan")
    best_return = float("nan")
    out_path = Path(args.out)

    print(
        f"[setup] env_device={resolved_env_device} num_envs={args.num_envs} "
        f"difficulty={args.difficulty} wall_obstacles={args.wall_obstacles} wall_mix={args.wall_mix}"
    )
    if resolved_env_device.startswith("cuda"):
        print(
            "[setup] GPU mode uses CUDA for the batched OBELIX environment. "
            "Clustered Q-updates still run on CPU because the paper algorithm is state-cluster based."
        )
    print(
        f"[setup] env_reward_scale={args.env_reward_scale} env_reward_clip={args.env_reward_clip} "
        f"epsilon={args.epsilon}->{args.epsilon_end} save_best={args.save_best}"
    )
    if args.init_policy:
        print(f"[setup] init_policy={args.init_policy}")

    try:
        while env_steps < args.total_env_steps:
            progress = min(1.0, env_steps / max(1, args.total_env_steps))
            current_epsilon = float(args.epsilon + progress * (args.epsilon_end - args.epsilon))
            set_controller_epsilon(controller, current_epsilon)
            obs_bits = np.asarray([binarize_observation(ob) for ob in obs], dtype=np.int8)

            behaviors = []
            action_idx = np.empty((args.num_envs,), dtype=np.int64)
            for env_id in range(args.num_envs):
                behavior, push_timer, unwedge_timer = controller.behavior_for_state(
                    obs_bits=obs_bits[env_id],
                    push_timer=int(push_timers[env_id]),
                    unwedge_timer=int(unwedge_timers[env_id]),
                )
                behaviors.append(behavior)
                push_timers[env_id] = push_timer
                unwedge_timers[env_id] = unwedge_timer

                if not bool(obs_bits[env_id, STUCK_BIT]):
                    unwedge_turn_action[env_id] = -1
                    unwedge_probe_forward[env_id] = False
                elif unwedge_turn_action[env_id] < 0:
                    turn_action, turn_sign = choose_unwedge_turn_action(
                        obs_bits=obs_bits[env_id],
                        last_turn_sign=int(last_unwedge_turn_sign[env_id]),
                    )
                    unwedge_turn_action[env_id] = int(turn_action)
                    last_unwedge_turn_sign[env_id] = int(turn_sign)

                if bool(obs_bits[env_id, STUCK_BIT]) and not bool(unwedge_probe_forward[env_id]):
                    action_idx[env_id] = int(unwedge_turn_action[env_id])
                    unwedge_probe_forward[env_id] = True
                elif bool(obs_bits[env_id, STUCK_BIT]):
                    action_idx[env_id] = ACTIONS.index("FW")
                    unwedge_probe_forward[env_id] = False
                else:
                    action_idx[env_id] = controller.select_action(
                        behavior=behavior,
                        obs_bits=obs_bits[env_id],
                        rng=rng,
                    )
                behavior_counts_since_log[behavior] += 1

            next_obs, env_rewards, dones = vec_env.step(action_idx)
            next_obs_bits = np.asarray([binarize_observation(ob) for ob in next_obs], dtype=np.int8)

            next_behaviors = []
            next_push_timers = push_timers.copy()
            next_unwedge_timers = unwedge_timers.copy()
            for env_id in range(args.num_envs):
                next_behavior, next_push_timer, next_unwedge_timer = controller.behavior_for_state(
                    obs_bits=next_obs_bits[env_id],
                    push_timer=int(next_push_timers[env_id]),
                    unwedge_timer=int(next_unwedge_timers[env_id]),
                )
                next_behaviors.append(next_behavior)
                next_push_timers[env_id] = next_push_timer
                next_unwedge_timers[env_id] = next_unwedge_timer

            for env_id in range(args.num_envs):
                behavior = behaviors[env_id]
                intrinsic_reward = behavior_reward(
                    behavior=behavior,
                    obs_bits=obs_bits[env_id],
                    action_idx=int(action_idx[env_id]),
                    next_obs_bits=next_obs_bits[env_id],
                    prev_stuck_bit=prev_stuck_memory[env_id],
                    config=reward_config,
                )
                training_reward = intrinsic_reward + mix_env_reward(
                    env_reward=float(env_rewards[env_id]),
                    env_reward_scale=args.env_reward_scale,
                    env_reward_clip=args.env_reward_clip,
                )
                intrinsic_reward_history[behavior].append(float(intrinsic_reward))
                train_reward_history[behavior].append(float(training_reward))
                controller.modules[behavior].update(
                    state_bits=obs_bits[env_id],
                    action_idx=int(action_idx[env_id]),
                    reward=float(training_reward),
                    next_state_bits=next_obs_bits[env_id],
                    next_applicable=bool((not dones[env_id]) and (next_behaviors[env_id] == behavior)),
                )

            push_timers[:] = next_push_timers
            unwedge_timers[:] = next_unwedge_timers
            prev_stuck_memory[:] = obs_bits[:, STUCK_BIT]
            env_returns += env_rewards
            env_lengths += 1
            env_steps += args.num_envs
            obs = next_obs

            done_idx = np.flatnonzero(dones)
            if done_idx.size > 0:
                for idx in done_idx:
                    recent_returns.append(float(env_returns[idx]))
                    recent_lengths.append(int(env_lengths[idx]))
                    recent_successes.append(int(float(env_rewards[idx]) >= args.success_reward_threshold))
                    total_completed_eps += 1

                reset_map = vec_env.reset(
                    env_indices=done_idx.tolist(),
                    seed=args.seed * 10_000 + env_steps,
                )
                for idx, reset_obs in reset_map.items():
                    obs[idx] = reset_obs
                    prev_stuck_memory[idx] = 0
                    unwedge_turn_action[idx] = -1
                    unwedge_probe_forward[idx] = False
                    last_unwedge_turn_sign[idx] = 0
                    push_timers[idx] = 0
                    unwedge_timers[idx] = 0
                    env_returns[idx] = 0.0
                    env_lengths[idx] = 0

            if env_steps - last_log_step >= args.log_interval:
                elapsed = max(1e-6, time.time() - start_time)
                sps = env_steps / elapsed
                mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
                mean_length = float(np.mean(recent_lengths)) if recent_lengths else float("nan")
                success_rate = float(np.mean(recent_successes)) if recent_successes else float("nan")
                train_reward_parts = " ".join(
                    f"{name}:{(float(np.mean(history)) if history else 0.0):.2f}"
                    for name, history in train_reward_history.items()
                )
                intrinsic_reward_parts = " ".join(
                    f"{name}:{(float(np.mean(history)) if history else 0.0):.2f}"
                    for name, history in intrinsic_reward_history.items()
                )
                count_total = max(1, sum(behavior_counts_since_log.values()))
                behavior_mix = " ".join(
                    f"{name}:{behavior_counts_since_log[name] / count_total:.2f}" for name in BEHAVIORS
                )
                cluster_summary = " ".join(
                    f"{behavior}:{sum(len(model.clusters) for model in controller.modules[behavior].actions)}"
                    for behavior in BEHAVIORS
                )
                print(
                    f"[train] env_steps={env_steps} sps={sps:.1f} "
                    f"recent_return={mean_return:.1f} recent_len={mean_length:.1f} "
                    f"success={success_rate:.3f} episodes={total_completed_eps} "
                    f"train_reward=[{train_reward_parts}] intrinsic_reward=[{intrinsic_reward_parts}] "
                    f"behavior_mix=[{behavior_mix}] "
                    f"clusters=[{cluster_summary}] elapsed={format_hms(elapsed)}"
                )

                if (
                    args.save_best
                    and len(recent_returns) >= args.best_checkpoint_min_episodes
                    and not np.isnan(mean_return)
                    and not np.isnan(success_rate)
                ):
                    score = (success_rate, mean_return)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_step = env_steps
                        best_success = success_rate
                        best_return = mean_return
                        controller.save(
                            out_path,
                            metadata={
                                "checkpoint_kind": "best",
                                "best_env_steps": best_step,
                                "best_recent_return": best_return,
                                "best_success_rate": best_success,
                                "train": vars(args),
                                "resolved_env_device": resolved_env_device,
                                "epsilon_at_save": current_epsilon,
                            },
                        )
                        print(
                            f"[checkpoint] saved best -> {out_path} env_steps={best_step} "
                            f"recent_return={best_return:.1f} success={best_success:.3f}"
                        )

                for behavior in BEHAVIORS:
                    behavior_counts_since_log[behavior] = 0
                last_log_step = env_steps

    finally:
        vec_env.close()

    total_elapsed = max(0.0, time.time() - start_time)
    metadata = {
        "algorithm": "paper_clustered_subsumption_q_learning",
        "paper": "Automatic Programming of Behaviour-based Robots using Reinforcement Learning (AAAI 1991)",
        "env": {
            "difficulty": args.difficulty,
            "wall_obstacles": args.wall_obstacles,
            "box_speed": args.box_speed,
            "scaling_factor": args.scaling_factor,
            "arena_size": args.arena_size,
            "max_steps": args.max_steps,
        },
        "train": vars(args),
        "reward_config": reward_config.__dict__,
        "resolved_env_device": resolved_env_device,
        "elapsed_seconds": total_elapsed,
        "completed_episodes": total_completed_eps,
        "best_checkpoint": {
            "saved": bool(best_score is not None),
            "env_steps": best_step,
            "recent_return": best_return,
            "success_rate": best_success,
        },
    }
    final_out_path = out_path
    if args.save_best and best_score is not None:
        final_out_path = last_checkpoint_path(out_path)
    controller.save(final_out_path, metadata=metadata)
    if args.save_best and best_score is not None:
        print(f"[done] saved best checkpoint -> {out_path}")
        print(f"[done] saved final checkpoint -> {final_out_path}")
    else:
        print(f"[done] saved checkpoint -> {final_out_path}")
    print(f"[done] train_time={format_hms(total_elapsed)}")


if __name__ == "__main__":
    main()
