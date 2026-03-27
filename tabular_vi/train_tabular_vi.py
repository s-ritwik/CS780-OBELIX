from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


OBS_BITS = 18
NUM_ACTIONS = 5
NUM_OBS_STATES = 1 << OBS_BITS
TERMINAL_STATE_ID = NUM_OBS_STATES
NUM_STATES = NUM_OBS_STATES + 1
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
FW_ACTION = ACTIONS.index("FW")
BIT_WEIGHTS = (1 << np.arange(OBS_BITS, dtype=np.uint32)).astype(np.uint32)
ACTION_TIE_BREAK = np.array([0.0, 1e-6, 2e-6, 1e-6, 0.0], dtype=np.float64)


def import_parallel_obelix(parallel_dir: Path):
    parallel_dir_str = str(parallel_dir)
    if parallel_dir_str not in sys.path:
        sys.path.insert(0, parallel_dir_str)
    from parallel_env import ParallelOBELIX

    return ParallelOBELIX


def encode_observations(obs: np.ndarray) -> np.ndarray:
    obs_bits = (np.asarray(obs, dtype=np.float32) > 0.5).astype(np.uint32, copy=False)
    return np.sum(obs_bits * BIT_WEIGHTS, axis=1, dtype=np.uint32)


class EmpiricalMDP:
    def __init__(self, num_states: int, num_actions: int):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.visit_counts = np.zeros((self.num_states, self.num_actions), dtype=np.uint32)
        self.state_visit_counts = np.zeros((self.num_states,), dtype=np.uint32)
        self.transitions: dict[int, dict[int, list[float]]] = {}
        self.total_transitions = 0

    def update_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        for state, action, reward, next_state in zip(states, actions, rewards, next_states):
            s = int(state)
            a = int(action)
            ns = int(next_state)
            self.state_visit_counts[s] += 1
            self.visit_counts[s, a] += 1
            sa_index = s * self.num_actions + a
            bucket = self.transitions.get(sa_index)
            if bucket is None:
                bucket = {}
                self.transitions[sa_index] = bucket
            stats = bucket.get(ns)
            if stats is None:
                bucket[ns] = [1.0, float(reward)]
            else:
                stats[0] += 1.0
                stats[1] += float(reward)
        self.total_transitions += int(states.shape[0])

    def visited_states(self) -> np.ndarray:
        return np.flatnonzero(self.state_visit_counts)

    def unique_edges(self) -> int:
        return int(sum(len(bucket) for bucket in self.transitions.values()))


def choose_actions(
    state_ids: np.ndarray,
    q_values: np.ndarray,
    visit_counts: np.ndarray,
    epsilon: float,
    explore_bonus: float,
    rng: np.random.Generator,
) -> np.ndarray:
    q_rows = q_values[state_ids].astype(np.float64, copy=False)
    bonus = float(explore_bonus) / np.sqrt(visit_counts[state_ids].astype(np.float64) + 1.0)
    greedy_actions = np.argmax(q_rows + bonus + ACTION_TIE_BREAK, axis=1)
    if epsilon <= 0.0:
        return greedy_actions.astype(np.int64, copy=False)

    explore_mask = rng.random(state_ids.shape[0]) < float(epsilon)
    random_actions = rng.integers(0, NUM_ACTIONS, size=state_ids.shape[0], dtype=np.int64)
    actions = np.where(explore_mask, random_actions, greedy_actions)
    return actions.astype(np.int64, copy=False)


def value_iteration(
    model: EmpiricalMDP,
    gamma: float,
    max_iterations: int,
    tolerance: float,
    q_init: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    values = np.zeros((NUM_STATES,), dtype=np.float64)
    if q_init is not None:
        q_values = np.asarray(q_init, dtype=np.float64).copy()
        values[:-1] = np.max(q_values[:-1], axis=1)
    else:
        q_values = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)

    visited_states = model.visited_states()
    if visited_states.size == 0:
        policy = np.full((NUM_STATES,), FW_ACTION, dtype=np.uint8)
        return (
            q_values.astype(np.float32),
            values.astype(np.float32),
            policy,
            0,
            0.0,
        )

    last_delta = 0.0
    for iteration in range(1, int(max_iterations) + 1):
        delta = 0.0
        for state in visited_states:
            state_int = int(state)
            best_value = 0.0
            base_index = state_int * NUM_ACTIONS
            for action in range(NUM_ACTIONS):
                q_estimate = 0.0
                bucket = model.transitions.get(base_index + action)
                if bucket:
                    total = float(model.visit_counts[state_int, action])
                    inv_total = 1.0 / total
                    for next_state, stats in bucket.items():
                        count = float(stats[0])
                        reward_mean = float(stats[1]) / count
                        q_estimate += count * inv_total * (
                            reward_mean + float(gamma) * values[int(next_state)]
                        )
                q_values[state_int, action] = q_estimate
                if action == 0 or q_estimate > best_value:
                    best_value = q_estimate

            state_delta = abs(best_value - values[state_int])
            if state_delta > delta:
                delta = state_delta
            values[state_int] = best_value

        last_delta = delta
        if delta < float(tolerance):
            break

    q_values = q_values.astype(np.float32)
    values = values.astype(np.float32)
    policy = np.argmax(q_values + ACTION_TIE_BREAK.astype(np.float32), axis=1).astype(np.uint8)
    policy[TERMINAL_STATE_ID] = np.uint8(FW_ACTION)
    return q_values, values, policy, iteration, float(last_delta)


def evaluate_policy(
    parallel_env_cls,
    obelix_py: Path,
    env_kwargs: dict,
    policy: np.ndarray,
    num_workers: int,
    num_episodes: int,
    seed: int,
    mp_start_method: str,
    success_threshold: float,
) -> dict[str, float]:
    env = parallel_env_cls(
        obelix_py=str(obelix_py),
        num_envs=int(num_workers),
        base_seed=int(seed),
        env_kwargs=env_kwargs,
        mp_start_method=mp_start_method,
    )

    returns = []
    lengths = []
    successes = 0

    episode_returns = np.zeros((num_workers,), dtype=np.float64)
    episode_lengths = np.zeros((num_workers,), dtype=np.int32)

    try:
        obs = env.reset_all(seed=int(seed))
        state_ids = encode_observations(obs)
        completed = 0

        while completed < int(num_episodes):
            action_ids = policy[state_ids].astype(np.int64, copy=False)
            actions = [ACTIONS[int(action_id)] for action_id in action_ids]
            next_obs, rewards, dones = env.step(actions)
            episode_returns += rewards.astype(np.float64, copy=False)
            episode_lengths += 1

            done_indices = np.flatnonzero(dones)
            if done_indices.size > 0:
                for idx in done_indices.tolist():
                    if completed >= int(num_episodes):
                        break
                    returns.append(float(episode_returns[idx]))
                    lengths.append(int(episode_lengths[idx]))
                    if episode_returns[idx] >= float(success_threshold):
                        successes += 1
                    completed += 1

                reset_map = env.reset(env_indices=done_indices.tolist(), seed=None)
                for idx in done_indices.tolist():
                    if idx in reset_map:
                        next_obs[idx] = reset_map[idx]
                    episode_returns[idx] = 0.0
                    episode_lengths[idx] = 0

            state_ids = encode_observations(next_obs)
    finally:
        env.close()

    return {
        "episodes": float(len(returns)),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "success_rate": float(successes / len(returns)) if returns else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate an observation-level OBELIX MDP and solve it with value iteration.",
    )
    base_dir = Path(__file__).resolve().parent
    repo_dir = base_dir.parent
    default_obelix = repo_dir / "obelix.py"
    default_out = base_dir / "artifacts" / "tabular_vi_table.npz"

    parser.add_argument("--obelix_py", type=str, default=str(default_obelix))
    parser.add_argument("--out", type=str, default=str(default_out))

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--total_env_steps", type=int, default=160000)
    parser.add_argument("--planning_interval", type=int, default=16000)
    parser.add_argument("--planning_iterations", type=int, default=30)
    parser.add_argument("--final_planning_iterations", type=int, default=300)

    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon_start", type=float, default=0.30)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--explore_bonus", type=float, default=1.5)
    parser.add_argument("--value_tolerance", type=float, default=1e-5)

    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--wall_obstacles", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mp_start_method", type=str, default="fork", choices=["fork", "spawn", "forkserver"])
    parser.add_argument("--log_interval", type=int, default=16000)

    parser.add_argument("--eval_episodes", type=int, default=128)
    parser.add_argument("--success_threshold", type=float, default=1000.0)
    return parser


def linear_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 0:
        return float(end)
    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    return float(start + progress * (end - start))


def save_outputs(
    out_path: Path,
    q_values: np.ndarray,
    values: np.ndarray,
    policy: np.ndarray,
    visit_counts: np.ndarray,
    state_visit_counts: np.ndarray,
    summary: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        q_values=q_values,
        values=values,
        policy=policy,
        visit_counts=visit_counts,
        state_visit_counts=state_visit_counts,
        actions=np.asarray(ACTIONS),
        terminal_state=np.asarray([TERMINAL_STATE_ID], dtype=np.int32),
    )
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    out_path = Path(args.out).resolve()
    obelix_py = Path(args.obelix_py).resolve()
    repo_dir = obelix_py.parent
    parallel_env_dir = repo_dir / "parallel_ddqn"

    if not obelix_py.exists():
        raise FileNotFoundError(f"Environment file not found: {obelix_py}")
    if not parallel_env_dir.exists():
        raise FileNotFoundError(f"parallel_ddqn directory not found: {parallel_env_dir}")

    parallel_env_cls = import_parallel_obelix(parallel_env_dir)
    env_kwargs = {
        "scaling_factor": int(args.scaling_factor),
        "arena_size": int(args.arena_size),
        "max_steps": int(args.max_steps),
        "wall_obstacles": bool(args.wall_obstacles),
        "difficulty": int(args.difficulty),
        "box_speed": int(args.box_speed),
    }

    rng = np.random.default_rng(int(args.seed))
    model = EmpiricalMDP(num_states=NUM_STATES, num_actions=NUM_ACTIONS)
    q_values = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float32)
    values = np.zeros((NUM_STATES,), dtype=np.float32)
    policy = np.full((NUM_STATES,), FW_ACTION, dtype=np.uint8)

    env = parallel_env_cls(
        obelix_py=str(obelix_py),
        num_envs=int(args.num_workers),
        base_seed=int(args.seed),
        env_kwargs=env_kwargs,
        mp_start_method=args.mp_start_method,
    )

    episode_returns = np.zeros((args.num_workers,), dtype=np.float64)
    completed_returns = []
    successful_episodes = 0
    total_episodes = 0
    planning_runs = 0

    start_time = time.time()
    last_log_time = start_time

    try:
        obs = env.reset_all(seed=int(args.seed))
        state_ids = encode_observations(obs)
        env_steps = 0
        next_plan_at = max(int(args.planning_interval), int(args.num_workers))
        next_log_at = max(int(args.log_interval), int(args.num_workers))

        while env_steps < int(args.total_env_steps):
            epsilon = linear_schedule(
                step=env_steps,
                total_steps=int(args.total_env_steps),
                start=float(args.epsilon_start),
                end=float(args.epsilon_end),
            )
            action_ids = choose_actions(
                state_ids=state_ids,
                q_values=q_values,
                visit_counts=model.visit_counts,
                epsilon=epsilon,
                explore_bonus=float(args.explore_bonus),
                rng=rng,
            )
            actions = [ACTIONS[int(action_id)] for action_id in action_ids]
            next_obs, rewards, dones = env.step(actions)
            next_state_ids = encode_observations(next_obs)
            next_state_ids[dones] = TERMINAL_STATE_ID

            model.update_batch(
                states=state_ids,
                actions=action_ids,
                rewards=rewards,
                next_states=next_state_ids,
            )

            episode_returns += rewards.astype(np.float64, copy=False)
            done_indices = np.flatnonzero(dones)
            if done_indices.size > 0:
                for idx in done_indices.tolist():
                    total_episodes += 1
                    completed_returns.append(float(episode_returns[idx]))
                    if episode_returns[idx] >= float(args.success_threshold):
                        successful_episodes += 1
                    episode_returns[idx] = 0.0

                reset_map = env.reset(env_indices=done_indices.tolist(), seed=None)
                for idx in done_indices.tolist():
                    if idx in reset_map:
                        next_obs[idx] = reset_map[idx]

            state_ids = encode_observations(next_obs)
            env_steps += int(args.num_workers)

            if env_steps >= next_plan_at and model.total_transitions > 0:
                q_values, values, policy, vi_iters, vi_delta = value_iteration(
                    model=model,
                    gamma=float(args.gamma),
                    max_iterations=int(args.planning_iterations),
                    tolerance=float(args.value_tolerance),
                    q_init=q_values,
                )
                planning_runs += 1
                print(
                    f"[plan] env_steps={env_steps} visited_states={model.visited_states().size} "
                    f"transitions={model.total_transitions} vi_iters={vi_iters} vi_delta={vi_delta:.6f}",
                    flush=True,
                )
                next_plan_at += max(int(args.planning_interval), int(args.num_workers))

            if env_steps >= next_log_at:
                elapsed = time.time() - start_time
                recent_rate = float(env_steps / max(elapsed, 1e-6))
                mean_return = float(np.mean(completed_returns[-64:])) if completed_returns else 0.0
                success_rate = (
                    float(successful_episodes / total_episodes) if total_episodes > 0 else 0.0
                )
                print(
                    f"[collect] env_steps={env_steps} visited_states={model.visited_states().size} "
                    f"unique_edges={model.unique_edges()} episodes={total_episodes} "
                    f"recent_mean_return={mean_return:.2f} success_rate={success_rate:.3f} "
                    f"steps_per_sec={recent_rate:.1f} dt={time.time() - last_log_time:.1f}s",
                    flush=True,
                )
                last_log_time = time.time()
                next_log_at += max(int(args.log_interval), int(args.num_workers))

        q_values, values, policy, vi_iters, vi_delta = value_iteration(
            model=model,
            gamma=float(args.gamma),
            max_iterations=int(args.final_planning_iterations),
            tolerance=float(args.value_tolerance),
            q_init=q_values,
        )
        planning_runs += 1
        print(
            f"[final-plan] env_steps={env_steps} visited_states={model.visited_states().size} "
            f"transitions={model.total_transitions} vi_iters={vi_iters} vi_delta={vi_delta:.6f}",
            flush=True,
        )
    finally:
        env.close()

    eval_seed = int(args.seed) + 50_000
    eval_metrics = evaluate_policy(
        parallel_env_cls=parallel_env_cls,
        obelix_py=obelix_py,
        env_kwargs=env_kwargs,
        policy=policy,
        num_workers=int(args.num_workers),
        num_episodes=int(args.eval_episodes),
        seed=eval_seed,
        mp_start_method=args.mp_start_method,
        success_threshold=float(args.success_threshold),
    )

    elapsed = time.time() - start_time
    visited_states = model.visited_states()
    summary = {
        "config": {
            "obelix_py": str(obelix_py),
            "num_workers": int(args.num_workers),
            "total_env_steps": int(args.total_env_steps),
            "planning_interval": int(args.planning_interval),
            "planning_iterations": int(args.planning_iterations),
            "final_planning_iterations": int(args.final_planning_iterations),
            "gamma": float(args.gamma),
            "epsilon_start": float(args.epsilon_start),
            "epsilon_end": float(args.epsilon_end),
            "explore_bonus": float(args.explore_bonus),
            "value_tolerance": float(args.value_tolerance),
            "env_kwargs": env_kwargs,
            "seed": int(args.seed),
            "mp_start_method": args.mp_start_method,
        },
        "training": {
            "elapsed_sec": float(elapsed),
            "env_steps": int(model.total_transitions),
            "planning_runs": int(planning_runs),
            "visited_states": int(visited_states.size),
            "unique_edges": int(model.unique_edges()),
            "episodes_completed": int(total_episodes),
            "success_rate": float(successful_episodes / total_episodes) if total_episodes else 0.0,
            "mean_return_last_128": float(np.mean(completed_returns[-128:])) if completed_returns else 0.0,
        },
        "evaluation": eval_metrics,
        "artifacts": {
            "table_path": str(out_path),
            "summary_path": str(out_path.with_suffix(".summary.json")),
        },
    }
    save_outputs(
        out_path=out_path,
        q_values=q_values,
        values=values,
        policy=policy,
        visit_counts=model.visit_counts,
        state_visit_counts=model.state_visit_counts,
        summary=summary,
    )

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
