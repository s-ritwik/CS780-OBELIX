from __future__ import annotations

import argparse
import importlib.util
import os
from statistics import mean

import numpy as np

from train_mixed_policy import SCENARIO_LIBRARY, parse_scenarios


def load_policy(agent_file: str):
    spec = importlib.util.spec_from_file_location("submitted_agent", agent_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module from: {agent_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "policy"):
        raise AttributeError("Submission must define policy(obs, rng) -> action_str")
    return getattr(module, "policy")


def import_obelix(obelix_py: str):
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import OBELIX from {obelix_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OBELIX


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an agent across multiple OBELIX scenarios")
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--agent_file", type=str, required=True)
    parser.add_argument("--obelix_py", type=str, default=os.path.join(repo_dir, "obelix.py"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIO_LIBRARY.keys()),
    )
    args = parser.parse_args()

    agent_path = os.path.abspath(args.agent_file)
    policy_fn = load_policy(agent_path)
    OBELIX = import_obelix(args.obelix_py)
    scenarios = parse_scenarios(list(args.scenarios))

    overall_scores: list[float] = []
    print(f"[suite] agent={agent_path}")
    print(f"[suite] runs={args.runs} max_steps={args.max_steps}")

    for scenario_idx, scenario in enumerate(scenarios):
        scores: list[float] = []
        for run_idx in range(int(args.runs)):
            episode_seed = int(args.seed + scenario_idx * 10_000 + run_idx)
            env = OBELIX(
                scaling_factor=args.scaling_factor,
                arena_size=args.arena_size,
                max_steps=args.max_steps,
                wall_obstacles=scenario.wall_obstacles,
                difficulty=scenario.difficulty,
                box_speed=scenario.box_speed,
                seed=episode_seed,
            )
            obs = env.reset(seed=episode_seed)
            rng = np.random.default_rng(episode_seed)
            total_reward = 0.0
            done = False
            while not done:
                action = policy_fn(obs, rng)
                obs, reward, done = env.step(action, render=False)
                total_reward += float(reward)
            scores.append(total_reward)
            overall_scores.append(total_reward)

        print(
            f"[suite] {scenario.tag} mean={float(np.mean(scores)):.3f} "
            f"std={float(np.std(scores)):.3f} best={float(np.max(scores)):.3f} "
            f"worst={float(np.min(scores)):.3f}"
        )

    print(
        f"[suite] overall_mean={float(np.mean(overall_scores)):.3f} "
        f"overall_std={float(np.std(overall_scores)):.3f} "
        f"scenario_mean_of_means={mean([float(np.mean(overall_scores[i*args.runs:(i+1)*args.runs])) for i in range(len(scenarios))]):.3f}"
    )


if __name__ == "__main__":
    main()
