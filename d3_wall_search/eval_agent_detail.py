from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from obelix import OBELIX


def load_policy(path: str):
    spec = importlib.util.spec_from_file_location("candidate_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_file", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--wall_obstacles", action="store_true")
    args = parser.parse_args()

    policy = load_policy(args.agent_file)
    env = OBELIX(
        max_steps=args.max_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        wall_obstacles=args.wall_obstacles,
        seed=args.seed,
    )
    scores: list[float] = []
    steps: list[int] = []
    for i in range(args.runs):
        seed = args.seed + i
        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        total = 0.0
        done = False
        step_count = 0
        while not done:
            action = policy(obs, rng)
            obs, reward, done = env.step(action, render=False)
            total += float(reward)
            step_count += 1
        scores.append(total)
        steps.append(step_count)
        print(f"seed={seed} score={total:.1f} steps={step_count}", flush=True)
    arr = np.asarray(scores, dtype=np.float64)
    print(f"mean={arr.mean():.3f} std={arr.std():.3f} min={arr.min():.1f} max={arr.max():.1f}", flush=True)


if __name__ == "__main__":
    main()
