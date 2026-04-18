from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import sys

import numpy as np

BEST_POLICY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BEST_POLICY_DIR not in sys.path:
    sys.path.insert(0, BEST_POLICY_DIR)

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


def load_policy(path: str):
    spec = importlib.util.spec_from_file_location("agent_prefix_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy


def score_prefix(policy, seed: int, prefix: tuple[str, ...], candidate_idx: int) -> float:
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=2000,
        wall_obstacles=True,
        difficulty=3,
        box_speed=2,
        seed=seed,
    )
    obs = env.reset(seed=seed)
    rng = np.random.default_rng(seed * 1_000_000 + candidate_idx)
    total = 0.0
    done = False
    for action in prefix:
        obs, reward, done = env.step(action, render=False)
        total += float(reward)
        if done:
            return total
    while not done:
        action = policy(obs, rng)
        obs, reward, done = env.step(action, render=False)
        total += float(reward)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_file", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    best: list[tuple[float, tuple[str, ...]]] = []
    policy = load_policy(args.agent_file)
    candidate_idx = 0
    for length in range(1, args.length + 1):
        for prefix in itertools.product(ACTIONS, repeat=length):
            candidate_idx += 1
            score = score_prefix(policy, args.seed, prefix, candidate_idx)
            best.append((score, prefix))
            best.sort(reverse=True, key=lambda x: x[0])
            best = best[: args.limit]
        print(f"done length={length}", best[: min(5, len(best))], flush=True)
    print("BEST")
    for score, prefix in best:
        print(f"{score:.1f} {prefix}", flush=True)


if __name__ == "__main__":
    main()
