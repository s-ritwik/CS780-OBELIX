from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from obelix import OBELIX


@dataclass
class SuiteResult:
    scores: list[float]
    steps: list[int]

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std(self) -> float:
        return float(np.std(self.scores))


def load_policy(path: str):
    spec = importlib.util.spec_from_file_location("suite_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy


def run_suite(
    agent_file: str,
    seeds: list[int],
    *,
    wall_obstacles: bool,
    max_steps: int,
    difficulty: int,
    box_speed: int,
    scaling_factor: int,
    arena_size: int,
) -> SuiteResult:
    policy = load_policy(agent_file)
    env = OBELIX(
        max_steps=max_steps,
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        difficulty=difficulty,
        box_speed=box_speed,
        wall_obstacles=wall_obstacles,
        seed=seeds[0] if seeds else 0,
    )
    scores: list[float] = []
    steps: list[int] = []
    for seed in seeds:
        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        total = 0.0
        done = False
        n = 0
        while not done:
            action = policy(obs, rng)
            obs, reward, done = env.step(action, render=False)
            total += float(reward)
            n += 1
        scores.append(total)
        steps.append(n)
        print(
            f"{'wall' if wall_obstacles else 'nowall'} seed={seed} score={total:.1f} steps={n}",
            flush=True,
        )
    return SuiteResult(scores=scores, steps=steps)


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.replace(",", " ").split():
        if not part:
            continue
        seeds.append(int(part))
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_file", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    result = run_suite(
        args.agent_file,
        seeds,
        wall_obstacles=args.wall_obstacles,
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
    )
    print(
        f"mean={result.mean:.3f} std={result.std:.3f} min={min(result.scores):.1f} max={max(result.scores):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
