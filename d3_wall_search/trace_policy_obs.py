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
    spec = importlib.util.spec_from_file_location("trace_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_file", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wall", action="store_true")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    policy = load_policy(args.agent_file)
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=2000,
        wall_obstacles=args.wall,
        difficulty=3,
        box_speed=2,
        seed=args.seed,
    )
    obs = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"t=0 obs={tuple(int(x) for x in obs.tolist())}", flush=True)
    total = 0.0
    for t in range(1, args.steps + 1):
        action = policy(obs, rng)
        obs, reward, done = env.step(action, render=False)
        total += float(reward)
        bits = tuple(int(x) for x in np.asarray(obs).tolist())
        if t <= 20 or t % 10 == 0 or bits[16] or bits[17] or sum(bits[:16]) > 0:
            print(f"t={t} a={action} r={reward:.1f} total={total:.1f} obs={bits}", flush=True)
        if done:
            print(f"done t={t} total={total:.1f}", flush=True)
            break


if __name__ == "__main__":
    main()
