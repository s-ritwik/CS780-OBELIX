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


def load_agent(path: str):
    spec = importlib.util.spec_from_file_location("probe_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_seeds(raw: str) -> list[int]:
    return [int(part) for part in raw.replace(",", " ").split() if part]


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

    agent = load_agent(args.agent_file)
    env = OBELIX(
        max_steps=args.max_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        wall_obstacles=args.wall_obstacles,
        seed=0,
    )

    label = "wall" if args.wall_obstacles else "nowall"
    rng_refs: list[np.random.Generator] = []
    for seed in parse_seeds(args.seeds):
        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        # Keep old RNG objects alive so Python cannot recycle their object id.
        rng_refs.append(rng)
        total = 0.0
        done = False
        steps = 0
        # The policy detects reset only when called, so do not inspect _STEP
        # until after the first call of this episode.
        probe_steps = int(getattr(agent, "_PROBE_STEPS", 20))
        while not done and steps < probe_steps:
            action = agent.policy(obs, rng)
            obs, reward, done = env.step(action, render=False)
            total += float(reward)
            steps += 1

        obs_arr = np.asarray(obs, dtype=np.float32)
        wall_prob = float(agent._p_wall(obs_arr)) if hasattr(agent, "_p_wall") else float("nan")
        stats = getattr(agent, "_STATS", np.zeros((12,), dtype=np.float32))
        first_obs = getattr(agent, "_FIRST_OBS", np.zeros((18,), dtype=np.float32))
        print(
            f"{label} seed={seed} steps={steps} probe_total={total:.1f} "
            f"p_wall={wall_prob:.6f} first_sum={float(np.sum(first_obs[:16])):.1f} "
            f"first_front={float(np.sum(first_obs[4:12])):.1f} "
            f"stats_sum={float(stats[11]):.1f} stats_front={float(stats[1]):.1f} "
            f"obs_sum={float(np.sum(obs_arr[:16])):.1f} obs_front={float(np.sum(obs_arr[4:12])):.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
