from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import Counter

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(THIS_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from obelix import OBELIX


def load_agent(agent_path: str):
    spec = importlib.util.spec_from_file_location("agent_mod_stats", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent from {agent_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect(agent_mod, wall: bool, episodes: int, max_steps: int):
    init = Counter()
    later = Counter()
    for seed in range(episodes):
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=max_steps,
            difficulty=3,
            wall_obstacles=wall,
            seed=seed,
        )
        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        init[tuple(obs.astype(int).tolist())] += 1
        done = False
        while not done:
            action = agent_mod.policy(obs, rng)
            obs, _, done = env.step(action, render=False)
            if not done:
                later[tuple(obs.astype(int).tolist())] += 1
    return init, later


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_file", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=2000)
    args = parser.parse_args()

    agent_mod = load_agent(os.path.abspath(args.agent_file))
    for wall in [False, True]:
        init, later = collect(agent_mod, wall=wall, episodes=args.episodes, max_steps=args.max_steps)
        print(f"wall={wall} unique_init={len(init)} unique_later={len(later)}")
        for pat, count in init.most_common(12):
            bits = "".join(map(str, pat))
            print(f"init_count={count} later_count={later.get(pat, 0)} pattern={bits}")
        only_init = [(pat, count) for pat, count in init.items() if later.get(pat, 0) == 0]
        only_init.sort(key=lambda item: item[1], reverse=True)
        top_only_init = [(count, "".join(map(str, pat))) for pat, count in only_init[:12]]
        print(f"only_init_top={top_only_init}")
        print()


if __name__ == "__main__":
    main()
