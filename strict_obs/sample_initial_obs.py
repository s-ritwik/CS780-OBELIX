from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(THIS_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from obelix import OBELIX


def collect(wall: bool, episodes: int, max_steps: int):
    counts = Counter()
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
        counts[tuple(obs.astype(int).tolist())] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=2000)
    args = parser.parse_args()

    for wall in [False, True]:
        counts = collect(wall=wall, episodes=args.episodes, max_steps=args.max_steps)
        print(f"wall={wall} unique={len(counts)}")
        for pat, count in counts.most_common(20):
            bits = "".join(map(str, pat))
            print(f"count={count} pattern={bits}")
        print()


if __name__ == "__main__":
    main()
