from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


def rollout(seed: int, actions: tuple[str, ...], *, max_steps: int, difficulty: int, wall: bool) -> tuple[tuple[int, ...], float, bool]:
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=max_steps,
        wall_obstacles=wall,
        difficulty=difficulty,
        box_speed=2,
        seed=seed,
    )
    obs = env.reset(seed=seed)
    total = 0.0
    done = False
    for action in actions:
        if done:
            break
        obs, reward, done = env.step(action, render=False)
        total += float(reward)
    return tuple(int(x) for x in np.asarray(obs).tolist()), total, done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    found = 0
    for length in range(1, args.length + 1):
        for seq in itertools.product(ACTIONS, repeat=length):
            rows = [rollout(seed, seq, max_steps=args.max_steps, difficulty=args.difficulty, wall=args.wall) for seed in args.seeds]
            obs_set = {row[0] for row in rows}
            if len(obs_set) == len(args.seeds) and all((not done) and reward > -300 for _, reward, done in rows):
                print(seq, rows, flush=True)
                found += 1
                if found >= args.limit:
                    return


if __name__ == "__main__":
    main()
