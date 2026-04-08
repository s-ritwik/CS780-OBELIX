from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASYM = os.path.join(ROOT, "assym_ppo")
for path in (ROOT, ASYM):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

from obelix import OBELIX
from teacher import ScriptedTeacherState, scripted_teacher_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--box_speed", type=int, default=2)
    args = parser.parse_args()

    rows = {}
    for i in range(args.runs):
        seed = args.seed + i
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=args.max_steps,
            wall_obstacles=True,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=seed,
        )
        obs = env.reset(seed=seed)
        init_obs = tuple(int(x) for x in np.asarray(obs).tolist())
        state = ScriptedTeacherState()
        actions: list[str] = []
        total = 0.0
        done = False
        while not done:
            action = scripted_teacher_action(env, state)
            actions.append(action)
            obs, reward, done = env.step(action, render=False)
            total += float(reward)
        rows[str(seed)] = {
            "seed": seed,
            "score": total,
            "steps": len(actions),
            "init_obs": init_obs,
            "actions": actions,
        }
        print(f"seed={seed} teacher_score={total:.1f} steps={len(actions)} init={init_obs}", flush=True)
    with open(args.out, "w") as f:
        json.dump(rows, f)
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
