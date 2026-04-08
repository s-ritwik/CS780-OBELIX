from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASYM = os.path.join(ROOT, "assym_ppo")
for path in (ROOT, ASYM):
    if path not in sys.path:
        sys.path.insert(0, path)

from obelix import OBELIX
from teacher import ScriptedTeacherState, scripted_teacher_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--probe", nargs="*", default=[])
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    args = parser.parse_args()

    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=args.max_steps,
        wall_obstacles=True,
        difficulty=args.difficulty,
        box_speed=2,
        seed=args.seed,
    )
    obs = env.reset(seed=args.seed)
    total = 0.0
    done = False
    for action in args.probe:
        obs, reward, done = env.step(action, render=False)
        total += float(reward)
    probe_obs = tuple(int(x) for x in np.asarray(obs).tolist())
    state = ScriptedTeacherState()
    actions: list[str] = []
    while not done:
        action = scripted_teacher_action(env, state)
        actions.append(action)
        obs, reward, done = env.step(action, render=False)
        total += float(reward)
    row = {
        "seed": args.seed,
        "probe": args.probe,
        "probe_obs": probe_obs,
        "score": total,
        "steps_after_probe": len(actions),
        "actions": actions,
    }
    print(row["seed"], row["probe"], row["probe_obs"], row["score"], row["steps_after_probe"], flush=True)
    with open(args.out, "w") as f:
        json.dump(row, f)


if __name__ == "__main__":
    main()
