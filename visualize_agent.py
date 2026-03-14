import argparse
import importlib.util
import os
from types import ModuleType
from typing import Callable

import numpy as np

from obelix import OBELIX


ActionFn = Callable[[np.ndarray, np.random.Generator], str]


def load_agent_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("submitted_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one episode for a trained OBELIX agent."
    )
    parser.add_argument("--agent_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="Print rollout stats every N steps.",
    )
    args = parser.parse_args()

    agent_path = os.path.abspath(args.agent_file)
    agent_mod = load_agent_module(agent_path)
    if not hasattr(agent_mod, "policy"):
        raise AttributeError("Submission must define: policy(obs, rng) -> action_str")

    policy_fn: ActionFn = getattr(agent_mod, "policy")
    rng = np.random.default_rng(args.seed)

    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed,
    )
    obs = env.reset(seed=args.seed)
    env.render_frame()

    total_reward = 0.0
    step = 0
    done = False

    print(f"[viewer] agent={agent_path}")
    print(
        f"[viewer] seed={args.seed} difficulty={args.difficulty} "
        f"wall_obstacles={args.wall_obstacles} max_steps={args.max_steps}"
    )

    while not done:
        action = policy_fn(obs, rng)
        obs, reward, done = env.step(action, render=True)
        total_reward += float(reward)
        step += 1

        if args.print_every > 0 and (step % args.print_every == 0 or done):
            print(
                f"[viewer] step={step} action={action} reward={float(reward):.1f} "
                f"total={total_reward:.1f} done={done}"
            )

    print(f"[viewer] episode_return={total_reward:.1f} steps={step}")


if __name__ == "__main__":
    main()
