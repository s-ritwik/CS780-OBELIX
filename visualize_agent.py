import argparse
import importlib.util
import os
from types import ModuleType
from typing import Callable

import cv2
import numpy as np
import time
from obelix import OBELIX


ActionFn = Callable[[np.ndarray, np.random.Generator], str]


def load_agent_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("submitted_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_video_writer(path: str, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    height, width = frame.shape[:2]
    ext = os.path.splitext(path)[1].lower()
    codec_names = ["mp4v", "avc1"] if ext in {".mp4", ".m4v", ".mov"} else ["MJPG", "XVID"]

    for codec_name in codec_names:
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*codec_name),
            float(fps),
            (int(width), int(height)),
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(f"Could not open video writer for: {path}")


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
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the live OpenCV window while rolling out.",
    )
    parser.add_argument(
        "--record_path",
        type=str,
        default=None,
        help="Optional output video path, e.g. rollout.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Recording FPS when --record_path is set.",
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.05,
        help="Sleep between rendered steps when --display is enabled.",
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
    display_enabled = bool(args.display)
    video_writer: cv2.VideoWriter | None = None

    if display_enabled:
        try:
            env.render_frame()
        except cv2.error:
            if args.record_path:
                print("[viewer] OpenCV GUI unavailable, continuing with recording only.")
                display_enabled = False
                env._update_frames(show=False)
            else:
                raise
    else:
        env._update_frames(show=False)

    if args.record_path:
        video_writer = make_video_writer(args.record_path, env.frame, args.fps)
        video_writer.write(env.frame)

    total_reward = 0.0
    step = 0
    done = False

    print(f"[viewer] agent={agent_path}")
    print(
        f"[viewer] seed={args.seed} difficulty={args.difficulty} "
        f"wall_obstacles={args.wall_obstacles} max_steps={args.max_steps}"
    )
    if args.record_path:
        print(f"[viewer] recording={os.path.abspath(args.record_path)} fps={args.fps}")

    try:
        while not done:
            action = policy_fn(obs, rng)
            obs, reward, done = env.step(action, render=display_enabled)
            total_reward += float(reward)
            step += 1

            if video_writer is not None:
                video_writer.write(env.frame)

            if args.print_every > 0 and (step % args.print_every == 0 or done):
                print(
                    f"[viewer] step={step} action={action} reward={float(reward):.1f} "
                    f"total={total_reward:.1f} done={done}"
                )
            if display_enabled and args.step_delay > 0.0:
                time.sleep(args.step_delay)
    finally:
        if video_writer is not None:
            video_writer.release()

    print(f"[viewer] episode_return={total_reward:.1f} steps={step}")


if __name__ == "__main__":
    main()
