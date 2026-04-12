import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    out_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    height, width = frame.shape[:2]
    ext = os.path.splitext(out_path)[1].lower()
    codec_names = ["mp4v", "avc1"] if ext in {".mp4", ".m4v", ".mov"} else ["MJPG", "XVID"]

    for codec_name in codec_names:
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*codec_name),
            float(fps),
            (int(width), int(height)),
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(f"Could not open video writer for: {out_path}")


def overlay_frame(
    frame: np.ndarray,
    *,
    title: str,
    seed: int,
    wall: bool,
    step: int,
    action: str,
    reward: float,
    total: float,
) -> np.ndarray:
    out_w = 1920
    out_h = 1080
    panel_w = 420
    left_w = out_w - panel_w
    viewport_w = 1180
    viewport_h = 820

    h, w = frame.shape[:2]
    scale = min(viewport_w / float(w), viewport_h / float(h))
    render_w = max(1, int(round(w * scale)))
    render_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (render_w, render_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:, :] = (18, 18, 18)

    x0 = (left_w - render_w) // 2
    y0 = (out_h - render_h) // 2
    canvas[y0:y0 + render_h, x0:x0 + render_w] = resized

    panel_x1 = left_w
    cv2.rectangle(canvas, (panel_x1, 0), (out_w, out_h), (28, 28, 28), thickness=-1)
    cv2.line(canvas, (panel_x1, 0), (panel_x1, out_h), (210, 210, 210), thickness=2)

    lines = [
        title,
        "",
        f"seed   : {seed}",
        f"wall   : {int(wall)}",
        f"step   : {step}",
        f"action : {action}",
        f"reward : {reward:.1f}",
        f"total  : {total:.1f}",
    ]
    y = 70
    for i, line in enumerate(lines):
        if not line:
            y += 20
            continue
        scale = 0.95 if i == 0 else 0.78
        thickness = 2 if i == 0 else 2
        cv2.putText(
            canvas,
            line,
            (panel_x1 + 28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )
        y += 52 if i == 0 else 62
    return canvas


def render_episode(
    *,
    agent_file: str,
    output_path: str,
    title: str,
    seed: int,
    wall_obstacles: bool,
    difficulty: int,
    max_steps: int,
    scaling_factor: int,
    arena_size: int,
    box_speed: int,
    fps: float,
) -> tuple[float, int]:
    agent_mod = load_agent_module(agent_file)
    if not hasattr(agent_mod, "policy"):
        raise AttributeError("Submission must define: policy(obs, rng) -> action_str")
    policy_fn: ActionFn = getattr(agent_mod, "policy")

    rng = np.random.default_rng(seed)
    env = OBELIX(
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        max_steps=max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=box_speed,
        seed=seed,
    )
    obs = env.reset(seed=seed)
    env._update_frames(show=False)

    total_reward = 0.0
    step = 0

    initial = overlay_frame(
        env.frame,
        title=title,
        seed=seed,
        wall=wall_obstacles,
        step=step,
        action="RESET",
        reward=0.0,
        total=total_reward,
    )
    writer = make_video_writer(output_path, initial, fps)
    writer.write(initial)

    try:
        done = False
        while not done:
            action = policy_fn(obs, rng)
            obs, reward, done = env.step(action, render=False)
            step += 1
            total_reward += float(reward)
            frame = overlay_frame(
                env.frame,
                title=title,
                seed=seed,
                wall=wall_obstacles,
                step=step,
                action=action,
                reward=float(reward),
                total=total_reward,
            )
            writer.write(frame)
    finally:
        writer.release()
    return total_reward, step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    total, steps = render_episode(
        agent_file=os.path.abspath(args.agent_file),
        output_path=os.path.abspath(args.output),
        title=args.title,
        seed=args.seed,
        wall_obstacles=bool(args.wall_obstacles),
        difficulty=args.difficulty,
        max_steps=args.max_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        box_speed=args.box_speed,
        fps=args.fps,
    )
    print(
        f"saved={os.path.abspath(args.output)} seed={args.seed} wall={int(args.wall_obstacles)} "
        f"steps={steps} total={total:.1f}"
    )


if __name__ == "__main__":
    main()
