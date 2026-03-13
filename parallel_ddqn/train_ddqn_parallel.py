"""Parallel DDQN trainer for OBELIX.

This trainer is intended for local/offline learning. Codabench will still
evaluate with its own OBELIX runtime and your submitted policy(obs, rng).
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from ddqn_model import ACTIONS, QNetwork
from parallel_env import ParallelOBELIX


class ReplayBuffer:
    """Numpy replay buffer with vectorized batch insertion."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self.s = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.a = np.empty((self.capacity,), dtype=np.int64)
        self.r = np.empty((self.capacity,), dtype=np.float32)
        self.s2 = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.d = np.empty((self.capacity,), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        n = int(states.shape[0])
        if n <= 0:
            return

        idx = (np.arange(n) + self.pos) % self.capacity
        self.s[idx] = states
        self.a[idx] = actions
        self.r[idx] = rewards
        self.s2[idx] = next_states
        self.d[idx] = dones

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=int(batch_size))
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return end
    if step >= duration:
        return end
    alpha = float(step) / float(duration)
    return float(start + alpha * (end - start))


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel DDQN trainer for OBELIX")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_obelix_np = os.path.join(os.path.dirname(base_dir), "obelix.py")
    default_obelix_torch = os.path.join(os.path.dirname(base_dir), "obelix_torch.py")

    parser.add_argument(
        "--env_backend",
        type=str,
        choices=["numpy", "torch"],
        default="numpy",
        help="Environment implementation backend: numpy=obelix.py, torch=obelix_torch.py",
    )
    parser.add_argument(
        "--obelix_py",
        type=str,
        default=None,
        help=(
            "Optional explicit OBELIX implementation path. "
            f"Defaults to {default_obelix_np} for numpy backend, "
            f"or {default_obelix_torch} for torch backend."
        ),
    )
    parser.add_argument(
        "--env_device",
        type=str,
        default="cpu",
        help="Device passed to torch backend envs (cpu|cuda|auto). Ignored for numpy backend.",
    )
    parser.add_argument("--out", type=str, default="weights_parallel_ddqn.pth")

    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--total_env_steps", type=int, default=2_000_000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--mp_start_method", type=str, default="spawn")

    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer sizes, e.g. --hidden_dims 128 64 32",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--replay_size", type=int, default=1_000_000)
    parser.add_argument("--warmup_steps", type=int, default=20_000)
    parser.add_argument("--updates_per_iter", type=int, default=1)
    parser.add_argument("--target_sync_every", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=10.0)

    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_steps", type=int, default=1_000_000)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100_000)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_dir)
    default_impl = {
        "numpy": os.path.join(repo_dir, "obelix.py"),
        "torch": os.path.join(repo_dir, "obelix_torch.py"),
    }
    obelix_py = args.obelix_py if args.obelix_py is not None else default_impl[args.env_backend]
    if not os.path.exists(obelix_py):
        raise FileNotFoundError(f"Environment file not found: {obelix_py}")

    print(f"[setup] device={device} num_envs={args.num_envs} difficulty={args.difficulty}")
    print(f"[setup] env_backend={args.env_backend} obelix_py={obelix_py}")
    if args.env_backend == "torch":
        print(f"[setup] env_device={args.env_device}")
        if args.env_device in ("cuda", "auto") and args.num_envs > 8:
            print(
                "[warn] torch env backend on CUDA with many subprocess envs can exhaust GPU memory. "
                "Start with small num_envs (e.g. 1-8)."
            )

    hidden_dims = tuple(int(h) for h in args.hidden_dims)
    q_net = QNetwork(hidden_dims=hidden_dims).to(device)
    target_net = QNetwork(hidden_dims=hidden_dims).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    if args.torch_compile and hasattr(torch, "compile"):
        q_net = torch.compile(q_net)

    def online_state_dict():
        if hasattr(q_net, "_orig_mod"):
            return q_net._orig_mod.state_dict()
        return q_net.state_dict()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(capacity=args.replay_size, obs_dim=18)
    rng = np.random.default_rng(args.seed)

    env_kwargs = dict(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )
    if args.env_backend == "torch":
        if args.env_device != "auto":
            env_kwargs["device"] = args.env_device

    vec_env = ParallelOBELIX(
        obelix_py=obelix_py,
        num_envs=args.num_envs,
        base_seed=args.seed * 10_000,
        env_kwargs=env_kwargs,
        mp_start_method=args.mp_start_method,
    )

    obs = vec_env.reset_all(seed=args.seed * 10_000)
    episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.num_envs,), dtype=np.int32)
    recent_returns = deque(maxlen=200)
    recent_lengths = deque(maxlen=200)

    env_steps = 0
    grad_steps = 0
    start_time = time.time()

    try:
        while env_steps < args.total_env_steps:
            epsilon = linear_schedule(
                env_steps, args.eps_start, args.eps_end, args.eps_decay_steps
            )

            with torch.no_grad():
                obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
                q_values = q_net(obs_t)
                greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()

            random_actions = rng.integers(0, len(ACTIONS), size=args.num_envs)
            explore_mask = rng.random(args.num_envs) < epsilon
            action_idx = np.where(explore_mask, random_actions, greedy_actions)
            actions = [ACTIONS[int(a)] for a in action_idx]

            next_obs, rewards, dones = vec_env.step(actions)
            replay.add_batch(
                obs,
                action_idx.astype(np.int64, copy=False),
                rewards.astype(np.float32, copy=False),
                next_obs,
                dones.astype(np.float32, copy=False),
            )

            episode_returns += rewards
            episode_lengths += 1

            done_idx = np.nonzero(dones)[0]
            if done_idx.size > 0:
                for idx in done_idx:
                    recent_returns.append(float(episode_returns[idx]))
                    recent_lengths.append(int(episode_lengths[idx]))
                episode_returns[done_idx] = 0.0
                episode_lengths[done_idx] = 0

                reset_map = vec_env.reset(
                    env_indices=done_idx.tolist(),
                    seed=args.seed * 10_000 + env_steps,
                )
                for idx, reset_obs in reset_map.items():
                    next_obs[idx] = reset_obs

            obs = next_obs
            env_steps += args.num_envs

            if replay.size >= max(args.warmup_steps, args.batch_size):
                for _ in range(args.updates_per_iter):
                    sb, ab, rb, s2b, db = replay.sample(args.batch_size, rng)

                    sb_t = torch.from_numpy(sb).to(device=device, dtype=torch.float32)
                    ab_t = torch.from_numpy(ab).to(device=device, dtype=torch.int64)
                    rb_t = torch.from_numpy(rb).to(device=device, dtype=torch.float32)
                    s2b_t = torch.from_numpy(s2b).to(device=device, dtype=torch.float32)
                    db_t = torch.from_numpy(db).to(device=device, dtype=torch.float32)

                    with torch.no_grad():
                        next_online_q = q_net(s2b_t)
                        next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)
                        next_target_q = target_net(s2b_t).gather(1, next_actions).squeeze(1)
                        targets = rb_t + args.gamma * (1.0 - db_t) * next_target_q

                    pred_q = q_net(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                    loss = F.smooth_l1_loss(pred_q, targets)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(q_net.parameters(), args.grad_clip)
                    optimizer.step()

                    grad_steps += 1
                    if grad_steps % args.target_sync_every == 0:
                        target_net.load_state_dict(online_state_dict())

            if env_steps % args.log_interval < args.num_envs:
                elapsed = max(1e-6, time.time() - start_time)
                sps = env_steps / elapsed
                mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
                mean_len = float(np.mean(recent_lengths)) if recent_lengths else float("nan")
                print(
                    f"[train] env_steps={env_steps} grad_steps={grad_steps} "
                    f"eps={epsilon:.3f} replay={replay.size} sps={sps:.1f} "
                    f"recent_return={mean_ret:.1f} recent_len={mean_len:.1f} "
                    f"elapsed={format_hms(elapsed)}"
                )

    finally:
        vec_env.close()

    total_elapsed = max(0.0, time.time() - start_time)

    checkpoint = {
        "state_dict": online_state_dict(),
        "obs_dim": 18,
        "action_dim": len(ACTIONS),
        "hidden_dims": [int(h) for h in args.hidden_dims],
        "actions": ACTIONS,
        "config": vars(args),
    }
    torch.save(checkpoint, args.out)
    print(
        f"[done] total_train_time={format_hms(total_elapsed)} "
        f"({total_elapsed / 60.0:.2f} min)"
    )
    print(f"[done] saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
