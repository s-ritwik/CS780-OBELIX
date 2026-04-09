from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from obelix import OBELIX


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.Tanh())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_random_probe_module():
    spec = importlib.util.spec_from_file_location(
        "random_probe_for_router",
        os.path.join(HERE, "agent_random_router_probe.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load agent_random_router_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._load_once()
    return module


def collect_features(
    *,
    probe_mod,
    seeds: list[int],
    probe_steps: int,
    wall_obstacles: bool,
    difficulty: int,
    box_speed: int,
    max_steps: int,
    scaling_factor: int,
    arena_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    env = OBELIX(
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        max_steps=max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=box_speed,
        seed=seeds[0] if seeds else 0,
    )
    for seed in seeds:
        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        obs_arr = np.asarray(obs, dtype=np.float32)
        probe_mod._reset(obs_arr, rng)
        done = False
        for _ in range(probe_steps):
            obs_arr = np.asarray(obs, dtype=np.float32)
            probe_mod._STATS += probe_mod._obs_stats(obs_arr)
            probe_mod._PROBE_REWARD_PROXY -= 1.0
            action = probe_mod._wall_action(obs_arr, rng)
            obs, _, done = env.step(action, render=False)
            if done:
                break
        obs_arr = np.asarray(obs, dtype=np.float32)
        feat = np.concatenate(
            [
                np.asarray(probe_mod._FIRST_OBS, dtype=np.float32),
                obs_arr.astype(np.float32, copy=False),
                (probe_mod._STATS / float(max(1, probe_steps))).astype(np.float32, copy=False),
                np.asarray([probe_mod._PROBE_REWARD_PROXY / 1000.0], dtype=np.float32),
            ]
        )
        xs.append(feat)
        ys.append(1 if wall_obstacles else 0)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def parse_seed_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.replace(",", " ").split():
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(HERE, "router_d3_cost_sensitive_v1.pth"))
    parser.add_argument("--seeds", default="0-999")
    parser.add_argument("--extra_seeds", default="")
    parser.add_argument("--probe_steps", type=int, default=20)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wall_weight", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_seeds = parse_seed_list(args.seeds)
    if args.extra_seeds.strip():
        train_seeds += parse_seed_list(args.extra_seeds)
    train_seeds = sorted(set(train_seeds))
    probe_mod = load_random_probe_module()

    x0, y0 = collect_features(
        probe_mod=probe_mod,
        seeds=train_seeds,
        probe_steps=args.probe_steps,
        wall_obstacles=False,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        max_steps=args.max_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
    )
    x1, y1 = collect_features(
        probe_mod=probe_mod,
        seeds=train_seeds,
        probe_steps=args.probe_steps,
        wall_obstacles=True,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        max_steps=args.max_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
    )
    x = torch.as_tensor(np.concatenate([x0, x1], axis=0), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.concatenate([y0, y1], axis=0), dtype=torch.long, device=device)

    model = RouterMLP(input_dim=x.shape[1], hidden_dims=tuple(args.hidden_dims)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = torch.as_tensor([1.0, float(args.wall_weight)], dtype=torch.float32, device=device)
    for epoch in range(args.epochs):
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                pred = (torch.softmax(logits, dim=1)[:, 1] >= args.threshold).long()
                fp = int(torch.sum((pred == 1) & (y == 0)).item())
                fn = int(torch.sum((pred == 0) & (y == 1)).item())
                acc = float(torch.mean((pred == y).float()).item())
            print(f"epoch={epoch} loss={float(loss.item()):.4f} acc={acc:.3f} fp={fp} fn={fn}", flush=True)

    ckpt = {
        "state_dict": model.cpu().state_dict(),
        "input_dim": int(x.shape[1]),
        "hidden_dims": tuple(int(v) for v in args.hidden_dims),
        "threshold": float(args.threshold),
        "probe_steps": int(args.probe_steps),
        "wall_weight": float(args.wall_weight),
        "train_seeds": train_seeds,
    }
    torch.save(ckpt, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
