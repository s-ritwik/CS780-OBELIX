from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from obelix import OBELIX


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
ACTION_DIM = len(ACTIONS)
OBS_DIM = 18


def parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.replace(",", " ").split():
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def load_probe_agent(path: str):
    spec = importlib.util.spec_from_file_location("detector_probe_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load probe agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "_load_once"):
        module._load_once()
    return module


def obs_features(obs: np.ndarray, action_idx: int, step: int, probe_steps: int) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32)
    left = float(np.sum(x[:4]))
    front = float(np.sum(x[4:12]))
    right = float(np.sum(x[12:16]))
    far_front = float(np.sum(x[4:12:2]))
    near_front = float(np.sum(x[5:12:2]))
    derived = np.asarray(
        [
            left,
            front,
            right,
            far_front,
            near_front,
            float(np.sum(x[:16]) == 0.0),
            float(np.sum(x[:16])),
            float(step) / float(max(1, probe_steps - 1)),
        ],
        dtype=np.float32,
    )
    action = np.zeros((ACTION_DIM,), dtype=np.float32)
    if action_idx >= 0:
        action[action_idx] = 1.0
    return np.concatenate([x, derived, action], axis=0).astype(np.float32)


def fixed_scan_action(step: int, obs: np.ndarray) -> str:
    # A legal scan intended to expose long continuous walls: rotate through
    # 360 degrees, occasionally move forward when blind, then scan the other way.
    cycle = step % 32
    if cycle in {8, 17, 26} and float(np.sum(obs[:16])) == 0.0:
        return "FW"
    if cycle < 16:
        return "L45"
    return "R45"


def spin_action(step: int, obs: np.ndarray) -> str:
    return "L45"


def wall_policy_action(probe_agent, obs: np.ndarray, rng: np.random.Generator) -> str:
    if hasattr(probe_agent, "_wall_action"):
        return str(probe_agent._wall_action(obs, rng))
    return str(probe_agent.policy(obs, rng))


@dataclass
class TraceBatch:
    x: np.ndarray
    y: np.ndarray
    seeds: np.ndarray


def collect_traces(
    *,
    seeds: list[int],
    wall_obstacles: bool,
    probe_steps: int,
    probe: str,
    probe_agent,
    difficulty: int,
    box_speed: int,
    max_steps: int,
    scaling_factor: int,
    arena_size: int,
) -> TraceBatch:
    env = OBELIX(
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        max_steps=max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=box_speed,
        seed=seeds[0] if seeds else 0,
    )
    xs: list[np.ndarray] = []
    ys: list[int] = []
    out_seeds: list[int] = []
    rng_refs: list[np.random.Generator] = []
    for seed in seeds:
        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        rng_refs.append(rng)
        if probe_agent is not None and hasattr(probe_agent, "_reset"):
            probe_agent._reset(np.asarray(obs, dtype=np.float32), rng)
        trace: list[np.ndarray] = []
        last_action = -1
        done = False
        for step in range(probe_steps):
            obs_arr = np.asarray(obs, dtype=np.float32)
            trace.append(obs_features(obs_arr, last_action, step, probe_steps))
            if probe == "wall_policy":
                action = wall_policy_action(probe_agent, obs_arr, rng)
            elif probe == "spin":
                action = spin_action(step, obs_arr)
            elif probe == "scan":
                action = fixed_scan_action(step, obs_arr)
            else:
                raise ValueError(f"unknown probe: {probe}")
            last_action = ACTION_TO_INDEX[action]
            obs, _, done = env.step(action, render=False)
            if done:
                # Pad terminal traces with the final observation and last action.
                obs = np.asarray(obs, dtype=np.float32)
                for pad_step in range(step + 1, probe_steps):
                    trace.append(obs_features(obs, last_action, pad_step, probe_steps))
                break
        xs.append(np.stack(trace[:probe_steps], axis=0))
        ys.append(1 if wall_obstacles else 0)
        out_seeds.append(seed)
    return TraceBatch(
        x=np.stack(xs, axis=0).astype(np.float32),
        y=np.asarray(ys, dtype=np.int64),
        seeds=np.asarray(out_seeds, dtype=np.int64),
    )


class WallDetectorGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(layers),
            batch_first=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def train_detector(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dim: int,
    layers: int,
    dropout: float,
    epochs: int,
    lr: float,
    wall_weight: float,
    seed: int,
) -> WallDetectorGRU:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WallDetectorGRU(x.shape[-1], hidden_dim, layers, dropout).to(device)
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y, dtype=torch.long, device=device)
    weights = torch.as_tensor([1.0, wall_weight], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        logits = model(tx)
        loss = F.cross_entropy(logits, ty, weight=weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if epoch % 100 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                acc = torch.mean((pred == ty).float()).item()
                fn = int(torch.sum((pred == 0) & (ty == 1)).item())
                fp = int(torch.sum((pred == 1) & (ty == 0)).item())
            print(f"epoch={epoch} loss={float(loss.item()):.4f} acc={acc:.3f} fp={fp} fn={fn}", flush=True)
    return model.cpu().eval()


@torch.no_grad()
def evaluate(model: WallDetectorGRU, batch: TraceBatch, threshold: float) -> tuple[float, int, int, np.ndarray]:
    logits = model(torch.as_tensor(batch.x, dtype=torch.float32))
    prob = torch.softmax(logits, dim=1)[:, 1].numpy()
    pred = (prob >= float(threshold)).astype(np.int64)
    acc = float(np.mean(pred == batch.y))
    fp = int(np.sum((pred == 1) & (batch.y == 0)))
    fn = int(np.sum((pred == 0) & (batch.y == 1)))
    return acc, fp, fn, prob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(HERE, "wall_detector_gru_v1.pth"))
    parser.add_argument("--probe", choices=["wall_policy", "scan", "spin"], default="wall_policy")
    parser.add_argument("--probe_agent", default=os.path.join(HERE, "agent_random_router_probe.py"))
    parser.add_argument("--train_seeds", default="0-1999")
    parser.add_argument("--test_seeds", default="2000-2499")
    parser.add_argument("--extra_test_seeds", default="")
    parser.add_argument("--probe_steps", type=int, default=80)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wall_weight", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    args = parser.parse_args()

    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    np.random.seed(args.seed)
    probe_agent = load_probe_agent(args.probe_agent) if args.probe == "wall_policy" else None
    train_seeds = parse_seed_list(args.train_seeds)
    test_seeds = parse_seed_list(args.test_seeds)
    if args.extra_test_seeds.strip():
        test_seeds = sorted(set(test_seeds + parse_seed_list(args.extra_test_seeds)))

    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'} probe={args.probe}", flush=True)
    train_nowall = collect_traces(
        seeds=train_seeds,
        wall_obstacles=False,
        probe_steps=args.probe_steps,
        probe=args.probe,
        probe_agent=probe_agent,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        max_steps=args.max_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
    )
    train_wall = collect_traces(
        seeds=train_seeds,
        wall_obstacles=True,
        probe_steps=args.probe_steps,
        probe=args.probe,
        probe_agent=probe_agent,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        max_steps=args.max_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
    )
    x_train = np.concatenate([train_nowall.x, train_wall.x], axis=0)
    y_train = np.concatenate([train_nowall.y, train_wall.y], axis=0)

    model = train_detector(
        x_train,
        y_train,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        wall_weight=args.wall_weight,
        seed=args.seed,
    )

    test_nowall = collect_traces(
        seeds=test_seeds,
        wall_obstacles=False,
        probe_steps=args.probe_steps,
        probe=args.probe,
        probe_agent=probe_agent,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        max_steps=args.max_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
    )
    test_wall = collect_traces(
        seeds=test_seeds,
        wall_obstacles=True,
        probe_steps=args.probe_steps,
        probe=args.probe,
        probe_agent=probe_agent,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        max_steps=args.max_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
    )
    test_batch = TraceBatch(
        x=np.concatenate([test_nowall.x, test_wall.x], axis=0),
        y=np.concatenate([test_nowall.y, test_wall.y], axis=0),
        seeds=np.concatenate([test_nowall.seeds, test_wall.seeds], axis=0),
    )
    acc, fp, fn, prob = evaluate(model, test_batch, args.threshold)
    print(f"test acc={acc:.4f} fp={fp} fn={fn} threshold={args.threshold:.3f}", flush=True)
    y = test_batch.y
    hard = np.where(((prob >= args.threshold).astype(np.int64) != y))[0]
    for idx in hard[:40]:
        label = "wall" if int(y[idx]) == 1 else "nowall"
        print(f"miss {label} seed={int(test_batch.seeds[idx])} p_wall={float(prob[idx]):.6f}", flush=True)

    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": int(x_train.shape[-1]),
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
        "dropout": float(args.dropout),
        "probe": str(args.probe),
        "probe_steps": int(args.probe_steps),
        "threshold": float(args.threshold),
        "wall_weight": float(args.wall_weight),
        "train_seeds": train_seeds,
        "test_seeds": test_seeds,
        "test_acc": float(acc),
        "test_fp": int(fp),
        "test_fn": int(fn),
    }
    torch.save(ckpt, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
