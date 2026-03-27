from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


LABEL_STATIC = 0
LABEL_MOVE_NOWALL = 1
LABEL_MOVE_WALL = 2
NUM_CLASSES = 3
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
INPUT_DIM = 18 + len(ACTIONS)


def import_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scenario_label(difficulty: int, wall: bool) -> int:
    if difficulty == 3:
        return LABEL_MOVE_WALL if wall else LABEL_MOVE_NOWALL
    return LABEL_STATIC


@dataclass(frozen=True)
class Scenario:
    difficulty: int
    wall: bool


SCENARIOS = [
    Scenario(0, False),
    Scenario(0, True),
    Scenario(2, False),
    Scenario(2, True),
    Scenario(3, False),
    Scenario(3, True),
]


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 96,
        num_layers: int = 1,
        rnn_type: str = "gru",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.rnn_type = str(rnn_type).lower()
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=float(dropout) if self.num_layers > 1 else 0.0,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=float(dropout) if self.num_layers > 1 else 0.0,
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, hidden = self.rnn(packed)
        if self.rnn_type == "lstm":
            h_n = hidden[0]
        else:
            h_n = hidden
        last = h_n[-1]
        return self.head(last)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train recurrent mode classifier for OBELIX routing")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_dir)

    parser.add_argument("--obelix_py", type=str, default=os.path.join(repo_dir, "obelix.py"))
    parser.add_argument(
        "--expert_file",
        type=str,
        default=os.path.join(repo_dir, "ppo_lab", "expert_conservative.py"),
    )
    parser.add_argument("--out", type=str, default=os.path.join(base_dir, "mode_classifier_gru.pth"))
    parser.add_argument("--probe_steps", type=int, default=64)
    parser.add_argument("--train_seeds", type=int, default=400)
    parser.add_argument("--val_seeds", type=int, default=120)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--rnn_type", type=str, choices=["gru", "lstm"], default="gru")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--balance_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def collect_split(
    *,
    obelix_cls,
    expert_policy,
    scenarios: list[Scenario],
    seed_start: int,
    num_seeds: int,
    probe_steps: int,
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    box_speed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    lengths: list[int] = []

    for scenario_idx, spec in enumerate(scenarios):
        for offset in range(int(num_seeds)):
            seed = int(seed_start + scenario_idx * 100_000 + offset)
            env = obelix_cls(
                scaling_factor=scaling_factor,
                arena_size=arena_size,
                max_steps=max_steps,
                wall_obstacles=spec.wall,
                difficulty=spec.difficulty,
                box_speed=box_speed,
                seed=seed,
            )
            obs = env.reset(seed=seed)
            rng = np.random.default_rng(seed)

            seq = np.zeros((probe_steps, INPUT_DIM), dtype=np.float32)
            prev_action = np.zeros((len(ACTIONS),), dtype=np.float32)
            done = False
            actual_len = 0

            for step in range(int(probe_steps)):
                seq[step, :18] = np.asarray(obs, dtype=np.float32)
                seq[step, 18:] = prev_action
                actual_len += 1
                if done:
                    continue
                action_name = expert_policy(obs, rng)
                action_idx = ACTIONS.index(action_name)
                prev_action.fill(0.0)
                prev_action[action_idx] = 1.0
                obs, _, done = env.step(action_name, render=False)

            sequences.append(seq)
            labels.append(scenario_label(spec.difficulty, spec.wall))
            lengths.append(actual_len)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(lengths, dtype=np.int64),
    )


@torch.no_grad()
def evaluate(
    model: SequenceClassifier,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    total = int(x.shape[0])
    correct = 0
    loss_sum = 0.0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    for start in range(0, total, batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        lb = lengths[start : start + batch_size]
        logits = model(xb, lb)
        loss = nn.functional.cross_entropy(logits, yb)
        preds = torch.argmax(logits, dim=1)
        loss_sum += float(loss.item()) * int(xb.shape[0])
        correct += int((preds == yb).sum().item())
        for t, p in zip(yb.tolist(), preds.tolist()):
            confusion[t, p] += 1
    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        "confusion": confusion.cpu().numpy(),
    }


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

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from obelix import OBELIX

    expert_mod = import_module(args.expert_file, "mode_switch_expert")
    expert_policy = getattr(expert_mod, "policy")

    t0 = time.time()
    x_train_np, y_train_np, len_train_np = collect_split(
        obelix_cls=OBELIX,
        expert_policy=expert_policy,
        scenarios=SCENARIOS,
        seed_start=args.seed,
        num_seeds=args.train_seeds,
        probe_steps=args.probe_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        box_speed=args.box_speed,
    )
    x_val_np, y_val_np, len_val_np = collect_split(
        obelix_cls=OBELIX,
        expert_policy=expert_policy,
        scenarios=SCENARIOS,
        seed_start=args.seed + 1_000_000,
        num_seeds=args.val_seeds,
        probe_steps=args.probe_steps,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        box_speed=args.box_speed,
    )
    print(
        f"[data] train={x_train_np.shape[0]} val={x_val_np.shape[0]} "
        f"probe_steps={args.probe_steps} build_s={time.time() - t0:.1f}"
    )

    x_train = torch.as_tensor(x_train_np, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train_np, dtype=torch.long, device=device)
    len_train = torch.as_tensor(len_train_np, dtype=torch.long, device=device)
    x_val = torch.as_tensor(x_val_np, dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y_val_np, dtype=torch.long, device=device)
    len_val = torch.as_tensor(len_val_np, dtype=torch.long, device=device)

    model = SequenceClassifier(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    class_weight = None
    if args.balance_loss:
        counts = np.bincount(y_train_np, minlength=NUM_CLASSES).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / np.mean(weights)
        class_weight = torch.as_tensor(weights, dtype=torch.float32, device=device)
        print(f"[train] class_weight={weights.tolist()}")

    best_acc = -1.0
    best_payload = None
    batch_size = int(args.batch_size)
    total = int(x_train.shape[0])

    for epoch in range(int(args.epochs)):
        model.train()
        perm = torch.randperm(total, device=device)
        loss_sum = 0.0
        correct = 0
        seen = 0

        for start in range(0, total, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            lb = len_train[idx]
            logits = model(xb, lb)
            loss = nn.functional.cross_entropy(logits, yb, weight=class_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            loss_sum += float(loss.item()) * int(xb.shape[0])
            correct += int((preds == yb).sum().item())
            seen += int(xb.shape[0])

        train_acc = correct / max(1, seen)
        train_loss = loss_sum / max(1, seen)
        val_stats = evaluate(model, x=x_val, y=y_val, lengths=len_val, batch_size=batch_size)
        print(
            f"[train] epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.3f}"
        )

        if val_stats["acc"] > best_acc:
            best_acc = float(val_stats["acc"])
            best_payload = {
                "state_dict": model.state_dict(),
                "probe_steps": int(args.probe_steps),
                "hidden_dim": int(args.hidden_dim),
                "num_layers": int(args.num_layers),
                "rnn_type": str(args.rnn_type),
                "dropout": float(args.dropout),
                "label_names": ["static_or_blink", "move_nowall", "move_wall"],
                "val_acc": float(val_stats["acc"]),
                "val_confusion": val_stats["confusion"],
                "config": vars(args),
            }
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            torch.save(best_payload, args.out)
            print(f"[train] new best -> {args.out}")

    if best_payload is not None:
        final_path = args.out[:-4] + "_final.pth" if args.out.endswith(".pth") else args.out + "_final"
        torch.save(best_payload, final_path)
        print(f"[train] final -> {final_path} best_val_acc={best_acc:.3f}")


if __name__ == "__main__":
    main()
