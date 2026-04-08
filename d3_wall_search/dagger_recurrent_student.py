from __future__ import annotations

import argparse
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASYM_DIR = os.path.join(ROOT, "assym_ppo")
if ASYM_DIR not in sys.path:
    sys.path.insert(0, ASYM_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from obelix import OBELIX
from recurrent import ACTIONS, PoseMemoryTracker, RecurrentAsymmetricActorCritic, RecurrentFeatureConfig, make_checkpoint, save_checkpoint
from teacher import ScriptedTeacherState, scripted_teacher_action
from train_recurrent_sequence_bc import train_sequence_bc


def parse_seed_list(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            step = 1 if hi >= lo else -1
            seeds.extend(range(lo, hi + step, step))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("Empty seed list")
    return seeds


def build_model(
    *,
    feature_config: RecurrentFeatureConfig,
    encoder_dims: tuple[int, ...],
    gru_hidden_dim: int,
    critic_hidden_dims: tuple[int, ...],
    gru_layers: int,
    gru_dropout: float,
    device: torch.device,
) -> RecurrentAsymmetricActorCritic:
    return RecurrentAsymmetricActorCritic(
        actor_dim=feature_config.feature_dim,
        privileged_dim=34,
        encoder_dims=encoder_dims,
        gru_hidden_dim=gru_hidden_dim,
        critic_hidden_dims=critic_hidden_dims,
        gru_layers=gru_layers,
        gru_dropout=gru_dropout,
        fw_bias_init=0.0,
    ).to(device)


def load_actor_if_present(model: RecurrentAsymmetricActorCritic, path: str | None, device: torch.device) -> None:
    if not path:
        return
    ckpt = torch.load(path, map_location=device)
    model.actor_encoder.load_state_dict(ckpt["actor_encoder_state_dict"], strict=True)
    model.actor_rnn.load_state_dict(ckpt["actor_rnn_state_dict"], strict=True)
    model.policy_head.load_state_dict(ckpt["policy_head_state_dict"], strict=True)
    print(f"[setup] warm-started actor from {path}", flush=True)


def checkpoint_shapes(path: str | None, fallback_encoder: tuple[int, ...], fallback_gru: int, fallback_critic: tuple[int, ...], fallback_layers: int, fallback_dropout: float):
    if not path:
        return fallback_encoder, fallback_gru, fallback_critic, fallback_layers, fallback_dropout
    ckpt = torch.load(path, map_location="cpu")
    return (
        tuple(int(x) for x in ckpt.get("encoder_dims", fallback_encoder)),
        int(ckpt.get("gru_hidden_dim", fallback_gru)),
        tuple(int(x) for x in ckpt.get("critic_hidden_dims", fallback_critic)),
        int(ckpt.get("gru_layers", fallback_layers)),
        float(ckpt.get("gru_dropout", fallback_dropout)),
    )


@torch.no_grad()
def rollout_collect(
    *,
    model: RecurrentAsymmetricActorCritic,
    feature_config: RecurrentFeatureConfig,
    seeds: list[int],
    repeats: int,
    env_kwargs: dict,
    device: torch.device,
    teacher_mix: float,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    model.eval()
    sequences_x: list[np.ndarray] = []
    sequences_y: list[np.ndarray] = []
    returns: list[float] = []
    lengths: list[int] = []

    for repeat in range(int(repeats)):
        for seed in seeds:
            env = OBELIX(seed=int(seed), **env_kwargs)
            obs = np.asarray(env.reset(seed=int(seed)), dtype=np.float32)
            tracker = PoseMemoryTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
            tracker.reset_all(obs[None, :])
            hidden = model.initial_state(1, device)
            teacher_state = ScriptedTeacherState()
            rng = np.random.default_rng(int(seed) + 1000003 * repeat)

            seq_x: list[np.ndarray] = []
            seq_y: list[int] = []
            total = 0.0
            done = False
            steps = 0
            starts = torch.ones((1,), dtype=torch.float32, device=device)
            while not done:
                feat_cpu = tracker.features().cpu().numpy()[0].astype(np.float32, copy=True)
                teacher_action = scripted_teacher_action(env, teacher_state)
                teacher_idx = ACTIONS.index(teacher_action)
                feat_t = torch.as_tensor(feat_cpu[None, :], dtype=torch.float32, device=device)
                logits, hidden = model.actor_step(feat_t, hidden, starts)
                student_idx = int(torch.argmax(logits, dim=1).item())
                starts = torch.zeros((1,), dtype=torch.float32, device=device)

                action_idx = teacher_idx if rng.random() < float(teacher_mix) else student_idx
                action_name = ACTIONS[action_idx]
                seq_x.append(feat_cpu)
                seq_y.append(teacher_idx)
                obs, reward, done = env.step(action_name, render=False)
                obs = np.asarray(obs, dtype=np.float32)
                total += float(reward)
                steps += 1
                if not done:
                    tracker.post_step(
                        actions=torch.tensor([action_idx], dtype=torch.long),
                        next_obs=obs[None, :],
                        dones=np.asarray([False]),
                    )

            sequences_x.append(np.asarray(seq_x, dtype=np.float32))
            sequences_y.append(np.asarray(seq_y, dtype=np.int64))
            returns.append(total)
            lengths.append(steps)
            print(
                f"[collect] repeat={repeat} seed={seed} return={total:.1f} steps={steps}",
                flush=True,
            )

    return sequences_x, sequences_y, {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
    }


@torch.no_grad()
def eval_model(
    *,
    model: RecurrentAsymmetricActorCritic,
    feature_config: RecurrentFeatureConfig,
    seeds: list[int],
    env_kwargs: dict,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    scores: list[float] = []
    for seed in seeds:
        env = OBELIX(seed=int(seed), **env_kwargs)
        obs = np.asarray(env.reset(seed=int(seed)), dtype=np.float32)
        tracker = PoseMemoryTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
        tracker.reset_all(obs[None, :])
        hidden = model.initial_state(1, device)
        total = 0.0
        done = False
        steps = 0
        starts = torch.ones((1,), dtype=torch.float32, device=device)
        while not done:
            feat_t = tracker.features().to(device)
            logits, hidden = model.actor_step(feat_t, hidden, starts)
            starts = torch.zeros((1,), dtype=torch.float32, device=device)
            action_idx = int(torch.argmax(logits, dim=1).item())
            obs, reward, done = env.step(ACTIONS[action_idx], render=False)
            obs = np.asarray(obs, dtype=np.float32)
            total += float(reward)
            steps += 1
            if not done:
                tracker.post_step(
                    actions=torch.tensor([action_idx], dtype=torch.long),
                    next_obs=obs[None, :],
                    dones=np.asarray([False]),
                )
        scores.append(total)
        print(f"[eval_detail] seed={seed} return={total:.1f} steps={steps}", flush=True)
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DAgger recurrent student for OBELIX D3 random seeds.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--no_wall_obstacles", dest="wall_obstacles", action="store_false")
    parser.set_defaults(wall_obstacles=True)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--repeats_per_iter", type=int, default=1)
    parser.add_argument("--teacher_mix_start", type=float, default=0.5)
    parser.add_argument("--teacher_mix_end", type=float, default=0.0)
    parser.add_argument("--epochs_per_iter", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--encoder_dims", type=int, nargs="+", default=[768, 384])
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[768, 384, 192])
    parser.add_argument("--gru_hidden_dim", type=int, default=384)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--gru_dropout", type=float, default=0.0)
    parser.add_argument("--last_action_hist", type=int, default=8)
    parser.add_argument("--max_sequences", type=int, default=160)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    seeds = parse_seed_list(args.seeds)
    feature_config = RecurrentFeatureConfig(max_steps=args.max_steps, last_action_hist=args.last_action_hist)
    encoder_dims, gru_hidden_dim, critic_hidden_dims, gru_layers, gru_dropout = checkpoint_shapes(
        args.init_checkpoint,
        tuple(int(x) for x in args.encoder_dims),
        int(args.gru_hidden_dim),
        tuple(int(x) for x in args.critic_hidden_dims),
        int(args.gru_layers),
        float(args.gru_dropout),
    )
    model = build_model(
        feature_config=feature_config,
        encoder_dims=encoder_dims,
        gru_hidden_dim=gru_hidden_dim,
        critic_hidden_dims=critic_hidden_dims,
        gru_layers=gru_layers,
        gru_dropout=gru_dropout,
        device=device,
    )
    load_actor_if_present(model, args.init_checkpoint, device)
    env_kwargs = {
        "scaling_factor": int(args.scaling_factor),
        "arena_size": int(args.arena_size),
        "max_steps": int(args.max_steps),
        "wall_obstacles": bool(args.wall_obstacles),
        "difficulty": int(args.difficulty),
        "box_speed": int(args.box_speed),
    }

    aggregate_x: list[np.ndarray] = []
    aggregate_y: list[np.ndarray] = []
    best_mean = -1e18
    best_ckpt = None
    for iteration in range(int(args.iterations)):
        frac = 0.0 if args.iterations <= 1 else iteration / float(args.iterations - 1)
        teacher_mix = float(args.teacher_mix_start) * (1.0 - frac) + float(args.teacher_mix_end) * frac
        new_x, new_y, stats = rollout_collect(
            model=model,
            feature_config=feature_config,
            seeds=seeds,
            repeats=int(args.repeats_per_iter),
            env_kwargs=env_kwargs,
            device=device,
            teacher_mix=teacher_mix,
        )
        aggregate_x.extend(new_x)
        aggregate_y.extend(new_y)
        if len(aggregate_x) > int(args.max_sequences):
            aggregate_x = aggregate_x[-int(args.max_sequences) :]
            aggregate_y = aggregate_y[-int(args.max_sequences) :]
        print(
            f"[iter] {iteration + 1}/{args.iterations} teacher_mix={teacher_mix:.3f} "
            f"collect_mean={stats['mean_return']:.1f} collect_std={stats['std_return']:.1f} "
            f"dataset={len(aggregate_x)}",
            flush=True,
        )
        model.train()
        train_sequence_bc(
            model=model,
            sequences_x=aggregate_x,
            sequences_y=aggregate_y,
            device=device,
            epochs=int(args.epochs_per_iter),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            grad_clip=float(args.grad_clip),
        )
        eval_stats = eval_model(
            model=model,
            feature_config=feature_config,
            seeds=seeds,
            env_kwargs=env_kwargs,
            device=device,
        )
        print(
            f"[eval] iter={iteration + 1} mean={eval_stats['mean']:.1f} std={eval_stats['std']:.1f} "
            f"min={eval_stats['min']:.1f} max={eval_stats['max']:.1f}",
            flush=True,
        )
        if eval_stats["mean"] > best_mean:
            best_mean = eval_stats["mean"]
            best_ckpt = make_checkpoint(
                model=model,
                feature_config=feature_config,
                encoder_dims=encoder_dims,
                gru_hidden_dim=gru_hidden_dim,
                critic_hidden_dims=critic_hidden_dims,
                privileged_dim=34,
                gru_layers=gru_layers,
                gru_dropout=gru_dropout,
                args=SimpleNamespace(**vars(args)),
                best_eval=float(best_mean),
            )
            save_checkpoint(args.out, best_ckpt)
            print(f"[save] best_mean={best_mean:.1f} -> {args.out}", flush=True)

    if best_ckpt is None:
        raise RuntimeError("No checkpoint was produced")
    final_path = os.path.splitext(args.out)[0] + "_final.pth"
    save_checkpoint(final_path, best_ckpt)
    print(f"[done] best_mean={best_mean:.1f} final={final_path}", flush=True)


if __name__ == "__main__":
    main()
