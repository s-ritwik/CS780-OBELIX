from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASYM_DIR = os.path.join(ROOT, "assym_ppo")
if ASYM_DIR not in sys.path:
    sys.path.insert(0, ASYM_DIR)

from recurrent import ACTIONS, RecurrentAsymmetricActorCritic, RecurrentFeatureConfig, PoseMemoryTracker, make_checkpoint, save_checkpoint
from train_recurrent_sequence_bc import (
    collect_teacher_sequences,
    evaluate_actor,
    import_module,
    import_symbol,
    train_sequence_bc,
)
from teacher import ScriptedTeacherState, scripted_teacher_action


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
        raise ValueError("--collect_seeds produced an empty seed list")
    return seeds


def collect_seed_list_sequences(
    *,
    obelix_py: str,
    env_kwargs: dict,
    feature_config: RecurrentFeatureConfig,
    expert_path: str,
    seeds: list[int],
    repeats: int,
    min_return: float | None,
    success_dup_factor: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    obelix_cls = import_symbol(obelix_py, "OBELIX")
    use_scripted_teacher = expert_path == "scripted_teacher"
    expert_policy = None
    if not use_scripted_teacher:
        expert_mod = import_module(expert_path, f"macro_student_expert_{abs(hash(expert_path))}")
        if not hasattr(expert_mod, "policy"):
            raise AttributeError(f"Expert module {expert_path} does not define policy(obs, rng)")
        expert_policy = getattr(expert_mod, "policy")

    tracker = PoseMemoryTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
    sequences_x: list[np.ndarray] = []
    sequences_y: list[np.ndarray] = []
    returns: list[float] = []
    lengths: list[int] = []
    accepted = 0
    attempts = 0

    for _ in range(int(repeats)):
        for episode_seed in seeds:
            env = obelix_cls(seed=int(episode_seed), **env_kwargs)
            obs = np.asarray(env.reset(seed=int(episode_seed)), dtype=np.float32)
            tracker.reset_all(obs[None, :])
            rng = np.random.default_rng(int(episode_seed))
            teacher_state = ScriptedTeacherState()

            seq_x: list[np.ndarray] = []
            seq_y: list[int] = []
            total_reward = 0.0
            done = False
            steps = 0
            while not done:
                seq_x.append(tracker.features().cpu().numpy()[0].astype(np.float32, copy=True))
                if use_scripted_teacher:
                    action_name = scripted_teacher_action(env, teacher_state)
                else:
                    action_name = expert_policy(obs, rng)
                action_idx = ACTIONS.index(action_name)
                seq_y.append(action_idx)
                obs, reward, done = env.step(action_name, render=False)
                obs = np.asarray(obs, dtype=np.float32)
                total_reward += float(reward)
                steps += 1
                if not done:
                    tracker.post_step(
                        actions=torch.tensor([action_idx], dtype=torch.long),
                        next_obs=obs[None, :],
                        dones=np.asarray([False]),
                    )

            attempts += 1
            returns.append(total_reward)
            lengths.append(steps)
            accepted_episode = min_return is None or total_reward >= float(min_return)
            if accepted_episode:
                seq_x_arr = np.asarray(seq_x, dtype=np.float32)
                seq_y_arr = np.asarray(seq_y, dtype=np.int64)
                sequences_x.append(seq_x_arr)
                sequences_y.append(seq_y_arr)
                accepted += 1
                if total_reward >= 1000.0 and int(success_dup_factor) > 1:
                    for _ in range(int(success_dup_factor) - 1):
                        sequences_x.append(seq_x_arr.copy())
                        sequences_y.append(seq_y_arr.copy())
            print(
                f"[seed_collect] seed={episode_seed} return={total_reward:.1f} steps={steps} "
                f"accepted={accepted}/{attempts}",
                flush=True,
            )

    return sequences_x, sequences_y, {
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        "accepted": float(accepted),
        "attempts": float(attempts),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill macro wall policy into a recurrent legal-observation student.")
    parser.add_argument(
        "--expert_path",
        default=os.path.join(HERE, "submission_d3_wall_macro_51", "agent.py"),
    )
    parser.add_argument("--obelix_py", default=os.path.join(ROOT, "obelix.py"))
    parser.add_argument("--out", default=os.path.join(HERE, "wall_d3_macro_student_v1.pth"))
    parser.add_argument("--collect_seed", type=int, default=0)
    parser.add_argument("--collect_seeds", type=str, default=None)
    parser.add_argument("--collect_repeats", type=int, default=1)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--max_attempts", type=int, default=120)
    parser.add_argument("--min_return", type=float, default=None)
    parser.add_argument("--success_dup_factor", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval_runs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true", default=True)
    parser.add_argument("--no_wall_obstacles", dest="wall_obstacles", action="store_false")
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--encoder_dims", type=int, nargs="+", default=[384, 192])
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[768, 384, 192])
    parser.add_argument("--gru_hidden_dim", type=int, default=192)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--gru_dropout", type=float, default=0.0)
    parser.add_argument("--pose_clip", type=float, default=500.0)
    parser.add_argument("--blind_clip", type=float, default=120.0)
    parser.add_argument("--stuck_clip", type=float, default=24.0)
    parser.add_argument("--contact_clip", type=float, default=24.0)
    parser.add_argument("--same_obs_clip", type=float, default=64.0)
    parser.add_argument("--wall_hit_clip", type=float, default=24.0)
    parser.add_argument("--last_action_hist", type=int, default=8)
    parser.add_argument("--heading_bins", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    feature_config = RecurrentFeatureConfig(
        max_steps=int(args.max_steps),
        pose_clip=float(args.pose_clip),
        blind_clip=float(args.blind_clip),
        stuck_clip=float(args.stuck_clip),
        contact_clip=float(args.contact_clip),
        same_obs_clip=float(args.same_obs_clip),
        wall_hit_clip=float(args.wall_hit_clip),
        last_action_hist=int(args.last_action_hist),
        heading_bins=int(args.heading_bins),
    )
    env_kwargs = {
        "scaling_factor": int(args.scaling_factor),
        "arena_size": int(args.arena_size),
        "max_steps": int(args.max_steps),
        "wall_obstacles": bool(args.wall_obstacles),
        "difficulty": int(args.difficulty),
        "box_speed": int(args.box_speed),
    }

    if args.collect_seeds:
        sequences_x, sequences_y, stats = collect_seed_list_sequences(
            obelix_py=args.obelix_py,
            env_kwargs=env_kwargs,
            feature_config=feature_config,
            expert_path=args.expert_path,
            seeds=parse_seed_list(args.collect_seeds),
            repeats=int(args.collect_repeats),
            min_return=args.min_return,
            success_dup_factor=int(args.success_dup_factor),
        )
    else:
        sequences_x, sequences_y, stats = collect_teacher_sequences(
            obelix_py=args.obelix_py,
            env_kwargs=env_kwargs,
            feature_config=feature_config,
            expert_path=args.expert_path,
            episodes=int(args.episodes),
            seed=int(args.collect_seed),
            min_return=args.min_return,
            max_attempts=int(args.max_attempts),
            success_dup_factor=int(args.success_dup_factor),
        )
    if not sequences_x:
        raise RuntimeError("No teacher sequences were collected.")
    print(
        f"[collect] mean_return={stats['mean_return']:.1f} mean_length={stats['mean_length']:.1f} "
        f"accepted={stats['accepted']:.0f}/{stats['attempts']:.0f}",
        flush=True,
    )

    encoder_dims = tuple(int(x) for x in args.encoder_dims)
    critic_hidden_dims = tuple(int(x) for x in args.critic_hidden_dims)
    model = RecurrentAsymmetricActorCritic(
        actor_dim=feature_config.feature_dim,
        privileged_dim=34,
        encoder_dims=encoder_dims,
        gru_hidden_dim=int(args.gru_hidden_dim),
        critic_hidden_dims=critic_hidden_dims,
        gru_layers=int(args.gru_layers),
        gru_dropout=float(args.gru_dropout),
        fw_bias_init=0.0,
    ).to(device)

    train_sequence_bc(
        model=model,
        sequences_x=sequences_x,
        sequences_y=sequences_y,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        grad_clip=float(args.grad_clip),
    )

    eval_stats = evaluate_actor(
        model=model,
        feature_config=feature_config,
        obelix_py=args.obelix_py,
        env_kwargs=env_kwargs,
        runs=int(args.eval_runs),
        seed=int(args.eval_seed),
        device=device,
    )
    print(
        f"[eval] mean={eval_stats['mean_reward']:.1f} std={eval_stats['std_reward']:.1f} "
        f"success={eval_stats['success_rate']:.3f} mean_len={eval_stats['mean_length']:.1f}",
        flush=True,
    )

    checkpoint = make_checkpoint(
        model=model,
        feature_config=feature_config,
        encoder_dims=encoder_dims,
        gru_hidden_dim=int(args.gru_hidden_dim),
        critic_hidden_dims=critic_hidden_dims,
        privileged_dim=34,
        gru_layers=int(args.gru_layers),
        gru_dropout=float(args.gru_dropout),
        args=args,
        best_eval=float(eval_stats["mean_reward"]),
    )
    save_checkpoint(args.out, checkpoint)
    final_path = os.path.splitext(args.out)[0] + "_final.pth"
    save_checkpoint(final_path, checkpoint)
    print(f"[done] saved -> {args.out}", flush=True)
    print(f"[done] final checkpoint -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
