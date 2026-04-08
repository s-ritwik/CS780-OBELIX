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

from recurrent import RecurrentAsymmetricActorCritic, RecurrentFeatureConfig, make_checkpoint, save_checkpoint
from distill_macro_wall_student import collect_seed_list_sequences, parse_seed_list
from train_recurrent_sequence_bc import evaluate_actor, train_sequence_bc


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill separate wall/no-wall teachers into one recurrent student.")
    parser.add_argument("--wall_expert", default=os.path.join(HERE, "submission_d3_wall_macro_51", "agent.py"))
    parser.add_argument("--nowall_expert", default=os.path.join(ASYM_DIR, "submission_switch_d3_v3base", "agent.py"))
    parser.add_argument("--obelix_py", default=os.path.join(ROOT, "obelix.py"))
    parser.add_argument("--out", default=os.path.join(HERE, "combo_d3_student_v1.pth"))
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--wall_repeats", type=int, default=4)
    parser.add_argument("--nowall_repeats", type=int, default=6)
    parser.add_argument("--wall_success_dup", type=int, default=80)
    parser.add_argument("--nowall_success_dup", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--encoder_dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[1024, 512, 256])
    parser.add_argument("--gru_hidden_dim", type=int, default=512)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--gru_dropout", type=float, default=0.0)
    parser.add_argument("--last_action_hist", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    feature_config = RecurrentFeatureConfig(max_steps=args.max_steps, last_action_hist=args.last_action_hist)
    seeds = parse_seed_list(args.seeds)
    base_env = {
        "scaling_factor": args.scaling_factor,
        "arena_size": args.arena_size,
        "max_steps": args.max_steps,
        "difficulty": args.difficulty,
        "box_speed": args.box_speed,
    }
    wall_x, wall_y, wall_stats = collect_seed_list_sequences(
        obelix_py=args.obelix_py,
        env_kwargs={**base_env, "wall_obstacles": True},
        feature_config=feature_config,
        expert_path=args.wall_expert,
        seeds=seeds,
        repeats=args.wall_repeats,
        min_return=None,
        success_dup_factor=args.wall_success_dup,
    )
    nowall_x, nowall_y, nowall_stats = collect_seed_list_sequences(
        obelix_py=args.obelix_py,
        env_kwargs={**base_env, "wall_obstacles": False},
        feature_config=feature_config,
        expert_path=args.nowall_expert,
        seeds=seeds,
        repeats=args.nowall_repeats,
        min_return=None,
        success_dup_factor=args.nowall_success_dup,
    )
    sequences_x = wall_x + nowall_x
    sequences_y = wall_y + nowall_y
    print(
        f"[collect] wall_mean={wall_stats['mean_return']:.1f} nowall_mean={nowall_stats['mean_return']:.1f} "
        f"seqs={len(sequences_x)}",
        flush=True,
    )

    encoder_dims = tuple(int(x) for x in args.encoder_dims)
    critic_hidden_dims = tuple(int(x) for x in args.critic_hidden_dims)
    model = RecurrentAsymmetricActorCritic(
        actor_dim=feature_config.feature_dim,
        privileged_dim=34,
        encoder_dims=encoder_dims,
        gru_hidden_dim=args.gru_hidden_dim,
        critic_hidden_dims=critic_hidden_dims,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        fw_bias_init=0.0,
    ).to(device)
    train_sequence_bc(
        model=model,
        sequences_x=sequences_x,
        sequences_y=sequences_y,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )
    wall_eval = evaluate_actor(
        model=model,
        feature_config=feature_config,
        obelix_py=args.obelix_py,
        env_kwargs={**base_env, "wall_obstacles": True},
        runs=10,
        seed=0,
        device=device,
    )
    nowall_eval = evaluate_actor(
        model=model,
        feature_config=feature_config,
        obelix_py=args.obelix_py,
        env_kwargs={**base_env, "wall_obstacles": False},
        runs=10,
        seed=0,
        device=device,
    )
    weighted = 0.6 * wall_eval["mean_reward"] + 0.4 * nowall_eval["mean_reward"]
    print(
        f"[eval] wall={wall_eval['mean_reward']:.1f} nowall={nowall_eval['mean_reward']:.1f} weighted={weighted:.1f}",
        flush=True,
    )
    checkpoint = make_checkpoint(
        model=model,
        feature_config=feature_config,
        encoder_dims=encoder_dims,
        gru_hidden_dim=args.gru_hidden_dim,
        critic_hidden_dims=critic_hidden_dims,
        privileged_dim=34,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        args=args,
        best_eval=float(weighted),
    )
    save_checkpoint(args.out, checkpoint)
    save_checkpoint(os.path.splitext(args.out)[0] + "_final.pth", checkpoint)
    print(f"[done] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
