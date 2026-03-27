from __future__ import annotations

import argparse
import os

import torch

from privileged import privileged_obs_dim
from recurrent_v2 import RecurrentAsymmetricActorCritic, RecurrentFeatureConfig, load_checkpoint
from train_mixed_recurrent_asym_ppo_v2 import evaluate_suite, parse_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a recurrent asymmetric PPO checkpoint on the exact OBELIX suite")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_dir)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--obelix_py", type=str, default=os.path.join(repo_dir, "obelix.py"))
    parser.add_argument("--scenarios", type=str, default="0:nw,0:w,2:nw,2:w,3:nw,3:w")
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    feature_payload = checkpoint.get("feature_config", {})
    feature_config = RecurrentFeatureConfig(**feature_payload)
    encoder_dims = tuple(int(x) for x in checkpoint.get("encoder_dims", [384, 256, 128]))
    critic_hidden_dims = tuple(int(x) for x in checkpoint.get("critic_hidden_dims", [1024, 512, 256]))
    privileged_dim = int(checkpoint.get("privileged_dim", privileged_obs_dim()))

    model = RecurrentAsymmetricActorCritic(
        actor_dim=feature_config.feature_dim,
        privileged_dim=privileged_dim,
        encoder_dims=encoder_dims,
        rnn_hidden_dim=int(checkpoint.get("rnn_hidden_dim", 192)),
        critic_hidden_dims=critic_hidden_dims,
        rnn_layers=int(checkpoint.get("rnn_layers", 1)),
        rnn_dropout=float(checkpoint.get("rnn_dropout", 0.0)),
        actor_dropout=float(checkpoint.get("actor_dropout", 0.0)),
        critic_dropout=float(checkpoint.get("critic_dropout", 0.0)),
        feature_dropout=float(checkpoint.get("feature_dropout", 0.0)),
        aux_target_dim=int(checkpoint.get("aux_target_dim", 0)),
        aux_hidden_dim=int(checkpoint.get("aux_hidden_dim", 0)),
        fw_bias_init=0.0,
        rnn_type=str(checkpoint.get("rnn_type", "gru")),
    ).to(device)
    model.load_state_dict(checkpoint["full_state_dict"], strict=True)

    stats = evaluate_suite(
        model=model,
        feature_config=feature_config,
        obelix_py=args.obelix_py,
        scenarios=parse_scenarios(args.scenarios),
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        box_speed=args.box_speed,
        runs=args.runs,
        seed=args.seed,
        device=device,
        stochastic=bool(args.stochastic),
    )

    scenario_summary = " ".join(
        f"{key.replace('_mean', '')}:{value:.1f}"
        for key, value in stats.items()
        if key.endswith("_mean") and key != "mean_reward"
    )
    print(
        f"checkpoint={args.checkpoint} mean={stats['mean_reward']:.3f} std={stats['std_reward']:.3f} "
        f"{scenario_summary}"
    )


if __name__ == "__main__":
    main()
