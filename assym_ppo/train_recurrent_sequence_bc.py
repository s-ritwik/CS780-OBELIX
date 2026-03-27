from __future__ import annotations

import argparse
import importlib.util
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from recurrent import (
    ACTIONS,
    ACTION_DIM,
    PoseMemoryTracker,
    RecurrentAsymmetricActorCritic,
    RecurrentFeatureConfig,
    format_hms,
    make_checkpoint,
    save_checkpoint,
)
from teacher import ScriptedTeacherState, scripted_teacher_action


def import_symbol(py_file: str, symbol: str):
    spec = importlib.util.spec_from_file_location("obelix_module", py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {symbol} from {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise AttributeError(f"{symbol} not found in {py_file}")
    return getattr(module, symbol)


def import_module(py_file: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_teacher_sequences(
    *,
    obelix_py: str,
    env_kwargs: dict,
    feature_config: RecurrentFeatureConfig,
    expert_path: str,
    episodes: int,
    seed: int,
    min_return: float | None,
    max_attempts: int,
    success_dup_factor: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    obelix_cls = import_symbol(obelix_py, "OBELIX")
    expert_policy = None
    use_scripted_teacher = expert_path == "scripted_teacher"
    if not use_scripted_teacher:
        expert_mod = import_module(expert_path, f"seqbc_expert_{abs(hash(expert_path))}")
        if not hasattr(expert_mod, "policy"):
            raise AttributeError(f"Expert module {expert_path} does not define policy(obs, rng)")
        expert_policy = getattr(expert_mod, "policy")
    tracker = PoseMemoryTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
    sequences_x: list[np.ndarray] = []
    sequences_y: list[np.ndarray] = []
    returns = deque(maxlen=50)
    lengths = deque(maxlen=50)

    attempts = 0
    accepted = 0
    while accepted < int(episodes) and attempts < int(max_attempts):
        episode_seed = int(seed + attempts)
        env = obelix_cls(seed=episode_seed, **env_kwargs)
        obs = np.asarray(env.reset(seed=episode_seed), dtype=np.float32)
        tracker.reset_all(obs[None, :])
        teacher_state = ScriptedTeacherState()
        rng = np.random.default_rng(episode_seed)

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

        accepted_episode = min_return is None or total_reward >= float(min_return)
        success = total_reward >= 1000.0
        if accepted_episode:
            seq_x_arr = np.asarray(seq_x, dtype=np.float32)
            seq_y_arr = np.asarray(seq_y, dtype=np.int64)
            sequences_x.append(seq_x_arr)
            sequences_y.append(seq_y_arr)
            accepted += 1
            if success and int(success_dup_factor) > 1:
                for _ in range(int(success_dup_factor) - 1):
                    sequences_x.append(seq_x_arr.copy())
                    sequences_y.append(seq_y_arr.copy())
        returns.append(total_reward)
        lengths.append(steps)
        attempts += 1
        if attempts % max(1, min(20, max_attempts)) == 0:
            print(
                f"[collect] attempts={attempts}/{max_attempts} accepted={accepted}/{episodes} "
                f"recent_return={float(np.mean(returns)):.1f} recent_len={float(np.mean(lengths)):.1f}",
                flush=True,
            )

    return sequences_x, sequences_y, {
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        "accepted": float(accepted),
        "attempts": float(attempts),
    }


def masked_sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, ACTION_DIM)
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    losses = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    losses = losses * flat_mask.to(dtype=losses.dtype)
    return losses.sum() / flat_mask.to(dtype=losses.dtype).sum().clamp_min(1.0)


def actor_logits_sequence(
    model: RecurrentAsymmetricActorCritic,
    x_batch: torch.Tensor,
) -> torch.Tensor:
    seq_len, batch_size, _ = x_batch.shape
    hidden = model.initial_state(batch_size, x_batch.device)
    logits_list = []
    for t in range(seq_len):
        starts = torch.ones((batch_size,), dtype=torch.float32, device=x_batch.device) if t == 0 else torch.zeros(
            (batch_size,),
            dtype=torch.float32,
            device=x_batch.device,
        )
        logits_t, hidden = model.actor_step(x_batch[t], hidden, starts)
        logits_list.append(logits_t)
    return torch.stack(logits_list, dim=0)


def train_sequence_bc(
    *,
    model: RecurrentAsymmetricActorCritic,
    sequences_x: list[np.ndarray],
    sequences_y: list[np.ndarray],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
) -> None:
    params = list(model.actor_encoder.parameters()) + list(model.actor_rnn.parameters()) + list(model.policy_head.parameters())
    optimizer = torch.optim.Adam(params, lr=float(lr))

    for epoch in range(int(epochs)):
        order = np.random.permutation(len(sequences_x))
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0.0

        for start in range(0, len(order), int(batch_size)):
            idxs = order[start : start + int(batch_size)]
            max_len = max(int(sequences_y[idx].shape[0]) for idx in idxs)
            cur_batch = len(idxs)

            x_batch = np.zeros((max_len, cur_batch, sequences_x[0].shape[1]), dtype=np.float32)
            y_batch = np.zeros((max_len, cur_batch), dtype=np.int64)
            mask_batch = np.zeros((max_len, cur_batch), dtype=np.float32)

            for b, seq_idx in enumerate(idxs):
                seq_x = sequences_x[int(seq_idx)]
                seq_y = sequences_y[int(seq_idx)]
                seq_len = int(seq_y.shape[0])
                x_batch[:seq_len, b] = seq_x
                y_batch[:seq_len, b] = seq_y
                mask_batch[:seq_len, b] = 1.0

            x_t = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
            y_t = torch.as_tensor(y_batch, dtype=torch.long, device=device)
            mask_t = torch.as_tensor(mask_batch, dtype=torch.float32, device=device)

            logits = actor_logits_sequence(model, x_t)
            loss = masked_sequence_cross_entropy(logits, y_t, mask_t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                correct = ((pred == y_t).to(torch.float32) * mask_t).sum().item()
                count = mask_t.sum().item()
            total_loss += float(loss.item()) * cur_batch
            total_correct += float(correct)
            total_count += float(count)

        print(
            f"[train] epoch={epoch + 1}/{epochs} loss={total_loss / max(1, len(order)):.4f} "
            f"acc={total_correct / max(1.0, total_count):.3f}",
            flush=True,
        )


@torch.no_grad()
def evaluate_actor(
    model: RecurrentAsymmetricActorCritic,
    *,
    feature_config: RecurrentFeatureConfig,
    obelix_py: str,
    env_kwargs: dict,
    runs: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    obelix_cls = import_symbol(obelix_py, "OBELIX")
    model.eval()
    scores: list[float] = []
    lengths: list[int] = []
    successes = 0

    for run_idx in range(int(runs)):
        episode_seed = int(seed + run_idx)
        env = obelix_cls(seed=episode_seed, **env_kwargs)
        obs = np.asarray(env.reset(seed=episode_seed), dtype=np.float32)
        tracker = PoseMemoryTracker(num_envs=1, config=feature_config, device=device)
        tracker.reset_all(obs[None, :])
        hidden = model.initial_state(1, device)

        total_reward = 0.0
        steps = 0
        done = False
        pending_action: int | None = None
        while not done:
            if pending_action is None:
                starts = torch.ones((1,), dtype=torch.float32, device=device)
            else:
                tracker.post_step(
                    actions=torch.tensor([pending_action], dtype=torch.long, device=device),
                    next_obs=obs[None, :],
                    dones=np.asarray([False]),
                )
                starts = torch.zeros((1,), dtype=torch.float32, device=device)
            features = tracker.features()
            logits, hidden = model.actor_step(features, hidden, starts)
            action_idx = int(torch.argmax(logits, dim=1).item())
            pending_action = action_idx
            obs, reward, done = env.step(ACTIONS[action_idx], render=False)
            obs = np.asarray(obs, dtype=np.float32)
            total_reward += float(reward)
            steps += 1

        scores.append(total_reward)
        lengths.append(steps)
        if total_reward >= 1000.0:
            successes += 1

    return {
        "mean_reward": float(np.mean(scores)),
        "std_reward": float(np.std(scores)),
        "success_rate": float(successes) / float(max(1, runs)),
        "mean_length": float(np.mean(lengths)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequence BC for recurrent asym actor")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_dir)
    parser.add_argument("--obelix_py", type=str, default=os.path.join(repo_dir, "obelix.py"))
    parser.add_argument("--out", type=str, default=os.path.join(base_dir, "recurrent_seqbc_teacher.pth"))
    parser.add_argument("--expert_path", type=str, default="scripted_teacher")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=500)
    parser.add_argument("--min_return", type=float, default=-500.0)
    parser.add_argument("--success_dup_factor", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--eval_runs", type=int, default=10)
    parser.add_argument("--encoder_dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[768, 384, 192])
    parser.add_argument("--gru_hidden_dim", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--gru_dropout", type=float, default=0.0)
    parser.add_argument("--pose_clip", type=float, default=500.0)
    parser.add_argument("--blind_clip", type=float, default=120.0)
    parser.add_argument("--stuck_clip", type=float, default=24.0)
    parser.add_argument("--contact_clip", type=float, default=24.0)
    parser.add_argument("--same_obs_clip", type=float, default=64.0)
    parser.add_argument("--wall_hit_clip", type=float, default=24.0)
    parser.add_argument("--last_action_hist", type=int, default=6)
    parser.add_argument("--heading_bins", type=int, default=8)
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
    encoder_dims = tuple(int(x) for x in args.encoder_dims)
    critic_hidden_dims = tuple(int(x) for x in args.critic_hidden_dims)

    env_kwargs = {
        "scaling_factor": args.scaling_factor,
        "arena_size": args.arena_size,
        "max_steps": args.max_steps,
        "wall_obstacles": args.wall_obstacles,
        "difficulty": args.difficulty,
        "box_speed": args.box_speed,
    }

    sequences_x, sequences_y, stats = collect_teacher_sequences(
        obelix_py=args.obelix_py,
        env_kwargs=env_kwargs,
        feature_config=feature_config,
        expert_path=args.expert_path,
        episodes=args.episodes,
        seed=args.seed + 200_000,
        min_return=args.min_return,
        max_attempts=args.max_attempts,
        success_dup_factor=args.success_dup_factor,
    )
    print(
        f"[collect] mean_return={stats['mean_return']:.1f} mean_length={stats['mean_length']:.1f} "
        f"accepted={stats['accepted']:.0f}/{stats['attempts']:.0f}",
        flush=True,
    )

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
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )

    eval_stats = evaluate_actor(
        model=model,
        feature_config=feature_config,
        obelix_py=args.obelix_py,
        env_kwargs=env_kwargs,
        runs=args.eval_runs,
        seed=args.seed,
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
