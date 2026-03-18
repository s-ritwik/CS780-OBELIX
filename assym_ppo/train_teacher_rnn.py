from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from rnn_model import ACTIONS, ACTION_DIM, RecurrentActor, RecurrentConfig, build_input_tensor, save_checkpoint, sequence_cross_entropy
from teacher import PrivilegedTeacher, ScriptedTeacherState, scripted_teacher_action


def import_symbol(py_file: str, symbol: str):
    spec = importlib.util.spec_from_file_location("obelix_module", py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {symbol} from {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise AttributeError(f"{symbol} not found in {py_file}")
    return getattr(module, symbol)


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


@torch.no_grad()
def evaluate_model(
    model: RecurrentActor,
    config: RecurrentConfig,
    *,
    obelix_py: str,
    runs: int,
    seed: int,
    env_kwargs: dict,
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
        total_reward = 0.0
        steps = 0
        done = False
        hidden = None
        prev_action = torch.zeros((1, 1, ACTION_DIM), dtype=torch.float32, device=device)

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            step_frac = torch.full(
                (1, 1, 1),
                float(steps) / max(1.0, float(config.max_steps)),
                dtype=torch.float32,
                device=device,
            )
            inp = build_input_tensor(
                obs_t,
                prev_action_one_hot=prev_action,
                step_frac=step_frac,
                config=config,
            )
            logits, hidden = model(inp, hidden)
            action_idx = int(torch.argmax(logits[:, -1], dim=1).item())
            obs, reward, done = env.step(ACTIONS[action_idx], render=False)
            obs = np.asarray(obs, dtype=np.float32)
            total_reward += float(reward)
            steps += 1

            prev_action.zero_()
            prev_action[0, 0, action_idx] = 1.0

        scores.append(total_reward)
        lengths.append(steps)
        if total_reward >= 1000.0:
            successes += 1

    return {
        "mean_reward": float(np.mean(scores)),
        "std_reward": float(np.std(scores)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(successes) / float(max(1, runs)),
    }


class EpisodeBuffer:
    def __init__(self, input_dim: int) -> None:
        self.input_dim = int(input_dim)
        self.inputs: list[np.ndarray] = []
        self.targets: list[int] = []

    def add(self, inp: np.ndarray, target: int) -> None:
        self.inputs.append(inp.astype(np.float32, copy=True))
        self.targets.append(int(target))

    def finalize(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.targets:
            return None
        return (
            np.asarray(self.inputs, dtype=np.float32),
            np.asarray(self.targets, dtype=np.int64),
        )

    def reset(self) -> None:
        self.inputs.clear()
        self.targets.clear()


def collect_dataset(
    *,
    model: RecurrentActor | None,
    config: RecurrentConfig,
    teacher: PrivilegedTeacher,
    vec_env,
    total_episodes: int,
    beta: float,
    seed: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    obs = vec_env.reset_all(seed=seed)
    teacher.reset_all()
    num_envs = int(vec_env.num_envs)
    episode_buffers = [EpisodeBuffer(config.input_dim) for _ in range(num_envs)]
    prev_action = torch.zeros((num_envs, 1, ACTION_DIM), dtype=torch.float32, device=device)
    hidden = None

    episode_returns = np.zeros((num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((num_envs,), dtype=np.int32)
    recent_returns: list[float] = []
    recent_lengths: list[int] = []

    sequences_x: list[np.ndarray] = []
    sequences_y: list[np.ndarray] = []
    completed = 0

    model_was_training = None
    if model is not None:
        model_was_training = model.training
        model.eval()

    while completed < total_episodes:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(1)
        step_frac = torch.as_tensor(
            vec_env.current_step.to(dtype=torch.float32).detach().cpu().numpy() / max(1.0, float(config.max_steps)),
            dtype=torch.float32,
            device=device,
        ).view(num_envs, 1, 1)
        inp = build_input_tensor(
            obs_t,
            prev_action_one_hot=prev_action,
            step_frac=step_frac,
            config=config,
        )
        teacher_actions = teacher.act(vec_env)

        if model is None:
            exec_actions = teacher_actions.clone()
        else:
            logits, hidden = model(inp, hidden)
            student_actions = torch.argmax(logits[:, -1], dim=1)
            if beta >= 1.0:
                exec_actions = teacher_actions.clone()
            elif beta <= 0.0:
                exec_actions = student_actions
            else:
                chooser = torch.rand((num_envs,), device=device) < float(beta)
                exec_actions = torch.where(chooser, teacher_actions, student_actions)

        inp_np = inp[:, 0].detach().cpu().numpy()
        teacher_np = teacher_actions.detach().cpu().numpy()
        exec_np = exec_actions.detach().cpu().numpy()

        for env_idx in range(num_envs):
            if completed >= total_episodes:
                break
            if bool(vec_env.done[env_idx]):
                continue
            episode_buffers[env_idx].add(inp_np[env_idx], int(teacher_np[env_idx]))

        next_obs, rewards, dones = vec_env.step(exec_np)
        episode_returns += rewards
        episode_lengths += 1

        done_idx = np.flatnonzero(dones)
        if done_idx.size > 0:
            reset_map = vec_env.reset(
                env_indices=done_idx.tolist(),
                seed=seed + 100_000 + completed * 17,
            )
            for idx in done_idx.tolist():
                payload = episode_buffers[idx].finalize()
                if payload is not None and completed < total_episodes:
                    seq_x, seq_y = payload
                    sequences_x.append(seq_x)
                    sequences_y.append(seq_y)
                    recent_returns.append(float(episode_returns[idx]))
                    recent_lengths.append(int(episode_lengths[idx]))
                    completed += 1
                episode_buffers[idx].reset()
                episode_returns[idx] = 0.0
                episode_lengths[idx] = 0
                prev_action[idx] = 0.0
                if idx in reset_map:
                    next_obs[idx] = reset_map[idx]

            teacher.reset_indices(done_idx.tolist())
            if hidden is not None:
                hidden[:, done_idx, :] = 0.0

        prev_action.zero_()
        prev_action[torch.arange(num_envs, device=device), 0, exec_actions] = 1.0
        obs = next_obs

    if model is not None and model_was_training:
        model.train()

    summary = {
        "episodes": float(completed),
        "mean_return": float(np.mean(recent_returns)) if recent_returns else float("nan"),
        "mean_length": float(np.mean(recent_lengths)) if recent_lengths else float("nan"),
    }
    return sequences_x, sequences_y, summary


def collect_cpu_teacher_dataset(
    *,
    obelix_py: str,
    env_kwargs: dict,
    config: RecurrentConfig,
    accepted_episodes: int,
    success_only: bool,
    max_attempts: int,
    min_return: float | None,
    success_dup_factor: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    obelix_cls = import_symbol(obelix_py, "OBELIX")
    sequences_x: list[np.ndarray] = []
    sequences_y: list[np.ndarray] = []
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    attempts = 0

    while len(sequences_x) < int(accepted_episodes) and attempts < int(max_attempts):
        episode_seed = 50_000 + attempts
        env = obelix_cls(seed=episode_seed, **env_kwargs)
        obs = np.asarray(env.reset(seed=episode_seed), dtype=np.float32)
        teacher_state = ScriptedTeacherState()

        seq_x: list[np.ndarray] = []
        seq_y: list[int] = []
        prev_action = np.zeros((ACTION_DIM,), dtype=np.float32)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
            prev_action_t = torch.as_tensor(prev_action, dtype=torch.float32).view(1, 1, -1)
            step_frac = torch.full(
                (1, 1, 1),
                float(steps) / max(1.0, float(config.max_steps)),
                dtype=torch.float32,
            )
            inp = build_input_tensor(
                obs_t,
                prev_action_one_hot=prev_action_t,
                step_frac=step_frac,
                config=config,
            )
            action_name = scripted_teacher_action(env, teacher_state)
            action_idx = ACTIONS.index(action_name)
            seq_x.append(inp[0, 0].cpu().numpy().astype(np.float32, copy=True))
            seq_y.append(action_idx)
            obs, reward, done = env.step(action_name, render=False)
            obs = np.asarray(obs, dtype=np.float32)
            total_reward += float(reward)
            steps += 1
            prev_action.fill(0.0)
            prev_action[action_idx] = 1.0

        attempts += 1
        success = total_reward >= 1000.0
        successes += int(success)
        returns.append(total_reward)
        lengths.append(steps)
        accepted = False
        if success_only:
            accepted = success
        elif min_return is None:
            accepted = True
        else:
            accepted = total_reward >= float(min_return)

        if accepted:
            seq_x_arr = np.asarray(seq_x, dtype=np.float32)
            seq_y_arr = np.asarray(seq_y, dtype=np.int64)
            sequences_x.append(seq_x_arr)
            sequences_y.append(seq_y_arr)
            if success and int(success_dup_factor) > 1:
                for _ in range(int(success_dup_factor) - 1):
                    sequences_x.append(seq_x_arr.copy())
                    sequences_y.append(seq_y_arr.copy())
        if attempts % 25 == 0:
            print(
                f"[collect] cpu attempts={attempts} accepted={len(sequences_x)} "
                f"successes={successes} last_return={total_reward:.1f}",
                flush=True,
            )

    stats = {
        "accepted": float(len(sequences_x)),
        "attempts": float(attempts),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        "success_rate": float(successes) / float(max(1, attempts)),
    }
    return sequences_x, sequences_y, stats


def train_epoch(
    model: RecurrentActor,
    optimizer: torch.optim.Optimizer,
    *,
    config: RecurrentConfig,
    sequences_x: list[np.ndarray],
    sequences_y: list[np.ndarray],
    batch_size: int,
    device: torch.device,
    grad_clip: float,
) -> dict[str, float]:
    order = np.random.permutation(len(sequences_x))
    total_loss = 0.0
    total_correct = 0
    total_steps = 0

    model.train()
    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        max_len = max(int(sequences_y[idx].shape[0]) for idx in idxs)
        x_batch = np.zeros((len(idxs), max_len, config.input_dim), dtype=np.float32)
        y_batch = np.zeros((len(idxs), max_len), dtype=np.int64)
        mask_batch = np.zeros((len(idxs), max_len), dtype=np.float32)

        for batch_idx, seq_idx in enumerate(idxs):
            seq_x = sequences_x[seq_idx]
            seq_y = sequences_y[seq_idx]
            seq_len = int(seq_y.shape[0])
            x_batch[batch_idx, :seq_len] = seq_x
            y_batch[batch_idx, :seq_len] = seq_y
            mask_batch[batch_idx, :seq_len] = 1.0

        x_t = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y_batch, dtype=torch.long, device=device)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.float32, device=device)
        logits, _ = model(x_t)
        loss = sequence_cross_entropy(logits, y_t, mask_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            correct = ((pred == y_t) & (mask_t > 0.5)).sum().item()
            total_correct += int(correct)
            total_steps += int(mask_t.sum().item())
            total_loss += float(loss.item()) * float(mask_t.sum().item())

    return {
        "loss": total_loss / max(1, total_steps),
        "acc": float(total_correct) / float(max(1, total_steps)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recurrent teacher-imitation trainer for OBELIX")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_dir)

    parser.add_argument("--obelix_py", type=str, default=os.path.join(repo_dir, "obelix.py"))
    parser.add_argument("--obelix_torch_py", type=str, default=os.path.join(repo_dir, "obelix_torch.py"))
    parser.add_argument("--out", type=str, default=os.path.join(base_dir, "teacher_rnn_best.pth"))

    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--env_device", type=str, default="auto")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--include_step_frac", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_prev_action", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--teacher_episodes", type=int, default=1024)
    parser.add_argument("--teacher_source", choices=["vector", "cpu"], default="vector")
    parser.add_argument("--teacher_success_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher_max_attempts", type=int, default=0)
    parser.add_argument("--teacher_min_return", type=float, default=float("-inf"))
    parser.add_argument("--teacher_success_dup", type=int, default=1)
    parser.add_argument("--dagger_rounds", type=int, default=4)
    parser.add_argument("--dagger_episodes", type=int, default=512)
    parser.add_argument("--dagger_beta_start", type=float, default=0.7)
    parser.add_argument("--dagger_beta_end", type=float, default=0.0)
    parser.add_argument("--epochs_per_round", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_sequences", type=int, default=20000)

    parser.add_argument("--eval_runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
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

    config = RecurrentConfig(
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        include_step_frac=bool(args.include_step_frac),
        include_prev_action=bool(args.include_prev_action),
        max_steps=int(args.max_steps),
    )
    model = RecurrentActor(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    env_device = str(device) if args.env_device == "auto" else args.env_device
    vec_env_cls = import_symbol(args.obelix_torch_py, "OBELIXVectorized")
    vec_env = vec_env_cls(
        num_envs=args.num_envs,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed * 10_000,
        device=env_device,
    )
    teacher = PrivilegedTeacher(num_envs=args.num_envs, device=torch.device(env_device))

    env_kwargs = {
        "scaling_factor": args.scaling_factor,
        "arena_size": args.arena_size,
        "max_steps": args.max_steps,
        "wall_obstacles": args.wall_obstacles,
        "difficulty": args.difficulty,
        "box_speed": args.box_speed,
    }

    print(
        f"[setup] device={device} env_device={env_device} num_envs={args.num_envs} "
        f"wall_obstacles={args.wall_obstacles} difficulty={args.difficulty}"
    )
    print(
        f"[setup] input_dim={config.input_dim} hidden_size={config.hidden_size} "
        f"num_layers={config.num_layers} teacher_source={args.teacher_source} "
        f"teacher_episodes={args.teacher_episodes}"
    )

    start_time = time.time()
    best_eval = -float("inf")
    dataset_x: list[np.ndarray] = []
    dataset_y: list[np.ndarray] = []

    try:
        if args.teacher_source == "cpu":
            teacher_max_attempts = int(args.teacher_max_attempts) if int(args.teacher_max_attempts) > 0 else int(args.teacher_episodes) * 20
            teacher_x, teacher_y, teacher_stats = collect_cpu_teacher_dataset(
                obelix_py=args.obelix_py,
                env_kwargs=env_kwargs,
                config=config,
                accepted_episodes=int(args.teacher_episodes),
                success_only=bool(args.teacher_success_only),
                max_attempts=teacher_max_attempts,
                min_return=None if math.isinf(float(args.teacher_min_return)) else float(args.teacher_min_return),
                success_dup_factor=int(args.teacher_success_dup),
            )
        else:
            teacher_x, teacher_y, teacher_stats = collect_dataset(
                model=None,
                config=config,
                teacher=teacher,
                vec_env=vec_env,
                total_episodes=int(args.teacher_episodes),
                beta=1.0,
                seed=args.seed * 10_000,
                device=device,
            )
        dataset_x.extend(teacher_x)
        dataset_y.extend(teacher_y)
        if args.teacher_source == "cpu":
            print(
                f"[collect] cpu teacher accepted={int(teacher_stats['accepted'])} attempts={int(teacher_stats['attempts'])} "
                f"success_rate={teacher_stats['success_rate']:.3f} mean_return={teacher_stats['mean_return']:.1f} "
                f"mean_len={teacher_stats['mean_length']:.1f}"
            )
        else:
            print(
                f"[collect] teacher episodes={int(teacher_stats['episodes'])} "
                f"mean_return={teacher_stats['mean_return']:.1f} mean_len={teacher_stats['mean_length']:.1f}"
            )

        total_rounds = 1 + int(args.dagger_rounds)
        for round_idx in range(total_rounds):
            if round_idx == 0:
                beta = 1.0
            else:
                frac = float(round_idx - 1) / float(max(1, args.dagger_rounds - 1))
                beta = float(args.dagger_beta_start) + frac * float(args.dagger_beta_end - args.dagger_beta_start)
                dagger_x, dagger_y, stats = collect_dataset(
                    model=model,
                    config=config,
                    teacher=teacher,
                    vec_env=vec_env,
                    total_episodes=int(args.dagger_episodes),
                    beta=beta,
                    seed=args.seed * 20_000 + round_idx * 1_000,
                    device=device,
                )
                dataset_x.extend(dagger_x)
                dataset_y.extend(dagger_y)
                if len(dataset_x) > int(args.max_sequences):
                    dataset_x = dataset_x[-int(args.max_sequences) :]
                    dataset_y = dataset_y[-int(args.max_sequences) :]
                print(
                    f"[collect] round={round_idx} beta={beta:.2f} episodes={int(stats['episodes'])} "
                    f"mean_return={stats['mean_return']:.1f} mean_len={stats['mean_length']:.1f} "
                    f"dataset={len(dataset_x)}"
                )

            for epoch_idx in range(int(args.epochs_per_round)):
                metrics = train_epoch(
                    model,
                    optimizer,
                    config=config,
                    sequences_x=dataset_x,
                    sequences_y=dataset_y,
                    batch_size=int(args.batch_size),
                    device=device,
                    grad_clip=float(args.grad_clip),
                )
                print(
                    f"[train] round={round_idx} epoch={epoch_idx + 1}/{args.epochs_per_round} "
                    f"loss={metrics['loss']:.4f} acc={metrics['acc']:.3f} dataset={len(dataset_x)}"
                )

            eval_stats = evaluate_model(
                model,
                config,
                obelix_py=args.obelix_py,
                runs=args.eval_runs,
                seed=args.seed,
                env_kwargs=env_kwargs,
                device=device,
            )
            print(
                f"[eval] round={round_idx} mean={eval_stats['mean_reward']:.1f} "
                f"std={eval_stats['std_reward']:.1f} success={eval_stats['success_rate']:.3f} "
                f"mean_len={eval_stats['mean_length']:.1f}"
            )
            if eval_stats["mean_reward"] > best_eval:
                best_eval = eval_stats["mean_reward"]
                save_checkpoint(
                    args.out,
                    model=model,
                    config=config,
                    args=args,
                    best_eval=best_eval,
                )
                print(f"[eval] new best -> {args.out} ({best_eval:.1f})")

    finally:
        vec_env.close()

    elapsed = max(0.0, time.time() - start_time)
    final_path = os.path.splitext(args.out)[0] + "_final.pth"
    save_checkpoint(
        final_path,
        model=model,
        config=config,
        args=args,
        best_eval=best_eval,
    )
    print(f"[done] total_train_time={format_hms(elapsed)} ({elapsed / 60.0:.2f} min)")
    print(f"[done] final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
