from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HERE)
GRU_DIR = os.path.join(REPO_DIR, "gru_pose")
if GRU_DIR not in sys.path:
    sys.path.insert(0, GRU_DIR)

from common import ACTIONS, ACTION_DIM, FeatureConfig, GRUPolicy, PoseGRUFeatureTracker, make_checkpoint, save_checkpoint
from common import load_checkpoint


ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}


def import_symbol(py_file: str, symbol: str):
    spec = importlib.util.spec_from_file_location("obelix_module", py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {symbol} from {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise AttributeError(f"{symbol} not found in {py_file}")
    return getattr(module, symbol)


class SequenceBuffer:
    def __init__(self, rollout_steps: int, num_envs: int, feat_dim: int, device: torch.device) -> None:
        self.rollout_steps = int(rollout_steps)
        self.num_envs = int(num_envs)
        self.feat_dim = int(feat_dim)
        self.device = device
        self.features = torch.zeros((self.rollout_steps, self.num_envs, self.feat_dim), dtype=torch.float32, device=device)
        self.starts = torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=device)
        self.teacher_actions = torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.long, device=device)

    def add(self, step: int, features: torch.Tensor, starts: torch.Tensor, teacher_actions: torch.Tensor) -> None:
        self.features[step].copy_(features)
        self.starts[step].copy_(starts)
        self.teacher_actions[step].copy_(teacher_actions)

    def sequence_batches(self, seq_len: int, batch_size: int, epochs: int):
        if self.rollout_steps % int(seq_len) != 0:
            raise ValueError("rollout_steps must be divisible by seq_len")

        items: list[tuple[int, int]] = []
        for env_id in range(self.num_envs):
            for t0 in range(0, self.rollout_steps, int(seq_len)):
                items.append((t0, env_id))

        seqs_per_batch = max(1, int(batch_size))
        total = len(items)
        for _ in range(int(epochs)):
            perm = torch.randperm(total, device=self.device)
            for start_idx in range(0, total, seqs_per_batch):
                batch_ids = perm[start_idx : start_idx + seqs_per_batch].tolist()
                chosen = [items[int(i)] for i in batch_ids]
                bs = len(chosen)
                feat = torch.zeros((seq_len, bs, self.feat_dim), dtype=torch.float32, device=self.device)
                starts = torch.zeros((seq_len, bs), dtype=torch.float32, device=self.device)
                labels = torch.zeros((seq_len, bs), dtype=torch.long, device=self.device)
                for b, (t0, env_id) in enumerate(chosen):
                    sl = slice(t0, t0 + seq_len)
                    feat[:, b] = self.features[sl, env_id]
                    starts[:, b] = self.starts[sl, env_id]
                    labels[:, b] = self.teacher_actions[sl, env_id]
                yield feat, starts, labels


class NoWallPrivilegedTeacher:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)

    @staticmethod
    def _wrap_deg(angle: float) -> float:
        return ((angle + 180.0) % 360.0) - 180.0

    def act_batch(self, vec_env) -> np.ndarray:
        actions = np.zeros((self.num_envs,), dtype=np.int64)
        half = max(1, int(vec_env.box_size) // 2)
        offset = float(vec_env.bot_radius + half + 12)

        for env_id in range(self.num_envs):
            if bool(vec_env.done[env_id]):
                actions[env_id] = ACTION_TO_INDEX["L22"]
                continue

            box_x = float(vec_env.box_center_x[env_id])
            box_y = float(vec_env.box_center_y[env_id])
            bot_x = float(vec_env.bot_center_x[env_id])
            bot_y = float(vec_env.bot_center_y[env_id])
            facing = float(vec_env.facing_angle[env_id])

            distances = {
                "left": box_x - (10 + half),
                "right": (vec_env.frame_size[1] - 10 - half) - box_x,
                "bottom": box_y - (10 + half),
                "top": (vec_env.frame_size[0] - 10 - half) - box_y,
            }
            wall = min(distances, key=distances.get)
            push_angle = {"left": 180.0, "right": 0.0, "bottom": -90.0, "top": 90.0}[wall]

            if bool(vec_env.enable_push[env_id]):
                desired = push_angle
            else:
                ux = math.cos(math.radians(push_angle))
                uy = math.sin(math.radians(push_angle))
                stage_x = box_x - offset * ux
                stage_y = box_y - offset * uy
                stage_dist = math.hypot(bot_x - stage_x, bot_y - stage_y)
                if stage_dist <= 20.0:
                    desired = push_angle
                else:
                    desired = math.degrees(math.atan2(stage_y - bot_y, stage_x - bot_x))

            diff = self._wrap_deg(desired - facing)
            if diff > 50.0:
                action = "L45"
            elif diff > 12.0:
                action = "L22"
            elif diff < -50.0:
                action = "R45"
            elif diff < -12.0:
                action = "R22"
            else:
                action = "FW"
            actions[env_id] = ACTION_TO_INDEX[action]
        return actions


@torch.no_grad()
def evaluate_model(
    model: GRUPolicy,
    feature_config: FeatureConfig,
    obelix_py: str,
    *,
    runs: int,
    seed: int,
    env_kwargs: dict,
    device: torch.device,
    stochastic: bool,
) -> dict[str, float]:
    OBELIX = import_symbol(obelix_py, "OBELIX")
    tracker = PoseGRUFeatureTracker(num_envs=1, config=feature_config, device=device)
    scores: list[float] = []
    successes = 0

    for run_idx in range(int(runs)):
        episode_seed = int(seed + run_idx)
        env = OBELIX(seed=episode_seed, **env_kwargs)
        obs = env.reset(seed=episode_seed)
        tracker.reset_all(np.asarray(obs, dtype=np.float32)[None, :])
        hidden = model.initial_state(1, device)
        starts = torch.ones((1,), dtype=torch.float32, device=device)
        pending_action = None
        rng = np.random.default_rng(episode_seed)
        total = 0.0
        done = False

        while not done:
            feat = tracker.features()
            logits, _, hidden = model.forward_step(feat, hidden, starts)
            if stochastic:
                probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)
                probs /= probs.sum()
                action_idx = int(rng.choice(len(ACTIONS), p=probs))
            else:
                action_idx = int(torch.argmax(logits, dim=1).item())
            action_name = ACTIONS[action_idx]
            next_obs, reward, done = env.step(action_name, render=False)
            total += float(reward)
            if not done:
                tracker.post_step(
                    actions=torch.tensor([action_idx], dtype=torch.long, device=device),
                    next_obs=np.asarray(next_obs, dtype=np.float32)[None, :],
                    dones=np.asarray([False]),
                )
                starts = torch.zeros((1,), dtype=torch.float32, device=device)
            else:
                starts = torch.ones((1,), dtype=torch.float32, device=device)
            obs = next_obs
            pending_action = action_idx
            if done:
                del pending_action

        scores.append(total)
        if total >= 1000.0:
            successes += 1

    return {
        "mean_reward": float(np.mean(scores)),
        "std_reward": float(np.std(scores)),
        "success_rate": float(successes) / float(max(1, runs)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-wall GRU DAgger trainer")
    parser.add_argument("--obelix_py", type=str, default=os.path.join(REPO_DIR, "obelix.py"))
    parser.add_argument("--obelix_torch_py", type=str, default=os.path.join(REPO_DIR, "obelix_torch.py"))
    parser.add_argument("--out", type=str, default=os.path.join(HERE, "weights_nowall_gru_dagger.pth"))
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--total_env_steps", type=int, default=2_000_000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--gru_hidden_dim", type=int, default=128)
    parser.add_argument("--fw_bias_init", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta_final", type=float, default=0.2)
    parser.add_argument("--beta_decay_steps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env_device", type=str, default="cpu")
    parser.add_argument("--eval_runs", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=262144)
    parser.add_argument("--log_interval", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    feature_config = FeatureConfig(
        max_steps=int(args.max_steps),
        pose_clip=500.0,
        blind_clip=200.0,
        stuck_clip=20.0,
        contact_clip=20.0,
        same_obs_clip=100.0,
        wall_hit_clip=20.0,
        last_action_hist=5,
        heading_bins=8,
    )
    encoder_dims = tuple(int(h) for h in args.hidden_dims)
    model = GRUPolicy(
        input_dim=feature_config.feature_dim,
        encoder_dims=encoder_dims,
        gru_hidden_dim=int(args.gru_hidden_dim),
        fw_bias_init=float(args.fw_bias_init),
    ).to(device)
    if args.init_checkpoint:
        init_checkpoint = load_checkpoint(args.init_checkpoint, device=device)
        model.load_state_dict(init_checkpoint["state_dict"], strict=False)
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    VecEnvCls = import_symbol(args.obelix_torch_py, "OBELIXVectorized")
    vec_env = VecEnvCls(
        num_envs=int(args.num_envs),
        scaling_factor=int(args.scaling_factor),
        arena_size=int(args.arena_size),
        max_steps=int(args.max_steps),
        wall_obstacles=False,
        difficulty=int(args.difficulty),
        box_speed=int(args.box_speed),
        seed=int(args.seed * 10000),
        device=str(args.env_device),
    )

    env_kwargs = {
        "scaling_factor": int(args.scaling_factor),
        "arena_size": int(args.arena_size),
        "max_steps": int(args.max_steps),
        "wall_obstacles": False,
        "difficulty": int(args.difficulty),
        "box_speed": int(args.box_speed),
    }

    print(
        f"[setup] device={device} env_device={args.env_device} num_envs={args.num_envs} "
        f"feature_dim={feature_config.feature_dim} hidden_dims={encoder_dims} gru_hidden_dim={args.gru_hidden_dim}"
    )

    teacher = NoWallPrivilegedTeacher(args.num_envs)
    tracker = PoseGRUFeatureTracker(num_envs=args.num_envs, config=feature_config, device=device)
    obs = vec_env.reset_all(seed=args.seed * 10000)
    tracker.reset_all(obs)
    starts = torch.ones((args.num_envs,), dtype=torch.float32, device=device)
    hidden = model.initial_state(args.num_envs, device)
    episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    recent_returns = deque(maxlen=200)
    recent_successes = deque(maxlen=200)
    env_steps = 0
    best_eval = -float("inf")
    last_log = 0
    last_eval = 0
    start_time = time.time()

    while env_steps < int(args.total_env_steps):
        buffer = SequenceBuffer(args.rollout_steps, args.num_envs, feature_config.feature_dim, device)
        model.eval()
        for step in range(int(args.rollout_steps)):
            features = tracker.features()
            teacher_idx_np = teacher.act_batch(vec_env)
            teacher_actions = torch.as_tensor(teacher_idx_np, dtype=torch.long, device=device)

            with torch.no_grad():
                logits, _, next_hidden = model.forward_step(features, hidden, starts)
                student_actions = torch.argmax(logits, dim=1)

            if args.beta_decay_steps > 0:
                progress = min(1.0, float(env_steps) / float(max(1, args.beta_decay_steps)))
            else:
                progress = 1.0
            beta = (1.0 - progress) * float(args.beta) + progress * float(args.beta_final)
            if beta >= 1.0:
                exec_actions = teacher_actions
            elif beta <= 0.0:
                exec_actions = student_actions
            else:
                mix_mask = torch.rand((args.num_envs,), device=device) < float(beta)
                exec_actions = torch.where(mix_mask, teacher_actions, student_actions)

            buffer.add(step, features, starts, teacher_actions)

            action_idx_np = exec_actions.detach().cpu().numpy()
            next_obs, rewards, dones = vec_env.step(action_idx_np)
            episode_returns += rewards

            done_idx = np.flatnonzero(dones)
            if done_idx.size > 0:
                for idx in done_idx:
                    ret = float(episode_returns[idx])
                    recent_returns.append(ret)
                    recent_successes.append(int(ret >= 1000.0))
                reset_seed = int(args.seed * 10000 + env_steps + step * args.num_envs)
                reset_map = vec_env.reset(env_indices=done_idx.tolist(), seed=reset_seed)
                for idx, reset_obs in reset_map.items():
                    next_obs[idx] = reset_obs
                episode_returns[done_idx] = 0.0

            tracker.post_step(actions=exec_actions, next_obs=next_obs, dones=dones)
            hidden = next_hidden
            if done_idx.size > 0:
                hidden[:, done_idx] = 0.0
            starts = torch.as_tensor(dones.astype(np.float32, copy=False), dtype=torch.float32, device=device)
            env_steps += int(args.num_envs)

        model.train()
        mean_loss = 0.0
        num_batches = 0
        for feat_batch, starts_batch, labels_batch in buffer.sequence_batches(args.seq_len, args.batch_size, args.epochs):
            bs = feat_batch.shape[1]
            h0 = model.initial_state(bs, device)
            logits_seq = []
            hidden_seq = h0
            for t in range(args.seq_len):
                logits_t, _, hidden_seq = model.forward_step(feat_batch[t], hidden_seq, starts_batch[t])
                logits_seq.append(logits_t)
            logits = torch.stack(logits_seq, dim=0)
            loss = F.cross_entropy(logits.reshape(-1, ACTION_DIM), labels_batch.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            mean_loss += float(loss.item())
            num_batches += 1

        if env_steps - last_log >= int(args.log_interval):
            elapsed = max(1e-6, time.time() - start_time)
            sps = float(env_steps) / elapsed
            recent_mean = float(np.mean(recent_returns)) if recent_returns else float("nan")
            recent_success = float(np.mean(recent_successes)) if recent_successes else float("nan")
            print(
                f"[train] steps={env_steps} loss={mean_loss / max(1, num_batches):.4f} "
                f"recent_return={recent_mean:.1f} recent_success={recent_success:.2f} "
                f"sps={sps:.1f} beta={beta:.3f}"
            )
            last_log = env_steps

        if env_steps - last_eval >= int(args.eval_interval):
            model.eval()
            metrics = evaluate_model(
                model=model,
                feature_config=feature_config,
                obelix_py=args.obelix_py,
                runs=int(args.eval_runs),
                seed=0,
                env_kwargs=env_kwargs,
                device=device,
                stochastic=False,
            )
            print(
                f"[eval] steps={env_steps} mean={metrics['mean_reward']:.1f} "
                f"std={metrics['std_reward']:.1f} success={metrics['success_rate']:.2f}"
            )
            if metrics["mean_reward"] > best_eval:
                best_eval = metrics["mean_reward"]
                checkpoint = make_checkpoint(
                    model=model,
                    feature_config=feature_config,
                    encoder_dims=encoder_dims,
                    gru_hidden_dim=int(args.gru_hidden_dim),
                    args=args,
                    best_eval=best_eval,
                )
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                save_checkpoint(args.out, checkpoint)
                print(f"[eval] new best -> {args.out} ({best_eval:.1f})")
            last_eval = env_steps

    final_path = os.path.splitext(args.out)[0] + "_final.pth"
    checkpoint = make_checkpoint(
        model=model,
        feature_config=feature_config,
        encoder_dims=encoder_dims,
        gru_hidden_dim=int(args.gru_hidden_dim),
        args=args,
        best_eval=best_eval,
    )
    save_checkpoint(final_path, checkpoint)
    print(f"[done] wrote final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
