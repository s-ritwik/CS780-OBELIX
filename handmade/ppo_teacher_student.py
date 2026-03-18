"""Teacher-student PPO trainer guided by the handmade controller.

The student is a standard PPO policy trained on reward in the batched torch
environment from ``obelix_torch.py``. A decaying behavior-cloning loss from the
handmade controller acts only as a prior: it should help exploration early,
then get out of the way so PPO can improve beyond the heuristic.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(THIS_DIR)
PPO_DIR = os.path.join(REPO_DIR, "PPO")

if PPO_DIR not in sys.path:
    sys.path.insert(0, PPO_DIR)

import train_ppo as ppo_base


ACTIONS = list(ppo_base.ACTIONS)
ACTION_TO_IDX = {name: idx for idx, name in enumerate(ACTIONS)}


def import_handmade_module(teacher_path: str):
    spec = importlib.util.spec_from_file_location("handmade_teacher_module", teacher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import handmade teacher from {teacher_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class TeacherBatch:
    def __init__(self, num_envs: int, teacher_module):
        self.num_envs = int(num_envs)
        if not hasattr(teacher_module, "ACTIONS"):
            raise AttributeError("Teacher module must define ACTIONS")
        teacher_actions = list(getattr(teacher_module, "ACTIONS"))
        if teacher_actions != ACTIONS:
            raise RuntimeError(f"Teacher action order mismatch: teacher={teacher_actions} expected={ACTIONS}")

        if hasattr(teacher_module, "HandmadeController"):
            controller_cls = getattr(teacher_module, "HandmadeController")
        elif hasattr(teacher_module, "WallTeacher"):
            controller_cls = getattr(teacher_module, "WallTeacher")
        else:
            raise AttributeError("Teacher module must define HandmadeController or WallTeacher")

        self.controllers = [controller_cls() for _ in range(self.num_envs)]
        self.episode_counts = np.zeros((self.num_envs,), dtype=np.int64)
        self.reset_indices(list(range(self.num_envs)))

    def reset_indices(self, env_indices: list[int]) -> None:
        for idx in env_indices:
            self.episode_counts[idx] += 1
            episode_key = int((idx + 1) * 1_000_000_000 + self.episode_counts[idx])
            self.controllers[idx].reset(episode_key)

    def act(self, obs_batch: np.ndarray) -> np.ndarray:
        actions = np.zeros((self.num_envs,), dtype=np.int64)
        for idx in range(self.num_envs):
            action_name = self.controllers[idx].act(obs_batch[idx])
            actions[idx] = ACTION_TO_IDX[action_name]
        return actions


class RolloutBuffer(ppo_base.RolloutBuffer):
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        super().__init__(num_steps=num_steps, num_envs=num_envs, obs_dim=obs_dim, action_dim=action_dim, device=device)
        self.teacher_actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long, device=device)

    def add(
        self,
        step: int,
        obs: torch.Tensor,
        actions: torch.Tensor,
        teacher_actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        super().add(
            step=step,
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            dones=dones,
            values=values,
            logits=logits,
        )
        self.teacher_actions[step].copy_(teacher_actions)

    def mini_batch_generator(self, minibatch_size: int, num_learning_epochs: int):
        batch_size = self.num_steps * self.num_envs
        mb_size = min(int(minibatch_size), batch_size)

        obs = self.obs.reshape(batch_size, self.obs_dim)
        actions = self.actions.reshape(batch_size)
        teacher_actions = self.teacher_actions.reshape(batch_size)
        old_log_probs = self.log_probs.reshape(batch_size)
        old_values = self.values.reshape(batch_size)
        returns = self.returns.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        old_logits = self.logits.reshape(batch_size, self.action_dim)

        for _ in range(int(num_learning_epochs)):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mb_size):
                mb_idx = indices[start : start + mb_size]
                yield (
                    obs[mb_idx],
                    actions[mb_idx],
                    teacher_actions[mb_idx],
                    old_log_probs[mb_idx],
                    old_values[mb_idx],
                    returns[mb_idx],
                    advantages[mb_idx],
                    old_logits[mb_idx],
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teacher-student PPO trainer for OBELIX")
    default_obelix_torch = os.path.join(REPO_DIR, "obelix_torch.py")

    parser.add_argument("--obelix_py", type=str, default=default_obelix_torch)
    parser.add_argument("--out", type=str, default="weights_teacher_student.pth")
    parser.add_argument(
        "--teacher_agent_file",
        type=str,
        default=os.path.join(THIS_DIR, "agent.py"),
        help="Teacher controller file used for auxiliary behavior cloning.",
    )

    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--total_env_steps", type=int, default=2_000_000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)

    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--minibatch_size", type=int, default=4096)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--normalize_advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize_advantages_per_minibatch", action="store_true")
    parser.add_argument("--schedule", type=str, choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--desired_kl", type=float, default=0.01)
    parser.add_argument("--use_clipped_value_loss", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--teacher_coef", type=float, default=1.0)
    parser.add_argument("--teacher_coef_final", type=float, default=0.05)
    parser.add_argument(
        "--teacher_decay_steps",
        type=int,
        default=1_000_000,
        help="Linear decay horizon for the BC coefficient. Use <=0 to decay across the full run.",
    )

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--env_device", type=str, default="cpu")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--fw_bias_init", type=float, default=1.0)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--rec_enco", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100_000)
    parser.add_argument("--success_reward_threshold", type=float, default=1000.0)
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

    obelix_py = args.obelix_py
    if not os.path.exists(obelix_py):
        raise FileNotFoundError(f"Environment file not found: {obelix_py}")
    if not os.path.exists(args.teacher_agent_file):
        raise FileNotFoundError(f"Teacher file not found: {args.teacher_agent_file}")

    print(f"[setup] device={device} trainer=teacher_student num_envs={args.num_envs}")
    print(f"[setup] obelix_py={obelix_py}")
    print(f"[setup] env_device={args.env_device}")
    print(f"[setup] teacher_agent={args.teacher_agent_file}")

    hidden_dims = tuple(int(h) for h in args.hidden_dims)
    model = ppo_base.ActorCritic(
        hidden_dims=hidden_dims,
        fw_bias_init=args.fw_bias_init,
        use_rec_encoder=args.rec_enco,
    ).to(device)
    if args.rec_enco:
        print(f"[setup] recommended encoder enabled (input_dim={ppo_base.ENCODED_OBS_DIM})")
    if args.init_checkpoint is not None:
        state_dict, metadata = ppo_base.load_checkpoint_state(args.init_checkpoint)
        ckpt_hidden_dims = metadata.get("hidden_dims")
        if ckpt_hidden_dims is not None and tuple(int(h) for h in ckpt_hidden_dims) != hidden_dims:
            raise ValueError(
                "Checkpoint hidden_dims do not match requested model shape: "
                f"checkpoint={tuple(int(h) for h in ckpt_hidden_dims)} requested={hidden_dims}"
            )
        ckpt_rec_enco = metadata.get("use_rec_encoder")
        if ckpt_rec_enco is None:
            ckpt_rec_enco = ppo_base.infer_use_rec_encoder_from_state_dict(state_dict)
        if bool(ckpt_rec_enco) != bool(args.rec_enco):
            raise ValueError(
                "Checkpoint encoder setting does not match this run: "
                f"checkpoint_use_rec_encoder={bool(ckpt_rec_enco)} requested={bool(args.rec_enco)}"
            )
        model.load_state_dict(state_dict, strict=True)
        print(f"[setup] warm-started model weights from {args.init_checkpoint}")
    if args.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    def online_state_dict():
        if hasattr(model, "_orig_mod"):
            return model._orig_mod.state_dict()
        return model.state_dict()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    current_lr = float(args.lr)

    VecEnvCls = ppo_base.import_symbol(obelix_py, "OBELIXVectorized")
    env_kwargs = dict(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        device=str(device) if args.env_device == "auto" else args.env_device,
    )
    vec_env = VecEnvCls(
        num_envs=args.num_envs,
        seed=args.seed * 10_000,
        **env_kwargs,
    )
    teacher_module = import_handmade_module(args.teacher_agent_file)
    teacher = TeacherBatch(args.num_envs, teacher_module)

    obs = vec_env.reset_all(seed=args.seed * 10_000)
    teacher.reset_indices(list(range(args.num_envs)))

    episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.num_envs,), dtype=np.int32)
    recent_returns = deque(maxlen=200)
    recent_lengths = deque(maxlen=200)
    recent_successes = deque(maxlen=200)
    recent_timeouts = deque(maxlen=200)
    recent_terminal_rewards = deque(maxlen=200)

    env_steps = 0
    update_idx = 0
    last_log_env_step = 0
    start_time = time.time()
    total_completed_eps = 0
    total_successes = 0
    total_timeouts = 0

    try:
        while env_steps < args.total_env_steps:
            buffer = RolloutBuffer(
                num_steps=args.rollout_steps,
                num_envs=args.num_envs,
                obs_dim=ppo_base.RAW_OBS_DIM,
                action_dim=len(ACTIONS),
                device=device,
            )

            model.eval()
            rollout_student_counts = np.zeros((len(ACTIONS),), dtype=np.int64)
            rollout_teacher_counts = np.zeros((len(ACTIONS),), dtype=np.int64)

            for step in range(args.rollout_steps):
                teacher_action_idx = teacher.act(obs)
                rollout_teacher_counts += np.bincount(teacher_action_idx, minlength=len(ACTIONS))

                obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    actions_t, log_probs_t, _, values_t, logits_t = model.act(obs_t)

                action_idx = actions_t.cpu().numpy()
                rollout_student_counts += np.bincount(action_idx, minlength=len(ACTIONS))

                next_obs, rewards, dones = vec_env.step(action_idx)
                teacher_actions_t = torch.as_tensor(teacher_action_idx, dtype=torch.long, device=device)

                buffer.add(
                    step=step,
                    obs=obs_t,
                    actions=actions_t,
                    teacher_actions=teacher_actions_t,
                    log_probs=log_probs_t,
                    rewards=rewards.astype(np.float32, copy=False),
                    dones=dones.astype(np.float32, copy=False),
                    values=values_t,
                    logits=logits_t,
                )

                episode_returns += rewards
                episode_lengths += 1

                done_idx = np.nonzero(dones)[0]
                if done_idx.size > 0:
                    for idx in done_idx:
                        recent_returns.append(float(episode_returns[idx]))
                        recent_lengths.append(int(episode_lengths[idx]))
                        terminal_reward = float(rewards[idx])
                        timeout_flag = int(episode_lengths[idx] >= args.max_steps)
                        success_flag = int(terminal_reward >= args.success_reward_threshold)
                        recent_successes.append(success_flag)
                        recent_timeouts.append(timeout_flag)
                        recent_terminal_rewards.append(terminal_reward)
                        total_completed_eps += 1
                        total_successes += success_flag
                        total_timeouts += timeout_flag
                    episode_returns[done_idx] = 0.0
                    episode_lengths[done_idx] = 0

                    reset_map = vec_env.reset(
                        env_indices=done_idx.tolist(),
                        seed=args.seed * 10_000 + env_steps + step * args.num_envs,
                    )
                    teacher.reset_indices(done_idx.tolist())
                    for idx, reset_obs in reset_map.items():
                        next_obs[idx] = reset_obs

                obs = next_obs
                env_steps += args.num_envs

            with torch.no_grad():
                next_obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
                _, last_values = model(next_obs_t)

            buffer.compute_returns(
                last_values=last_values,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                normalize_advantages=args.normalize_advantages,
                normalize_advantages_per_minibatch=args.normalize_advantages_per_minibatch,
            )

            decay_steps = args.teacher_decay_steps if args.teacher_decay_steps > 0 else args.total_env_steps
            teacher_progress = min(1.0, float(env_steps) / float(max(1, decay_steps)))
            teacher_coef = args.teacher_coef + (args.teacher_coef_final - args.teacher_coef) * teacher_progress

            model.train()
            mean_policy_loss = 0.0
            mean_value_loss = 0.0
            mean_entropy = 0.0
            mean_kl = 0.0
            mean_bc_loss = 0.0
            mean_bc_acc = 0.0
            num_minibatches = 0

            for (
                obs_batch,
                actions_batch,
                teacher_actions_batch,
                old_log_probs_batch,
                old_values_batch,
                returns_batch,
                advantages_batch,
                old_logits_batch,
            ) in buffer.mini_batch_generator(args.minibatch_size, args.update_epochs):
                if args.normalize_advantages_per_minibatch:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std(unbiased=False) + 1e-8
                    )

                new_log_probs, entropy, values, new_logits = model.evaluate_actions(obs_batch, actions_batch)

                if args.schedule == "adaptive" and args.desired_kl > 0.0:
                    with torch.no_grad():
                        kl_mean = float(ppo_base.categorical_kl(old_logits_batch, new_logits).mean().item())
                        if kl_mean > args.desired_kl * 2.0:
                            current_lr = max(1e-5, current_lr / 1.5)
                        elif 0.0 < kl_mean < args.desired_kl / 2.0:
                            current_lr = min(1e-2, current_lr * 1.5)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = current_lr
                else:
                    with torch.no_grad():
                        kl_mean = float(ppo_base.categorical_kl(old_logits_batch, new_logits).mean().item())

                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surrogate = -advantages_batch * ratio
                surrogate_clipped = -advantages_batch * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                policy_loss = torch.max(surrogate, surrogate_clipped).mean()

                if args.use_clipped_value_loss:
                    value_clipped = old_values_batch + (values - old_values_batch).clamp(
                        -args.clip_coef, args.clip_coef
                    )
                    value_losses = (values - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - values).pow(2).mean()

                entropy_loss = entropy.mean()
                bc_loss = F.cross_entropy(new_logits, teacher_actions_batch)
                bc_acc = (torch.argmax(new_logits, dim=-1) == teacher_actions_batch).to(torch.float32).mean()

                loss = (
                    policy_loss
                    + args.vf_coef * value_loss
                    - args.ent_coef * entropy_loss
                    + teacher_coef * bc_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                mean_policy_loss += float(policy_loss.item())
                mean_value_loss += float(value_loss.item())
                mean_entropy += float(entropy_loss.item())
                mean_kl += kl_mean
                mean_bc_loss += float(bc_loss.item())
                mean_bc_acc += float(bc_acc.item())
                num_minibatches += 1

            update_idx += 1
            if num_minibatches > 0:
                mean_policy_loss /= num_minibatches
                mean_value_loss /= num_minibatches
                mean_entropy /= num_minibatches
                mean_kl /= num_minibatches
                mean_bc_loss /= num_minibatches
                mean_bc_acc /= num_minibatches

            if env_steps - last_log_env_step >= args.log_interval:
                elapsed = max(1e-6, time.time() - start_time)
                sps = env_steps / elapsed
                mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
                mean_len = float(np.mean(recent_lengths)) if recent_lengths else float("nan")
                recent_success_rate = float(np.mean(recent_successes)) if recent_successes else float("nan")
                recent_timeout_rate = float(np.mean(recent_timeouts)) if recent_timeouts else float("nan")
                mean_terminal_reward = (
                    float(np.mean(recent_terminal_rewards)) if recent_terminal_rewards else float("nan")
                )
                total_success_rate = (
                    float(total_successes) / float(total_completed_eps) if total_completed_eps > 0 else float("nan")
                )
                total_timeout_rate = (
                    float(total_timeouts) / float(total_completed_eps) if total_completed_eps > 0 else float("nan")
                )
                rollout_student_total = max(1, int(np.sum(rollout_student_counts)))
                rollout_teacher_total = max(1, int(np.sum(rollout_teacher_counts)))
                student_mix = " ".join(
                    f"{name}:{(count / rollout_student_total):.2f}"
                    for name, count in zip(ACTIONS, rollout_student_counts.tolist())
                )
                teacher_mix = " ".join(
                    f"{name}:{(count / rollout_teacher_total):.2f}"
                    for name, count in zip(ACTIONS, rollout_teacher_counts.tolist())
                )
                print(
                    f"[train] update={update_idx} env_steps={env_steps} "
                    f"policy_loss={mean_policy_loss:.4f} value_loss={mean_value_loss:.4f} "
                    f"bc_loss={mean_bc_loss:.4f} bc_acc={mean_bc_acc:.3f} bc_coef={teacher_coef:.4f} "
                    f"entropy={mean_entropy:.4f} kl={mean_kl:.5f} lr={current_lr:.6f} "
                    f"sps={sps:.1f} recent_return={mean_ret:.1f} recent_len={mean_len:.1f} "
                    f"recent_success={recent_success_rate:.3f} recent_timeout={recent_timeout_rate:.3f} "
                    f"total_success={total_success_rate:.3f} total_timeout={total_timeout_rate:.3f} "
                    f"terminal_reward={mean_terminal_reward:.1f} completed_eps={total_completed_eps} "
                    f"student=[{student_mix}] teacher=[{teacher_mix}] "
                    f"elapsed={ppo_base.format_hms(elapsed)}"
                )
                last_log_env_step = env_steps

    finally:
        vec_env.close()

    total_elapsed = max(0.0, time.time() - start_time)
    checkpoint = {
        "state_dict": online_state_dict(),
        "obs_dim": ppo_base.RAW_OBS_DIM,
        "model_input_dim": ppo_base.ENCODED_OBS_DIM if args.rec_enco else ppo_base.RAW_OBS_DIM,
        "action_dim": len(ACTIONS),
        "hidden_dims": [int(h) for h in hidden_dims],
        "actions": ACTIONS,
        "use_rec_encoder": bool(args.rec_enco),
        "teacher_agent": args.teacher_agent_file,
        "teacher_coef": float(args.teacher_coef),
        "teacher_coef_final": float(args.teacher_coef_final),
        "config": vars(args),
    }
    torch.save(checkpoint, args.out)
    print(f"[done] total_train_time={ppo_base.format_hms(total_elapsed)} ({total_elapsed / 60.0:.2f} min)")
    print(f"[done] saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
