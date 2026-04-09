from __future__ import annotations

import argparse
import importlib.util
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from privileged import extract_privileged_obs, extract_shaping_metrics, privileged_obs_dim
from recurrent_v2 import (
    ACTIONS,
    ACTION_DIM,
    PoseMemoryTracker,
    RecurrentAsymmetricActorCritic,
    RecurrentFeatureConfig,
    RecurrentRolloutBuffer,
    RecurrentState,
    categorical_kl,
    format_hms,
    load_checkpoint,
    make_checkpoint,
    save_checkpoint,
)
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


def import_module(py_file: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TeacherPool:
    def __init__(self, teacher_path: str, num_envs: int, seed: int) -> None:
        teacher_mod = import_module(teacher_path, f"mixed_teacher_{abs(hash(teacher_path))}")
        if not hasattr(teacher_mod, "policy"):
            raise AttributeError(f"Teacher module {teacher_path} does not define policy(obs, rng)")
        self.policy = getattr(teacher_mod, "policy")
        self.rngs = [np.random.default_rng(int(seed + 1_000_000 + i)) for i in range(int(num_envs))]

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        actions = np.zeros((obs.shape[0],), dtype=np.int64)
        for idx in range(obs.shape[0]):
            actions[idx] = ACTIONS.index(self.policy(obs[idx], self.rngs[idx]))
        return actions

    def reset_indices(self, env_indices: list[int], base_seed: int) -> None:
        for offset, env_id in enumerate(env_indices):
            self.rngs[int(env_id)] = np.random.default_rng(int(base_seed + offset))


def resolve_teacher_for_scenario(expert_key: str, spec: "ScenarioSpec", repo_dir: str) -> str:
    if expert_key != "mixed_teacher_map_v1":
        return expert_key
    if spec.difficulty in {0, 2}:
        return os.path.join(repo_dir, "ppo_lab", "expert_conservative.py")
    if spec.difficulty == 3 and (not spec.wall_obstacles):
        return os.path.join(repo_dir, "ppo_lab", "agent_nowall_submission.py")
    return os.path.join(repo_dir, "ppo_lab", "agent_seenmask_submission.py")


def collect_expert_sequences_mixed(
    *,
    expert_path: str,
    episodes: int,
    max_attempts: int,
    min_return: float | None,
    success_dup_factor: int,
    seed: int,
    obelix_py: str,
    repo_dir: str,
    scenarios: list["ScenarioSpec"],
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    box_speed: int,
    feature_config: RecurrentFeatureConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    if episodes <= 0:
        return [], [], {"accepted": 0.0, "attempts": 0.0, "mean_return": float("nan")}

    obelix_cls = import_symbol(obelix_py, "OBELIX")
    tracker = PoseMemoryTracker(num_envs=1, config=feature_config, device=torch.device("cpu"))
    sequence_features: list[np.ndarray] = []
    sequence_actions: list[np.ndarray] = []
    returns = deque(maxlen=50)
    policy_cache: dict[str, object] = {}
    progress_interval = max(1, max_attempts // 4)

    attempts = 0
    accepted = 0
    print(f"[warm_start] collecting up to {episodes} expert episode(s) from {expert_path}", flush=True)
    while accepted < int(episodes) and attempts < int(max_attempts):
        spec = scenarios[attempts % len(scenarios)]
        resolved_expert = resolve_teacher_for_scenario(expert_path, spec, repo_dir)
        use_privileged_teacher = resolved_expert == "privileged_teacher"
        expert_policy = None
        if not use_privileged_teacher:
            if resolved_expert not in policy_cache:
                expert_mod = import_module(resolved_expert, f"mixed_warm_start_{abs(hash(resolved_expert))}")
                if not hasattr(expert_mod, "policy"):
                    raise AttributeError(f"Expert module {resolved_expert} does not define policy(obs, rng)")
                policy_cache[resolved_expert] = getattr(expert_mod, "policy")
            expert_policy = policy_cache[resolved_expert]
        episode_seed = int(seed + 700_000 + attempts)
        env = obelix_cls(
            scaling_factor=scaling_factor,
            arena_size=arena_size,
            max_steps=max_steps,
            wall_obstacles=spec.wall_obstacles,
            difficulty=spec.difficulty,
            box_speed=box_speed,
            seed=episode_seed,
        )
        obs = env.reset(seed=episode_seed)
        tracker.reset_all(np.asarray(obs, dtype=np.float32)[None, :])

        rng = np.random.default_rng(episode_seed)
        teacher_state = ScriptedTeacherState()
        total_reward = 0.0
        done = False
        episode_feats: list[np.ndarray] = []
        episode_actions: list[int] = []
        while not done:
            episode_feats.append(tracker.features().cpu().numpy()[0].astype(np.float32, copy=True))
            if use_privileged_teacher:
                action_name = scripted_teacher_action(env, teacher_state)
            else:
                action_name = expert_policy(obs, rng)
            action_idx = ACTIONS.index(action_name)
            episode_actions.append(action_idx)
            obs, reward, done = env.step(action_name, render=False)
            total_reward += float(reward)
            if not done:
                tracker.post_step(
                    actions=torch.tensor([action_idx], dtype=torch.long),
                    next_obs=np.asarray(obs, dtype=np.float32)[None, :],
                    dones=np.asarray([False]),
                )

        returns.append(total_reward)
        accepted_episode = min_return is None or total_reward >= float(min_return)
        success = total_reward >= 1000.0
        if accepted_episode and episode_feats and episode_actions:
            seq_x = np.asarray(episode_feats, dtype=np.float32)
            seq_y = np.asarray(episode_actions, dtype=np.int64)
            sequence_features.append(seq_x)
            sequence_actions.append(seq_y)
            accepted += 1
            if success and int(success_dup_factor) > 1:
                for _ in range(int(success_dup_factor) - 1):
                    sequence_features.append(seq_x.copy())
                    sequence_actions.append(seq_y.copy())
        attempts += 1
        if attempts % progress_interval == 0 or accepted == episodes or attempts == max_attempts:
            print(
                f"[warm_start] attempts={attempts}/{max_attempts} accepted={accepted}/{episodes} "
                f"recent_return={float(np.mean(returns)):.1f}",
                flush=True,
            )

    total_steps = int(sum(seq.shape[0] for seq in sequence_actions))
    print(f"[warm_start] collected {len(sequence_actions)} sequence(s), {total_steps} state-action pairs", flush=True)
    return sequence_features, sequence_actions, {
        "accepted": float(accepted),
        "attempts": float(attempts),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
    }


def flatten_demo_sequences(
    sequences_x: list[np.ndarray],
    sequences_y: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not sequences_x or not sequences_y:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    feats = np.concatenate(sequences_x, axis=0).astype(np.float32, copy=False)
    acts = np.concatenate(sequences_y, axis=0).astype(np.int64, copy=False)
    return feats, acts


def warm_start_policy(
    model: RecurrentAsymmetricActorCritic,
    optimizer: optim.Optimizer,
    feats: np.ndarray,
    acts: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    grad_clip: float,
) -> None:
    if feats.size == 0 or acts.size == 0 or epochs <= 0:
        return

    x = torch.as_tensor(feats, dtype=torch.float32, device=device)
    y = torch.as_tensor(acts, dtype=torch.long, device=device)
    total = int(x.shape[0])
    batch = min(int(batch_size), total)

    print(f"[warm_start] behavior cloning on {total} samples", flush=True)
    for epoch in range(int(epochs)):
        perm = torch.randperm(total, device=device)
        total_loss = 0.0
        total_correct = 0
        seen = 0

        for start in range(0, total, batch):
            idx = perm[start : start + batch]
            state0 = model.initial_state(idx.numel(), device)
            starts = torch.ones((idx.numel(),), dtype=torch.float32, device=device)
            logits, _ = model.actor_step(x[idx], state0, starts)
            loss = F.cross_entropy(logits, y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += float(loss.item()) * idx.numel()
            total_correct += int((torch.argmax(logits, dim=1) == y[idx]).sum().item())
            seen += int(idx.numel())

        print(
            f"[warm_start] epoch={epoch + 1}/{epochs} "
            f"loss={total_loss / max(1, seen):.4f} acc={total_correct / max(1, seen):.3f}",
            flush=True,
        )


def masked_sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, ACTION_DIM)
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    losses = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    losses = losses * flat_mask.to(dtype=losses.dtype)
    return losses.sum() / flat_mask.to(dtype=losses.dtype).sum().clamp_min(1.0)


def warm_start_sequence_policy(
    model: RecurrentAsymmetricActorCritic,
    optimizer: optim.Optimizer,
    sequences_x: list[np.ndarray],
    sequences_y: list[np.ndarray],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    grad_clip: float,
) -> None:
    if not sequences_x or not sequences_y or epochs <= 0:
        return

    total_sequences = len(sequences_x)
    batch = min(int(batch_size), total_sequences)
    print(f"[warm_start] recurrent BC on {total_sequences} sequence(s)", flush=True)
    for epoch in range(int(epochs)):
        order = np.random.permutation(total_sequences)
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0.0

        for start in range(0, total_sequences, batch):
            idxs = order[start : start + batch]
            max_len = max(int(sequences_y[int(idx)].shape[0]) for idx in idxs)
            cur_batch = len(idxs)
            feat_dim = int(sequences_x[0].shape[1])

            x_batch = np.zeros((max_len, cur_batch, feat_dim), dtype=np.float32)
            y_batch = np.zeros((max_len, cur_batch), dtype=np.int64)
            mask_batch = np.zeros((max_len, cur_batch), dtype=np.float32)
            starts_batch = np.zeros((max_len, cur_batch), dtype=np.float32)

            for b, seq_idx in enumerate(idxs):
                seq_x = sequences_x[int(seq_idx)]
                seq_y = sequences_y[int(seq_idx)]
                seq_len = int(seq_y.shape[0])
                x_batch[:seq_len, b] = seq_x
                y_batch[:seq_len, b] = seq_y
                mask_batch[:seq_len, b] = 1.0
                starts_batch[0, b] = 1.0

            x_t = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
            y_t = torch.as_tensor(y_batch, dtype=torch.long, device=device)
            mask_t = torch.as_tensor(mask_batch, dtype=torch.float32, device=device)
            starts_t = torch.as_tensor(starts_batch, dtype=torch.float32, device=device)
            hidden = model.initial_state(cur_batch, device)

            logits_list = []
            for t in range(max_len):
                logits_t, hidden = model.actor_step(x_t[t], hidden, starts_t[t])
                logits_list.append(logits_t)
            logits = torch.stack(logits_list, dim=0)
            loss = masked_sequence_cross_entropy(logits, y_t, mask_t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                correct = ((pred == y_t).to(torch.float32) * mask_t).sum().item()
                count = mask_t.sum().item()
            total_loss += float(loss.item()) * cur_batch
            total_correct += float(correct)
            total_count += float(count)

        print(
            f"[warm_start] epoch={epoch + 1}/{epochs} "
            f"loss={total_loss / max(1, total_sequences):.4f} acc={total_correct / max(1.0, total_count):.3f}",
            flush=True,
        )


@dataclass(frozen=True)
class ScenarioSpec:
    difficulty: int
    wall_obstacles: bool

    @property
    def tag(self) -> str:
        wall = "wall" if self.wall_obstacles else "nowall"
        if self.difficulty == 0:
            diff = "static"
        elif self.difficulty == 2:
            diff = "blink"
        elif self.difficulty == 3:
            diff = "move"
        else:
            diff = f"d{self.difficulty}"
        return f"{diff}_{wall}"


@dataclass
class ScenarioRuntime:
    spec: ScenarioSpec
    vec_env: object
    tracker: PoseMemoryTracker
    obs: np.ndarray
    state: RecurrentState
    starts: torch.Tensor
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    privileged_teacher: PrivilegedTeacher | None
    teacher_pool: TeacherPool | None


class RecurrentRunner:
    def __init__(
        self,
        model: RecurrentAsymmetricActorCritic,
        feature_config: RecurrentFeatureConfig,
        device: torch.device,
        stochastic: bool,
    ) -> None:
        self.model = model
        self.device = device
        self.stochastic = bool(stochastic)
        self.tracker = PoseMemoryTracker(num_envs=1, config=feature_config, device=device)
        self.state = self.model.initial_state(1, device)
        self.pending_action: int | None = None
        self.started = False

    def reset(self, obs: np.ndarray) -> None:
        self.tracker.reset_all(np.asarray(obs, dtype=np.float32)[None, :])
        self.state = self.model.initial_state(1, self.device)
        self.pending_action = None
        self.started = True

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if not self.started:
            self.reset(obs_arr)
            starts = torch.ones((1,), dtype=torch.float32, device=self.device)
        else:
            if self.pending_action is not None:
                self.tracker.post_step(
                    actions=torch.tensor([self.pending_action], dtype=torch.long, device=self.device),
                    next_obs=obs_arr[None, :],
                    dones=np.asarray([False]),
                )
            starts = torch.zeros((1,), dtype=torch.float32, device=self.device)

        features = self.tracker.features()
        logits, self.state = self.model.actor_step(features, self.state, starts)
        if self.stochastic:
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64, copy=False)
            probs /= probs.sum()
            action_idx = int(rng.choice(len(ACTIONS), p=probs))
        else:
            action_idx = int(torch.argmax(logits, dim=1).item())
        self.pending_action = action_idx
        return action_idx


def parse_scenarios(spec: str) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid scenario token: {token}")
        difficulty = int(parts[0])
        wall_token = parts[1].lower()
        if wall_token not in {"w", "wall", "nw", "nowall"}:
            raise ValueError(f"Invalid wall flag in scenario token: {token}")
        scenarios.append(ScenarioSpec(difficulty=difficulty, wall_obstacles=wall_token in {"w", "wall"}))
    if not scenarios:
        raise ValueError("No scenarios parsed")
    return scenarios


COMPACT_AUX_INDICES = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 15, 18, 19, 24, 26, 28, 29, 30, 31, 32, 33,
)


def auxiliary_target_dim(mode: str, privileged_dim: int) -> int:
    if mode == "none":
        return 0
    if mode == "full" or privileged_dim < max(COMPACT_AUX_INDICES) + 1:
        return int(privileged_dim)
    return len(COMPACT_AUX_INDICES)


def build_auxiliary_targets(privileged_obs: torch.Tensor, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "full" or privileged_obs.shape[-1] < max(COMPACT_AUX_INDICES) + 1:
        return privileged_obs
    return privileged_obs[..., list(COMPACT_AUX_INDICES)]


def shape_rewards(
    *,
    args: argparse.Namespace,
    env_rewards: np.ndarray,
    next_obs: np.ndarray,
    actions: torch.Tensor,
    prev_metrics: dict[str, torch.Tensor],
    next_metrics: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    rewards_t = torch.as_tensor(env_rewards, dtype=torch.float32, device=device)

    if args.visible_bonus != 0.0:
        rewards_t = rewards_t + float(args.visible_bonus) * next_metrics["box_visible"]

    if args.ir_bonus != 0.0:
        ir_seen = torch.as_tensor(next_obs[:, 16] > 0.5, dtype=torch.float32, device=device)
        rewards_t = rewards_t + float(args.ir_bonus) * ir_seen

    if args.blind_turn_penalty != 0.0:
        blind = torch.as_tensor(np.sum(next_obs[:, :16], axis=1) == 0.0, dtype=torch.bool, device=device)
        turn_mask = actions != ACTIONS.index("FW")
        rewards_t = rewards_t - float(args.blind_turn_penalty) * (blind & turn_mask).to(torch.float32)

    if args.blind_forward_penalty != 0.0:
        blind = torch.as_tensor(np.sum(next_obs[:, :16], axis=1) == 0.0, dtype=torch.bool, device=device)
        fw_mask = actions == ACTIONS.index("FW")
        rewards_t = rewards_t - float(args.blind_forward_penalty) * (blind & fw_mask).to(torch.float32)

    if args.stuck_extra_penalty != 0.0:
        rewards_t = rewards_t - float(args.stuck_extra_penalty) * next_metrics["stuck"]

    if args.approach_progress_bonus != 0.0:
        nonpush = 1.0 - prev_metrics["push_active"]
        progress = torch.clamp(
            prev_metrics["bot_box_distance"] - next_metrics["bot_box_distance"],
            min=-0.05,
            max=0.05,
        )
        rewards_t = rewards_t + float(args.approach_progress_bonus) * progress * nonpush

    if args.alignment_bonus != 0.0:
        nonpush = 1.0 - prev_metrics["push_active"]
        align_gain = torch.clamp(
            next_metrics["heading_alignment"] - prev_metrics["heading_alignment"],
            min=-0.5,
            max=0.5,
        )
        rewards_t = rewards_t + float(args.alignment_bonus) * align_gain * nonpush

    if args.push_progress_bonus != 0.0:
        push_mask = torch.maximum(prev_metrics["push_active"], next_metrics["push_active"])
        goal_progress = torch.clamp(
            prev_metrics["goal_distance"] - next_metrics["goal_distance"],
            min=-0.05,
            max=0.05,
        )
        rewards_t = rewards_t + float(args.push_progress_bonus) * goal_progress * push_mask

    if args.gap_progress_bonus != 0.0:
        route_mask = (1.0 - prev_metrics["push_active"]) * prev_metrics["opposite_wall_side"]
        gap_progress = torch.clamp(
            prev_metrics["gap_target_distance"] - next_metrics["gap_target_distance"],
            min=-0.05,
            max=0.05,
        )
        rewards_t = rewards_t + float(args.gap_progress_bonus) * gap_progress * route_mask

    if args.gap_alignment_bonus != 0.0:
        route_mask = (1.0 - prev_metrics["push_active"]) * prev_metrics["opposite_wall_side"]
        gap_align = torch.clamp(
            next_metrics["gap_alignment"] - prev_metrics["gap_alignment"],
            min=-0.5,
            max=0.5,
        )
        rewards_t = rewards_t + float(args.gap_alignment_bonus) * gap_align * route_mask

    if args.reward_scale != 1.0:
        rewards_t = rewards_t / float(args.reward_scale)
    if args.reward_clip > 0.0:
        rewards_t = torch.clamp(rewards_t, min=-float(args.reward_clip), max=float(args.reward_clip))
    return rewards_t.detach().cpu().numpy().astype(np.float32, copy=False)


@torch.no_grad()
def evaluate_suite(
    model: RecurrentAsymmetricActorCritic,
    *,
    feature_config: RecurrentFeatureConfig,
    obelix_py: str,
    scenarios: list[ScenarioSpec],
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    box_speed: int,
    runs: int,
    seed: int,
    device: torch.device,
    stochastic: bool,
) -> dict[str, float]:
    obelix_cls = import_symbol(obelix_py, "OBELIX")
    results: dict[str, float] = {}
    overall_scores: list[float] = []

    model.eval()
    for spec in scenarios:
        runner = RecurrentRunner(model=model, feature_config=feature_config, device=device, stochastic=stochastic)
        scores: list[float] = []
        for run_idx in range(int(runs)):
            episode_seed = int(seed + run_idx)
            env = obelix_cls(
                scaling_factor=scaling_factor,
                arena_size=arena_size,
                max_steps=max_steps,
                wall_obstacles=spec.wall_obstacles,
                difficulty=spec.difficulty,
                box_speed=box_speed,
                seed=episode_seed,
            )
            obs = env.reset(seed=episode_seed)
            runner.reset(obs)

            total_reward = 0.0
            done = False
            rng = np.random.default_rng(episode_seed)
            while not done:
                action_idx = runner.act(obs, rng)
                obs, reward, done = env.step(ACTIONS[action_idx], render=False)
                total_reward += float(reward)

            scores.append(total_reward)
            overall_scores.append(total_reward)

        results[f"{spec.tag}_mean"] = float(np.mean(scores))
        results[f"{spec.tag}_std"] = float(np.std(scores))

    results["mean_reward"] = float(np.mean(overall_scores))
    results["std_reward"] = float(np.std(overall_scores))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mixed-scenario recurrent asymmetric PPO trainer for OBELIX")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_dir)

    parser.add_argument("--obelix_py", type=str, default=os.path.join(repo_dir, "obelix.py"))
    parser.add_argument("--obelix_torch_py", type=str, default=os.path.join(repo_dir, "obelix_torch.py"))
    parser.add_argument("--out", type=str, default=os.path.join(base_dir, "mixed_recurrent_v2_best.pth"))
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--reset_best_eval", action="store_true")

    parser.add_argument("--scenarios", type=str, default="0:nw,0:w,2:nw,2:w,3:nw,3:w")
    parser.add_argument("--num_envs", type=int, default=192)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--total_env_steps", type=int, default=4_000_000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--env_device", type=str, default="auto")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch_compile", action="store_true")

    parser.add_argument("--encoder_dims", type=int, nargs="+", default=[384, 256, 128])
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[1024, 512, 256])
    parser.add_argument("--rnn_hidden_dim", type=int, default=192)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--rnn_dropout", type=float, default=0.0)
    parser.add_argument("--actor_dropout", type=float, default=0.0)
    parser.add_argument("--critic_dropout", type=float, default=0.0)
    parser.add_argument("--feature_dropout", type=float, default=0.0)
    parser.add_argument("--aux_coef", type=float, default=0.0)
    parser.add_argument("--aux_hidden_dim", type=int, default=128)
    parser.add_argument("--aux_target_mode", type=str, choices=["none", "compact", "full"], default="compact")
    parser.add_argument("--rnn_type", type=str, choices=["gru", "lstm"], default="gru")
    parser.add_argument("--fw_bias_init", type=float, default=1.0)
    parser.add_argument("--policy_temperature", type=float, default=1.0)

    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.004)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--minibatch_size", type=int, default=8192)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--normalize_advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward_scale", type=float, default=5.0)
    parser.add_argument("--reward_clip", type=float, default=100.0)
    parser.add_argument("--visible_bonus", type=float, default=0.75)
    parser.add_argument("--ir_bonus", type=float, default=1.0)
    parser.add_argument("--blind_turn_penalty", type=float, default=0.05)
    parser.add_argument("--blind_forward_penalty", type=float, default=0.0)
    parser.add_argument("--stuck_extra_penalty", type=float, default=1.0)
    parser.add_argument("--approach_progress_bonus", type=float, default=30.0)
    parser.add_argument("--alignment_bonus", type=float, default=1.0)
    parser.add_argument("--push_progress_bonus", type=float, default=140.0)
    parser.add_argument("--gap_progress_bonus", type=float, default=100.0)
    parser.add_argument("--gap_alignment_bonus", type=float, default=1.25)
    parser.add_argument("--schedule", type=str, choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--desired_kl", type=float, default=0.01)
    parser.add_argument("--use_clipped_value_loss", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--pose_clip", type=float, default=500.0)
    parser.add_argument("--blind_clip", type=float, default=120.0)
    parser.add_argument("--stuck_clip", type=float, default=24.0)
    parser.add_argument("--contact_clip", type=float, default=24.0)
    parser.add_argument("--same_obs_clip", type=float, default=64.0)
    parser.add_argument("--wall_hit_clip", type=float, default=24.0)
    parser.add_argument("--blind_turn_clip", type=float, default=24.0)
    parser.add_argument("--stuck_memory_clip", type=float, default=24.0)
    parser.add_argument("--turn_streak_clip", type=float, default=24.0)
    parser.add_argument("--forward_streak_clip", type=float, default=24.0)
    parser.add_argument("--last_action_hist", type=int, default=6)
    parser.add_argument("--heading_bins", type=int, default=8)
    parser.add_argument("--use_current_obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_delta_obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_derived_obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_pose_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_counter_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_action_history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_same_obs_feature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_wall_hit_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_last_seen_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_sensor_seen_mask", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--teacher_agent_file", type=str, default="privileged_teacher")
    parser.add_argument("--teacher_coef", type=float, default=0.15)
    parser.add_argument("--teacher_coef_final", type=float, default=0.02)
    parser.add_argument("--teacher_decay_steps", type=int, default=2_000_000)
    parser.add_argument("--teacher_action_prob", type=float, default=0.35)
    parser.add_argument("--teacher_action_prob_final", type=float, default=0.05)
    parser.add_argument("--teacher_action_prob_decay_steps", type=int, default=1_500_000)
    parser.add_argument("--warm_start_expert", type=str, default="privileged_teacher")
    parser.add_argument("--warm_start_episodes", type=int, default=120)
    parser.add_argument("--warm_start_max_attempts", type=int, default=360)
    parser.add_argument("--warm_start_min_return", type=float, default=None)
    parser.add_argument("--warm_start_success_dup_factor", type=int, default=1)
    parser.add_argument("--warm_start_mode", type=str, choices=["flat", "sequence"], default="sequence")
    parser.add_argument("--warm_start_epochs", type=int, default=4)
    parser.add_argument("--warm_start_batch_size", type=int, default=8192)

    parser.add_argument("--eval_runs", type=int, default=6)
    parser.add_argument("--eval_interval", type=int, default=250_000)
    parser.add_argument("--log_interval", type=int, default=100_000)
    parser.add_argument("--eval_stochastic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def clone_state(state: RecurrentState) -> RecurrentState:
    h = state[0].clone()
    c = state[1].clone() if state[1] is not None else None
    return h, c


def zero_done_state(state: RecurrentState, done_idx: np.ndarray) -> None:
    if done_idx.size == 0:
        return
    state[0][:, done_idx] = 0.0
    if state[1] is not None:
        state[1][:, done_idx] = 0.0


def main() -> None:
    args = build_parser().parse_args()
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.rollout_steps % args.seq_len != 0:
        raise ValueError("rollout_steps must be divisible by seq_len")

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

    scenarios = parse_scenarios(args.scenarios)
    feature_config = RecurrentFeatureConfig(
        max_steps=int(args.max_steps),
        pose_clip=float(args.pose_clip),
        blind_clip=float(args.blind_clip),
        stuck_clip=float(args.stuck_clip),
        contact_clip=float(args.contact_clip),
        same_obs_clip=float(args.same_obs_clip),
        wall_hit_clip=float(args.wall_hit_clip),
        blind_turn_clip=float(args.blind_turn_clip),
        stuck_memory_clip=float(args.stuck_memory_clip),
        turn_streak_clip=float(args.turn_streak_clip),
        forward_streak_clip=float(args.forward_streak_clip),
        last_action_hist=int(args.last_action_hist),
        heading_bins=int(args.heading_bins),
        use_current_obs=bool(args.use_current_obs),
        use_delta_obs=bool(args.use_delta_obs),
        use_derived_obs=bool(args.use_derived_obs),
        use_pose_features=bool(args.use_pose_features),
        use_counter_features=bool(args.use_counter_features),
        use_action_history=bool(args.use_action_history),
        use_same_obs_feature=bool(args.use_same_obs_feature),
        use_wall_hit_memory=bool(args.use_wall_hit_memory),
        use_last_seen_features=bool(args.use_last_seen_features),
        use_sensor_seen_mask=bool(args.use_sensor_seen_mask),
    )
    encoder_dims = tuple(int(x) for x in args.encoder_dims)
    critic_hidden_dims = tuple(int(x) for x in args.critic_hidden_dims)
    privileged_dim = privileged_obs_dim()
    aux_target_dim = auxiliary_target_dim(args.aux_target_mode, privileged_dim) if float(args.aux_coef) > 0.0 else 0

    model = RecurrentAsymmetricActorCritic(
        actor_dim=feature_config.feature_dim,
        privileged_dim=privileged_dim,
        encoder_dims=encoder_dims,
        rnn_hidden_dim=int(args.rnn_hidden_dim),
        critic_hidden_dims=critic_hidden_dims,
        rnn_layers=int(args.rnn_layers),
        rnn_dropout=float(args.rnn_dropout),
        actor_dropout=float(args.actor_dropout),
        critic_dropout=float(args.critic_dropout),
        feature_dropout=float(args.feature_dropout),
        aux_target_dim=int(aux_target_dim),
        aux_hidden_dim=int(args.aux_hidden_dim),
        fw_bias_init=float(args.fw_bias_init),
        rnn_type=args.rnn_type,
    ).to(device)
    if args.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_eval = -float("inf")
    current_lr = float(args.lr)

    if args.load:
        checkpoint = load_checkpoint(args.load, device=device)
        missing, unexpected = model.load_state_dict(checkpoint["full_state_dict"], strict=False)
        allowed_missing = set()
        if aux_target_dim > 0:
            allowed_missing |= {name for name in missing if name.startswith("aux_head.")}
        if unexpected or (set(missing) - allowed_missing):
            raise RuntimeError(
                f"Incompatible checkpoint load for {args.load}: missing={missing} unexpected={unexpected}"
            )
        if not args.reset_best_eval:
            best_eval = float(checkpoint.get("best_eval", best_eval))
        print(f"[setup] loaded checkpoint {args.load}")

    if args.warm_start_expert and args.warm_start_episodes > 0:
        sequences_x, sequences_y, warm_stats = collect_expert_sequences_mixed(
            expert_path=args.warm_start_expert,
            episodes=args.warm_start_episodes,
            max_attempts=max(int(args.warm_start_max_attempts), int(args.warm_start_episodes)),
            min_return=args.warm_start_min_return,
            success_dup_factor=int(args.warm_start_success_dup_factor),
            seed=args.seed,
            obelix_py=args.obelix_py,
            repo_dir=repo_dir,
            scenarios=scenarios,
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            box_speed=args.box_speed,
            feature_config=feature_config,
        )
        print(
            f"[warm_start] accepted={warm_stats['accepted']:.0f}/{warm_stats['attempts']:.0f} "
            f"mean_return={warm_stats['mean_return']:.1f}"
        )
        if args.warm_start_mode == "sequence":
            warm_start_sequence_policy(
                model=model,
                optimizer=optimizer,
                sequences_x=sequences_x,
                sequences_y=sequences_y,
                device=device,
                epochs=args.warm_start_epochs,
                batch_size=args.warm_start_batch_size,
                grad_clip=args.grad_clip,
            )
        else:
            feats, acts = flatten_demo_sequences(sequences_x, sequences_y)
            warm_start_policy(
                model=model,
                optimizer=optimizer,
                feats=feats,
                acts=acts,
                device=device,
                epochs=args.warm_start_epochs,
                batch_size=args.warm_start_batch_size,
                grad_clip=args.grad_clip,
            )

    env_device = str(device) if args.env_device == "auto" else args.env_device
    vec_env_cls = import_symbol(args.obelix_torch_py, "OBELIXVectorized")

    runtimes: list[ScenarioRuntime] = []
    for idx, spec in enumerate(scenarios):
        vec_env = vec_env_cls(
            num_envs=args.num_envs,
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=spec.wall_obstacles,
            difficulty=spec.difficulty,
            box_speed=args.box_speed,
            seed=args.seed * 10_000 + idx * 1_000_000,
            device=env_device,
        )
        obs = vec_env.reset_all(seed=args.seed * 10_000 + idx * 1_000_000)
        tracker = PoseMemoryTracker(num_envs=args.num_envs, config=feature_config, device=device)
        tracker.reset_all(obs)
        privileged_teacher = None
        teacher_pool = None
        if args.teacher_agent_file:
            resolved_teacher = resolve_teacher_for_scenario(args.teacher_agent_file, spec, repo_dir)
            if resolved_teacher == "privileged_teacher":
                privileged_teacher = PrivilegedTeacher(args.num_envs, device)
                privileged_teacher.reset_all()
            else:
                teacher_pool = TeacherPool(resolved_teacher, args.num_envs, args.seed + idx * 13)
        runtimes.append(
            ScenarioRuntime(
                spec=spec,
                vec_env=vec_env,
                tracker=tracker,
                obs=obs,
                state=model.initial_state(args.num_envs, device),
                starts=torch.ones((args.num_envs,), dtype=torch.float32, device=device),
                episode_returns=np.zeros((args.num_envs,), dtype=np.float32),
                episode_lengths=np.zeros((args.num_envs,), dtype=np.int32),
                privileged_teacher=privileged_teacher,
                teacher_pool=teacher_pool,
            )
        )

    print(
        f"[setup] device={device} env_device={env_device} scenarios={[s.tag for s in scenarios]} "
        f"num_envs={args.num_envs} actor_dim={feature_config.feature_dim} privileged_dim={privileged_dim} "
        f"features={feature_config.enabled_feature_groups()}"
    )
    print(
        f"[setup] encoder={encoder_dims} rnn_type={args.rnn_type} rnn_hidden={args.rnn_hidden_dim} "
        f"critic_hidden={critic_hidden_dims} actor_dropout={args.actor_dropout} "
        f"critic_dropout={args.critic_dropout} feature_dropout={args.feature_dropout} "
        f"rnn_dropout={args.rnn_dropout} aux_coef={args.aux_coef} aux_target_mode={args.aux_target_mode} "
        f"aux_target_dim={aux_target_dim}"
    )

    scenario_recent_returns = {spec.tag: deque(maxlen=100) for spec in scenarios}
    recent_returns = deque(maxlen=300)
    recent_lengths = deque(maxlen=300)

    env_steps = 0
    update_idx = 0
    last_log_env_step = 0
    last_eval_env_step = 0
    start_time = time.time()

    try:
        while env_steps < args.total_env_steps:
            runtime = runtimes[update_idx % len(runtimes)]
            spec = runtime.spec
            buffer = RecurrentRolloutBuffer(
                num_steps=args.rollout_steps,
                num_envs=args.num_envs,
                actor_dim=feature_config.feature_dim,
                privileged_dim=privileged_dim,
                rnn_hidden_dim=int(args.rnn_hidden_dim),
                rnn_layers=int(args.rnn_layers),
                action_dim=ACTION_DIM,
                device=device,
                use_lstm=args.rnn_type == "lstm",
            )
            rollout_action_counts = np.zeros((ACTION_DIM,), dtype=np.int64)
            teacher_action_counts = np.zeros((ACTION_DIM,), dtype=np.int64)

            model.eval()
            for step in range(args.rollout_steps):
                actor_obs = runtime.tracker.features()
                privileged_obs = extract_privileged_obs(runtime.vec_env, target_device=device)
                prev_metrics = extract_shaping_metrics(runtime.vec_env, target_device=device)
                state_before = clone_state(runtime.state)

                if runtime.privileged_teacher is not None:
                    teacher_actions_t = runtime.privileged_teacher.act(runtime.vec_env)
                    teacher_action_counts += np.bincount(
                        teacher_actions_t.detach().cpu().numpy(),
                        minlength=ACTION_DIM,
                    )
                elif runtime.teacher_pool is not None:
                    teacher_idx_np = runtime.teacher_pool.act_batch(runtime.obs)
                    teacher_actions_t = torch.as_tensor(teacher_idx_np, dtype=torch.long, device=device)
                    teacher_action_counts += np.bincount(teacher_idx_np, minlength=ACTION_DIM)
                else:
                    teacher_actions_t = torch.full((args.num_envs,), -1, dtype=torch.long, device=device)

                with torch.no_grad():
                    logits_t, values_t, next_state, _, _ = model.forward_step(
                        actor_obs,
                        privileged_obs,
                        runtime.state,
                        runtime.starts,
                    )
                    temp = max(1e-4, float(args.policy_temperature))
                    dist_t = Categorical(logits=logits_t / temp)
                    actions_t = dist_t.sample()
                    log_probs_t = dist_t.log_prob(actions_t)

                    if args.teacher_action_prob_decay_steps > 0:
                        teacher_act_progress = min(
                            1.0,
                            float(env_steps) / float(max(1, args.teacher_action_prob_decay_steps)),
                        )
                    else:
                        teacher_act_progress = 1.0
                    current_teacher_action_prob = (
                        (1.0 - teacher_act_progress) * float(args.teacher_action_prob)
                        + teacher_act_progress * float(args.teacher_action_prob_final)
                    )
                    if current_teacher_action_prob > 0.0:
                        valid_teacher = teacher_actions_t >= 0
                        if bool(torch.any(valid_teacher)):
                            teacher_take_mask = (
                                torch.rand((args.num_envs,), device=device) < float(current_teacher_action_prob)
                            ) & valid_teacher
                            if bool(torch.any(teacher_take_mask)):
                                actions_t = torch.where(teacher_take_mask, teacher_actions_t, actions_t)
                                log_probs_t = dist_t.log_prob(actions_t)

                action_idx = actions_t.detach().cpu().numpy()
                rollout_action_counts += np.bincount(action_idx, minlength=ACTION_DIM)
                next_obs, rewards, dones = runtime.vec_env.step(action_idx)
                next_metrics = extract_shaping_metrics(runtime.vec_env, target_device=device)
                train_rewards = shape_rewards(
                    args=args,
                    env_rewards=rewards,
                    next_obs=next_obs,
                    actions=actions_t,
                    prev_metrics=prev_metrics,
                    next_metrics=next_metrics,
                    device=device,
                )

                done_idx = np.flatnonzero(dones)
                if done_idx.size > 0:
                    for idx in done_idx:
                        terminal_reward = float(runtime.episode_returns[idx] + rewards[idx])
                        terminal_length = int(runtime.episode_lengths[idx] + 1)
                        recent_returns.append(terminal_reward)
                        recent_lengths.append(terminal_length)
                        scenario_recent_returns[spec.tag].append(terminal_reward)

                    reset_seed = args.seed * 10_000 + env_steps + step * args.num_envs + (update_idx * 7919)
                    reset_map = runtime.vec_env.reset(env_indices=done_idx.tolist(), seed=reset_seed)
                    for idx, reset_obs in reset_map.items():
                        next_obs[idx] = reset_obs
                    if runtime.privileged_teacher is not None:
                        runtime.privileged_teacher.reset_indices(done_idx.tolist())
                    elif runtime.teacher_pool is not None:
                        runtime.teacher_pool.reset_indices(done_idx.tolist(), base_seed=reset_seed)

                buffer.add(
                    step=step,
                    actor_obs=actor_obs,
                    privileged_obs=privileged_obs,
                    starts=runtime.starts,
                    state=state_before,
                    actions=actions_t,
                    teacher_actions=teacher_actions_t,
                    log_probs=log_probs_t,
                    rewards=train_rewards,
                    dones=dones.astype(np.float32, copy=False),
                    values=values_t,
                    logits=logits_t,
                )

                runtime.episode_returns += rewards
                runtime.episode_lengths += 1
                if done_idx.size > 0:
                    runtime.episode_returns[done_idx] = 0.0
                    runtime.episode_lengths[done_idx] = 0

                runtime.tracker.post_step(actions=actions_t, next_obs=next_obs, dones=dones)
                runtime.state = next_state
                zero_done_state(runtime.state, done_idx)
                runtime.starts = torch.as_tensor(dones.astype(np.float32, copy=False), dtype=torch.float32, device=device)
                runtime.obs = next_obs
                env_steps += args.num_envs

            with torch.no_grad():
                next_actor_obs = runtime.tracker.features()
                next_privileged_obs = extract_privileged_obs(runtime.vec_env, target_device=device)
                _, last_values, _, _, _ = model.forward_step(
                    next_actor_obs,
                    next_privileged_obs,
                    runtime.state,
                    runtime.starts,
                )

            buffer.compute_returns(
                last_values=last_values,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                normalize_advantages=args.normalize_advantages,
            )

            model.train()
            mean_policy_loss = 0.0
            mean_value_loss = 0.0
            mean_entropy = 0.0
            mean_kl = 0.0
            mean_bc_loss = 0.0
            mean_aux_loss = 0.0
            num_minibatches = 0

            for (
                actor_batch,
                privileged_batch,
                starts_batch,
                state0_batch,
                actions_batch,
                teacher_actions_batch,
                old_log_probs_batch,
                old_values_batch,
                returns_batch,
                advantages_batch,
                old_logits_batch,
            ) in buffer.sequence_mini_batches(args.minibatch_size, args.update_epochs, args.seq_len):
                log_probs, entropy, values, new_logits, aux_preds, _ = model.evaluate_sequence(
                    actor_batch,
                    privileged_batch,
                    actions_batch,
                    state0_batch,
                    starts_batch,
                    temperature=args.policy_temperature,
                )

                flat_log_probs = log_probs.reshape(-1)
                flat_entropy = entropy.reshape(-1)
                flat_values = values.reshape(-1)
                flat_old_log_probs = old_log_probs_batch.reshape(-1)
                flat_old_values = old_values_batch.reshape(-1)
                flat_returns = returns_batch.reshape(-1)
                flat_advantages = advantages_batch.reshape(-1)
                flat_old_logits = old_logits_batch.reshape(-1, ACTION_DIM)
                flat_new_logits = new_logits.reshape(-1, ACTION_DIM)

                ratio = torch.exp(flat_log_probs - flat_old_log_probs)
                pg_loss1 = -flat_advantages * ratio
                pg_loss2 = -flat_advantages * torch.clamp(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + args.clip_coef,
                )
                policy_loss = torch.mean(torch.maximum(pg_loss1, pg_loss2))

                if args.use_clipped_value_loss:
                    value_delta = flat_values - flat_old_values
                    value_clipped = flat_old_values + torch.clamp(value_delta, -args.clip_coef, args.clip_coef)
                    value_loss_unclipped = (flat_values - flat_returns) ** 2
                    value_loss_clipped = (value_clipped - flat_returns) ** 2
                    value_loss = 0.5 * torch.mean(torch.maximum(value_loss_unclipped, value_loss_clipped))
                else:
                    value_loss = 0.5 * F.mse_loss(flat_values, flat_returns)

                entropy_loss = torch.mean(flat_entropy)
                kl = categorical_kl(flat_old_logits, flat_new_logits).mean()
                if args.schedule == "adaptive":
                    if kl > args.desired_kl * 2.0:
                        current_lr = max(1e-5, current_lr / 1.5)
                    elif kl < args.desired_kl / 2.0:
                        current_lr = min(1e-2, current_lr * 1.2)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr

                teacher_mask = teacher_actions_batch.reshape(-1) >= 0
                if bool(torch.any(teacher_mask)):
                    bc_loss = F.cross_entropy(
                        flat_new_logits[teacher_mask],
                        teacher_actions_batch.reshape(-1)[teacher_mask],
                    )
                else:
                    bc_loss = torch.zeros((), dtype=torch.float32, device=device)

                aux_targets = build_auxiliary_targets(privileged_batch, args.aux_target_mode)
                if float(args.aux_coef) > 0.0 and aux_preds is not None and aux_targets is not None:
                    aux_loss = F.smooth_l1_loss(aux_preds, aux_targets)
                else:
                    aux_loss = torch.zeros((), dtype=torch.float32, device=device)

                if args.teacher_decay_steps > 0:
                    decay_progress = min(1.0, float(env_steps) / float(max(1, args.teacher_decay_steps)))
                else:
                    decay_progress = 1.0
                current_teacher_coef = (1.0 - decay_progress) * float(args.teacher_coef) + decay_progress * float(
                    args.teacher_coef_final
                )

                loss = (
                    policy_loss
                    + args.vf_coef * value_loss
                    - args.ent_coef * entropy_loss
                    + current_teacher_coef * bc_loss
                    + float(args.aux_coef) * aux_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                mean_policy_loss += float(policy_loss.item())
                mean_value_loss += float(value_loss.item())
                mean_entropy += float(entropy_loss.item())
                mean_kl += float(kl.item())
                mean_bc_loss += float(bc_loss.item())
                mean_aux_loss += float(aux_loss.item())
                num_minibatches += 1

            update_idx += 1

            if env_steps - last_log_env_step >= args.log_interval:
                elapsed = max(1e-6, time.time() - start_time)
                sps = float(env_steps) / elapsed
                mean_policy_loss /= max(1, num_minibatches)
                mean_value_loss /= max(1, num_minibatches)
                mean_entropy /= max(1, num_minibatches)
                mean_kl /= max(1, num_minibatches)
                mean_bc_loss /= max(1, num_minibatches)
                mean_aux_loss /= max(1, num_minibatches)
                recent_return_mean = float(np.mean(recent_returns)) if recent_returns else float("nan")
                recent_len_mean = float(np.mean(recent_lengths)) if recent_lengths else float("nan")
                scenario_return_mean = float(np.mean(scenario_recent_returns[spec.tag])) if scenario_recent_returns[spec.tag] else float("nan")
                action_mix = " ".join(
                    f"{name}:{(count / max(1, np.sum(rollout_action_counts))):.2f}"
                    for name, count in zip(ACTIONS, rollout_action_counts.tolist())
                )
                teacher_mix = "n/a"
                if (runtime.teacher_pool is not None) or (runtime.privileged_teacher is not None):
                    total_teacher = max(1, int(np.sum(teacher_action_counts)))
                    teacher_mix = " ".join(
                        f"{name}:{(count / total_teacher):.2f}"
                        for name, count in zip(ACTIONS, teacher_action_counts.tolist())
                    )
                eta = format_hms((args.total_env_steps - env_steps) / max(1.0, sps))
                print(
                    f"[train] env_steps={env_steps} update={update_idx} scenario={spec.tag} "
                    f"ret={recent_return_mean:.1f} len={recent_len_mean:.1f} scen_ret={scenario_return_mean:.1f} "
                    f"policy={mean_policy_loss:.4f} value={mean_value_loss:.4f} entropy={mean_entropy:.4f} "
                    f"kl={mean_kl:.5f} bc={mean_bc_loss:.4f} aux={mean_aux_loss:.4f} lr={current_lr:.6f} "
                    f"actions=[{action_mix}] teacher=[{teacher_mix}] sps={sps:.1f} eta={eta}"
                )
                last_log_env_step = env_steps

            if env_steps - last_eval_env_step >= args.eval_interval:
                eval_stats = evaluate_suite(
                    model=model,
                    feature_config=feature_config,
                    obelix_py=args.obelix_py,
                    scenarios=scenarios,
                    scaling_factor=args.scaling_factor,
                    arena_size=args.arena_size,
                    max_steps=args.max_steps,
                    box_speed=args.box_speed,
                    runs=args.eval_runs,
                    seed=args.seed,
                    device=device,
                    stochastic=args.eval_stochastic,
                )
                scenario_summary = " ".join(
                    f"{spec_i.tag}:{eval_stats[f'{spec_i.tag}_mean']:.1f}" for spec_i in scenarios
                )
                print(
                    f"[eval] env_steps={env_steps} mean={eval_stats['mean_reward']:.1f} "
                    f"std={eval_stats['std_reward']:.1f} {scenario_summary}"
                )
                if eval_stats["mean_reward"] > best_eval:
                    best_eval = eval_stats["mean_reward"]
                    checkpoint = make_checkpoint(
                        model=model,
                        feature_config=feature_config,
                        encoder_dims=encoder_dims,
                        rnn_hidden_dim=int(args.rnn_hidden_dim),
                        critic_hidden_dims=critic_hidden_dims,
                        privileged_dim=privileged_dim,
                        rnn_layers=int(args.rnn_layers),
                        rnn_dropout=float(args.rnn_dropout),
                        actor_dropout=float(args.actor_dropout),
                        critic_dropout=float(args.critic_dropout),
                        feature_dropout=float(args.feature_dropout),
                        aux_target_dim=int(aux_target_dim),
                        aux_hidden_dim=int(args.aux_hidden_dim),
                        rnn_type=args.rnn_type,
                        args=args,
                        best_eval=best_eval,
                    )
                    save_checkpoint(args.out, checkpoint)
                    print(f"[eval] new best -> {args.out} ({best_eval:.1f})")
                last_eval_env_step = env_steps

    except KeyboardInterrupt:
        print("[train] interrupted; saving final checkpoint")

    final_checkpoint = make_checkpoint(
        model=model,
        feature_config=feature_config,
        encoder_dims=encoder_dims,
        rnn_hidden_dim=int(args.rnn_hidden_dim),
        critic_hidden_dims=critic_hidden_dims,
        privileged_dim=privileged_dim,
        rnn_layers=int(args.rnn_layers),
        rnn_dropout=float(args.rnn_dropout),
        actor_dropout=float(args.actor_dropout),
        critic_dropout=float(args.critic_dropout),
        feature_dropout=float(args.feature_dropout),
        aux_target_dim=int(aux_target_dim),
        aux_hidden_dim=int(args.aux_hidden_dim),
        rnn_type=args.rnn_type,
        args=args,
        best_eval=best_eval,
    )
    if best_eval == -float("inf"):
        save_checkpoint(args.out, final_checkpoint)
    final_path = args.out[:-4] + "_final.pth" if args.out.endswith(".pth") else args.out + "_final"
    save_checkpoint(final_path, final_checkpoint)
    print(f"[train] final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
