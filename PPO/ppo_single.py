"""Offline PPO trainer (single environment, CPU) for OBELIX.

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
  python ppo_single.py --obelix_py ../obelix.py --out weights.pth --episodes 2000 --difficulty 0
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
FW_ACTION_INDEX = ACTIONS.index("FW")


def apply_forward_bias(layer: nn.Linear, fw_bias_init: float) -> None:
    if fw_bias_init == 0.0:
        return
    with torch.no_grad():
        layer.bias[FW_ACTION_INDEX] += float(fw_bias_init)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 18, action_dim: int = 5, hidden_dims=(64, 64), fw_bias_init: float = 0.0):
        super().__init__()
        h1, h2 = int(hidden_dims[0]), int(hidden_dims[1])
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(h2, action_dim)
        self.value_head = nn.Linear(h2, 1)
        apply_forward_bias(self.policy_head, fw_bias_init)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value


def import_obelix(obelix_py: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_nonterminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_adv = delta + gamma * gae_lambda * next_nonterminal * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


def load_checkpoint_state(checkpoint_path: str):
    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = raw["state_dict"]
        metadata = raw
    elif isinstance(raw, dict):
        state_dict = raw
        metadata = {}
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    return state_dict, metadata


def heuristic_action(obs: np.ndarray, step_idx: int) -> int:
    obs = np.asarray(obs, dtype=np.float32)

    left = float(np.sum(obs[:4]) + 0.5 * np.sum(obs[4:8]))
    right = float(np.sum(obs[12:16]) + 0.5 * np.sum(obs[8:12]))
    front_far = float(np.sum(obs[4:12:2]))
    front_near = float(np.sum(obs[5:12:2]))
    ir_contact = float(obs[16])
    stuck = float(obs[17])
    diff = left - right

    if stuck > 0.5:
        return 0 if diff >= 0.0 else 4
    if ir_contact > 0.5 or front_near >= 2.0:
        return 2
    if diff >= 2.0:
        return 0
    if diff >= 0.75:
        return 1
    if diff <= -2.0:
        return 4
    if diff <= -0.75:
        return 3
    if front_far > 0.0 or front_near > 0.0:
        return 2
    phase = step_idx % 12
    if phase < 8:
        return 2
    if phase < 10:
        return 1
    return 3


def warm_start_policy(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    obelix_cls,
    args: argparse.Namespace,
) -> None:
    if args.warm_start_episodes <= 0:
        return

    demo_obs = []
    demo_actions = []
    demo_returns: Deque[float] = deque(maxlen=20)
    action_counts = np.zeros((len(ACTIONS),), dtype=np.int64)

    print(f"Warm start: collecting {args.warm_start_episodes} heuristic episodes...")
    for ep in range(args.warm_start_episodes):
        env = obelix_cls(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + 100_000 + ep,
        )
        obs = env.reset(seed=args.seed + 100_000 + ep)
        ep_ret = 0.0

        for step_idx in range(args.max_steps):
            action = heuristic_action(obs, step_idx)
            demo_obs.append(np.asarray(obs, dtype=np.float32))
            demo_actions.append(action)
            action_counts[action] += 1
            obs, reward, done = env.step(ACTIONS[action], render=False)
            ep_ret += float(reward)
            if done:
                break

        demo_returns.append(ep_ret)
        if (ep + 1) % max(1, min(20, args.warm_start_episodes)) == 0:
            print(
                f"[warm_start] demos={ep+1}/{args.warm_start_episodes} "
                f"recent_mean_return={np.mean(demo_returns):.1f}"
            )

    if not demo_obs:
        return

    obs_t = torch.tensor(np.asarray(demo_obs, dtype=np.float32), dtype=torch.float32)
    act_t = torch.tensor(np.asarray(demo_actions, dtype=np.int64), dtype=torch.int64)
    batch_size = int(obs_t.shape[0])
    minibatch_size = min(args.warm_start_batch_size, batch_size)

    print(
        "[warm_start] action_mix="
        + " ".join(
            f"{name}:{count / max(1, int(np.sum(action_counts))):.2f}"
            for name, count in zip(ACTIONS, action_counts)
        )
    )

    for epoch in range(args.warm_start_epochs):
        indices = np.random.permutation(batch_size)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            logits, _ = model(obs_t[mb_idx])
            loss = F.cross_entropy(logits, act_t[mb_idx])

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += float(loss.item()) * len(mb_idx)
            total_correct += int((logits.argmax(dim=1) == act_t[mb_idx]).sum().item())
            total_seen += len(mb_idx)

        print(
            f"[warm_start] epoch={epoch+1}/{args.warm_start_epochs} "
            f"bc_loss={total_loss / max(1, total_seen):.4f} "
            f"bc_acc={total_correct / max(1, total_seen):.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip_coef", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--update_epochs", type=int, default=10)
    ap.add_argument("--minibatch_size", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--normalize_advantages", action="store_true")
    ap.add_argument("--hidden_dims", type=int, nargs=2, default=[64, 64])
    ap.add_argument(
        "--fw_bias_init",
        type=float,
        default=1.0,
        help="Initial logit bias added to the FW action in the actor head.",
    )
    ap.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path used to warm-start model weights before training.",
    )
    ap.add_argument("--warm_start_episodes", type=int, default=0)
    ap.add_argument("--warm_start_epochs", type=int, default=5)
    ap.add_argument("--warm_start_batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    hidden_dims = tuple(int(h) for h in args.hidden_dims)
    model = ActorCritic(hidden_dims=hidden_dims, fw_bias_init=args.fw_bias_init)
    if args.init_checkpoint is not None:
        state_dict, metadata = load_checkpoint_state(args.init_checkpoint)
        ckpt_hidden_dims = metadata.get("hidden_dims")
        if ckpt_hidden_dims is not None and tuple(int(h) for h in ckpt_hidden_dims) != hidden_dims:
            raise ValueError(
                "Checkpoint hidden_dims do not match requested model shape: "
                f"checkpoint={tuple(int(h) for h in ckpt_hidden_dims)} requested={hidden_dims}"
            )
        model.load_state_dict(state_dict, strict=True)
        print(f"Warm-started model weights from {args.init_checkpoint}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    warm_start_policy(model, optimizer, OBELIX, args)

    recent_returns: Deque[float] = deque(maxlen=100)
    total_steps = 0
    start_time = time.time()
    print(f"Training PPO for {args.episodes} episodes...")
    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )
        obs = env.reset(seed=args.seed + ep)

        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []
        logp_buf = []
        val_buf = []

        ep_ret = 0.0
        ep_len = 0

        for _ in range(args.max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_t)
                dist = Categorical(logits=logits)
                action_t = dist.sample()
                logp_t = dist.log_prob(action_t)

            action = int(action_t.item())
            obs2, reward, done = env.step(ACTIONS[action], render=False)

            obs_buf.append(np.asarray(obs, dtype=np.float32))
            act_buf.append(action)
            rew_buf.append(float(reward))
            done_buf.append(float(done))
            logp_buf.append(float(logp_t.item()))
            val_buf.append(float(value.item()))

            obs = obs2
            ep_ret += float(reward)
            ep_len += 1
            total_steps += 1

            if done:
                break

        if done:
            next_value = 0.0
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                _, bootstrap_value = model(obs_t)
                next_value = float(bootstrap_value.item())

        obs_arr = np.asarray(obs_buf, dtype=np.float32)
        act_arr = np.asarray(act_buf, dtype=np.int64)
        rew_arr = np.asarray(rew_buf, dtype=np.float32)
        done_arr = np.asarray(done_buf, dtype=np.float32)
        logp_arr = np.asarray(logp_buf, dtype=np.float32)
        val_arr = np.asarray(val_buf, dtype=np.float32)

        adv_arr, ret_arr = compute_gae(
            rewards=rew_arr,
            values=val_arr,
            dones=done_arr,
            next_value=next_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        if args.normalize_advantages and len(adv_arr) > 1:
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        obs_t = torch.tensor(obs_arr, dtype=torch.float32)
        act_t = torch.tensor(act_arr, dtype=torch.int64)
        old_logp_t = torch.tensor(logp_arr, dtype=torch.float32)
        adv_t = torch.tensor(adv_arr, dtype=torch.float32)
        ret_t = torch.tensor(ret_arr, dtype=torch.float32)

        batch_size = len(obs_arr)
        minibatch_size = min(args.minibatch_size, batch_size)

        for _ in range(args.update_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]

                logits, values = model(obs_t[mb_idx])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp_t[mb_idx])
                surr1 = ratio * adv_t[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * adv_t[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, ret_t[mb_idx])
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        recent_returns.append(ep_ret)
        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} "
                f"recent_mean={np.mean(recent_returns):.1f} len={ep_len} steps={total_steps}"
            )
            print("Time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    checkpoint = {
        "state_dict": model.state_dict(),
        "obs_dim": 18,
        "action_dim": len(ACTIONS),
        "hidden_dims": [int(h) for h in hidden_dims],
        "actions": ACTIONS,
        "config": vars(args),
    }
    torch.save(checkpoint, args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
