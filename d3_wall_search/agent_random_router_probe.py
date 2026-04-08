from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PPO_AGENT = os.path.join(ROOT, "PPO", "agent.py")
WALL_AGENT = os.path.join(ROOT, "assym_ppo", "agent_mixed_rnn_v2_best.py")
WALL_WEIGHTS = os.path.join(HERE, "wall_d3_random_asym_v1.pth")
NOWALL_WEIGHTS = os.path.join(HERE, "nowall_d3_ppo_random_ft_v1.pth")
NOWALL_V1_WEIGHTS = os.path.join(ROOT, "ppo_lab", "nowall_d3_ppo_v1.pth")
ROUTER_WEIGHTS = os.path.join(HERE, "router_d3_random_probe20_mlp_v2.pth")


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.Tanh())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_WALL_MOD = None
_NOWALL_MOD = None
_NOWALL_V1_MOD = None
_ROUTER: Optional[RouterMLP] = None
_ROUTER_THRESHOLD = 0.8
_PROBE_STEPS = 80

_LAST_RNG_ID: Optional[int] = None
_STEP = 0
_MODE: Optional[str] = None
_FIRST_OBS: Optional[np.ndarray] = None
_STATS = np.zeros((12,), dtype=np.float32)
_PROBE_REWARD_PROXY = 0.0
_SPIN_COUNT = 0
_SPIN_SEEN = 0
_SPIN_LAST_SUM = 0.0
_LONG_PROBE_TARGET = 75


def _load_once() -> None:
    global _WALL_MOD, _NOWALL_MOD, _NOWALL_V1_MOD, _ROUTER, _ROUTER_THRESHOLD, _PROBE_STEPS
    if _WALL_MOD is None:
        _WALL_MOD = _load_module(WALL_AGENT, "wall_random_agent")
        os.environ["OBELIX_WEIGHTS"] = WALL_WEIGHTS
    if _NOWALL_MOD is None:
        _NOWALL_MOD = _load_module(PPO_AGENT, "nowall_random_agent")
        os.environ["OBELIX_WEIGHTS"] = NOWALL_WEIGHTS
    if _NOWALL_V1_MOD is None:
        _NOWALL_V1_MOD = _load_module(PPO_AGENT, "nowall_v1_agent")
        os.environ["OBELIX_WEIGHTS"] = NOWALL_V1_WEIGHTS
    if _ROUTER is None:
        ckpt = torch.load(ROUTER_WEIGHTS, map_location="cpu")
        _ROUTER = RouterMLP(
            input_dim=int(ckpt["input_dim"]),
            hidden_dims=tuple(int(x) for x in ckpt["hidden_dims"]),
        )
        state_dict = ckpt["state_dict"]
        if state_dict and not next(iter(state_dict)).startswith("net."):
            state_dict = {f"net.{key}": value for key, value in state_dict.items()}
        _ROUTER.load_state_dict(state_dict, strict=True)
        _ROUTER.eval()
        _ROUTER_THRESHOLD = float(ckpt.get("threshold", 0.8))
        _PROBE_STEPS = int(ckpt.get("probe_steps", 80))


def _reset(obs_arr: np.ndarray, rng) -> None:
    global _LAST_RNG_ID, _STEP, _MODE, _FIRST_OBS, _STATS, _PROBE_REWARD_PROXY, _SPIN_COUNT, _SPIN_SEEN, _SPIN_LAST_SUM
    _LAST_RNG_ID = id(rng)
    _STEP = 0
    _MODE = None
    _FIRST_OBS = obs_arr.astype(np.float32, copy=True)
    _STATS = np.zeros((12,), dtype=np.float32)
    _PROBE_REWARD_PROXY = 0.0
    _SPIN_COUNT = 0
    _SPIN_SEEN = 0
    _SPIN_LAST_SUM = 0.0


def _obs_stats(obs: np.ndarray) -> np.ndarray:
    left = float(np.sum(obs[:4]))
    front = float(np.sum(obs[4:12]))
    right = float(np.sum(obs[12:16]))
    return np.asarray(
        [
            left,
            front,
            right,
            float(np.sum(obs[:16]) == 0.0),
            float(obs[16]),
            float(obs[17]),
            float(left > 0.0),
            float(front > 0.0),
            float(right > 0.0),
            float(np.sum(obs[5:12:2])),
            float(np.sum(obs[4:12:2])),
            float(np.sum(obs[:16])),
        ],
        dtype=np.float32,
    )


def _router_feature(obs_arr: np.ndarray) -> torch.Tensor:
    step_80_stats = _STATS / float(max(1, _PROBE_STEPS))
    x = np.concatenate(
        [
            np.asarray(_FIRST_OBS, dtype=np.float32),
            obs_arr.astype(np.float32, copy=False),
            step_80_stats.astype(np.float32, copy=False),
            np.asarray([_PROBE_REWARD_PROXY / 1000.0], dtype=np.float32),
        ]
    )
    return torch.as_tensor(x[None, :], dtype=torch.float32)


@torch.no_grad()
def _p_wall(obs_arr: np.ndarray) -> float:
    assert _ROUTER is not None
    logits = _ROUTER(_router_feature(obs_arr))
    return float(torch.softmax(logits, dim=1)[0, 1].item())


def _wall_action(obs_arr: np.ndarray, rng) -> str:
    os.environ["OBELIX_WEIGHTS"] = WALL_WEIGHTS
    return _WALL_MOD.policy(obs_arr, rng)


def _nowall_action(obs_arr: np.ndarray, rng) -> str:
    os.environ["OBELIX_WEIGHTS"] = NOWALL_WEIGHTS
    return _NOWALL_MOD.policy(obs_arr, rng)


def _nowall_v1_action(obs_arr: np.ndarray, rng) -> str:
    os.environ["OBELIX_WEIGHTS"] = NOWALL_V1_WEIGHTS
    return _NOWALL_V1_MOD.policy(obs_arr, rng)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _STEP, _MODE, _STATS, _PROBE_REWARD_PROXY, _SPIN_COUNT, _SPIN_SEEN
    _load_once()
    obs_arr = np.asarray(obs, dtype=np.float32)
    if _LAST_RNG_ID != id(rng):
        _reset(obs_arr, rng)

    if _MODE == "wall":
        _STEP += 1
        return _wall_action(obs_arr, rng)
    if _MODE == "nowall":
        _STEP += 1
        return _nowall_action(obs_arr, rng)
    if _MODE == "nowall_v1":
        _STEP += 1
        return _nowall_v1_action(obs_arr, rng)

    if _MODE == "spin_probe":
        global _SPIN_LAST_SUM
        _SPIN_LAST_SUM = float(np.sum(obs_arr[:16]))
        _SPIN_SEEN += int(np.sum(obs_arr[:16]) > 0.0)
        if _SPIN_COUNT >= 20:
            _MODE = "wall" if _SPIN_LAST_SUM > 0.0 else "nowall_v1"
            _STEP += 1
            return _wall_action(obs_arr, rng) if _MODE == "wall" else _nowall_v1_action(obs_arr, rng)
        _SPIN_COUNT += 1
        _STEP += 1
        return "L45"

    if _MODE == "nowall_spin":
        if _SPIN_COUNT >= 20:
            _MODE = "nowall"
            _STEP += 1
            return _nowall_action(obs_arr, rng)
        _SPIN_COUNT += 1
        _STEP += 1
        return "L45"

    if _MODE == "long_probe":
        if _STEP < _LONG_PROBE_TARGET:
            _STEP += 1
            return _wall_action(obs_arr, rng)
        _MODE = "spin_probe"
        _SPIN_COUNT = 1
        _SPIN_SEEN = int(np.sum(obs_arr[:16]) > 0.0)
        _STEP += 1
        return "L45"

    if _STEP < _PROBE_STEPS:
        _STATS += _obs_stats(obs_arr)
        _PROBE_REWARD_PROXY -= 1.0
        _STEP += 1
        return _wall_action(obs_arr, rng)

    wall_prob = _p_wall(obs_arr)
    blind_probe = float(_STATS[11]) <= 0.05
    if wall_prob >= _ROUTER_THRESHOLD:
        _MODE = "wall"
        action = _wall_action(obs_arr, rng)
    elif blind_probe:
        _MODE = "long_probe"
        action = _wall_action(obs_arr, rng)
    else:
        _MODE = "nowall_spin"
        _SPIN_COUNT = 1
        action = "L45"

    _STEP += 1
    return action
