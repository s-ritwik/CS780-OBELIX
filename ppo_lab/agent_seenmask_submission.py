from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18
ACTION_DIM = len(ACTIONS)
SENSOR_MEMORY_DIM = 17
META_DIM = 8
DEFAULT_WEIGHT_CANDIDATES = (
    "weights.pth",
    "weights_seenmask_best.pth",
    "mixed_seenmask_ft1_snapshot4.pth",
    "mixed_seenmask_ft1.pth",
    "weights_best.pth",
)
STOCHASTIC_POLICY = os.environ.get("OBELIX_STOCHASTIC", "1") != "0"


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureConfig:
    def __init__(
        self,
        obs_stack: int = 8,
        action_hist: int = 4,
        max_steps: int = 500,
        contact_clip: float = 12.0,
        blind_clip: float = 50.0,
        stuck_clip: float = 10.0,
        repeat_clip: float = 20.0,
        turn_clip: float = 12.0,
        stuck_memory_clip: float = 12.0,
    ) -> None:
        self.obs_stack = int(obs_stack)
        self.action_hist = int(action_hist)
        self.max_steps = int(max_steps)
        self.contact_clip = float(contact_clip)
        self.blind_clip = float(blind_clip)
        self.stuck_clip = float(stuck_clip)
        self.repeat_clip = float(repeat_clip)
        self.turn_clip = float(turn_clip)
        self.stuck_memory_clip = float(stuck_memory_clip)

    @property
    def feature_dim(self) -> int:
        return self.obs_stack * OBS_DIM + self.action_hist * ACTION_DIM + SENSOR_MEMORY_DIM + META_DIM


class FeatureTracker:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.obs_hist = np.zeros((self.config.obs_stack, OBS_DIM), dtype=np.float32)
        self.action_hist = np.zeros((self.config.action_hist, ACTION_DIM), dtype=np.float32)
        self.blind_steps = 0.0
        self.stuck_steps = 0.0
        self.step_count = 0.0
        self.last_seen_side = 0.0
        self.contact_memory = 0.0
        self.sensor_seen = np.zeros((SENSOR_MEMORY_DIM,), dtype=np.float32)
        self.repeat_obs_steps = 0.0
        self.blind_turn_steps = 0.0
        self.stuck_memory = 0.0

    def _seed_meta(self, obs: np.ndarray) -> None:
        left = float(np.sum(obs[:4]))
        right = float(np.sum(obs[12:16]))
        front = float(np.sum(obs[4:12]))
        front_near = float(np.sum(obs[5:12:2]))
        visible = bool(np.any(obs[:16] > 0.5))
        contact_like = bool(obs[16] > 0.5 or front_near > 0.0)

        if left > right and left > 0.0:
            self.last_seen_side = -1.0
        elif right > left and right > 0.0:
            self.last_seen_side = 1.0
        elif front > 0.0:
            self.last_seen_side = 0.0

        if visible:
            self.blind_steps = 0.0
        else:
            self.blind_steps = min(self.config.blind_clip, self.blind_steps + 1.0)

        if obs[17] > 0.5:
            self.stuck_steps = min(self.config.stuck_clip, self.stuck_steps + 1.0)
        else:
            self.stuck_steps = 0.0

        if contact_like:
            self.contact_memory = min(self.config.contact_clip, self.contact_memory + 1.0)
        else:
            self.contact_memory = max(0.0, self.contact_memory - 1.0)

    def reset(self, obs: np.ndarray) -> None:
        obs_arr = np.asarray(obs, dtype=np.float32)
        self.obs_hist.fill(0.0)
        self.action_hist.fill(0.0)
        self.blind_steps = 0.0
        self.stuck_steps = 0.0
        self.step_count = 0.0
        self.last_seen_side = 0.0
        self.contact_memory = 0.0
        self.sensor_seen.fill(0.0)
        self.repeat_obs_steps = 0.0
        self.blind_turn_steps = 0.0
        self.stuck_memory = 1.0 if obs_arr[17] > 0.5 else 0.0
        self.obs_hist[-1] = obs_arr
        self.sensor_seen = (obs_arr[:SENSOR_MEMORY_DIM] > 0.5).astype(np.float32, copy=False)
        self._seed_meta(obs_arr)

    def observe_transition(self, action_idx: int, obs: np.ndarray) -> None:
        obs_arr = np.asarray(obs, dtype=np.float32)
        prev_obs = self.obs_hist[-1].copy()
        self.obs_hist = np.roll(self.obs_hist, shift=-1, axis=0)
        self.action_hist = np.roll(self.action_hist, shift=-1, axis=0)
        self.obs_hist[-1] = obs_arr
        self.action_hist[-1] = 0.0
        self.action_hist[-1, int(action_idx)] = 1.0
        self.step_count = min(float(self.config.max_steps), self.step_count + 1.0)
        same_obs = np.array_equal((prev_obs[:SENSOR_MEMORY_DIM] > 0.5), (obs_arr[:SENSOR_MEMORY_DIM] > 0.5))
        self.repeat_obs_steps = min(self.config.repeat_clip, self.repeat_obs_steps + 1.0) if same_obs else 0.0
        blind = bool(np.sum(obs_arr[:16]) == 0.0)
        turned = int(action_idx) != ACTIONS.index("FW")
        self.blind_turn_steps = min(self.config.turn_clip, self.blind_turn_steps + 1.0) if blind and turned else 0.0
        if obs_arr[17] > 0.5:
            self.stuck_memory = min(self.config.stuck_memory_clip, self.stuck_memory + 1.0)
        else:
            self.stuck_memory = max(0.0, self.stuck_memory - 1.0)
        self.sensor_seen = np.maximum(self.sensor_seen, (obs_arr[:SENSOR_MEMORY_DIM] > 0.5).astype(np.float32, copy=False))
        self._seed_meta(obs_arr)

    def features(self) -> np.ndarray:
        meta = np.asarray(
            [
                np.clip(self.blind_steps / max(1.0, self.config.blind_clip), 0.0, 1.0),
                np.clip(self.stuck_steps / max(1.0, self.config.stuck_clip), 0.0, 1.0),
                self.last_seen_side,
                np.clip(self.contact_memory / max(1.0, self.config.contact_clip), 0.0, 1.0),
                np.clip(self.step_count / max(1.0, self.config.max_steps), 0.0, 1.0),
                np.clip(self.repeat_obs_steps / max(1.0, self.config.repeat_clip), 0.0, 1.0),
                np.clip(self.blind_turn_steps / max(1.0, self.config.turn_clip), 0.0, 1.0),
                np.clip(self.stuck_memory / max(1.0, self.config.stuck_memory_clip), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                self.obs_hist.reshape(-1),
                self.action_hist.reshape(-1),
                self.sensor_seen,
                meta,
            ],
            axis=0,
        ).astype(np.float32, copy=False)


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            h = int(hidden_dim)
            layers.append(layer_init(nn.Linear(last_dim, h)))
            layers.append(nn.Tanh())
            last_dim = h

        self.backbone = nn.Sequential(*layers)
        self.policy_head = layer_init(nn.Linear(last_dim, ACTION_DIM), std=0.01)
        self.value_head = layer_init(nn.Linear(last_dim, 1), std=1.0)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value


_MODEL: Optional[ActorCritic] = None
_TRACKER: Optional[FeatureTracker] = None
_LAST_RNG_ID: Optional[int] = None
_PENDING_ACTION: Optional[int] = None


def _checkpoint_path() -> str:
    here = os.path.dirname(__file__)
    override = os.environ.get("OBELIX_WEIGHTS")
    if override:
        return override if os.path.isabs(override) else os.path.join(here, override)

    for candidate in DEFAULT_WEIGHT_CANDIDATES:
        path = os.path.join(here, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Weights file not found. Expected one of: "
        + ", ".join(DEFAULT_WEIGHT_CANDIDATES)
    )


def _load_once() -> None:
    global _MODEL, _TRACKER
    if _MODEL is not None and _TRACKER is not None:
        return

    raw = torch.load(_checkpoint_path(), map_location="cpu")
    if not isinstance(raw, dict) or "state_dict" not in raw:
        raise RuntimeError("Unsupported checkpoint format")

    state_dict = raw["state_dict"]
    hidden_dims = tuple(int(x) for x in raw.get("hidden_dims", [256, 128]))
    feature_payload = raw.get("feature_config", {})
    feature_config = FeatureConfig(
        obs_stack=int(feature_payload.get("obs_stack", 8)),
        action_hist=int(feature_payload.get("action_hist", 4)),
        max_steps=int(feature_payload.get("max_steps", 500)),
        contact_clip=float(feature_payload.get("contact_clip", 12.0)),
        blind_clip=float(feature_payload.get("blind_clip", 50.0)),
        stuck_clip=float(feature_payload.get("stuck_clip", 10.0)),
        repeat_clip=float(feature_payload.get("repeat_clip", 20.0)),
        turn_clip=float(feature_payload.get("turn_clip", 12.0)),
        stuck_memory_clip=float(feature_payload.get("stuck_memory_clip", 12.0)),
    )

    model = ActorCritic(input_dim=feature_config.feature_dim, hidden_dims=hidden_dims)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _MODEL = model
    _TRACKER = FeatureTracker(feature_config)


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_RNG_ID, _PENDING_ACTION
    _load_once()

    obs_arr = np.asarray(obs, dtype=np.float32)
    rng_id = id(rng)
    if _LAST_RNG_ID != rng_id:
        _LAST_RNG_ID = rng_id
        _PENDING_ACTION = None
        _TRACKER.reset(obs_arr)
    elif _PENDING_ACTION is not None:
        _TRACKER.observe_transition(_PENDING_ACTION, obs_arr)

    x = torch.as_tensor(_TRACKER.features(), dtype=torch.float32).unsqueeze(0)
    logits, _ = _MODEL(x)
    if STOCHASTIC_POLICY:
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64, copy=False)
        probs /= probs.sum()
        action_idx = int(rng.choice(len(ACTIONS), p=probs))
    else:
        action_idx = int(torch.argmax(logits, dim=1).item())
    _PENDING_ACTION = action_idx
    return ACTIONS[action_idx]
