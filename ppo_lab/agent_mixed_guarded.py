from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
OBS_DIM = 18
ACTION_DIM = len(ACTIONS)
META_DIM = 5
DEFAULT_WEIGHT_CANDIDATES = ("weights.pth", "mixed_scaled_ft1.pth", "weights_best.pth")
STOCHASTIC_POLICY = os.environ.get("OBELIX_STOCHASTIC", "1") != "0"
TURN_DELTAS = {"L45": 45.0, "L22": 22.5, "FW": 0.0, "R22": -22.5, "R45": -45.0}
FORWARD_STEP = 5.0
POSE_BIN = 30.0
WALL_MEMORY_RADIUS = 55.0
WALL_MEMORY_ANGLE = 45.0


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _wrap_angle(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _angle_diff(a: float, b: float) -> float:
    return abs(_wrap_angle(a - b))


def _turn_action(sign: int, degrees: int) -> str:
    if degrees >= 45:
        return "L45" if sign > 0 else "R45"
    return "L22" if sign > 0 else "R22"


class FeatureConfig:
    def __init__(
        self,
        obs_stack: int = 8,
        action_hist: int = 4,
        max_steps: int = 500,
        contact_clip: float = 12.0,
        blind_clip: float = 50.0,
        stuck_clip: float = 10.0,
    ) -> None:
        self.obs_stack = int(obs_stack)
        self.action_hist = int(action_hist)
        self.max_steps = int(max_steps)
        self.contact_clip = float(contact_clip)
        self.blind_clip = float(blind_clip)
        self.stuck_clip = float(stuck_clip)

    @property
    def feature_dim(self) -> int:
        return self.obs_stack * OBS_DIM + self.action_hist * ACTION_DIM + META_DIM


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
        self.obs_hist[-1] = obs_arr
        self._seed_meta(obs_arr)

    def observe_transition(self, action_idx: int, obs: np.ndarray) -> None:
        obs_arr = np.asarray(obs, dtype=np.float32)
        self.obs_hist = np.roll(self.obs_hist, shift=-1, axis=0)
        self.action_hist = np.roll(self.action_hist, shift=-1, axis=0)
        self.obs_hist[-1] = obs_arr
        self.action_hist[-1] = 0.0
        self.action_hist[-1, int(action_idx)] = 1.0
        self.step_count = min(float(self.config.max_steps), self.step_count + 1.0)
        self._seed_meta(obs_arr)

    def features(self) -> np.ndarray:
        meta = np.asarray(
            [
                np.clip(self.blind_steps / max(1.0, self.config.blind_clip), 0.0, 1.0),
                np.clip(self.stuck_steps / max(1.0, self.config.stuck_clip), 0.0, 1.0),
                self.last_seen_side,
                np.clip(self.contact_memory / max(1.0, self.config.contact_clip), 0.0, 1.0),
                np.clip(self.step_count / max(1.0, self.config.max_steps), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                self.obs_hist.reshape(-1),
                self.action_hist.reshape(-1),
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


class GuardState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.prev_obs: Optional[np.ndarray] = None
        self.last_action: Optional[str] = None
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading_deg = 0.0
        self.wall_hits: list[tuple[float, float, float, int]] = []
        self.visit_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        self.sensor_seen = np.zeros((17,), dtype=bool)
        self.recovery_plan: list[str] = []
        self.scan_dir = 1
        self.escape_bias = 1
        self.blind_fw_streak = 0
        self.repeat_obs_steps = 0
        self.last_seen_side = 0
        self.stuck_events = 0
        self.caution_steps = 0
        self.caution_turn_sign = 1
        self.wall_mode = False
        self.wall_mode_turns = 0

    def pose_key(self) -> tuple[int, int, int]:
        return (
            int(np.round(self.pose_x / POSE_BIN)),
            int(np.round(self.pose_y / POSE_BIN)),
            int(np.round(_wrap_angle(self.heading_deg) / 45.0)) % 8,
        )

    def wall_ahead_turn(self) -> int:
        for hit_x, hit_y, hit_heading, turn_sign in reversed(self.wall_hits):
            dx = self.pose_x - hit_x
            dy = self.pose_y - hit_y
            if (dx * dx + dy * dy) > (WALL_MEMORY_RADIUS * WALL_MEMORY_RADIUS):
                continue
            if _angle_diff(self.heading_deg, hit_heading) <= WALL_MEMORY_ANGLE:
                return int(turn_sign)
        return 0


_MODEL: Optional[ActorCritic] = None
_TRACKER: Optional[FeatureTracker] = None
_GUARD = GuardState()
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
    )

    model = ActorCritic(input_dim=feature_config.feature_dim, hidden_dims=hidden_dims)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _MODEL = model
    _TRACKER = FeatureTracker(feature_config)


def _update_guard(obs_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current_bits = (obs_arr > 0.5).astype(np.int8, copy=False)
    prev_seen = _GUARD.sensor_seen.copy()

    if _GUARD.last_action is not None:
        if _GUARD.last_action == "FW":
            if current_bits[17]:
                sign = _GUARD.escape_bias if _GUARD.escape_bias != 0 else 1
                _GUARD.wall_hits.append((_GUARD.pose_x, _GUARD.pose_y, _GUARD.heading_deg, sign))
                if len(_GUARD.wall_hits) > 16:
                    _GUARD.wall_hits = _GUARD.wall_hits[-16:]
            else:
                rad = np.deg2rad(_GUARD.heading_deg)
                _GUARD.pose_x += FORWARD_STEP * float(np.cos(rad))
                _GUARD.pose_y += FORWARD_STEP * float(np.sin(rad))
                _GUARD.visit_counts[_GUARD.pose_key()] += 1
        else:
            _GUARD.heading_deg = _wrap_angle(_GUARD.heading_deg + TURN_DELTAS[_GUARD.last_action])

    if _GUARD.prev_obs is not None and np.array_equal(current_bits[:17], _GUARD.prev_obs[:17]):
        _GUARD.repeat_obs_steps += 1
    else:
        _GUARD.repeat_obs_steps = 0

    left = int(np.sum(current_bits[:4]))
    right = int(np.sum(current_bits[12:16]))
    front = int(np.sum(current_bits[4:12]))
    if left > right and left > 0:
        _GUARD.last_seen_side = -1
    elif right > left and right > 0:
        _GUARD.last_seen_side = 1
    elif front > 0:
        _GUARD.last_seen_side = 0

    _GUARD.sensor_seen |= current_bits[:17].astype(bool, copy=False)
    _GUARD.prev_obs = current_bits.copy()
    return current_bits, prev_seen


def _override_action(proposed: str, obs_bits: np.ndarray, prev_seen: np.ndarray) -> str:
    blind = not bool(np.any(obs_bits[:16]))
    front_near = int(np.sum(obs_bits[5:12:2]))
    front_any = int(np.sum(obs_bits[4:12]))
    ir = bool(obs_bits[16])
    wall_turn = _GUARD.wall_ahead_turn()
    revisit = _GUARD.visit_counts[_GUARD.pose_key()]
    new_bits = np.logical_and(obs_bits[:17].astype(bool), ~prev_seen)

    if obs_bits[17] and _GUARD.last_action == "FW":
        sign = wall_turn if wall_turn != 0 else _GUARD.escape_bias
        _GUARD.escape_bias *= -1
        _GUARD.stuck_events += 1
        _GUARD.caution_steps = 10
        _GUARD.caution_turn_sign = sign
        _GUARD.wall_mode = True
        _GUARD.wall_mode_turns = 0
        _GUARD.recovery_plan = [_turn_action(sign, 45), _turn_action(sign, 22), "FW"]
        _GUARD.blind_fw_streak = 0
        return _GUARD.recovery_plan.pop(0)

    if _GUARD.recovery_plan:
        return _GUARD.recovery_plan.pop(0)

    if _GUARD.wall_mode and (ir or front_near >= 2 and np.any(new_bits[4:17])):
        _GUARD.wall_mode = False
        _GUARD.wall_mode_turns = 0

    if _GUARD.wall_mode and not ir and front_near == 0:
        if blind and wall_turn == 0 and revisit == 0 and _GUARD.wall_mode_turns >= 12:
            _GUARD.wall_mode_turns = 0
            return "FW"
        _GUARD.wall_mode_turns += 1
        _GUARD.blind_fw_streak = 0
        return _turn_action(_GUARD.caution_turn_sign, 22)

    if _GUARD.caution_steps > 0 and not ir and front_near == 0:
        _GUARD.caution_steps -= 1
        _GUARD.blind_fw_streak = 0
        return _turn_action(_GUARD.caution_turn_sign, 22)

    if blind:
        if wall_turn != 0 or revisit >= 2 or _GUARD.repeat_obs_steps >= 2:
            _GUARD.blind_fw_streak = 0
            return _turn_action(wall_turn if wall_turn != 0 else -_GUARD.scan_dir, 45)

        if proposed == "FW":
            if _GUARD.blind_fw_streak >= 1:
                _GUARD.blind_fw_streak = 0
                _GUARD.scan_dir *= -1
                return _turn_action(_GUARD.scan_dir, 22)
            _GUARD.blind_fw_streak += 1
            return proposed

        if _GUARD.repeat_obs_steps >= 1:
            _GUARD.scan_dir *= -1
            return _turn_action(_GUARD.scan_dir, 22)

        _GUARD.blind_fw_streak = 0
        return proposed

    _GUARD.blind_fw_streak = 0

    if _GUARD.stuck_events >= 2 and not ir and front_near == 0 and proposed == "FW":
        return _turn_action(wall_turn if wall_turn != 0 else _GUARD.caution_turn_sign, 22)

    if front_near > 0 or ir or np.any(new_bits[4:17]):
        return proposed

    if proposed == "FW" and (wall_turn != 0 or _GUARD.repeat_obs_steps >= 3 or revisit >= 3):
        if _GUARD.last_seen_side < 0:
            return "L22"
        if _GUARD.last_seen_side > 0:
            return "R22"
        return _turn_action(wall_turn if wall_turn != 0 else _GUARD.scan_dir, 22)

    if front_any == 0 and _GUARD.last_seen_side != 0 and proposed == "FW":
        return "L22" if _GUARD.last_seen_side < 0 else "R22"

    return proposed


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
        _GUARD.reset()
    elif _PENDING_ACTION is not None:
        _TRACKER.observe_transition(_PENDING_ACTION, obs_arr)

    obs_bits, prev_seen = _update_guard(obs_arr)

    x = torch.as_tensor(_TRACKER.features(), dtype=torch.float32).unsqueeze(0)
    logits, _ = _MODEL(x)
    if STOCHASTIC_POLICY:
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64, copy=False)
        probs /= probs.sum()
        proposed_idx = int(rng.choice(len(ACTIONS), p=probs))
    else:
        proposed_idx = int(torch.argmax(logits, dim=1).item())
    action = _override_action(ACTIONS[proposed_idx], obs_bits, prev_seen)
    _GUARD.last_action = action
    _PENDING_ACTION = ACTION_TO_INDEX[action]
    return action
