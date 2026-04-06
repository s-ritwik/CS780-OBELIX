from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
RAW_OBS_DIM = 18
ENCODED_OBS_DIM = 32
TURN_DELTAS = {"L45": 45.0, "L22": 22.5, "FW": 0.0, "R22": -22.5, "R45": -45.0}
FORWARD_STEP = 5.0
POSE_BIN = 30.0
WALL_MEMORY_RADIUS = 55.0
WALL_MEMORY_ANGLE = 45.0
DEFAULT_WEIGHTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ppo_lab", "nowall_d3_ppo_v3.pth")
)


def encode_obs_tensor(obs: torch.Tensor) -> torch.Tensor:
    raw = obs.to(dtype=torch.float32)
    left = raw[..., 0:4]
    front = raw[..., 4:12]
    right = raw[..., 12:16]
    ir = raw[..., 16:17]
    stuck = raw[..., 17:18]

    left_count = left.sum(dim=-1, keepdim=True)
    front_count = front.sum(dim=-1, keepdim=True)
    right_count = right.sum(dim=-1, keepdim=True)
    front_far_count = front[..., ::2].sum(dim=-1, keepdim=True)
    front_near_count = front[..., 1::2].sum(dim=-1, keepdim=True)
    side_mean_count = 0.5 * (left_count + right_count)
    blind = (raw[..., :16].sum(dim=-1, keepdim=True) == 0.0).to(dtype=torch.float32)

    derived = torch.cat(
        [
            (left_count > 0.0).to(dtype=torch.float32),
            (front_count > 0.0).to(dtype=torch.float32),
            (right_count > 0.0).to(dtype=torch.float32),
            ir,
            stuck,
            blind,
            left_count,
            front_count,
            right_count,
            front_far_count,
            front_near_count,
            left_count - right_count,
            right_count - left_count,
            front_count - side_mean_count,
        ],
        dim=-1,
    )
    return torch.cat([raw, derived], dim=-1)


def _wrap_angle(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _angle_diff(a: float, b: float) -> float:
    return abs(_wrap_angle(a - b))


def _turn_action(sign: int, degrees: int) -> str:
    if degrees >= 45:
        return "L45" if sign > 0 else "R45"
    return "L22" if sign > 0 else "R22"


def _infer_hidden_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    dims: list[tuple[int, int]] = []
    for key, value in state_dict.items():
        if key.startswith("backbone.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                dims.append((int(parts[1]), int(value.shape[0])))
    dims.sort(key=lambda item: item[0])
    hidden = [dim for _, dim in dims]
    return tuple(hidden) if hidden else (128, 64)


def infer_use_rec_encoder_from_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    first_weight = state_dict.get("backbone.0.weight")
    if first_weight is None:
        return False
    in_features = int(first_weight.shape[1])
    if in_features == RAW_OBS_DIM:
        return False
    if in_features == ENCODED_OBS_DIM:
        return True
    raise ValueError(f"Unexpected first-layer input dim: {in_features}")


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = RAW_OBS_DIM,
        action_dim: int = 5,
        hidden_dims: tuple[int, ...] = (128, 64),
        use_rec_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.use_rec_encoder = bool(use_rec_encoder)
        last_dim = ENCODED_OBS_DIM if self.use_rec_encoder else int(obs_dim)
        layers: list[nn.Module] = []
        for hidden_dim in hidden_dims:
            h = int(hidden_dim)
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, int(action_dim))
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_rec_encoder:
            x = encode_obs_tensor(x)
        feat = self.backbone(x)
        return self.policy_head(feat), self.value_head(feat).squeeze(-1)


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
_GUARD = GuardState()
_LAST_RNG_ID: Optional[int] = None


def _load_once() -> None:
    global _MODEL
    if _MODEL is not None:
        return

    wpath = os.environ.get("OBELIX_WEIGHTS", DEFAULT_WEIGHTS)
    if not os.path.isabs(wpath):
        wpath = os.path.abspath(os.path.join(os.path.dirname(__file__), wpath))
    raw = torch.load(wpath, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = raw["state_dict"]
        hidden_dims = tuple(int(x) for x in raw.get("hidden_dims", _infer_hidden_dims(state_dict)))
        use_rec_encoder = raw.get("use_rec_encoder")
    elif isinstance(raw, dict):
        state_dict = raw
        hidden_dims = _infer_hidden_dims(state_dict)
        use_rec_encoder = None
    else:
        raise RuntimeError("Unsupported checkpoint format")

    if use_rec_encoder is None:
        use_rec_encoder = infer_use_rec_encoder_from_state_dict(state_dict)
    model = ActorCritic(hidden_dims=hidden_dims, use_rec_encoder=bool(use_rec_encoder))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _MODEL = model


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

    if _GUARD.wall_mode and (ir or (front_near >= 2 and np.any(new_bits[4:17]))):
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
    global _LAST_RNG_ID
    _load_once()

    obs_arr = np.asarray(obs, dtype=np.float32)
    rng_id = id(rng)
    if _LAST_RNG_ID != rng_id:
        _LAST_RNG_ID = rng_id
        _GUARD.reset()

    obs_bits, prev_seen = _update_guard(obs_arr)
    x = torch.as_tensor(obs_arr, dtype=torch.float32).unsqueeze(0)
    logits, _ = _MODEL(x)
    proposed = ACTIONS[int(torch.argmax(logits, dim=1).item())]
    action = _override_action(proposed, obs_bits, prev_seen)
    _GUARD.last_action = action
    return action
