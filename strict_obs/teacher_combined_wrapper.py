from __future__ import annotations

import importlib.util
import os
from types import ModuleType
from typing import Optional

import numpy as np
import torch


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

_TEACHER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "assym_ppo",
    "submission_switch_d3_v3base",
    "agent.py",
)
_DEVICE = torch.device("cpu")
_TEACHER_MODULE: Optional[ModuleType] = None
_WALL_MODEL: Optional[torch.nn.Module] = None
_NOWALL_MODEL: Optional[torch.nn.Module] = None
_FEATURE_CONFIG = None


def _load_teacher_module() -> ModuleType:
    global _TEACHER_MODULE
    if _TEACHER_MODULE is None:
        spec = importlib.util.spec_from_file_location("strict_obs_teacher_shared", _TEACHER_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import teacher from {_TEACHER_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TEACHER_MODULE = module
    return _TEACHER_MODULE


def _load_shared_models() -> tuple[torch.nn.Module, torch.nn.Module, object]:
    global _WALL_MODEL, _NOWALL_MODEL, _FEATURE_CONFIG
    module = _load_teacher_module()
    if _WALL_MODEL is not None and _NOWALL_MODEL is not None and _FEATURE_CONFIG is not None:
        return _WALL_MODEL, _NOWALL_MODEL, _FEATURE_CONFIG

    bundle = module._load_checkpoint(os.path.join(os.path.dirname(_TEACHER_PATH), "weights.pth"))
    wall_ckpt = bundle["wall"]
    nowall_ckpt = bundle["nowall"]

    feature_config = module.FeatureConfig(**wall_ckpt.get("feature_config", {}))
    wall_model = module.ActorOnly(
        actor_dim=feature_config.feature_dim,
        actor_hidden_dims=tuple(int(x) for x in wall_ckpt.get("actor_hidden_dims", [256, 128])),
        fw_bias_init=0.0,
    ).to(_DEVICE)
    wall_model.actor_backbone.load_state_dict(wall_ckpt["actor_backbone_state_dict"], strict=True)
    wall_model.policy_head.load_state_dict(wall_ckpt["policy_head_state_dict"], strict=True)
    wall_model.eval()

    if isinstance(nowall_ckpt, dict) and "state_dict" in nowall_ckpt and isinstance(nowall_ckpt["state_dict"], dict):
        nowall_state_dict = nowall_ckpt["state_dict"]
        hidden_dims = tuple(int(x) for x in nowall_ckpt.get("hidden_dims", module._infer_hidden_dims(nowall_state_dict)))
        use_rec_encoder = nowall_ckpt.get("use_rec_encoder")
    else:
        nowall_state_dict = nowall_ckpt
        hidden_dims = module._infer_hidden_dims(nowall_state_dict)
        use_rec_encoder = None
    if use_rec_encoder is None:
        use_rec_encoder = module.infer_use_rec_encoder_from_state_dict(nowall_state_dict)

    nowall_model = module.RecActorCritic(
        hidden_dims=hidden_dims,
        use_rec_encoder=bool(use_rec_encoder),
    ).to(_DEVICE)
    nowall_model.load_state_dict(nowall_state_dict, strict=True)
    nowall_model.eval()

    _WALL_MODEL = wall_model
    _NOWALL_MODEL = nowall_model
    _FEATURE_CONFIG = feature_config
    return _WALL_MODEL, _NOWALL_MODEL, _FEATURE_CONFIG


class _WallState:
    def __init__(self) -> None:
        module = _load_teacher_module()
        wall_model, _, feature_config = _load_shared_models()
        self._module = module
        self._wall_model = wall_model
        self._tracker = module.FeatureTracker(num_envs=1, config=feature_config, device=_DEVICE)
        self._pending_action: Optional[int] = None
        self._initialized = False

    def reset(self) -> None:
        self._pending_action = None
        self._initialized = False

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> str:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if not self._initialized:
            self._tracker.reset_all(obs_arr[None, :])
            self._initialized = True
        elif self._pending_action is not None:
            self._tracker.post_step(
                actions=torch.tensor([self._pending_action], dtype=torch.long, device=_DEVICE),
                next_obs=obs_arr[None, :],
                dones=np.asarray([False]),
            )
        logits = self._wall_model(self._tracker.features())
        action_idx = int(torch.argmax(logits, dim=1).item())
        self._pending_action = action_idx
        return ACTIONS[action_idx]


class _NoWallState:
    def __init__(self) -> None:
        _, nowall_model, _ = _load_shared_models()
        self._nowall_model = nowall_model

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> str:
        x = torch.tensor(obs, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
        logits, _ = self._nowall_model(x)
        action_idx = int(torch.argmax(logits, dim=1).item())
        return ACTIONS[action_idx]


class HandmadeController:
    def __init__(self) -> None:
        module = _load_teacher_module()
        self._module = module
        self._wall = _WallState()
        self._nowall = _NoWallState()
        self._handmade = module.ExactBoostHandmadeController()
        self.reset(0)

    def reset(self, episode_key: int) -> None:
        self._episode_key = int(episode_key)
        self._wall.reset()
        self._handmade.reset(self._episode_key)
        self._step_count = 0
        self._contact_count = 0
        self._blind_count = 0
        self._front_total = 0
        self._switched = False
        self._decided = False
        self._boost_initialized = False
        self._boost_active = False
        self._boost_step = 0
        self._boost_stuck = 0
        self._boost_seen_contact = False

    def _update_probe_stats(self, obs_arr: np.ndarray) -> None:
        front_count = int(np.sum(obs_arr[4:12] > 0.5))
        front_near = int(np.sum(obs_arr[5:12:2] > 0.5))
        self._contact_count += int(front_near >= 1 or front_count >= 4)
        self._blind_count += int(np.sum(obs_arr[:16]) == 0.0)
        self._front_total += front_count

    def _submission_policy(self, obs_arr: np.ndarray) -> str:
        if self._step_count < self._module.PROBE_STEPS:
            self._update_probe_stats(obs_arr)
            action = self._wall.act(obs_arr)
        else:
            if not self._decided:
                self._switched = (
                    (self._contact_count <= self._module.CONTACT_THRESHOLD)
                    and (self._blind_count >= self._module.BLIND_THRESHOLD)
                    and (self._front_total >= self._module.FRONT_TOTAL_THRESHOLD)
                )
                self._decided = True
            action = self._nowall.act(obs_arr) if self._switched else self._wall.act(obs_arr)
        self._step_count += 1
        return action

    def _init_boost(self, obs_arr: np.ndarray) -> None:
        bits = tuple((obs_arr > 0.5).astype(np.int8, copy=False).tolist())
        self._boost_active = bits in self._module.BOOST_PATTERNS
        self._boost_step = 0
        self._boost_stuck = 0
        self._boost_seen_contact = False
        self._boost_initialized = True

    def act(self, obs: np.ndarray) -> str:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if not self._boost_initialized:
            self._init_boost(obs_arr)

        if self._boost_active:
            front_count = int(np.sum(obs_arr[4:12] > 0.5))
            if obs_arr[17] > 0.5:
                self._boost_stuck += 1
            if obs_arr[16] > 0.5 or front_count > 0:
                self._boost_seen_contact = True
            if (
                (self._boost_stuck >= self._module.BOOST_STUCK_SWITCH and not self._boost_seen_contact)
                or (self._boost_step >= self._module.BOOST_STEPS and not self._boost_seen_contact)
            ):
                self._boost_active = False
            else:
                self._boost_step += 1
                return self._handmade.act(obs_arr)

        return self._submission_policy(obs_arr)
