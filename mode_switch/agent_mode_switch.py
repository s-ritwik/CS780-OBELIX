from __future__ import annotations

import importlib.util
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
INPUT_DIM = 18 + len(ACTIONS)
LABEL_STATIC = 0
LABEL_MOVE_NOWALL = 1
LABEL_MOVE_WALL = 2

HERE = os.path.dirname(__file__)
REPO_DIR = os.path.dirname(HERE)
EXPERT_PATH = os.path.join(REPO_DIR, "ppo_lab", "expert_conservative.py")
NOWALL_PATH = os.path.join(REPO_DIR, "ppo_lab", "agent_nowall_submission.py")
SAFE_PATH = os.path.join(REPO_DIR, "ppo_lab", "agent_seenmask_submission.py")
CLASSIFIER_PATH = os.path.join(HERE, "mode_classifier_gru.pth")

_EXPERT = None
_NOWALL = None
_SAFE = None
_CLASSIFIER = None
_PROBE_STEPS = 64


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 96,
        num_layers: int = 1,
        rnn_type: str = "gru",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rnn_type = str(rnn_type).lower()
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=int(input_dim),
                hidden_size=int(hidden_dim),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            )
        else:
            self.rnn = nn.GRU(
                input_size=int(input_dim),
                hidden_size=int(hidden_dim),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            )
        self.head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 3),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)
        if self.rnn_type == "lstm":
            last = hidden[0][-1]
        else:
            last = hidden[-1]
        return self.head(last)


def _load_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_once() -> None:
    global _EXPERT, _NOWALL, _SAFE, _CLASSIFIER, _PROBE_STEPS
    if _EXPERT is None:
        _EXPERT = _load_module(EXPERT_PATH, "mode_switch_expert")
    if _NOWALL is None:
        _NOWALL = _load_module(NOWALL_PATH, "mode_switch_nowall")
    if _SAFE is None:
        _SAFE = _load_module(SAFE_PATH, "mode_switch_safe")
    if _CLASSIFIER is None:
        raw = torch.load(CLASSIFIER_PATH, map_location="cpu")
        model = SequenceClassifier(
            hidden_dim=int(raw.get("hidden_dim", 96)),
            num_layers=int(raw.get("num_layers", 1)),
            rnn_type=str(raw.get("rnn_type", "gru")),
            dropout=float(raw.get("dropout", 0.0)),
        )
        model.load_state_dict(raw["state_dict"], strict=True)
        model.eval()
        _CLASSIFIER = model
        _PROBE_STEPS = int(raw.get("probe_steps", 64))


class EpisodeState:
    def __init__(self) -> None:
        self.rng_id: Optional[int] = None
        self.step_idx = 0
        self.prev_action = np.zeros((len(ACTIONS),), dtype=np.float32)
        self.seq = np.zeros((256, INPUT_DIM), dtype=np.float32)
        self.routed_label: Optional[int] = None

    def reset(self, rng_id: int) -> None:
        self.rng_id = rng_id
        self.step_idx = 0
        self.prev_action.fill(0.0)
        self.seq.fill(0.0)
        self.routed_label = None


_STATE = EpisodeState()


@torch.no_grad()
def _route_label() -> int:
    length = min(_STATE.step_idx, _PROBE_STEPS)
    x = torch.as_tensor(_STATE.seq[:length][None, :, :], dtype=torch.float32)
    lengths = torch.tensor([length], dtype=torch.long)
    logits = _CLASSIFIER(x, lengths)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    label = int(np.argmax(probs))
    if label == LABEL_STATIC and probs[LABEL_STATIC] < 0.55:
        label = LABEL_MOVE_WALL if probs[LABEL_MOVE_WALL] >= probs[LABEL_MOVE_NOWALL] else LABEL_MOVE_NOWALL
    return label


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    obs_arr = np.asarray(obs, dtype=np.float32)
    rng_id = id(rng)
    if _STATE.rng_id != rng_id:
        _STATE.reset(rng_id)

    if _STATE.step_idx < _STATE.seq.shape[0]:
        _STATE.seq[_STATE.step_idx, :18] = obs_arr
        _STATE.seq[_STATE.step_idx, 18:] = _STATE.prev_action

    if _STATE.routed_label is None and _STATE.step_idx >= _PROBE_STEPS:
        _STATE.routed_label = _route_label()

    if _STATE.routed_label == LABEL_MOVE_NOWALL:
        action = _NOWALL.policy(obs_arr, rng)
    elif _STATE.routed_label == LABEL_MOVE_WALL:
        action = _SAFE.policy(obs_arr, rng)
    else:
        action = _EXPERT.policy(obs_arr, rng)

    action_idx = ACTIONS.index(action)
    _STATE.prev_action.fill(0.0)
    _STATE.prev_action[action_idx] = 1.0
    _STATE.step_idx += 1
    return action
