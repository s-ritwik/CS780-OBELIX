from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch


HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rnn_model import ACTIONS, ACTION_DIM, RecurrentConfig, build_input_tensor, load_checkpoint


_MODEL = None
_CONFIG: Optional[RecurrentConfig] = None
_STATE_BY_RNG: dict[int, dict] = {}
_STOCHASTIC = os.environ.get("OBELIX_STOCHASTIC", "0") != "0"


def _checkpoint_path() -> str:
    override = os.environ.get("OBELIX_WEIGHTS")
    if override:
        return override if os.path.isabs(override) else os.path.join(HERE, override)
    return os.path.join(HERE, "teacher_rnn_best.pth")


def _load_once() -> None:
    global _MODEL, _CONFIG
    if _MODEL is not None and _CONFIG is not None:
        return
    model, config, _ = load_checkpoint(_checkpoint_path(), device=torch.device("cpu"))
    _MODEL = model
    _CONFIG = config


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()

    obs_arr = np.asarray(obs, dtype=np.float32)
    rng_id = id(rng)
    state = _STATE_BY_RNG.get(rng_id)
    if state is None:
        state = {
            "hidden": None,
            "prev_action": torch.zeros((1, 1, ACTION_DIM), dtype=torch.float32),
            "steps": 0,
        }
        _STATE_BY_RNG[rng_id] = state

    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32).view(1, 1, -1)
    step_frac = torch.full(
        (1, 1, 1),
        float(state["steps"]) / max(1.0, float(_CONFIG.max_steps)),
        dtype=torch.float32,
    )
    inp = build_input_tensor(
        obs_t,
        prev_action_one_hot=state["prev_action"],
        step_frac=step_frac,
        config=_CONFIG,
    )
    logits, next_hidden = _MODEL(inp, state["hidden"])
    logits = logits[:, -1]

    if _STOCHASTIC:
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64, copy=False)
        probs /= probs.sum()
        action_idx = int(rng.choice(len(ACTIONS), p=probs))
    else:
        action_idx = int(torch.argmax(logits, dim=1).item())

    state["hidden"] = next_hidden
    state["prev_action"].zero_()
    state["prev_action"][0, 0, action_idx] = 1.0
    state["steps"] = min(int(_CONFIG.max_steps), int(state["steps"]) + 1)
    return ACTIONS[action_idx]
