from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_DIM = len(ACTIONS)
RAW_OBS_DIM = 18
ENCODED_OBS_DIM = 32


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


@dataclass
class RecurrentConfig:
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    include_step_frac: bool = True
    include_prev_action: bool = True
    max_steps: int = 300

    @property
    def input_dim(self) -> int:
        dim = ENCODED_OBS_DIM
        if self.include_prev_action:
            dim += ACTION_DIM
        if self.include_step_frac:
            dim += 1
        return dim


class RecurrentActor(nn.Module):
    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size=config.input_dim,
            hidden_size=int(config.hidden_size),
            num_layers=int(config.num_layers),
            batch_first=True,
            dropout=float(config.dropout) if int(config.num_layers) > 1 else 0.0,
        )
        self.policy_head = nn.Linear(int(config.hidden_size), ACTION_DIM)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, next_hidden = self.rnn(x, hidden)
        logits = self.policy_head(out)
        return logits, next_hidden

    def step(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, next_hidden = self.forward(x, hidden)
        return logits[:, -1], next_hidden


def build_input_tensor(
    obs: torch.Tensor,
    *,
    prev_action_one_hot: torch.Tensor | None = None,
    step_frac: torch.Tensor | None = None,
    config: RecurrentConfig,
) -> torch.Tensor:
    pieces = [encode_obs_tensor(obs)]
    if config.include_prev_action:
        if prev_action_one_hot is None:
            prev_action_one_hot = torch.zeros(
                (*obs.shape[:-1], ACTION_DIM),
                dtype=torch.float32,
                device=obs.device,
            )
        pieces.append(prev_action_one_hot.to(dtype=torch.float32, device=obs.device))
    if config.include_step_frac:
        if step_frac is None:
            step_frac = torch.zeros((*obs.shape[:-1], 1), dtype=torch.float32, device=obs.device)
        pieces.append(step_frac.to(dtype=torch.float32, device=obs.device))
    return torch.cat(pieces, dim=-1)


def save_checkpoint(
    path: str,
    *,
    model: RecurrentActor,
    config: RecurrentConfig,
    args,
    best_eval: float,
) -> None:
    import os

    payload = {
        "state_dict": model.state_dict(),
        "actions": ACTIONS,
        "model_type": "recurrent_bc",
        "recurrent_config": asdict(config),
        "best_eval": float(best_eval),
        "config": vars(args),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> tuple[RecurrentActor, RecurrentConfig, dict]:
    raw = torch.load(path, map_location=device)
    if not isinstance(raw, dict) or "state_dict" not in raw:
        raise RuntimeError(f"Unsupported recurrent checkpoint format: {path}")
    cfg = RecurrentConfig(**raw.get("recurrent_config", {}))
    model = RecurrentActor(cfg).to(device)
    model.load_state_dict(raw["state_dict"], strict=True)
    model.eval()
    return model, cfg, raw


def sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, ACTION_DIM)
    flat_targets = targets.reshape(-1)
    flat_mask = valid_mask.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    loss = loss * flat_mask.to(dtype=loss.dtype)
    return loss.sum() / flat_mask.to(dtype=loss.dtype).sum().clamp_min(1.0)
