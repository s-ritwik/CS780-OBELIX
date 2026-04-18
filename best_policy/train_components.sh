#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/generated"

mkdir -p "${OUT_DIR}"

set +u
source /home/rycker/src/anaconda3/etc/profile.d/conda.sh
conda activate torch
set -u

cd "${SCRIPT_DIR}"

# Phase 1: difficulty-3 no-wall PPO branch.
python "${SCRIPT_DIR}/code/train_ppo.py" \
  --env_backend torch_vec \
  --env_device cuda \
  --device cuda \
  --out "${OUT_DIR}/nowall_d3_ppo_v1.pth" \
  --num_envs 256 \
  --rollout_steps 128 \
  --total_env_steps 2000000 \
  --max_steps 2000 \
  --difficulty 3 \
  --box_speed 2 \
  --hidden_dims 128 64 \
  --lr 0.0003 \
  --ent_coef 0.01 \
  --update_epochs 5 \
  --minibatch_size 4096 \
  --schedule adaptive \
  --fw_bias_init 1.0 \
  --rec_enco \
  --seed 0 \
  --log_interval 250000

python "${SCRIPT_DIR}/code/train_ppo.py" \
  --env_backend torch_vec \
  --env_device cuda \
  --device cuda \
  --out "${OUT_DIR}/nowall_d3_ppo_v3.pth" \
  --init_checkpoint "${OUT_DIR}/nowall_d3_ppo_v1.pth" \
  --num_envs 256 \
  --rollout_steps 128 \
  --total_env_steps 4194304 \
  --max_steps 2000 \
  --difficulty 3 \
  --box_speed 2 \
  --hidden_dims 128 64 \
  --lr 0.0002 \
  --ent_coef 0.005 \
  --update_epochs 6 \
  --minibatch_size 8192 \
  --schedule fixed \
  --fw_bias_init 1.0 \
  --rec_enco \
  --seed 41 \
  --log_interval 262144

# Phase 2: wall asymmetric PPO actor.
python "${SCRIPT_DIR}/code/train_asym_ppo.py" \
  --out "${OUT_DIR}/wall_tuned_v1.pth" \
  --env_device cuda \
  --device cuda \
  --num_envs 512 \
  --rollout_steps 128 \
  --total_env_steps 1000000 \
  --max_steps 300 \
  --difficulty 0 \
  --wall_obstacles \
  --obs_stack 12 \
  --action_hist 6 \
  --actor_hidden_dims 512 256 128 \
  --critic_hidden_dims 1024 512 256 \
  --lr 0.00015 \
  --ent_coef 0.002 \
  --reward_scale 5.0 \
  --reward_clip 100.0 \
  --visible_bonus 0.5 \
  --ir_bonus 1.5 \
  --stuck_extra_penalty 2.0 \
  --approach_progress_bonus 35.0 \
  --alignment_bonus 0.75 \
  --push_progress_bonus 120.0 \
  --blind_turn_penalty 0.15 \
  --warm_start_expert eval300 \
  --warm_start_episodes 20 \
  --warm_start_epochs 4 \
  --eval_runs 10 \
  --seed 0

# Pack the freshly trained family checkpoints into a local bundle.
python - <<'PY' "${OUT_DIR}"
import sys
from pathlib import Path

import torch

out_dir = Path(sys.argv[1])
bundle = {
    "wall": torch.load(out_dir / "wall_tuned_v1.pth", map_location="cpu", weights_only=False),
    "nowall": torch.load(out_dir / "nowall_d3_ppo_v3.pth", map_location="cpu", weights_only=False),
}
torch.save(bundle, out_dir / "weights_from_training.pth")
print(out_dir / "weights_from_training.pth")
PY
