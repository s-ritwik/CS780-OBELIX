#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source /home/rycker/src/anaconda3/etc/profile.d/conda.sh
conda activate torch
cd "${PROJECT_DIR}"

# Phase 1: no-wall PPO pretraining.
python CS780-OBELIX/PPO/train_ppo.py \
  --env_backend torch_vec \
  --env_device cuda \
  --device cuda \
  --out CS780-OBELIX/ppo_lab/nowall_fixed2m.pth \
  --num_envs 256 \
  --rollout_steps 128 \
  --total_env_steps 2097152 \
  --max_steps 300 \
  --difficulty 0 \
  --hidden_dims 128 64 \
  --lr 0.0003 \
  --ent_coef 0.005 \
  --fw_bias_init 1.0 \
  --rec_enco \
  --schedule fixed \
  --seed 0 \
  --log_interval 524288

python CS780-OBELIX/PPO/train_ppo.py \
  --env_backend torch_vec \
  --env_device cuda \
  --device cuda \
  --out CS780-OBELIX/ppo_lab/nowall_fixed8m.pth \
  --init_checkpoint CS780-OBELIX/ppo_lab/nowall_fixed2m.pth \
  --num_envs 256 \
  --rollout_steps 128 \
  --total_env_steps 6291456 \
  --max_steps 300 \
  --difficulty 0 \
  --hidden_dims 128 64 \
  --lr 0.0002 \
  --ent_coef 0.003 \
  --fw_bias_init 1.0 \
  --rec_enco \
  --schedule fixed \
  --seed 0 \
  --log_interval 1572864

# Phase 2: wall asymmetric PPO actor.
python CS780-OBELIX/assym_ppo/train_asym_ppo.py \
  --out CS780-OBELIX/assym_ppo/wall_tuned_v1.pth \
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

# Phase 3: exact submission rebuild.
"${SCRIPT_DIR}/rebuild_exact.sh"
