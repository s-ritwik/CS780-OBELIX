#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source /home/rycker/src/anaconda3/etc/profile.d/conda.sh
conda activate torch
cd "${PROJECT_DIR}"

# Phase 1: difficulty-3 no-wall PPO branch.
python CS780-OBELIX/PPO/train_ppo.py \
  --env_backend torch_vec \
  --env_device cuda \
  --device cuda \
  --out CS780-OBELIX/ppo_lab/nowall_d3_ppo_v1.pth \
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

python CS780-OBELIX/PPO/train_ppo.py \
  --env_backend torch_vec \
  --env_device cuda \
  --device cuda \
  --out CS780-OBELIX/ppo_lab/nowall_d3_ppo_v3.pth \
  --init_checkpoint CS780-OBELIX/ppo_lab/nowall_d3_ppo_v1.pth \
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

# Phase 2: wall asymmetric PPO actor, reused by the macro submission.
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

# Phase 3: legal-observation macro search.
python final_results/submission_d3_wall_macro_51/code/search_probe_sequences.py \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --length 4 \
  --difficulty 3 \
  --wall \
  --max_steps 2000

python final_results/submission_d3_wall_macro_51/code/search_prefix_with_agent.py \
  --agent_file CS780-OBELIX/d3_wall_search/submission_d3_wall_macro_51/agent.py \
  --seed 0 \
  --length 4 \
  --limit 20

# Phase 4: exact submission rebuild.
"${SCRIPT_DIR}/rebuild_exact.sh"
