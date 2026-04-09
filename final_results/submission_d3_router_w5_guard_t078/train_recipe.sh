#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/rycker/src/anaconda3/etc/profile.d/conda.sh
conda activate torch
cd "${ROOT_DIR}"

# Phase 1: strong local difficulty-3 no-wall branch used as nowall_v1.
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

# Phase 2: random-focused no-wall branch used as nowall.
python CS780-OBELIX/PPO/train_ppo.py \
  --env_backend torch_vec \
  --env_device cuda \
  --device cuda \
  --out CS780-OBELIX/d3_wall_search/nowall_d3_ppo_random_ft_v1.pth \
  --init_checkpoint CS780-OBELIX/ppo_lab/nowall_d3_ppo_v1.pth \
  --num_envs 128 \
  --rollout_steps 256 \
  --total_env_steps 1500000 \
  --max_steps 2000 \
  --difficulty 3 \
  --box_speed 2 \
  --hidden_dims 128 64 \
  --lr 0.0001 \
  --ent_coef 0.015 \
  --update_epochs 4 \
  --minibatch_size 4096 \
  --schedule adaptive \
  --rec_enco \
  --seed 91016 \
  --log_interval 100000

# Phase 3: recurrent wall actor trained with privileged critic/teacher.
python CS780-OBELIX/assym_ppo/train_mixed_recurrent_asym_ppo_v2.py \
  --out CS780-OBELIX/d3_wall_search/wall_d3_random_asym_v1.pth \
  --load CS780-OBELIX/d3_wall_search/mixed_wall_d3_teacherbc_gru_s503.pth \
  --reset_best_eval \
  --scenarios 3:w \
  --num_envs 96 \
  --rollout_steps 128 \
  --seq_len 16 \
  --total_env_steps 1500000 \
  --max_steps 2000 \
  --box_speed 2 \
  --scaling_factor 5 \
  --arena_size 500 \
  --env_device cuda \
  --device cuda \
  --encoder_dims 512 384 256 \
  --critic_hidden_dims 1024 512 256 \
  --rnn_hidden_dim 224 \
  --rnn_layers 1 \
  --rnn_dropout 0.0 \
  --actor_dropout 0.0 \
  --critic_dropout 0.0 \
  --feature_dropout 0.03 \
  --aux_coef 0.08 \
  --aux_hidden_dim 128 \
  --aux_target_mode compact \
  --rnn_type gru \
  --fw_bias_init 1.0 \
  --policy_temperature 1.0 \
  --gamma 0.995 \
  --gae_lambda 0.95 \
  --lr 0.0001 \
  --clip_coef 0.2 \
  --ent_coef 0.01 \
  --vf_coef 0.5 \
  --update_epochs 5 \
  --minibatch_size 8192 \
  --reward_scale 5.0 \
  --reward_clip 100.0 \
  --visible_bonus 0.75 \
  --ir_bonus 1.0 \
  --blind_turn_penalty 0.05 \
  --blind_forward_penalty 0.0 \
  --stuck_extra_penalty 2.0 \
  --approach_progress_bonus 30.0 \
  --alignment_bonus 1.0 \
  --push_progress_bonus 180.0 \
  --gap_progress_bonus 140.0 \
  --gap_alignment_bonus 1.25 \
  --schedule adaptive \
  --desired_kl 0.01 \
  --pose_clip 500.0 \
  --blind_clip 120.0 \
  --stuck_clip 24.0 \
  --contact_clip 24.0 \
  --same_obs_clip 64.0 \
  --wall_hit_clip 24.0 \
  --blind_turn_clip 24.0 \
  --stuck_memory_clip 24.0 \
  --turn_streak_clip 24.0 \
  --forward_streak_clip 24.0 \
  --last_action_hist 6 \
  --heading_bins 8 \
  --teacher_agent_file privileged_teacher \
  --teacher_coef 0.08 \
  --teacher_coef_final 0.02 \
  --teacher_decay_steps 1000000 \
  --teacher_action_prob 0.15 \
  --teacher_action_prob_final 0.02 \
  --teacher_action_prob_decay_steps 1000000 \
  --warm_start_expert privileged_teacher \
  --warm_start_episodes 0 \
  --warm_start_max_attempts 360 \
  --warm_start_mode sequence \
  --warm_start_epochs 0 \
  --warm_start_batch_size 8192 \
  --eval_runs 6 \
  --eval_interval 500000 \
  --log_interval 100000 \
  --seed 92016

# Phase 4: cost-sensitive 20-step router.
python CS780-OBELIX/d3_wall_search/train_cost_sensitive_router.py \
  --out CS780-OBELIX/d3_wall_search/router_d3_cost_sensitive_probe20_w5.pth \
  --seeds 0-399 \
  --extra_seeds 3049,4099,9780,11235,17686,24924,28001,36648,42017,53321,58337,60107,60209,65203,74419,81233,86519,91016,93007,97309,667116,667150,667151 \
  --probe_steps 20 \
  --hidden_dims 128 64 \
  --epochs 1000 \
  --lr 0.001 \
  --wall_weight 5.0 \
  --threshold 0.5 \
  --seed 123 \
  --difficulty 3 \
  --box_speed 2 \
  --max_steps 2000 \
  --scaling_factor 5 \
  --arena_size 500

# Phase 5: exact submission rebuild.
./final_results/submission_d3_router_w5_guard_t078/rebuild_exact.sh
