# Paper Algorithm

This directory implements the strongest method reported in the AAAI'91 OBELIX paper:

- subsumption control with three learned behaviors: `find`, `push`, `unwedge`
- per-behavior Q-learning
- clustering-based state generalization
- extra stabilization for this repo's OBELIX variant:
  better `find` shaping, clipped env-reward mixing, and best-checkpoint saving
  plus a runtime/training unwedge routine that uses the `stuck` bit explicitly

## Files

- `controller.py`: paper-style clustered Q-learning controller and serializer
- `train_paper_algo.py`: trainer using `OBELIXVectorized` from `obelix_torch.py`
- `agent.py`: CPU-only inference policy that loads `paper_policy.json`

## Train

```bash
python paper_algo/train_paper_algo.py \
  --num_envs 64 \
  --total_env_steps 500000 \
  --env_device cpu \
  --out paper_algo/paper_policy.json
```

## Train On GPU

Fast path:

```bash
python paper_algo/train_gpu.py
```

That wrapper expands to a CUDA-first launch with larger defaults:

```bash
python paper_algo/train_paper_algo.py \
  --env_device auto \
  --num_envs 256 \
  --total_env_steps 2000000 \
  --log_interval 50000 \
  --out paper_algo/paper_policy.json
```

You can still override anything:

```bash
python paper_algo/train_gpu.py \
  --difficulty 3 \
  --wall_obstacles \
  --num_envs 512 \
  --total_env_steps 4000000
```

Useful flags:

- `--difficulty 0|2|3`
- `--wall_obstacles`
- `--box_speed 2`
- `--env_device auto|cpu|cuda|cuda:N`
- `--env_reward_scale`, `--env_reward_clip`
- `--save_best`, `--best_checkpoint_min_episodes`
- `--match_threshold`, `--q_match_delta`, `--cluster_distance_threshold`

Checkpoint behavior:

- `--out .../paper_policy.json` stores the best checkpoint by default
- the final controller state is saved as `paper_policy.last.json`

## Use For Evaluation

`paper_algo/agent.py` expects `paper_policy.json` to be next to it.

If you submit this policy, include:

- `agent.py`
- `paper_policy.json`
