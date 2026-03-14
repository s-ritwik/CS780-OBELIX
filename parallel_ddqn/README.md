# Parallel DDQN Trainer (OBELIX)

This folder contains a subprocess-parallel DDQN trainer for OBELIX.

## Files

- `train_ddqn_parallel.py`: training entry point.
- `parallel_env.py`: vectorized OBELIX wrapper over multiple worker processes.
- `ddqn_model.py`: shared Q-network definition.
- `submission_agent_template.py`: standalone `agent.py` template for submission.

## Train (Level 1 / static box)

Run from `CS780-OBELIX/parallel_ddqn`:

```bash
python train_ddqn_parallel.py \
  --env_backend numpy \
  --difficulty 0 \
  --num_envs 64 \
  --max_steps 2000 \
  --hidden_dims 128 64 \
  --total_env_steps 2000000 \
  --batch_size 4096 \
  --out weights.pth
```

Recommended:

- Increase `--num_envs` until CPU utilization saturates.
- Keep `difficulty=0` while focusing on Problem 1.
- Use `--device cuda` (or `auto`) for GPU updates.

## Use torch environment backend

To run with `obelix_torch.py` instead of `obelix.py`:

```bash
python train_ddqn_parallel.py \
  --env_backend torch \
  --env_device cpu \
  --difficulty 0 \
  --num_envs 64 \
  --max_steps 2000 \
  --hidden_dims 128 64 \
  --total_env_steps 2000000 \
  --batch_size 4096 \
  --device cuda \
  --out weights.pth
```

Notes:

- `--env_backend numpy` uses `CS780-OBELIX/obelix.py`.
- `--env_backend torch` uses `CS780-OBELIX/obelix_torch.py`.
- `--env_backend torch_vec` uses `OBELIXVectorized` from `CS780-OBELIX/obelix_torch.py` (single-process batched).
- You can override either with `--obelix_py /path/to/file.py`.
- `--env_device` is only for the torch env backend (`cpu`, `cuda`, or `auto`).

## Use single-process vectorized torch backend (recommended for high `num_envs`)

```bash
python train_ddqn_parallel.py \
  --env_backend torch_vec \
  --env_device cuda \
  --difficulty 0 \
  --num_envs 512 \
  --max_steps 2000 \
  --hidden_dims 128 64 \
  --total_env_steps 4000000 \
  --batch_size 8192 \
  --device cuda \
  --out weights.pth
```

## Build submission

1. Copy `submission_agent_template.py` as `agent.py`.
2. Place trained weights as `weights.pth` next to `agent.py`.
3. Zip only `agent.py` and `weights.pth`.

Codabench runs inference only (CPU), not this trainer.
