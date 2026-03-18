# Asymmetric PPO

This directory contains a CUDA-friendly asymmetric PPO trainer for OBELIX:

- actor gets only deploy-time observation history/features
- critic gets actor features plus privileged simulator state from `OBELIXVectorized`
- training runs against `obelix_torch.py` in a single-process batched environment

Privileged critic inputs include:

- robot position and heading
- box position, relative direction, and distance
- exact push / visibility / stuck state
- step count, wall flag, difficulty, and box velocity
- exact boundary margins

## Train

```bash
python assym_ppo/train_asym_ppo.py \
  --env_device auto \
  --device auto \
  --num_envs 512 \
  --total_env_steps 4000000 \
  --max_steps 300 \
  --out assym_ppo/weights_best.pth
```

## Resume

```bash
python assym_ppo/train_asym_ppo.py \
  --load assym_ppo/weights_best.pth \
  --out assym_ppo/weights_best.pth
```

## Use The Trained Actor

```bash
python visualize_agent.py --agent_file assym_ppo/agent.py --difficulty 0 --max_steps 300
```

The saved checkpoint contains both full asymmetric model weights for training resume and actor-only weights for deployment through `agent.py`.
