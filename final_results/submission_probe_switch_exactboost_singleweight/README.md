# submission_probe_switch_exactboost_singleweight

Contents:

- `agent.py`: exact submission agent.
- `weights.pth`: exact bundled wall/no-wall checkpoint used by the submission.
- `switch.zip`: final packaged submission archive.
- `weights_rebuilt.pth`: rebuilt bundle generated from the saved component checkpoints already present in this repo.
- `switch_rebuilt.zip`: rebuilt archive using `agent.py` plus `weights_rebuilt.pth`.
- `rebuild_exact.sh`: restores `agent.py` from `switch.zip`, rebuilds `weights_rebuilt.pth`, checks semantic equality against `weights.pth`, and writes `switch_rebuilt.zip`.
- `train_recipe.sh`: documents the training pipeline for the two learned components.
- `code/`: copied trainer/source scripts used by this policy family.

This submission is composed of:

- a wall actor trained with asymmetric PPO,
- a no-wall actor trained with PPO on the CUDA vectorized environment,
- the probe-switch/exact-boost control logic in `agent.py`.

The final bundled `weights.pth` is a dictionary with two entries:

- `wall`: checkpoint payload corresponding to `CS780-OBELIX/assym_ppo/wall_tuned_v1_final.pth`
- `nowall`: checkpoint payload corresponding to `CS780-OBELIX/ppo_lab/nowall_fixed8m.pth`

Checkpoint rebuild command:

```bash
cd /home/rycker/study/CS780\ RL/CS780-OBELIX/final_results/submission_probe_switch_exactboost_singleweight
./rebuild_exact.sh
```

`weights_rebuilt.pth` is semantically identical to `weights.pth` after loading, but not byte-identical on disk because it is repacked with a fresh `torch.save`.
