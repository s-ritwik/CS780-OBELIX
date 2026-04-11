# submission_probe_switch_exactboost_singleweight

Contents:

- `agent.py`: exact submission agent.
- `weights.pth`: exact bundled wall/no-wall checkpoint used by the submission.
- `submission_probe_switch_exactboost_singleweight.zip`: packaged submission archive.
- `rebuild_exact.sh`: restores `agent.py` and `weights.pth` from the canonical repo copy.
- `train_recipe.sh`: reproduces the two learned components and then rebuilds the submission bundle.
- `code/`: copied trainer/source scripts used by this policy family.

This submission is composed of:

- a wall actor trained with asymmetric PPO,
- a no-wall actor trained with PPO on the CUDA vectorized environment,
- the probe-switch/exact-boost control logic in `agent.py`.

The final bundled `weights.pth` is a dictionary with two entries:

- `wall`: checkpoint payload corresponding to `CS780-OBELIX/assym_ppo/wall_tuned_v1.pth`
- `nowall`: checkpoint payload corresponding to `CS780-OBELIX/ppo_lab/nowall_fixed8m.pth`

Use `./rebuild_exact.sh` if you want the exact submission artifact back in place.
