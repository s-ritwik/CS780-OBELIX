# submission_d3_router_w5_guard_t078

Contents:

- `agent.py`: exact submission agent.
- `weights.pth`: exact bundled wall/no-wall/router checkpoint used by the submission.
- `rebuild_exact.sh`: restores `agent.py` and `weights.pth` from the canonical repo copy.
- `train_recipe.sh`: reproduces the learned components used by the router-guard submission and then restores the exact final bundle.
- `code/`: copied trainer/source scripts used by this policy family.

This submission is a packed mixture of four learned components:

- `wall`: recurrent asymmetric PPO wall actor (`wall_d3_random_asym_v1.pth`)
- `nowall`: random-seed-focused no-wall PPO branch (`nowall_d3_ppo_random_ft_v1.pth`)
- `nowall_v1`: strong local no-wall PPO branch (`nowall_d3_ppo_v3.pth`)
- `router`: cost-sensitive wall/no-wall router trained on 20-step probe statistics and then deployed with a more conservative runtime threshold (`0.78`)

The guard logic itself is in `agent.py`. The bundled router checkpoint is based on the `wall_weight=5` cost-sensitive probe model and then packaged with the stricter threshold used by the submission.
