# submission_d3_router_w5_guard_t078_v1

Contents:

- `agent.py`: exact submission agent.
- `weights.pth`: exact bundled wall/no-wall/router checkpoint used by the submission.
- `router_guard.zip`: final packaged submission archive.
- `weights_rebuilt.pth`: rebuilt bundle generated from the saved component checkpoints already present in this repo.
- `router_guard_rebuilt.zip`: rebuilt archive using `agent.py` plus `weights_rebuilt.pth`.
- `rebuild_exact.sh`: restores `agent.py` from `router_guard.zip`, rebuilds `weights_rebuilt.pth`, checks semantic equality against `weights.pth`, and writes `router_guard_rebuilt.zip`.
- `train_recipe.sh`: documents the training pipeline for the learned components used by the router-guard submission.
- `code/`: copied trainer/source scripts used by this policy family.

This submission is a packed mixture of four learned components:

- `wall`: recurrent asymmetric PPO wall actor (`wall_d3_random_asym_v1.pth`)
- `nowall`: random-seed-focused no-wall PPO branch (`nowall_d3_ppo_random_ft_v1.pth`)
- `nowall_v1`: strong local no-wall PPO branch (`nowall_d3_ppo_v1.pth`)
- `router`: cost-sensitive wall/no-wall router trained as `router_d3_cost_sensitive_probe20_w5.pth` and rebuilt with the deployed runtime threshold `0.78`

The guard logic itself is in `agent.py`. The bundled router checkpoint is based on the `wall_weight=5` cost-sensitive probe model and then packaged with the stricter threshold used by the submission.

Checkpoint rebuild command:

```bash
cd /home/rycker/study/CS780\ RL/CS780-OBELIX/final_results/submission_d3_router_w5_guard_t078_v1
./rebuild_exact.sh
```

`weights_rebuilt.pth` is semantically identical to `weights.pth` after loading, but not byte-identical on disk because it is repacked with a fresh `torch.save`.
