# submission_d3_wall_macro_51

Contents:

- `agent.py`: exact submission agent with the inlined macro layer.
- `weights.pth`: exact bundled wall/no-wall checkpoint used by the submission.
- `wall_macro.zip`: final packaged submission archive.
- `weights_rebuilt.pth`: exact semantic rebuild generated from the saved original component checkpoints.
- `wall_macro_rebuilt.zip`: rebuilt archive using `agent.py` plus `weights_rebuilt.pth`; this is the correct choice if you want the same policy as `wall_macro.zip`.
- `rebuild_exact.sh`: restores `agent.py` from `wall_macro.zip`, rebuilds `weights_rebuilt.pth`, checks semantic equality against `weights.pth`, and writes `wall_macro_rebuilt.zip`.
- `train_recipe.sh`: documents the learned-component training pipeline and the separate macro-search stage.
- `code/`: copied trainer/search scripts used by this policy family.

This submission reuses:

- the same wall actor family as the exact-boost policy (`wall_tuned_v1_final.pth`),
- a stronger difficulty-3 no-wall PPO branch (`nowall_d3_ppo_v3.pth`),
- and a manually inlined macro layer keyed only by legal 18-bit observations.

The macro sequences themselves live in `agent.py`; they were derived offline with legal-observation search scripts and then written directly into the submission agent.

Exact rebuild from original saved components:

```bash
cd /home/rycker/study/CS780\ RL/CS780-OBELIX/final_results/submission_d3_wall_macro_51
./rebuild_exact.sh
```

`weights_rebuilt.pth` is semantically identical to `weights.pth` after loading, but not byte-identical on disk because it is repacked with a fresh `torch.save`.
