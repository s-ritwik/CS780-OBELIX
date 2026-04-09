# Final Submission Artifacts

This directory collects the three submission variants selected for final review:

- `submission_d3_router_w5_guard_t078`
- `submission_probe_switch_exactboost`
- `submission_d3_wall_macro_51`

Each subdirectory contains:

- `agent.py`: the exact submission agent file.
- `weights.pth`: the exact submission weights bundle.
- `rebuild_exact.sh`: restores the exact submission files from the canonical source directory in this repo.
- `train_recipe.sh`: the training/assembly recipe that produced the learned components used by the submission.
- `code/`: copies of the training scripts used in that policy family.

Important distinction:

- `rebuild_exact.sh` gives back the exact submitted agent/weights already chosen in this repo.
- `train_recipe.sh` documents and launches the original training pipeline for the learned components. Because PPO and the later search stages are stochastic and iterative, retraining from scratch is not guaranteed to be byte-identical to the bundled submission unless you use the supplied rebuilt artifacts.
