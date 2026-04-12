# Final Submission Artifacts

This directory collects the three submissions for final phase capstone:

- `submission_probe_switch_exactboost_singleweight`
- `submission_d3_wall_macro_51`
- `submission_d3_router_w5_guard_t078_v1`

Each subdirectory contains:

- `agent.py`: the exact submission agent file.
- `weights.pth`: the exact submission weights bundle.
- the final packaged submission archive corresponding to that folder:
  - `switch.zip`
  - `wall_macro.zip`
  - `router_guard.zip`

- `train_recipe.sh`: the training/assembly recipe that produced the learned components used by the submission.
- `code/`: copies of the training scripts used in that policy family.
- `verify_seed_0_9.sh`: evaluates either the final artifacts or the rebuilt artifacts on seeds `0..9`.
- `verify_seed_0_9.py`: verification harness used by the shell wrapper.