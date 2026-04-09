# submission_d3_wall_macro_51

Contents:

- `agent.py`: exact submission agent with the inlined macro layer.
- `weights.pth`: exact bundled wall/no-wall checkpoint used by the submission.
- `rebuild_exact.sh`: restores `agent.py` and `weights.pth` from the canonical repo copy.
- `train_recipe.sh`: reproduces the learned components and shows the macro-search stage used to derive the final agent logic.
- `code/`: copied trainer/search scripts used by this policy family.

This submission reuses:

- the same wall actor family as the exact-boost policy (`wall_tuned_v1.pth`),
- a stronger difficulty-3 no-wall PPO branch (`nowall_d3_ppo_v3.pth`),
- and a manually inlined macro layer keyed only by legal 18-bit observations.

The macro sequences themselves live in `agent.py`; they were derived offline with legal-observation search scripts and then written directly into the submission agent.
