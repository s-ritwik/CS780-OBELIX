# best_policy

This folder packages the submission agent plus the preserved training and rebuild entrypoints for the `submission_d3_wall_macro_51` policy family. The training and helper scripts are localized so the folder can run on its own.

Files:

- `agent.py`: final submission agent.
- `code/train_ppo.py`: copied PPO trainer, adjusted so it can be launched from this folder.
- `code/train_asym_ppo.py`: copied asymmetric PPO trainer, adjusted so it can be launched from this folder.
- `code/obs_encoder.py`, `code/features.py`, `code/model.py`, `code/privileged.py`, `code/parallel_env.py`: local trainer dependencies copied into the bundle.
- `code/experts/`: local warm-start expert agents and checkpoints used by `train_asym_ppo.py`.
- `code/search_probe_sequences.py`: copied legal-observation macro search helper.
- `code/search_prefix_with_agent.py`: copied prefix search helper.
- `train_components.sh`: reruns the preserved component-family training commands with the same documented hyperparameters.
- `rebuild.sh`: repacks a local weights bundle from `generated/` outputs if present, otherwise reuses the existing local `weights.pth`.
- `checkpoint_provenance.json`: exact checkpoint mapping and saved hyperparameters used by the final bundle.

## What To Run

Train the preserved component family:

```bash
cd /home/rycker/study/CS780\ RL/final_submission_folder/best_policy
./train_components.sh
```

This writes:

- `generated/nowall_d3_ppo_v1.pth`
- `generated/nowall_d3_ppo_v3.pth`
- `generated/wall_tuned_v1.pth`
- `generated/weights_from_training.pth`

Rebuild the local submission bundle:

```bash
cd /home/rycker/study/CS780\ RL/final_submission_folder/best_policy
./rebuild.sh
```

This writes:

- `weights.pth`
- `best_policy.zip`

## Important Detail

`train_components.sh` reproduces the preserved training recipe locally and writes `generated/weights_from_training.pth`.

`rebuild.sh` is local-first:

- if `generated/weights_from_training.pth` exists, it repacks that
- else if `generated/wall_tuned_v1.pth` and `generated/nowall_d3_ppo_v3.pth` exist, it bundles those
- else it repacks the existing local `weights.pth`

`checkpoint_provenance.json` still records the original external checkpoint provenance of the archived final bundle.
