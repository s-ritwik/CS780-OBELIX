#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_WEIGHTS="${SCRIPT_DIR}/weights.pth"
FINAL_ZIP="${SCRIPT_DIR}/best_policy.zip"
GENERATED_DIR="${SCRIPT_DIR}/generated"

set +u
source /home/rycker/src/anaconda3/etc/profile.d/conda.sh
conda activate torch
set -u

python - <<'PY' "${FINAL_WEIGHTS}" "${FINAL_ZIP}" "${SCRIPT_DIR}/agent.py" "${GENERATED_DIR}"
import hashlib
import sys
import zipfile
from pathlib import Path

import torch

final_weights = Path(sys.argv[1])
final_zip = Path(sys.argv[2])
agent_path = Path(sys.argv[3])
generated_dir = Path(sys.argv[4])

generated_bundle = generated_dir / "weights_from_training.pth"
generated_wall = generated_dir / "wall_tuned_v1.pth"
generated_nowall = generated_dir / "nowall_d3_ppo_v3.pth"

if generated_bundle.exists():
    bundle = torch.load(generated_bundle, map_location="cpu", weights_only=False)
elif generated_wall.exists() and generated_nowall.exists():
    bundle = {
        "wall": torch.load(generated_wall, map_location="cpu", weights_only=False),
        "nowall": torch.load(generated_nowall, map_location="cpu", weights_only=False),
    }
elif final_weights.exists():
    bundle = torch.load(final_weights, map_location="cpu", weights_only=False)
else:
    raise FileNotFoundError(
        "No local weights source found. Expected one of: "
        f"{generated_bundle}, pair ({generated_wall}, {generated_nowall}), or existing {final_weights}"
    )

torch.save(bundle, final_weights)

with zipfile.ZipFile(final_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(agent_path, "agent.py")
    zf.write(final_weights, "weights.pth")

print("weights_sha256=", hashlib.sha256(final_weights.read_bytes()).hexdigest())
print("zip_sha256=", hashlib.sha256(final_zip.read_bytes()).hexdigest())
print("weights=", final_weights)
print("zip=", final_zip)
PY
