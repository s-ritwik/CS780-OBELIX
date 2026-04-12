#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_DIR="${PROJECT_DIR}/CS780-OBELIX"
SOURCE_ZIP="${SCRIPT_DIR}/router_guard.zip"
FINAL_WEIGHTS="${SCRIPT_DIR}/weights.pth"
REBUILT_WEIGHTS="${SCRIPT_DIR}/weights_rebuilt.pth"
REBUILT_ZIP="${SCRIPT_DIR}/router_guard_rebuilt.zip"

rm -f "${SCRIPT_DIR}/agent.py" "${REBUILT_WEIGHTS}" "${REBUILT_ZIP}"

python - <<'PY' "${SOURCE_ZIP}" "${SCRIPT_DIR}"
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as zf:
    zf.extract("agent.py", out_dir)
PY

set +u
source /home/rycker/src/anaconda3/etc/profile.d/conda.sh
conda activate torch
set -u

python - <<'PY' "${REPO_DIR}" "${FINAL_WEIGHTS}" "${REBUILT_WEIGHTS}" "${REBUILT_ZIP}" "${SCRIPT_DIR}/agent.py"
import hashlib
import sys
import zipfile
from pathlib import Path
from collections.abc import Mapping

import torch

repo_dir = Path(sys.argv[1])
final_weights = Path(sys.argv[2])
rebuilt_weights = Path(sys.argv[3])
rebuilt_zip = Path(sys.argv[4])
agent_path = Path(sys.argv[5])

router = torch.load(repo_dir / "d3_wall_search" / "router_d3_cost_sensitive_probe20_w5.pth", map_location="cpu", weights_only=False)
router = {**router, "threshold": 0.78}
bundle = {
    "wall": torch.load(repo_dir / "d3_wall_search" / "wall_d3_random_asym_v1.pth", map_location="cpu", weights_only=False),
    "nowall": torch.load(repo_dir / "d3_wall_search" / "nowall_d3_ppo_random_ft_v1.pth", map_location="cpu", weights_only=False),
    "nowall_v1": torch.load(repo_dir / "ppo_lab" / "nowall_d3_ppo_v1.pth", map_location="cpu", weights_only=False),
    "router": router,
    "meta": {"name": "submission_d3_random_probe_25p9", "random_seed_eval_weighted": 25.9},
}
torch.save(bundle, rebuilt_weights)

def same(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, Mapping):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

final_obj = torch.load(final_weights, map_location="cpu", weights_only=False)
rebuilt_obj = torch.load(rebuilt_weights, map_location="cpu", weights_only=False)
semantic_match = same(final_obj, rebuilt_obj)
if not semantic_match:
    raise SystemExit("semantic weight mismatch")

with zipfile.ZipFile(rebuilt_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(agent_path, "agent.py")
    zf.write(rebuilt_weights, "weights.pth")

print("semantic_match=True")
print("final_sha256=", hashlib.sha256(final_weights.read_bytes()).hexdigest())
print("rebuilt_sha256=", hashlib.sha256(rebuilt_weights.read_bytes()).hexdigest())
print("rebuilt_zip=", rebuilt_zip)
PY
