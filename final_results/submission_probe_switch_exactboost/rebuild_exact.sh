#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${ROOT_DIR}/CS780-OBELIX/assym_ppo/submission_probe_switch_exactboost_singleweight"

cp "${SRC_DIR}/agent.py" "${SCRIPT_DIR}/agent.py"
cp "${SRC_DIR}/weights.pth" "${SCRIPT_DIR}/weights.pth"
echo "Restored exact submission_probe_switch_exactboost bundle in ${SCRIPT_DIR}"
