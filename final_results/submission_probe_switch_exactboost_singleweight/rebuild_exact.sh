#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_DIR="${PROJECT_DIR}/CS780-OBELIX"
SRC_DIR="${REPO_DIR}/assym_ppo/submission_probe_switch_exactboost_singleweight"

cp "${SRC_DIR}/agent.py" "${SCRIPT_DIR}/agent.py"
cp "${SRC_DIR}/weights.pth" "${SCRIPT_DIR}/weights.pth"
cp "${SRC_DIR}/submission_probe_switch_exactboost_singleweight.zip" "${SCRIPT_DIR}/submission_probe_switch_exactboost_singleweight.zip"
echo "Restored exact submission_probe_switch_exactboost_singleweight bundle in ${SCRIPT_DIR}"
