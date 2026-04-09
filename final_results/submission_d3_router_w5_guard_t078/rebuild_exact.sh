#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${ROOT_DIR}/CS780-OBELIX/d3_wall_search/submission_d3_router_w5_guard_t078_v1"

cp "${SRC_DIR}/agent.py" "${SCRIPT_DIR}/agent.py"
cp "${SRC_DIR}/weights.pth" "${SCRIPT_DIR}/weights.pth"
echo "Restored exact submission_d3_router_w5_guard_t078 bundle in ${SCRIPT_DIR}"
