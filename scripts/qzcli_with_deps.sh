#!/usr/bin/env bash
set -euo pipefail

QZCLI_TOOL_ROOT="${QZCLI_TOOL_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/qzcli_tool}"
QZCLI_PY="${QZCLI_PY:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/bin/python}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

mkdir -p "${QZCLI_HOME}"
export HOME="${QZCLI_HOME}"
export PYTHONPATH="${QZCLI_TOOL_ROOT}:${PYTHONPATH:-}"
exec "${QZCLI_PY}" "${QZCLI_TOOL_ROOT}/bin/qzcli" "$@"
