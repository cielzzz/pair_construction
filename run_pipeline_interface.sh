#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJ_ROOT="$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash run_pipeline_interface.sh submit-qz [args...]
  bash run_pipeline_interface.sh pair-local <lang> <split> [config] [device] [run_root]

Modes:
  submit-qz
    Submit vcdata + edit to Qizhi using submit_vcdata_editx_batch_h200.sh.

  pair-local
    Build pairs locally from an existing RUN_ROOT produced by vcdata + edit.
    It sets:
      VCDATA_ROOT=<run_root>/vcdata/<lang>
      PAIR_OUTPUTS_ROOT=<run_root>/pair_outputs/<lang>
    and then calls scripts/run_pairs_local.sh.

Examples:
  bash run_pipeline_interface.sh submit-qz
  bash run_pipeline_interface.sh pair-local zh zh_slim_0001 configs/default.yaml cuda:0 \
    /path/to/run_root
EOF
}

MODE="${1:-}"
if [ -z "$MODE" ]; then
  usage >&2
  exit 1
fi
shift || true

case "$MODE" in
  submit-qz)
    exec bash "$PROJ_ROOT/submit_vcdata_editx_batch_h200.sh" "$@"
    ;;
  pair-local)
    LANG="${1:-}"
    SPLIT="${2:-}"
    CONFIG="${3:-$PROJ_ROOT/configs/default.yaml}"
    DEVICE="${4:-cuda:0}"
    RUN_ROOT="${5:-${PIPELINE_RUN_ROOT:-}}"

    if [ -z "$LANG" ] || [ -z "$SPLIT" ] || [ -z "$RUN_ROOT" ]; then
      usage >&2
      exit 1
    fi

    export VCDATA_ROOT="${VCDATA_ROOT:-$RUN_ROOT/vcdata/$LANG}"
    export PAIR_OUTPUTS_ROOT="${PAIR_OUTPUTS_ROOT:-$RUN_ROOT/pair_outputs/$LANG}"
    export EMOTION_EVAL_ROOT="${EMOTION_EVAL_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/code/emotion_eval}"

    exec bash "$PROJ_ROOT/scripts/run_pairs_local.sh" "$SPLIT" "$CONFIG" "$DEVICE"
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
