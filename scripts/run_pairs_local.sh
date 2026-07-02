#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJ_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

SPLIT="${1:?usage: run_pairs_local.sh <split> [config] [device]}"
PAIR_CONFIG="${2:-${PAIR_CONFIG:-$PROJ_ROOT/configs/default.yaml}}"
DEVICE="${3:-cuda:0}"
PAIR_RESUME="${PAIR_RESUME:-0}"
PAIR_CONSTRUCT_PARALLEL="${PAIR_CONSTRUCT_PARALLEL:-1}"
PAIR_CONSTRUCT_JOBS="${PAIR_CONSTRUCT_JOBS:-4}"
PAIR_RUN_QC="${PAIR_RUN_QC:-1}"
PAIR_GPU_GUARD_ENABLE="${PAIR_GPU_GUARD_ENABLE:-0}"
PAIR_GPU_GUARD_GPUS="${PAIR_GPU_GUARD_GPUS:-auto}"
PAIR_GPU_GUARD_PY="${PAIR_GPU_GUARD_PY:-}"
PAIR_GPU_GUARD_MATRIX_SIZE="${PAIR_GPU_GUARD_MATRIX_SIZE:-8192}"
PAIR_GPU_GUARD_ACTIVE_MS="${PAIR_GPU_GUARD_ACTIVE_MS:-900}"
PAIR_GPU_GUARD_IDLE_MS="${PAIR_GPU_GUARD_IDLE_MS:-150}"
PAIR_GPU_GUARD_DTYPE="${PAIR_GPU_GUARD_DTYPE:-bfloat16}"
PAIR_GPU_GUARD_RESERVE_MIB="${PAIR_GPU_GUARD_RESERVE_MIB:-0}"

EMOTION_PY_BIN="${EMOTION_PY_BIN:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/emotion/bin/python}"
WAVLMPY="${WAVLMPY:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/moss_ttsd_sglang/bin/python}"
PAIR_GPU_GUARD_PY="${PAIR_GPU_GUARD_PY:-$WAVLMPY}"

export PAIR_CONFIG

compute_pair_split_root() {
  if [ -n "${PAIR_OUTPUTS_ROOT:-}" ]; then
    printf '%s/%s\n' "${PAIR_OUTPUTS_ROOT%/}" "$SPLIT"
    return 0
  fi

  "$EMOTION_PY_BIN" - "$PAIR_CONFIG" "$SPLIT" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[3])
from _utils import load_config, project_split_root

cfg = load_config(sys.argv[1])
print(project_split_root(cfg, sys.argv[2]))
PY
}

PAIR_SPLIT_ROOT="$(compute_pair_split_root)"

echo "[pair-local] split=$SPLIT"
echo "[pair-local] config=$PAIR_CONFIG"
echo "[pair-local] device=$DEVICE"
echo "[pair-local] VCDATA_ROOT=${VCDATA_ROOT:-<config default>}"
echo "[pair-local] PAIR_OUTPUTS_ROOT=${PAIR_OUTPUTS_ROOT:-<config default>}"
echo "[pair-local] EMOTION_EVAL_ROOT=${EMOTION_EVAL_ROOT:-<config default>}"
echo "[pair-local] split_root=$PAIR_SPLIT_ROOT"
echo "[pair-local] resume=$PAIR_RESUME parallel=$PAIR_CONSTRUCT_PARALLEL jobs=$PAIR_CONSTRUCT_JOBS"
echo "[pair-local] run_qc=$PAIR_RUN_QC"

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

GPU_GUARD_PID=""
GPU_GUARD_LOG=""

start_gpu_guard() {
  truthy "$PAIR_GPU_GUARD_ENABLE" || return 0
  if [ ! -x "$PAIR_GPU_GUARD_PY" ]; then
    echo "[pair-local] [warn] gpu guard python not executable: $PAIR_GPU_GUARD_PY" >&2
    return 0
  fi
  mkdir -p "$PAIR_SPLIT_ROOT/logs"
  GPU_GUARD_LOG="$PAIR_SPLIT_ROOT/logs/gpu_util_guard_${SPLIT}.log"
  echo "[pair-local] start gpu guard: log=$GPU_GUARD_LOG gpus=$PAIR_GPU_GUARD_GPUS exclude=$DEVICE"
  "$PAIR_GPU_GUARD_PY" "$SCRIPT_DIR/gpu_util_guard.py" \
    --gpus "$PAIR_GPU_GUARD_GPUS" \
    --exclude-device "$DEVICE" \
    --matrix-size "$PAIR_GPU_GUARD_MATRIX_SIZE" \
    --active-ms "$PAIR_GPU_GUARD_ACTIVE_MS" \
    --idle-ms "$PAIR_GPU_GUARD_IDLE_MS" \
    --dtype "$PAIR_GPU_GUARD_DTYPE" \
    --reserve-mib "$PAIR_GPU_GUARD_RESERVE_MIB" \
    >"$GPU_GUARD_LOG" 2>&1 &
  GPU_GUARD_PID="$!"
  sleep 2
  if ! kill -0 "$GPU_GUARD_PID" 2>/dev/null; then
    echo "[pair-local] [warn] gpu guard exited early; see $GPU_GUARD_LOG" >&2
    GPU_GUARD_PID=""
  fi
}

stop_gpu_guard() {
  if [ -n "$GPU_GUARD_PID" ] && kill -0 "$GPU_GUARD_PID" 2>/dev/null; then
    echo "[pair-local] stop gpu guard pid=$GPU_GUARD_PID"
    kill "$GPU_GUARD_PID" 2>/dev/null || true
    wait "$GPU_GUARD_PID" 2>/dev/null || true
  fi
  GPU_GUARD_PID=""
}

cleanup() {
  local status=$?
  stop_gpu_guard
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mark_step_done() {
  local step_name="$1"
  mkdir -p "$PAIR_SPLIT_ROOT/.steps"
  "$EMOTION_PY_BIN" - "$step_name" "$PAIR_SPLIT_ROOT/.steps/${step_name}.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "step": sys.argv[1],
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
}
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
PY
}

should_skip_step() {
  local step_name="$1"
  local output_rel="${2:-}"
  [ "$PAIR_RESUME" = "1" ] || return 1
  [ -f "$PAIR_SPLIT_ROOT/.steps/${step_name}.json" ] || return 1
  if [ -n "$output_rel" ] && [ ! -e "$PAIR_SPLIT_ROOT/$output_rel" ]; then
    return 1
  fi
  return 0
}

run_python_step() {
  local step_name="$1"
  local output_rel="$2"
  local script_name="$3"
  shift 3
  if should_skip_step "$step_name" "$output_rel"; then
    echo "[pair-local] skip $step_name"
    return 0
  fi
  "$EMOTION_PY_BIN" "$SCRIPT_DIR/$script_name" --split "$SPLIT" "$@"
  mark_step_done "$step_name"
}

run_wavlm_step() {
  local step_name="$1"
  if should_skip_step "$step_name" ""; then
    echo "[pair-local] skip $step_name"
    return 0
  fi
  "$WAVLMPY" "$SCRIPT_DIR/11b_add_wavlm_sim.py" --split "$SPLIT" --device "$DEVICE"
  mark_step_done "$step_name"
}

run_emotion_step() {
  local step_name="$1"
  if should_skip_step "$step_name" "emotion/per_file_dual.csv"; then
    echo "[pair-local] skip $step_name"
    return 0
  fi
  bash "$SCRIPT_DIR/04_run_emotion_eval.sh" "$SPLIT" "$DEVICE"
  mark_step_done "$step_name"
}

run_dnsmos_step() {
  local step_name="$1"
  if should_skip_step "$step_name" ""; then
    echo "[pair-local] skip $step_name"
    return 0
  fi
  "$EMOTION_PY_BIN" "$SCRIPT_DIR/12_filter_dnsmos_bak.py" --split "$SPLIT"
  mark_step_done "$step_name"
}

run_qc_step() {
  local step_name="$1"
  if should_skip_step "$step_name" "quality_gate/summary.json"; then
    echo "[pair-local] skip $step_name"
    return 0
  fi
  echo "[pair-local] run $step_name"
  QWEN_ASR_DEVICE="$DEVICE" "$EMOTION_PY_BIN" "$SCRIPT_DIR/qc_pairs.py" --pair-root "$PAIR_SPLIT_ROOT" --config "$PAIR_CONFIG"
  mark_step_done "$step_name"
}

start_gpu_guard

run_python_step "01_build_vcdata_base" "intermediate/vcdata_base.jsonl" "01_build_vcdata_base.py"
run_python_step "02_build_editx_base" "intermediate/editx_base.jsonl" "02_build_editx_base.py"
run_python_step "03_join_editx_with_vcdata" "intermediate/joined_editx.jsonl" "03_join_editx_with_vcdata.py"
run_emotion_step "04_run_emotion_eval"

CONSTRUCT_SPECS=(
  "05_construct_A|pairs/A.jsonl|05_construct_A.py"
  "06_construct_B|pairs/B.jsonl|06_construct_B.py"
  "07_construct_C|pairs/C.jsonl|07_construct_C.py"
  "07b_construct_C_mixed|pairs/C_mixed.jsonl|07b_construct_C_mixed.py"
  "07c_construct_D|pairs/D.jsonl|07c_construct_D.py"
  "07d_construct_D_st|pairs/D_st.jsonl|07d_construct_D_st.py"
  "07e_construct_genre|pairs/Genre.jsonl|07e_construct_genre.py"
  "07e_construct_genre_conv|pairs/Genre_conv.jsonl|07e_construct_genre_conv.py"
  "07f_construct_D_cross_emo|pairs/D_cross_emo.jsonl|07f_construct_D_cross_emo.py"
  "08_construct_H1|pairs/H1.jsonl|08_construct_H1.py"
  "09_construct_H2|pairs/H2.jsonl|09_construct_H2.py"
  "10_construct_H3|pairs/H3.jsonl|10_construct_H3.py"
)

if [ "$PAIR_CONSTRUCT_PARALLEL" = "1" ]; then
  declare -a RUNNING=()
  construct_fail=0
  for spec in "${CONSTRUCT_SPECS[@]}"; do
    IFS='|' read -r step_name output_rel script_name <<<"$spec"
    run_python_step "$step_name" "$output_rel" "$script_name" &
    RUNNING+=("$!:$step_name")
    while [ "${#RUNNING[@]}" -ge "$PAIR_CONSTRUCT_JOBS" ]; do
      new_running=()
      for item in "${RUNNING[@]}"; do
        pid="${item%%:*}"
        name="${item#*:}"
        if kill -0 "$pid" 2>/dev/null; then
          new_running+=("$item")
          continue
        fi
        if ! wait "$pid"; then
          echo "[pair-local] failed: $name" >&2
          construct_fail=1
        fi
      done
      RUNNING=("${new_running[@]}")
      [ "${#RUNNING[@]}" -lt "$PAIR_CONSTRUCT_JOBS" ] || sleep 0.2
    done
  done
  for item in "${RUNNING[@]}"; do
    pid="${item%%:*}"
    name="${item#*:}"
    if ! wait "$pid"; then
      echo "[pair-local] failed: $name" >&2
      construct_fail=1
    fi
  done
  [ "$construct_fail" -eq 0 ] || exit 1
else
  for spec in "${CONSTRUCT_SPECS[@]}"; do
    IFS='|' read -r step_name output_rel script_name <<<"$spec"
    run_python_step "$step_name" "$output_rel" "$script_name"
  done
fi

run_wavlm_step "11b_add_wavlm_sim"
run_dnsmos_step "12_filter_dnsmos_bak"
if [ "$PAIR_RUN_QC" = "1" ] || [ "$PAIR_RUN_QC" = "true" ] || [ "$PAIR_RUN_QC" = "TRUE" ]; then
  run_qc_step "13_run_qc_pairs"
else
  echo "[pair-local] skip 13_run_qc_pairs (PAIR_RUN_QC=$PAIR_RUN_QC)"
fi

echo "[pair-local] done: $SPLIT"
