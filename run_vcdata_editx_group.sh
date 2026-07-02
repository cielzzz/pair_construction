#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

GROUP_DIR="${GROUP_DIR:?GROUP_DIR is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
GROUP_TAG="${GROUP_TAG:-$(basename "$GROUP_DIR")}"
JOB_NAME="${JOB_NAME:-$GROUP_TAG}"
LOG_FILE="${LOG_FILE:-}"

VCDATA_REPO="${VCDATA_REPO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction}"
VCDATA_SCRIPT="${VCDATA_SCRIPT:-$VCDATA_REPO/run8gpu_retries.sh}"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-$VCDATA_REPO/activate_moss_ttsd_vc.sh}"
VCDATA_MODEL_DIR="${VCDATA_MODEL_DIR:-${MODEL_DIR:-$VCDATA_REPO/MOSS-TTS}}"

VC_EDIT_REPO="${VC_EDIT_REPO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_edit}"
EDITX_SCRIPT="${EDITX_SCRIPT:-$VC_EDIT_REPO/vc_edit_framework/scripts/run_step_editx_split.sh}"
EDITX_MODEL_DIR="${EDITX_MODEL_DIR:-$VC_EDIT_REPO/models/Step-Audio-EditX}"
STEPX_PY="${STEPX_PY:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/step_audio_editx/bin/python}"

ZH_TEXT_FIELD="${ZH_TEXT_FIELD:-text}"
EN_TEXT_FIELD="${EN_TEXT_FIELD:-text}"
ZH_EDIT_PAIRS="${ZH_EDIT_PAIRS:-style:radio,style:news,style:chat}"
EN_EDIT_PAIRS="${EN_EDIT_PAIRS:-style:chat,style:news,style:radio}"
DISABLE_EDITX="${DISABLE_EDITX:-0}"

AUDIO_PATH_FIELD="${AUDIO_PATH_FIELD:-local_path}"
NUM_CANDIDATES="${NUM_CANDIDATES:-16}"
BATCH_SIZE="${BATCH_SIZE:-12}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.85}"
SEED_BASE="${SEED_BASE:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
MAX_RETRIES="${MAX_RETRIES:-3}"
VCDATA_GPU_MONITOR_ENABLE="${VCDATA_GPU_MONITOR_ENABLE:-1}"
VCDATA_GPU_MONITOR_INTERVAL_SEC="${VCDATA_GPU_MONITOR_INTERVAL_SEC:-10}"
EDITX_MAX_RETRIES="${EDITX_MAX_RETRIES:-3}"
EDITX_RETRY_BACKOFF_SEC="${EDITX_RETRY_BACKOFF_SEC:-30}"
EDITX_ATTEMPT_TIMEOUT_SEC="${EDITX_ATTEMPT_TIMEOUT_SEC:-0}"
EDITX_ATTEMPT_TIMEOUT_KILL_AFTER_SEC="${EDITX_ATTEMPT_TIMEOUT_KILL_AFTER_SEC:-120}"
EDITX_PREPARE_WORKERS="${EDITX_PREPARE_WORKERS:-2}"
EDITX_BATCH_SIZE="${EDITX_BATCH_SIZE:-6}"
EDITX_MAX_NUM_SEQS="${EDITX_MAX_NUM_SEQS:-6}"
EDITX_TENSOR_PARALLEL_SIZE="${EDITX_TENSOR_PARALLEL_SIZE:-1}"
EDITX_ENGINE_COUNT="${EDITX_ENGINE_COUNT:-16}"
EDITX_ENGINE_GPU_GROUPS="${EDITX_ENGINE_GPU_GROUPS:-0;0;1;1;2;2;3;3;4;4;5;5;6;6;7;7}"
EDITX_ENGINE_JOB_SPLIT_MODE="${EDITX_ENGINE_JOB_SPLIT_MODE:-round_robin}"
EDITX_ENGINE_STARTUP_STAGGER_SEC="${EDITX_ENGINE_STARTUP_STAGGER_SEC:-20}"
EDITX_ENGINE_PER_ATTEMPT_RETRIES="${EDITX_ENGINE_PER_ATTEMPT_RETRIES:-2}"
EDITX_ENGINE_PER_ATTEMPT_BACKOFF_SEC="${EDITX_ENGINE_PER_ATTEMPT_BACKOFF_SEC:-45}"
EDITX_AUDIO_DURATION_BUCKETING="${EDITX_AUDIO_DURATION_BUCKETING:-1}"
EDITX_DURATION_BUCKET_WINDOW="${EDITX_DURATION_BUCKET_WINDOW:-128}"
EDITX_NEXT_BATCH_PREFETCH="${EDITX_NEXT_BATCH_PREFETCH:-1}"
EDITX_PREFETCH_DEPTH="${EDITX_PREFETCH_DEPTH:-1}"
EDITX_AUDIO_CONDITION_BUILD_WORKERS="${EDITX_AUDIO_CONDITION_BUILD_WORKERS:-2}"
EDITX_DISABLE_AUDIO_CONDITION_ITEM_PARALLEL="${EDITX_DISABLE_AUDIO_CONDITION_ITEM_PARALLEL:-0}"
EDITX_USE_ASYNC_ENGINE="${EDITX_USE_ASYNC_ENGINE:-0}"
EDITX_STREAM_VOCODE="${EDITX_STREAM_VOCODE:-0}"
EDITX_PREPARE_BREAKDOWN="${EDITX_PREPARE_BREAKDOWN:-1}"
EDITX_ASYNC_WRITE_WORKERS="${EDITX_ASYNC_WRITE_WORKERS:-0}"
EDITX_ASYNC_VOCODER_WORKERS="${EDITX_ASYNC_VOCODER_WORKERS:-0}"
EDITX_GPU_MEMORY_UTILIZATION="${EDITX_GPU_MEMORY_UTILIZATION:-0.35}"
EDITX_MAX_MODEL_LEN="${EDITX_MAX_MODEL_LEN:-8192}"
EDITX_DTYPE="${EDITX_DTYPE:-bfloat16}"
EDITX_PREPROCESS_CACHE_SIZE="${EDITX_PREPROCESS_CACHE_SIZE:-512}"
EDITX_ENABLE_TASK10_FALLBACK="${EDITX_ENABLE_TASK10_FALLBACK:-1}"
EDITX_FALLBACK_PREPARE_WORKERS="${EDITX_FALLBACK_PREPARE_WORKERS:-4}"
EDITX_FALLBACK_BATCH_SIZE="${EDITX_FALLBACK_BATCH_SIZE:-12}"
EDITX_FALLBACK_MAX_NUM_SEQS="${EDITX_FALLBACK_MAX_NUM_SEQS:-12}"
EDITX_FALLBACK_AUDIO_DURATION_BUCKETING="${EDITX_FALLBACK_AUDIO_DURATION_BUCKETING:-1}"
EDITX_FALLBACK_DURATION_BUCKET_WINDOW="${EDITX_FALLBACK_DURATION_BUCKET_WINDOW:-128}"
EDITX_FALLBACK_NEXT_BATCH_PREFETCH="${EDITX_FALLBACK_NEXT_BATCH_PREFETCH:-1}"
EDITX_FALLBACK_PREFETCH_DEPTH="${EDITX_FALLBACK_PREFETCH_DEPTH:-1}"
EDITX_FALLBACK_AUDIO_CONDITION_BUILD_WORKERS="${EDITX_FALLBACK_AUDIO_CONDITION_BUILD_WORKERS:-}"
EDITX_FALLBACK_DISABLE_AUDIO_CONDITION_ITEM_PARALLEL="${EDITX_FALLBACK_DISABLE_AUDIO_CONDITION_ITEM_PARALLEL:-0}"
EDITX_FALLBACK_USE_ASYNC_ENGINE="${EDITX_FALLBACK_USE_ASYNC_ENGINE:-0}"
EDITX_FALLBACK_STREAM_VOCODE="${EDITX_FALLBACK_STREAM_VOCODE:-0}"
EDITX_FALLBACK_PREPARE_BREAKDOWN="${EDITX_FALLBACK_PREPARE_BREAKDOWN:-0}"
EDITX_FALLBACK_ASYNC_WRITE_WORKERS="${EDITX_FALLBACK_ASYNC_WRITE_WORKERS:-0}"
RUN_PAIRS_ON_QZ="${RUN_PAIRS_ON_QZ:-1}"
PAIR_LOCAL_ZH_CONFIG="${PAIR_LOCAL_ZH_CONFIG:-$SCRIPT_DIR/configs/default.yaml}"
PAIR_LOCAL_EN_CONFIG="${PAIR_LOCAL_EN_CONFIG:-$SCRIPT_DIR/configs/default_en.yaml}"
PAIR_LOCAL_DEVICE="${PAIR_LOCAL_DEVICE:-cuda:0}"
PAIR_LOCAL_RESUME="${PAIR_LOCAL_RESUME:-0}"
PAIR_LOCAL_PARALLEL="${PAIR_LOCAL_PARALLEL:-1}"
PAIR_LOCAL_JOBS="${PAIR_LOCAL_JOBS:-4}"
PAIR_GPU_GUARD_ENABLE="${PAIR_GPU_GUARD_ENABLE:-0}"
PAIR_GPU_GUARD_GPUS="${PAIR_GPU_GUARD_GPUS:-auto}"
PAIR_GPU_GUARD_PY="${PAIR_GPU_GUARD_PY:-}"
PAIR_GPU_GUARD_MATRIX_SIZE="${PAIR_GPU_GUARD_MATRIX_SIZE:-8192}"
PAIR_GPU_GUARD_ACTIVE_MS="${PAIR_GPU_GUARD_ACTIVE_MS:-900}"
PAIR_GPU_GUARD_IDLE_MS="${PAIR_GPU_GUARD_IDLE_MS:-150}"
PAIR_GPU_GUARD_DTYPE="${PAIR_GPU_GUARD_DTYPE:-bfloat16}"
PAIR_GPU_GUARD_RESERVE_MIB="${PAIR_GPU_GUARD_RESERVE_MIB:-0}"
RUN_IJ_ON_QZ="${RUN_IJ_ON_QZ:-1}"
IJ_LIMIT="${IJ_LIMIT:-0}"
IJ_DEVICE="${IJ_DEVICE:-$PAIR_LOCAL_DEVICE}"
IJ_RUN_SPEED="${IJ_RUN_SPEED:-1}"
IJ_RUN_PROSODY="${IJ_RUN_PROSODY:-1}"
IJ_RUN_WAVLM="${IJ_RUN_WAVLM:-1}"
IJ_RUN_QC="${IJ_RUN_QC:-1}"
IJ_RUN_PAIR_AUDIO_METRICS="${IJ_RUN_PAIR_AUDIO_METRICS:-$IJ_RUN_QC}"
IJ_SPEED_BATCH_SIZE="${IJ_SPEED_BATCH_SIZE:-4}"
IJ_SPEED_MAX_NUM_SEQS="${IJ_SPEED_MAX_NUM_SEQS:-$IJ_SPEED_BATCH_SIZE}"
IJ_SPEED_TENSOR_PARALLEL_SIZE="${IJ_SPEED_TENSOR_PARALLEL_SIZE:-8}"
IJ_SPEED_GPU_MEMORY_UTILIZATION="${IJ_SPEED_GPU_MEMORY_UTILIZATION:-0.75}"
IJ_SPEED_PREPARE_WORKERS="${IJ_SPEED_PREPARE_WORKERS:-4}"
IJ_SPEED_AUDIO_CONDITION_BUILD_WORKERS="${IJ_SPEED_AUDIO_CONDITION_BUILD_WORKERS:-$IJ_SPEED_PREPARE_WORKERS}"
IJ_SPEED_PREPROCESS_CACHE_SIZE="${IJ_SPEED_PREPROCESS_CACHE_SIZE:-1024}"
IJ_SEEDVC_SHARDED="${IJ_SEEDVC_SHARDED:-1}"
IJ_SEEDVC_GPU_IDS="${IJ_SEEDVC_GPU_IDS:-0,1,2,3,4,5,6,7}"
IJ_SEEDVC_SHARD_COUNT="${IJ_SEEDVC_SHARD_COUNT:-8}"
IJ_MAX_JOBS="${IJ_MAX_JOBS:-0}"
IJ_SKIP_EXISTING="${IJ_SKIP_EXISTING:-1}"
IJ_TIMBRE_PICK="${IJ_TIMBRE_PICK:-random}"
IJ_SEED="${IJ_SEED:-42}"
IJ_DIFFUSION_STEPS="${IJ_DIFFUSION_STEPS:-25}"
IJ_INFERENCE_CFG_RATE="${IJ_INFERENCE_CFG_RATE:-0.7}"

timestamp_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

truthy() {
  case "$1" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

jsonl_has_field() {
  local jsonl_path="$1"
  local field_name="$2"
  python - "$jsonl_path" "$field_name" <<'PY'
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
field_name = sys.argv[2]

with jsonl_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        raise SystemExit(0 if field_name in row else 1)

raise SystemExit(1)
PY
}

detect_audio_path_field() {
  local input_dir="$1"
  local candidate
  local first_jsonl

  first_jsonl=$(find "$input_dir" -maxdepth 1 -name '*.jsonl' | sort | head -n 1 || true)
  [ -n "$first_jsonl" ] || {
    echo "$AUDIO_PATH_FIELD"
    return 0
  }

  if jsonl_has_field "$first_jsonl" "$AUDIO_PATH_FIELD"; then
    echo "$AUDIO_PATH_FIELD"
    return 0
  fi

  for candidate in audio_path local_path wav_path path; do
    if jsonl_has_field "$first_jsonl" "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done

  echo "ERROR: failed to detect audio path field for $first_jsonl" >&2
  return 1
}

count_nonempty_lines() {
  awk 'NF { c += 1 } END { print c + 0 }' "$1"
}

count_manifest_cases() {
  local split_dir="$1"
  local manifest_paths=("$split_dir"/manifest_shard*.jsonl)
  if [ ! -e "${manifest_paths[0]}" ]; then
    echo 0
    return
  fi
  wc -l "${manifest_paths[@]}" | awk 'END { print $1 + 0 }'
}

has_jsonl_inputs() {
  local input_dir="$1"
  compgen -G "${input_dir}/*.jsonl" > /dev/null
}

split_dir_complete() {
  local split_dir="$1"
  local jsonl_path="$2"
  local expected_cases
  local actual_cases

  if [ ! -d "$split_dir" ]; then
    return 1
  fi

  expected_cases=$(count_nonempty_lines "$jsonl_path")
  actual_cases=$(count_manifest_cases "$split_dir")
  [ "$expected_cases" -gt 0 ] || return 1
  [ "$actual_cases" -eq "$expected_cases" ] || return 1
  [ -f "$split_dir/.stage1_generate_state.json" ]
}

prepare_vcdata_pending_input_dir() {
  local input_dir="$1"
  local job_output_root="$2"
  local final_root="$3"
  local pending_input_dir
  local jsonl_path
  local split_name

  pending_input_dir=$(mktemp -d "$job_output_root/.resume_inputs.XXXXXX")

  for jsonl_path in "$input_dir"/*.jsonl; do
    [ -e "$jsonl_path" ] || continue
    split_name=$(basename "$jsonl_path" .jsonl)
    if split_dir_complete "$final_root/$split_name" "$jsonl_path"; then
      echo "[$(timestamp_utc)] published vcdata already complete, skip rerun for $split_name" >&2
      continue
    fi
    ln -s "$jsonl_path" "$pending_input_dir/$(basename "$jsonl_path")"
  done

  printf '%s\n' "$pending_input_dir"
}

lang_text_field() {
  local lang="$1"
  local input_dir="${2:-}"
  local configured_field
  local first_jsonl
  local candidate

  case "$lang" in
    zh) configured_field="$ZH_TEXT_FIELD" ;;
    en) configured_field="$EN_TEXT_FIELD" ;;
    *)
      echo "ERROR: unsupported lang '$lang'" >&2
      return 1
      ;;
  esac

  if [ -z "$input_dir" ]; then
    printf '%s\n' "$configured_field"
    return 0
  fi

  first_jsonl=$(find "$input_dir" -maxdepth 1 -name '*.jsonl' | sort | head -n 1 || true)
  if [ -z "$first_jsonl" ]; then
    printf '%s\n' "$configured_field"
    return 0
  fi

  if jsonl_has_field "$first_jsonl" "$configured_field"; then
    printf '%s\n' "$configured_field"
    return 0
  fi

  for candidate in text mtd_transcript transcript asr_text; do
    if jsonl_has_field "$first_jsonl" "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "ERROR: failed to detect text field for $first_jsonl" >&2
  return 1
}

lang_edit_pairs() {
  case "$1" in
    zh) printf '%s\n' "$ZH_EDIT_PAIRS" ;;
    en) printf '%s\n' "$EN_EDIT_PAIRS" ;;
    *)
      echo "ERROR: unsupported lang '$1'" >&2
      return 1
      ;;
  esac
}

edit_tag_from_pair() {
  local edit_type="$1"
  local edit_info="$2"
  printf '%s_%s' "$edit_type" "$edit_info" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//'
}

editx_pair_complete() {
  local split_dir="$1"
  local split_name="$2"
  local jsonl_path="$3"
  local edit_type="$4"
  local edit_info="$5"
  local edit_tag
  local report_path
  local expected_cases
  local actual_cases

  edit_tag=$(edit_tag_from_pair "$edit_type" "$edit_info")
  report_path="$split_dir/stepaudio_${edit_tag}_${split_name}_all/paired_report.jsonl"
  [ -f "$report_path" ] || return 1

  expected_cases=$(count_nonempty_lines "$jsonl_path")
  actual_cases=$(count_nonempty_lines "$report_path")
  [ "$expected_cases" -gt 0 ] || return 1
  [ "$actual_cases" -eq "$expected_cases" ] || return 1
}

run_editx_once() {
  local split_dir="$1"
  local edit_type="$2"
  local edit_info="$3"
  local mode="${4:-primary}"
  local rc

  run_with_optional_timeout() {
    if [ "${EDITX_ATTEMPT_TIMEOUT_SEC}" -gt 0 ]; then
      timeout --signal=TERM --kill-after="${EDITX_ATTEMPT_TIMEOUT_KILL_AFTER_SEC}s" "${EDITX_ATTEMPT_TIMEOUT_SEC}s" "$@"
      return $?
    fi
    "$@"
  }

  if [ "$mode" = "fallback_task10" ]; then
    echo "[$(timestamp_utc)] editx fallback profile=task10 split=$(basename "$split_dir") pair=${edit_type}:${edit_info}"
    set +e
    PYTHON_BIN="$STEPX_PY" \
    MODEL_DIR="$EDITX_MODEL_DIR" \
    EDIT_TYPE="$edit_type" \
    EDIT_INFO="$edit_info" \
    VCDATA_ROOT="$VCDATA_REPO" \
    PREPARE_WORKERS="$EDITX_FALLBACK_PREPARE_WORKERS" \
    BATCH_SIZE="$EDITX_FALLBACK_BATCH_SIZE" \
    MAX_NUM_SEQS="$EDITX_FALLBACK_MAX_NUM_SEQS" \
    AUDIO_DURATION_BUCKETING="$EDITX_FALLBACK_AUDIO_DURATION_BUCKETING" \
    DURATION_BUCKET_WINDOW="$EDITX_FALLBACK_DURATION_BUCKET_WINDOW" \
    NEXT_BATCH_PREFETCH="$EDITX_FALLBACK_NEXT_BATCH_PREFETCH" \
    PREFETCH_DEPTH="$EDITX_FALLBACK_PREFETCH_DEPTH" \
    AUDIO_CONDITION_BUILD_WORKERS="$EDITX_FALLBACK_AUDIO_CONDITION_BUILD_WORKERS" \
    DISABLE_AUDIO_CONDITION_ITEM_PARALLEL="$EDITX_FALLBACK_DISABLE_AUDIO_CONDITION_ITEM_PARALLEL" \
    USE_ASYNC_ENGINE="$EDITX_FALLBACK_USE_ASYNC_ENGINE" \
    STREAM_VOCODE="$EDITX_FALLBACK_STREAM_VOCODE" \
    PREPARE_BREAKDOWN="$EDITX_FALLBACK_PREPARE_BREAKDOWN" \
    ASYNC_WRITE_WORKERS="$EDITX_FALLBACK_ASYNC_WRITE_WORKERS" \
    run_with_optional_timeout bash "$EDITX_SCRIPT" "$split_dir"
    rc=$?
    set -e
    return "$rc"
  fi

  set +e
  PYTHON_BIN="$STEPX_PY" \
  MODEL_DIR="$EDITX_MODEL_DIR" \
  EDIT_TYPE="$edit_type" \
  EDIT_INFO="$edit_info" \
  VCDATA_ROOT="$VCDATA_REPO" \
  PREPARE_WORKERS="$EDITX_PREPARE_WORKERS" \
  BATCH_SIZE="$EDITX_BATCH_SIZE" \
  MAX_NUM_SEQS="$EDITX_MAX_NUM_SEQS" \
  TENSOR_PARALLEL_SIZE="$EDITX_TENSOR_PARALLEL_SIZE" \
  ENGINE_COUNT="$EDITX_ENGINE_COUNT" \
  ENGINE_GPU_GROUPS="$EDITX_ENGINE_GPU_GROUPS" \
  ENGINE_JOB_SPLIT_MODE="$EDITX_ENGINE_JOB_SPLIT_MODE" \
  ENGINE_STARTUP_STAGGER_SEC="$EDITX_ENGINE_STARTUP_STAGGER_SEC" \
  ENGINE_PER_ATTEMPT_RETRIES="$EDITX_ENGINE_PER_ATTEMPT_RETRIES" \
  ENGINE_PER_ATTEMPT_BACKOFF_SEC="$EDITX_ENGINE_PER_ATTEMPT_BACKOFF_SEC" \
  AUDIO_DURATION_BUCKETING="$EDITX_AUDIO_DURATION_BUCKETING" \
  DURATION_BUCKET_WINDOW="$EDITX_DURATION_BUCKET_WINDOW" \
  NEXT_BATCH_PREFETCH="$EDITX_NEXT_BATCH_PREFETCH" \
  PREFETCH_DEPTH="$EDITX_PREFETCH_DEPTH" \
  AUDIO_CONDITION_BUILD_WORKERS="$EDITX_AUDIO_CONDITION_BUILD_WORKERS" \
  DISABLE_AUDIO_CONDITION_ITEM_PARALLEL="$EDITX_DISABLE_AUDIO_CONDITION_ITEM_PARALLEL" \
  USE_ASYNC_ENGINE="$EDITX_USE_ASYNC_ENGINE" \
  STREAM_VOCODE="$EDITX_STREAM_VOCODE" \
  PREPARE_BREAKDOWN="$EDITX_PREPARE_BREAKDOWN" \
  ASYNC_WRITE_WORKERS="$EDITX_ASYNC_WRITE_WORKERS" \
  ASYNC_VOCODER_WORKERS="$EDITX_ASYNC_VOCODER_WORKERS" \
  GPU_MEMORY_UTILIZATION="$EDITX_GPU_MEMORY_UTILIZATION" \
  MAX_MODEL_LEN="$EDITX_MAX_MODEL_LEN" \
  DTYPE="$EDITX_DTYPE" \
  PREPROCESS_CACHE_SIZE="$EDITX_PREPROCESS_CACHE_SIZE" \
  run_with_optional_timeout bash "$EDITX_SCRIPT" "$split_dir"
  rc=$?
  set -e
  return "$rc"
}

run_editx_with_retries() {
  local split_dir="$1"
  local edit_type="$2"
  local edit_info="$3"
  local attempt=1
  local max_attempts
  local rc

  max_attempts=$(( EDITX_MAX_RETRIES > 0 ? EDITX_MAX_RETRIES : 1 ))

  while [ "$attempt" -le "$max_attempts" ]; do
    echo "[$(timestamp_utc)] editx attempt ${attempt}/${max_attempts} split=$(basename "$split_dir") pair=${edit_type}:${edit_info}"
    run_editx_once "$split_dir" "$edit_type" "$edit_info" primary
    rc=$?
    if [ "$rc" -eq 0 ]; then
      return 0
    fi
    echo "[$(timestamp_utc)] WARNING: editx attempt ${attempt}/${max_attempts} failed split=$(basename "$split_dir") pair=${edit_type}:${edit_info}" >&2
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "[$(timestamp_utc)] WARNING: editx attempt hit timeout split=$(basename "$split_dir") pair=${edit_type}:${edit_info} timeout_sec=${EDITX_ATTEMPT_TIMEOUT_SEC}" >&2
    fi
    if [ "$attempt" -lt "$max_attempts" ]; then
      echo "[$(timestamp_utc)] retry after ${EDITX_RETRY_BACKOFF_SEC}s; rerun will resume from paired_report/existing wavs" >&2
      sleep "$EDITX_RETRY_BACKOFF_SEC"
    fi
    attempt=$((attempt + 1))
  done

  if [ "$EDITX_ENABLE_TASK10_FALLBACK" = "1" ] || [ "$EDITX_ENABLE_TASK10_FALLBACK" = "true" ] || [ "$EDITX_ENABLE_TASK10_FALLBACK" = "TRUE" ]; then
    echo "[$(timestamp_utc)] editx retries exhausted; fallback to task10 profile split=$(basename "$split_dir") pair=${edit_type}:${edit_info}" >&2
    if run_editx_once "$split_dir" "$edit_type" "$edit_info" fallback_task10; then
      return 0
    fi
    echo "[$(timestamp_utc)] ERROR: task10 fallback also failed split=$(basename "$split_dir") pair=${edit_type}:${edit_info}" >&2
  fi

  return 1
}

if [ -n "$LOG_FILE" ]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "=========================================="
echo "vcdata -> editx group worker"
echo "  started_at=$(timestamp_utc)"
echo "  job_name=$JOB_NAME"
echo "  group_tag=$GROUP_TAG"
echo "  group_dir=$GROUP_DIR"
echo "  run_root=$RUN_ROOT"
echo "  log_file=${LOG_FILE:-platform_only}"
echo "=========================================="

run_vcdata_for_lang() {
  local lang="$1"
  local input_dir="$GROUP_DIR/inputs/$lang"
  local final_root="$RUN_ROOT/vcdata/$lang"
  local job_output_root="$RUN_ROOT/vcdata_job_runs/$lang/$JOB_NAME"
  local vc_log_root="$RUN_ROOT/logs/vcdata/$JOB_NAME/$lang"
  local pending_input_dir
  local vc_status
  local text_field
  local audio_path_field

  if [ ! -d "$input_dir" ] || ! has_jsonl_inputs "$input_dir"; then
    echo "[$(timestamp_utc)] $lang: no assigned jsonl, skip vcdata"
    return 0
  fi

  text_field=$(lang_text_field "$lang" "$input_dir")
  audio_path_field=$(detect_audio_path_field "$input_dir")
  mkdir -p "$job_output_root" "$final_root" "$vc_log_root"
  pending_input_dir=$(prepare_vcdata_pending_input_dir "$input_dir" "$job_output_root" "$final_root")

  if ! has_jsonl_inputs "$pending_input_dir"; then
    rm -rf "$pending_input_dir"
    echo "[$(timestamp_utc)] $lang: all assigned splits already published, skip vcdata"
    return 0
  fi

  echo "[$(timestamp_utc)] $lang: start vcdata"
  echo "  input_dir=$input_dir"
  echo "  pending_input_dir=$pending_input_dir"
  echo "  job_output_root=$job_output_root"
  echo "  final_root=$final_root"
  echo "  text_field=$text_field"
  echo "  audio_path_field=$audio_path_field"
  echo "  vcdata_model_dir=$VCDATA_MODEL_DIR"

  set +e
  INPUT_DIR="$pending_input_dir" \
  OUTPUT_ROOT="$job_output_root" \
  ACTIVATE_SCRIPT="$ACTIVATE_SCRIPT" \
  MODEL_DIR="$VCDATA_MODEL_DIR" \
  AUDIO_PATH_FIELD="$audio_path_field" \
  TEXT_FIELD="$text_field" \
  NUM_CANDIDATES="$NUM_CANDIDATES" \
  BATCH_SIZE="$BATCH_SIZE" \
  SIMILARITY_THRESHOLD="$SIMILARITY_THRESHOLD" \
  SEED_BASE="$SEED_BASE" \
  NUM_GPUS="$NPROC_PER_NODE" \
  WORKERS_PER_GPU="$WORKERS_PER_GPU" \
  GPU_MONITOR_ENABLE="$VCDATA_GPU_MONITOR_ENABLE" \
  GPU_MONITOR_INTERVAL_SEC="$VCDATA_GPU_MONITOR_INTERVAL_SEC" \
  GPU_MONITOR_LOG="$job_output_root/gpu_metrics.csv" \
  MAX_RETRIES="$MAX_RETRIES" \
  SCHEDULER_MODE="dir_shards" \
  bash "$VCDATA_SCRIPT"
  vc_status=$?
  set -e

  rm -rf "$pending_input_dir"
  if [ "$vc_status" -ne 0 ]; then
    return "$vc_status"
  fi

  verify_vcdata_outputs "$lang" "$input_dir" "$job_output_root" "$final_root"
  publish_vcdata_outputs "$lang" "$input_dir" "$job_output_root" "$final_root" "$vc_log_root"
  echo "[$(timestamp_utc)] $lang: vcdata verified and published"
}

run_editx_for_lang() {
  local lang="$1"
  local input_dir="$GROUP_DIR/inputs/$lang"
  local output_root="$RUN_ROOT/vcdata/$lang"
  local job_output_root="$RUN_ROOT/vcdata_job_runs/$lang/$JOB_NAME"
  local edit_pairs
  local jsonl_path
  local split_name
  local split_dir
  local compat_split_dir
  local pair
  local edit_type
  local edit_info
  local -a edit_list=()

  if [ "$DISABLE_EDITX" = "1" ] || [ "$DISABLE_EDITX" = "true" ] || [ "$DISABLE_EDITX" = "TRUE" ]; then
    echo "[$(timestamp_utc)] $lang: DISABLE_EDITX=1, skip editx"
    return 0
  fi

  if [ ! -d "$input_dir" ] || ! has_jsonl_inputs "$input_dir"; then
    echo "[$(timestamp_utc)] $lang: no assigned jsonl, skip editx"
    return 0
  fi

  edit_pairs=$(lang_edit_pairs "$lang")
  if [ -z "$edit_pairs" ]; then
    echo "[$(timestamp_utc)] $lang: EDIT_PAIRS empty, skip editx"
    return 0
  fi

  IFS=',' read -r -a edit_list <<< "$edit_pairs"

  echo "[$(timestamp_utc)] $lang: start editx"
  echo "  edit_pairs=$edit_pairs"
  echo "  editx_model_dir=$EDITX_MODEL_DIR"

  for jsonl_path in "$input_dir"/*.jsonl; do
    [ -e "$jsonl_path" ] || continue
    split_name=$(basename "$jsonl_path" .jsonl)
    split_dir="$output_root/$split_name"
    compat_split_dir="$job_output_root/$split_name"
    if [ ! -d "$split_dir" ]; then
      echo "ERROR: vcdata split dir missing before editx: $split_dir" >&2
      return 1
    fi

    # Step-Audio-EditX jobs are derived from manifest rows that still carry
    # absolute paths under vcdata_job_runs. Keep that legacy path valid after
    # the split is published into the final vcdata tree.
    if [ ! -e "$compat_split_dir" ] && [ ! -L "$compat_split_dir" ]; then
      mkdir -p "$job_output_root"
      ln -s "$split_dir" "$compat_split_dir"
    fi

    for pair in "${edit_list[@]}"; do
      [ -n "$pair" ] || continue
      edit_type="${pair%%:*}"
      edit_info="${pair##*:}"
      if editx_pair_complete "$split_dir" "$split_name" "$jsonl_path" "$edit_type" "$edit_info"; then
        echo "[$(timestamp_utc)] $lang: skip editx $edit_type:$edit_info on $split_name (complete)"
        continue
      fi
      echo "[$(timestamp_utc)] $lang: editx $edit_type:$edit_info on $split_name"
      run_editx_with_retries "$split_dir" "$edit_type" "$edit_info"
    done
  done

  echo "[$(timestamp_utc)] $lang: editx done"
}

lang_pair_config() {
  case "$1" in
    zh) printf '%s\n' "$PAIR_LOCAL_ZH_CONFIG" ;;
    en) printf '%s\n' "$PAIR_LOCAL_EN_CONFIG" ;;
    *)
      echo "ERROR: unsupported lang '$1'" >&2
      return 1
      ;;
  esac
}

run_pairs_for_lang() {
  local lang="$1"
  local input_dir="$GROUP_DIR/inputs/$lang"
  local split_name
  local jsonl_path
  local split_dir
  local pair_cfg

  if [ "$RUN_PAIRS_ON_QZ" != "1" ] && [ "$RUN_PAIRS_ON_QZ" != "true" ] && [ "$RUN_PAIRS_ON_QZ" != "TRUE" ]; then
    return 0
  fi

  if [ ! -d "$input_dir" ] || ! has_jsonl_inputs "$input_dir"; then
    echo "[$(timestamp_utc)] $lang: no assigned jsonl, skip pair-local"
    return 0
  fi

  pair_cfg=$(lang_pair_config "$lang")
  echo "[$(timestamp_utc)] $lang: start pair-local"
  echo "  pair_config=$pair_cfg"
  echo "  pair_device=$PAIR_LOCAL_DEVICE"

  for jsonl_path in "$input_dir"/*.jsonl; do
    [ -e "$jsonl_path" ] || continue
    split_name=$(basename "$jsonl_path" .jsonl)
    split_dir="$RUN_ROOT/vcdata/$lang/$split_name"
    if [ ! -d "$split_dir" ]; then
      echo "ERROR: vcdata split dir missing before pair-local: $split_dir" >&2
      return 1
    fi
    echo "[$(timestamp_utc)] $lang: pair-local on $split_name"
    VCDATA_ROOT="$RUN_ROOT/vcdata/$lang" \
    PAIR_OUTPUTS_ROOT="$RUN_ROOT/pair_outputs/$lang" \
    EMOTION_EVAL_ROOT="${EMOTION_EVAL_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/code/emotion_eval}" \
    PAIR_RESUME="$PAIR_LOCAL_RESUME" \
    PAIR_CONSTRUCT_PARALLEL="$PAIR_LOCAL_PARALLEL" \
    PAIR_CONSTRUCT_JOBS="$PAIR_LOCAL_JOBS" \
    PAIR_GPU_GUARD_ENABLE="$PAIR_GPU_GUARD_ENABLE" \
    PAIR_GPU_GUARD_GPUS="$PAIR_GPU_GUARD_GPUS" \
    PAIR_GPU_GUARD_PY="$PAIR_GPU_GUARD_PY" \
    PAIR_GPU_GUARD_MATRIX_SIZE="$PAIR_GPU_GUARD_MATRIX_SIZE" \
    PAIR_GPU_GUARD_ACTIVE_MS="$PAIR_GPU_GUARD_ACTIVE_MS" \
    PAIR_GPU_GUARD_IDLE_MS="$PAIR_GPU_GUARD_IDLE_MS" \
    PAIR_GPU_GUARD_DTYPE="$PAIR_GPU_GUARD_DTYPE" \
    PAIR_GPU_GUARD_RESERVE_MIB="$PAIR_GPU_GUARD_RESERVE_MIB" \
    bash "$SCRIPT_DIR/scripts/run_pairs_local.sh" "$split_name" "$pair_cfg" "$PAIR_LOCAL_DEVICE"
  done

  echo "[$(timestamp_utc)] $lang: pair-local done"
}

run_ij_for_lang() {
  local lang="$1"
  local input_dir="$GROUP_DIR/inputs/$lang"
  local split_name
  local jsonl_path
  local pair_cfg

  if ! truthy "$RUN_IJ_ON_QZ"; then
    return 0
  fi

  if [ ! -d "$input_dir" ] || ! has_jsonl_inputs "$input_dir"; then
    echo "[$(timestamp_utc)] $lang: no assigned jsonl, skip I/J"
    return 0
  fi

  pair_cfg=$(lang_pair_config "$lang")
  echo "[$(timestamp_utc)] $lang: start I/J pair generation"
  echo "  ij_limit=$IJ_LIMIT"
  echo "  ij_device=$IJ_DEVICE"
  echo "  ij_speed_tp=$IJ_SPEED_TENSOR_PARALLEL_SIZE"
  echo "  ij_speed_gpu_memory_utilization=$IJ_SPEED_GPU_MEMORY_UTILIZATION"
  echo "  ij_seedvc_sharded=$IJ_SEEDVC_SHARDED"
  echo "  ij_seedvc_gpu_ids=$IJ_SEEDVC_GPU_IDS"

  for jsonl_path in "$input_dir"/*.jsonl; do
    [ -e "$jsonl_path" ] || continue
    split_name=$(basename "$jsonl_path" .jsonl)
    echo "[$(timestamp_utc)] $lang: I/J on $split_name"
    SOURCE_ROOT="$GROUP_DIR/inputs" \
    PAIR_OUTPUTS_ROOT="$RUN_ROOT/pair_outputs" \
    PROSODY_CONFIG="$SCRIPT_DIR/configs/prosody_routes.yaml" \
    SPLITS="$lang/$split_name" \
    LIMIT="$IJ_LIMIT" \
    DEVICE="$IJ_DEVICE" \
    RUN_SPEED="$IJ_RUN_SPEED" \
    RUN_PROSODY="$IJ_RUN_PROSODY" \
    RUN_WAVLM="$IJ_RUN_WAVLM" \
    RUN_QC="$IJ_RUN_QC" \
    RUN_PAIR_AUDIO_METRICS="$IJ_RUN_PAIR_AUDIO_METRICS" \
    SPEED_BATCH_SIZE="$IJ_SPEED_BATCH_SIZE" \
    SPEED_MAX_NUM_SEQS="$IJ_SPEED_MAX_NUM_SEQS" \
    SPEED_TENSOR_PARALLEL_SIZE="$IJ_SPEED_TENSOR_PARALLEL_SIZE" \
    SPEED_GPU_MEMORY_UTILIZATION="$IJ_SPEED_GPU_MEMORY_UTILIZATION" \
    SPEED_PREPARE_WORKERS="$IJ_SPEED_PREPARE_WORKERS" \
    SPEED_AUDIO_CONDITION_BUILD_WORKERS="$IJ_SPEED_AUDIO_CONDITION_BUILD_WORKERS" \
    SPEED_PREPROCESS_CACHE_SIZE="$IJ_SPEED_PREPROCESS_CACHE_SIZE" \
    SEEDVC_SHARDED="$IJ_SEEDVC_SHARDED" \
    SEEDVC_GPU_IDS="$IJ_SEEDVC_GPU_IDS" \
    SEEDVC_SHARD_COUNT="$IJ_SEEDVC_SHARD_COUNT" \
    MAX_JOBS="$IJ_MAX_JOBS" \
    SKIP_EXISTING="$IJ_SKIP_EXISTING" \
    TIMBRE_PICK="$IJ_TIMBRE_PICK" \
    SEED="$IJ_SEED" \
    DIFFUSION_STEPS="$IJ_DIFFUSION_STEPS" \
    INFERENCE_CFG_RATE="$IJ_INFERENCE_CFG_RATE" \
    PAIR_GPU_GUARD_ENABLE="$PAIR_GPU_GUARD_ENABLE" \
    PAIR_GPU_GUARD_GPUS="$PAIR_GPU_GUARD_GPUS" \
    PAIR_GPU_GUARD_PY="$PAIR_GPU_GUARD_PY" \
    PAIR_GPU_GUARD_MATRIX_SIZE="$PAIR_GPU_GUARD_MATRIX_SIZE" \
    PAIR_GPU_GUARD_ACTIVE_MS="$PAIR_GPU_GUARD_ACTIVE_MS" \
    PAIR_GPU_GUARD_IDLE_MS="$PAIR_GPU_GUARD_IDLE_MS" \
    PAIR_GPU_GUARD_DTYPE="$PAIR_GPU_GUARD_DTYPE" \
    PAIR_GPU_GUARD_RESERVE_MIB="$PAIR_GPU_GUARD_RESERVE_MIB" \
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    SEEDVC_HF_CACHE="${SEEDVC_HF_CACHE:-}" \
    SEEDVC_HF_HOME="${SEEDVC_HF_HOME:-}" \
    SEEDVC_TRANSFORMERS_CACHE="${SEEDVC_TRANSFORMERS_CACHE:-}" \
    SEEDVC_HUGGINGFACE_HUB_CACHE="${SEEDVC_HUGGINGFACE_HUB_CACHE:-}" \
    SEEDVC_HF_HUB_OFFLINE="${SEEDVC_HF_HUB_OFFLINE:-1}" \
    SEEDVC_TRANSFORMERS_OFFLINE="${SEEDVC_TRANSFORMERS_OFFLINE:-1}" \
    bash "$SCRIPT_DIR/scripts/run_run03_prosody_speed_pairs.sh"
  done

  echo "[$(timestamp_utc)] $lang: I/J done"
}

verify_vcdata_outputs() {
  local lang="$1"
  local input_dir="$2"
  local job_output_root="$3"
  local final_root="$4"
  local jsonl_path
  local split_name
  local split_dir
  local expected_cases
  local actual_cases

  for jsonl_path in "$input_dir"/*.jsonl; do
    [ -e "$jsonl_path" ] || continue
    split_name=$(basename "$jsonl_path" .jsonl)
    split_dir="$job_output_root/$split_name"

    if split_dir_complete "$final_root/$split_name" "$jsonl_path"; then
      continue
    fi

    expected_cases=$(count_nonempty_lines "$jsonl_path")
    if [ ! -d "$split_dir" ]; then
      echo "ERROR: $lang vcdata split dir missing: $split_dir" >&2
      return 1
    fi

    actual_cases=$(count_manifest_cases "$split_dir")
    if [ "$actual_cases" -ne "$expected_cases" ]; then
      echo "ERROR: $lang vcdata incomplete for $split_name: expected_cases=$expected_cases actual_cases=$actual_cases" >&2
      return 1
    fi

    if [ ! -f "$split_dir/.stage1_generate_state.json" ]; then
      echo "ERROR: $lang vcdata missing state file: $split_dir/.stage1_generate_state.json" >&2
      return 1
    fi
  done
}

publish_vcdata_outputs() {
  local lang="$1"
  local input_dir="$2"
  local job_output_root="$3"
  local final_root="$4"
  local vc_log_root="$5"
  local jsonl_path
  local split_name
  local src_split_dir
  local dst_split_dir
  local log_path

  for jsonl_path in "$input_dir"/*.jsonl; do
    [ -e "$jsonl_path" ] || continue
    split_name=$(basename "$jsonl_path" .jsonl)
    src_split_dir="$job_output_root/$split_name"
    dst_split_dir="$final_root/$split_name"
    if [ -e "$dst_split_dir" ]; then
      if split_dir_complete "$dst_split_dir" "$jsonl_path"; then
        continue
      fi
      echo "ERROR: published vcdata split dir exists but is incomplete: $dst_split_dir" >&2
      return 1
    fi
    if [ ! -d "$src_split_dir" ]; then
      continue
    fi
    mv "$src_split_dir" "$dst_split_dir"
    ln -s "$dst_split_dir" "$src_split_dir"
  done

  for log_path in "$job_output_root"/log_generate_*.txt; do
    [ -e "$log_path" ] || continue
    mv "$log_path" "$vc_log_root/$(basename "$log_path")"
  done
}

for lang in zh en; do
  run_vcdata_for_lang "$lang"
  run_editx_for_lang "$lang"
  run_pairs_for_lang "$lang"
  run_ij_for_lang "$lang"
done

echo "=========================================="
echo "group worker completed"
echo "  finished_at=$(timestamp_utc)"
echo "  job_name=$JOB_NAME"
echo "=========================================="
