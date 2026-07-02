#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
JOB_ROOT="$SCRIPT_DIR"

RUNNER_SCRIPT="${RUNNER_SCRIPT:-$JOB_ROOT/run_vcdata_editx_group.sh}"
RUNNER_SCRIPT_BASENAME=$(basename "$RUNNER_SCRIPT")

QZCLI="${QZCLI:-$JOB_ROOT/scripts/qzcli_with_deps.sh}"
WORKSPACE="${WORKSPACE:-CI-情境智能}"
PROJECT="${PROJECT:-CI-情境智能}"
# Default to MOSS-tts-data-tmp only. Users can still override via
# COMPUTE_GROUP/COMPUTE_GROUPS_CSV when they explicitly need another pool.
DEFAULT_MOSS_COMPUTE_GROUP="lcg-73efc9b6-8d94-406c-b150-50c91fda377f"
COMPUTE_GROUP="${COMPUTE_GROUP:-$DEFAULT_MOSS_COMPUTE_GROUP}"
DEFAULT_COMPUTE_GROUPS_CSV="${DEFAULT_COMPUTE_GROUPS_CSV:-$COMPUTE_GROUP}"
COMPUTE_GROUPS_CSV="${COMPUTE_GROUPS_CSV:-$DEFAULT_COMPUTE_GROUPS_CSV}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-3}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
INSTANCES=1

INPUT_DIR="${INPUT_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_data_temp/delivery_filter_10k}"
RUN_ROOT="${RUN_ROOT:-$JOB_ROOT/outputs/delivery_filter_10k_qz_20260604}"
ZH_MAX_INDEX="${ZH_MAX_INDEX:-197}"

VCDATA_REPO="${VCDATA_REPO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction}"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-$VCDATA_REPO/activate_moss_ttsd_vc.sh}"
MODEL_DIR="${MODEL_DIR:-$VCDATA_REPO/MOSS-TTS}"
VC_EDIT_REPO="${VC_EDIT_REPO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_edit}"
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
FORCE_INPUT_LANG="${FORCE_INPUT_LANG:-}"
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
RUN_PAIRS_ON_QZ="${RUN_PAIRS_ON_QZ:-1}"
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

TASK_COUNT="${TASK_COUNT:-16}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-xyzhang-delivery10k-vceditx}"
BATCH_ID="${BATCH_ID:-$(date -u +delivery10k-%m%d-%H%M%S)}"
GROUP_ROOT="${GROUP_ROOT:-$JOB_ROOT/.qz_vcdata_editx_batches/$BATCH_ID}"
AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-0}"
FORCE_NEW_BATCH="${FORCE_NEW_BATCH:-0}"
RESUME_BATCH_ID="${RESUME_BATCH_ID:-}"
RESUBMIT_GROUP_TAGS="${RESUBMIT_GROUP_TAGS:-}"

HF_HOME="${HF_HOME:-$JOB_ROOT/.hf_cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SEEDVC_HF_CACHE="${SEEDVC_HF_CACHE:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes/third_party/seed-vc/checkpoints/hf_cache}"
SEEDVC_HF_HOME="${SEEDVC_HF_HOME:-$SEEDVC_HF_CACHE}"
SEEDVC_TRANSFORMERS_CACHE="${SEEDVC_TRANSFORMERS_CACHE:-$SEEDVC_HF_CACHE}"
SEEDVC_HUGGINGFACE_HUB_CACHE="${SEEDVC_HUGGINGFACE_HUB_CACHE:-$SEEDVC_HF_CACHE}"
SEEDVC_HF_HUB_OFFLINE="${SEEDVC_HF_HUB_OFFLINE:-1}"
SEEDVC_TRANSFORMERS_OFFLINE="${SEEDVC_TRANSFORMERS_OFFLINE:-1}"
DRY_RUN="${DRY_RUN:-0}"

declare -a PENDING_JSONLS=()
declare -a PENDING_LANGS=()
declare -a PENDING_REMAINING=()
declare -a PENDING_TOTAL_CASES=()
declare -a PENDING_VCDATA_DONE=()
declare -a PENDING_EDIT_DONE=()
declare -a COMPUTE_GROUPS=()
declare -a GROUP_DIRS=()
declare -a GROUP_SPLIT_COUNTS=()
declare -a GROUP_REMAINING_CASES=()
declare -A SUBMITTED_GROUP_TAGS=()
declare -A RESUBMIT_GROUP_TAG_SET=()

TOTAL_SPLITS=0
SKIPPED_SPLITS=0
PENDING_COUNT=0
GROUP_COUNT=0
SUMMARY_PATH=""
RESUMING_BATCH=0
HAS_RESUBMIT_GROUP_TAGS=0

for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    DRY_RUN=1
  fi
done

trim_spaces() {
  printf '%s' "$1" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

normalize_group_tag() {
  local raw_tag="$1"
  raw_tag=$(trim_spaces "$raw_tag")
  raw_tag=${raw_tag#group_}
  raw_tag=${raw_tag#group-}
  raw_tag=${raw_tag#g}
  raw_tag=${raw_tag#G}

  if [[ ! "$raw_tag" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid group tag '$1' in RESUBMIT_GROUP_TAGS" >&2
    return 1
  fi

  printf '%02d\n' "$((10#$raw_tag))"
}

parse_resubmit_group_tags() {
  local raw_tag
  local normalized_tag

  RESUBMIT_GROUP_TAG_SET=()
  HAS_RESUBMIT_GROUP_TAGS=0

  [ -n "$RESUBMIT_GROUP_TAGS" ] || return 0

  IFS=',' read -r -a raw_tags <<< "$RESUBMIT_GROUP_TAGS"
  for raw_tag in "${raw_tags[@]}"; do
    raw_tag=$(trim_spaces "$raw_tag")
    [ -n "$raw_tag" ] || continue
    normalized_tag=$(normalize_group_tag "$raw_tag") || return 1
    RESUBMIT_GROUP_TAG_SET["$normalized_tag"]=1
    HAS_RESUBMIT_GROUP_TAGS=1
  done
}

count_nonempty_lines() {
  awk 'NF { c += 1 } END { print c + 0 }' "$1"
}

count_vcdata_done_cases() {
  local split_dir="$1"
  local manifest_paths=("$split_dir"/manifest_shard*.jsonl)
  if [ ! -e "${manifest_paths[0]}" ]; then
    echo 0
    return
  fi
  wc -l "${manifest_paths[@]}" | awk 'END { print $1 + 0 }'
}

edit_pairs_for_lang() {
  case "$1" in
    zh) printf '%s\n' "$ZH_EDIT_PAIRS" ;;
    en) printf '%s\n' "$EN_EDIT_PAIRS" ;;
    *)
      echo "ERROR: unsupported lang '$1'" >&2
      return 1
      ;;
  esac
}

count_edit_modes() {
  local edit_pairs="$1"
  local -a pairs=()
  if [ -z "$edit_pairs" ]; then
    echo 0
    return
  fi
  IFS=',' read -r -a pairs <<< "$edit_pairs"
  echo "${#pairs[@]}"
}

count_edit_done_cases() {
  local split_dir="$1"
  local edit_pairs="$2"
  local done=0
  local pair
  local edit_type
  local edit_info
  local edit_tag
  local report
  local -a pairs=()

  if [ -z "$edit_pairs" ]; then
    echo 0
    return
  fi

  IFS=',' read -r -a pairs <<< "$edit_pairs"
  for pair in "${pairs[@]}"; do
    [ -n "$pair" ] || continue
    edit_type="${pair%%:*}"
    edit_info="${pair##*:}"
    edit_tag=$(printf '%s_%s' "$edit_type" "$edit_info" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')
    for report in "$split_dir"/stepaudio_${edit_tag}_*/paired_report.jsonl; do
      [ -f "$report" ] || continue
      done=$((done + $(wc -l < "$report")))
      break
    done
  done
  echo "$done"
}

detect_lang_for_jsonl() {
  local jsonl_path="$1"
  local stem
  local idx

  if [ -n "$FORCE_INPUT_LANG" ]; then
    case "$FORCE_INPUT_LANG" in
      zh|en)
        echo "$FORCE_INPUT_LANG"
        return 0
        ;;
      *)
        echo "ERROR: unsupported FORCE_INPUT_LANG=$FORCE_INPUT_LANG" >&2
        return 1
        ;;
    esac
  fi

  stem=$(basename "$jsonl_path" .jsonl)
  case "$stem" in
    zh_*)
      echo zh
      return 0
      ;;
    en_*)
      echo en
      return 0
      ;;
    *_zh)
      echo zh
      return 0
      ;;
    *_en)
      echo en
      return 0
      ;;
  esac
  idx=$(printf '%s\n' "$stem" | sed -n 's/.*_\([0-9][0-9][0-9][0-9]\)$/\1/p')
  if [ -z "$idx" ]; then
    echo "ERROR: failed to parse manifest index from $jsonl_path" >&2
    return 1
  fi
  if [ "$idx" -le "$ZH_MAX_INDEX" ]; then
    echo zh
  else
    echo en
  fi
}

parse_compute_groups() {
  local raw_group
  COMPUTE_GROUPS=()

  if [ -n "$COMPUTE_GROUPS_CSV" ]; then
    IFS=',' read -r -a raw_groups <<< "$COMPUTE_GROUPS_CSV"
    for raw_group in "${raw_groups[@]}"; do
      raw_group=$(trim_spaces "$raw_group")
      if [ -n "$raw_group" ]; then
        COMPUTE_GROUPS+=("$raw_group")
      fi
    done
  fi

  if [ "${#COMPUTE_GROUPS[@]}" -eq 0 ]; then
    COMPUTE_GROUPS=("$COMPUTE_GROUP")
  fi
}

collect_pending_jsonls() {
  local jsonl_path
  local split_name
  local split_dir
  local lang
  local total_cases
  local vc_done
  local vc_remaining
  local edit_pairs
  local edit_modes
  local edit_expected
  local edit_done
  local edit_remaining
  local remaining

  PENDING_JSONLS=()
  PENDING_LANGS=()
  PENDING_REMAINING=()
  PENDING_TOTAL_CASES=()
  PENDING_VCDATA_DONE=()
  PENDING_EDIT_DONE=()
  TOTAL_SPLITS=0
  SKIPPED_SPLITS=0

  for jsonl_path in "$INPUT_DIR"/*.jsonl; do
    if [ ! -e "$jsonl_path" ]; then
      continue
    fi

    TOTAL_SPLITS=$((TOTAL_SPLITS + 1))
    lang=$(detect_lang_for_jsonl "$jsonl_path")
    split_name=$(basename "$jsonl_path" .jsonl)
    split_dir="$RUN_ROOT/vcdata/$lang/$split_name"
    total_cases=$(count_nonempty_lines "$jsonl_path")
    vc_done=0
    edit_done=0

    if [ -d "$split_dir" ]; then
      vc_done=$(count_vcdata_done_cases "$split_dir")
      edit_pairs=$(edit_pairs_for_lang "$lang")
      edit_done=$(count_edit_done_cases "$split_dir" "$edit_pairs")
    else
      edit_pairs=$(edit_pairs_for_lang "$lang")
    fi

    edit_modes=$(count_edit_modes "$edit_pairs")
    vc_remaining=$((total_cases - vc_done))
    if [ "$vc_remaining" -lt 0 ]; then
      vc_remaining=0
    fi
    edit_expected=$((total_cases * edit_modes))
    edit_remaining=$((edit_expected - edit_done))
    if [ "$edit_remaining" -lt 0 ]; then
      edit_remaining=0
    fi
    remaining=$((vc_remaining + edit_remaining))

    if [ "$remaining" -eq 0 ] && [ "$total_cases" -gt 0 ] && [ -f "$split_dir/.stage1_generate_state.json" ]; then
      SKIPPED_SPLITS=$((SKIPPED_SPLITS + 1))
      continue
    fi

    PENDING_JSONLS+=("$jsonl_path")
    PENDING_LANGS+=("$lang")
    PENDING_REMAINING+=("$remaining")
    PENDING_TOTAL_CASES+=("$total_cases")
    PENDING_VCDATA_DONE+=("$vc_done")
    PENDING_EDIT_DONE+=("$edit_done")
  done

  if [ "$TOTAL_SPLITS" -eq 0 ]; then
    echo "ERROR: No .jsonl files found under $INPUT_DIR"
    return 2
  fi

  PENDING_COUNT="${#PENDING_JSONLS[@]}"
  if [ "$PENDING_COUNT" -eq 0 ]; then
    return 3
  fi
}

latest_batch_dir() {
  ls -dt "$JOB_ROOT"/.qz_vcdata_editx_batches/* 2>/dev/null | head -n 1 || true
}

discover_resume_batch() {
  local candidate=""
  local group_count_on_disk=0
  local submitted_count=0

  if [ "$FORCE_NEW_BATCH" -eq 1 ]; then
    return 1
  fi

  if [ -n "$RESUME_BATCH_ID" ]; then
    candidate="$JOB_ROOT/.qz_vcdata_editx_batches/$RESUME_BATCH_ID"
  elif [ "$AUTO_RESUME_LATEST" -eq 1 ]; then
    candidate=$(latest_batch_dir)
  fi

  if [ -z "$candidate" ] || [ ! -d "$candidate" ]; then
    return 1
  fi

  group_count_on_disk=$(find "$candidate" -maxdepth 1 -type d -name 'group_*' | wc -l | awk '{print $1 + 0}')
  if [ "$group_count_on_disk" -le 0 ]; then
    return 1
  fi

  if [ -f "$candidate/submitted_jobs.tsv" ]; then
    submitted_count=$(awk 'END { print NR + 0 }' "$candidate/submitted_jobs.tsv")
  fi

  if [ "$submitted_count" -ge "$group_count_on_disk" ] && [ "$HAS_RESUBMIT_GROUP_TAGS" -eq 0 ]; then
    return 1
  fi

  BATCH_ID=$(basename "$candidate")
  GROUP_ROOT="$candidate"
  RESUMING_BATCH=1
  return 0
}

prepare_new_group_dirs() {
  local group_id
  local group_dir

  GROUP_COUNT="$TASK_COUNT"
  if [ "$GROUP_COUNT" -gt "$PENDING_COUNT" ]; then
    GROUP_COUNT="$PENDING_COUNT"
  fi
  if [ "$GROUP_COUNT" -le 0 ]; then
    echo "ERROR: TASK_COUNT must be >= 1"
    return 1
  fi

  mkdir -p "$GROUP_ROOT"
  GROUP_DIRS=()
  GROUP_SPLIT_COUNTS=()
  GROUP_REMAINING_CASES=()

  for group_id in $(seq 0 $((GROUP_COUNT - 1))); do
    group_dir="$GROUP_ROOT/group_$(printf '%02d' "$group_id")"
    mkdir -p "$group_dir/inputs/zh" "$group_dir/inputs/en"
    : > "$group_dir/plan.tsv"
    GROUP_DIRS+=("$group_dir")
    GROUP_SPLIT_COUNTS+=(0)
    GROUP_REMAINING_CASES+=(0)
  done
}

assign_new_groups_balanced() {
  local sort_rows_file
  local idx
  local line
  local remaining
  local lang
  local jsonl_path
  local total_cases
  local vc_done
  local edit_done
  local basename
  local best_group
  local best_load
  local group_id
  local group_load
  local group_dir

  sort_rows_file=$(mktemp)
  for idx in "${!PENDING_JSONLS[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${PENDING_REMAINING[$idx]}" \
      "${PENDING_LANGS[$idx]}" \
      "${PENDING_JSONLS[$idx]}" \
      "${PENDING_TOTAL_CASES[$idx]}" \
      "${PENDING_VCDATA_DONE[$idx]}" \
      "${PENDING_EDIT_DONE[$idx]}" >> "$sort_rows_file"
  done

  while IFS=$'\t' read -r remaining lang jsonl_path total_cases vc_done edit_done; do
    best_group=0
    best_load="${GROUP_REMAINING_CASES[0]}"
    for group_id in $(seq 1 $((GROUP_COUNT - 1))); do
      group_load="${GROUP_REMAINING_CASES[$group_id]}"
      if [ "$group_load" -lt "$best_load" ]; then
        best_group="$group_id"
        best_load="$group_load"
      fi
    done

    basename=$(basename "$jsonl_path")
    group_dir="${GROUP_DIRS[$best_group]}"
    ln -sf "$jsonl_path" "$group_dir/inputs/$lang/$basename"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$lang" \
      "$basename" \
      "$remaining" \
      "$total_cases" \
      "$vc_done" \
      "$edit_done" >> "$group_dir/plan.tsv"
    GROUP_SPLIT_COUNTS[$best_group]=$((GROUP_SPLIT_COUNTS[$best_group] + 1))
    GROUP_REMAINING_CASES[$best_group]=$((GROUP_REMAINING_CASES[$best_group] + remaining))
  done < <(sort -rn -k1,1 "$sort_rows_file")

  rm -f "$sort_rows_file"
}

load_existing_groups() {
  local group_dir
  local split_count
  local remaining_cases

  GROUP_DIRS=()
  GROUP_SPLIT_COUNTS=()
  GROUP_REMAINING_CASES=()

  mapfile -t GROUP_DIRS < <(find "$GROUP_ROOT" -maxdepth 1 -type d -name 'group_*' | sort)
  GROUP_COUNT="${#GROUP_DIRS[@]}"
  if [ "$GROUP_COUNT" -le 0 ]; then
    echo "ERROR: No existing group_* directories found under $GROUP_ROOT"
    return 1
  fi

  for group_dir in "${GROUP_DIRS[@]}"; do
    split_count=0
    remaining_cases=0
    if [ -f "$group_dir/plan.tsv" ]; then
      split_count=$(awk 'END { print NR + 0 }' "$group_dir/plan.tsv")
      remaining_cases=$(awk -F '\t' '{ s += $3 } END { print s + 0 }' "$group_dir/plan.tsv")
    fi
    GROUP_SPLIT_COUNTS+=("$split_count")
    GROUP_REMAINING_CASES+=("$remaining_cases")
  done
}

load_submitted_group_tags() {
  local job_name
  SUBMITTED_GROUP_TAGS=()
  if [ ! -f "$SUMMARY_PATH" ]; then
    return
  fi

  while IFS=$'\t' read -r job_name _rest; do
    if [[ "$job_name" =~ -g([0-9]{2})$ ]]; then
      SUBMITTED_GROUP_TAGS["${BASH_REMATCH[1]}"]=1
    fi
  done < "$SUMMARY_PATH"
}

job_command_for_group() {
  local group_dir="$1"
  local group_tag="$2"
  local job_name="$3"
  local seed="$4"

  cat <<EOF
bash -lc 'cd "$JOB_ROOT" && \
HF_HOME="$HF_HOME" \
TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
HF_ENDPOINT="$HF_ENDPOINT" \
SEEDVC_HF_CACHE="$SEEDVC_HF_CACHE" \
SEEDVC_HF_HOME="$SEEDVC_HF_HOME" \
SEEDVC_TRANSFORMERS_CACHE="$SEEDVC_TRANSFORMERS_CACHE" \
SEEDVC_HUGGINGFACE_HUB_CACHE="$SEEDVC_HUGGINGFACE_HUB_CACHE" \
SEEDVC_HF_HUB_OFFLINE="$SEEDVC_HF_HUB_OFFLINE" \
SEEDVC_TRANSFORMERS_OFFLINE="$SEEDVC_TRANSFORMERS_OFFLINE" \
GROUP_DIR="$group_dir" \
RUN_ROOT="$RUN_ROOT" \
GROUP_TAG="$group_tag" \
JOB_NAME="$job_name" \
LOG_FILE="$RUN_ROOT/logs/${job_name}.log" \
VCDATA_REPO="$VCDATA_REPO" \
ACTIVATE_SCRIPT="$ACTIVATE_SCRIPT" \
MODEL_DIR="$MODEL_DIR" \
VC_EDIT_REPO="$VC_EDIT_REPO" \
STEPX_PY="$STEPX_PY" \
ZH_TEXT_FIELD="$ZH_TEXT_FIELD" \
EN_TEXT_FIELD="$EN_TEXT_FIELD" \
ZH_EDIT_PAIRS="$ZH_EDIT_PAIRS" \
EN_EDIT_PAIRS="$EN_EDIT_PAIRS" \
DISABLE_EDITX="$DISABLE_EDITX" \
AUDIO_PATH_FIELD="$AUDIO_PATH_FIELD" \
NUM_CANDIDATES="$NUM_CANDIDATES" \
BATCH_SIZE="$BATCH_SIZE" \
SIMILARITY_THRESHOLD="$SIMILARITY_THRESHOLD" \
SEED_BASE="$seed" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
WORKERS_PER_GPU="$WORKERS_PER_GPU" \
MAX_RETRIES="$MAX_RETRIES" \
EDITX_PREPARE_WORKERS="$EDITX_PREPARE_WORKERS" \
EDITX_BATCH_SIZE="$EDITX_BATCH_SIZE" \
EDITX_MAX_NUM_SEQS="$EDITX_MAX_NUM_SEQS" \
EDITX_TENSOR_PARALLEL_SIZE="$EDITX_TENSOR_PARALLEL_SIZE" \
EDITX_ENGINE_COUNT="$EDITX_ENGINE_COUNT" \
EDITX_ENGINE_GPU_GROUPS="$EDITX_ENGINE_GPU_GROUPS" \
EDITX_ENGINE_JOB_SPLIT_MODE="$EDITX_ENGINE_JOB_SPLIT_MODE" \
EDITX_ENGINE_STARTUP_STAGGER_SEC="$EDITX_ENGINE_STARTUP_STAGGER_SEC" \
EDITX_ENGINE_PER_ATTEMPT_RETRIES="$EDITX_ENGINE_PER_ATTEMPT_RETRIES" \
EDITX_ENGINE_PER_ATTEMPT_BACKOFF_SEC="$EDITX_ENGINE_PER_ATTEMPT_BACKOFF_SEC" \
EDITX_AUDIO_DURATION_BUCKETING="$EDITX_AUDIO_DURATION_BUCKETING" \
EDITX_DURATION_BUCKET_WINDOW="$EDITX_DURATION_BUCKET_WINDOW" \
EDITX_NEXT_BATCH_PREFETCH="$EDITX_NEXT_BATCH_PREFETCH" \
EDITX_PREFETCH_DEPTH="$EDITX_PREFETCH_DEPTH" \
EDITX_AUDIO_CONDITION_BUILD_WORKERS="$EDITX_AUDIO_CONDITION_BUILD_WORKERS" \
EDITX_DISABLE_AUDIO_CONDITION_ITEM_PARALLEL="$EDITX_DISABLE_AUDIO_CONDITION_ITEM_PARALLEL" \
EDITX_USE_ASYNC_ENGINE="$EDITX_USE_ASYNC_ENGINE" \
EDITX_STREAM_VOCODE="$EDITX_STREAM_VOCODE" \
EDITX_PREPARE_BREAKDOWN="$EDITX_PREPARE_BREAKDOWN" \
EDITX_ASYNC_WRITE_WORKERS="$EDITX_ASYNC_WRITE_WORKERS" \
EDITX_ASYNC_VOCODER_WORKERS="$EDITX_ASYNC_VOCODER_WORKERS" \
EDITX_GPU_MEMORY_UTILIZATION="$EDITX_GPU_MEMORY_UTILIZATION" \
EDITX_MAX_MODEL_LEN="$EDITX_MAX_MODEL_LEN" \
EDITX_DTYPE="$EDITX_DTYPE" \
EDITX_PREPROCESS_CACHE_SIZE="$EDITX_PREPROCESS_CACHE_SIZE" \
RUN_PAIRS_ON_QZ="$RUN_PAIRS_ON_QZ" \
PAIR_LOCAL_DEVICE="$PAIR_LOCAL_DEVICE" \
PAIR_LOCAL_RESUME="$PAIR_LOCAL_RESUME" \
PAIR_LOCAL_PARALLEL="$PAIR_LOCAL_PARALLEL" \
PAIR_LOCAL_JOBS="$PAIR_LOCAL_JOBS" \
PAIR_GPU_GUARD_ENABLE="$PAIR_GPU_GUARD_ENABLE" \
PAIR_GPU_GUARD_GPUS="$PAIR_GPU_GUARD_GPUS" \
PAIR_GPU_GUARD_PY="$PAIR_GPU_GUARD_PY" \
PAIR_GPU_GUARD_MATRIX_SIZE="$PAIR_GPU_GUARD_MATRIX_SIZE" \
PAIR_GPU_GUARD_ACTIVE_MS="$PAIR_GPU_GUARD_ACTIVE_MS" \
PAIR_GPU_GUARD_IDLE_MS="$PAIR_GPU_GUARD_IDLE_MS" \
PAIR_GPU_GUARD_DTYPE="$PAIR_GPU_GUARD_DTYPE" \
PAIR_GPU_GUARD_RESERVE_MIB="$PAIR_GPU_GUARD_RESERVE_MIB" \
RUN_IJ_ON_QZ="$RUN_IJ_ON_QZ" \
IJ_LIMIT="$IJ_LIMIT" \
IJ_DEVICE="$IJ_DEVICE" \
IJ_RUN_SPEED="$IJ_RUN_SPEED" \
IJ_RUN_PROSODY="$IJ_RUN_PROSODY" \
IJ_RUN_WAVLM="$IJ_RUN_WAVLM" \
IJ_RUN_QC="$IJ_RUN_QC" \
IJ_RUN_PAIR_AUDIO_METRICS="$IJ_RUN_PAIR_AUDIO_METRICS" \
IJ_SPEED_BATCH_SIZE="$IJ_SPEED_BATCH_SIZE" \
IJ_SPEED_MAX_NUM_SEQS="$IJ_SPEED_MAX_NUM_SEQS" \
IJ_SPEED_TENSOR_PARALLEL_SIZE="$IJ_SPEED_TENSOR_PARALLEL_SIZE" \
IJ_SPEED_GPU_MEMORY_UTILIZATION="$IJ_SPEED_GPU_MEMORY_UTILIZATION" \
IJ_SPEED_PREPARE_WORKERS="$IJ_SPEED_PREPARE_WORKERS" \
IJ_SPEED_AUDIO_CONDITION_BUILD_WORKERS="$IJ_SPEED_AUDIO_CONDITION_BUILD_WORKERS" \
IJ_SPEED_PREPROCESS_CACHE_SIZE="$IJ_SPEED_PREPROCESS_CACHE_SIZE" \
IJ_SEEDVC_SHARDED="$IJ_SEEDVC_SHARDED" \
IJ_SEEDVC_GPU_IDS="$IJ_SEEDVC_GPU_IDS" \
IJ_SEEDVC_SHARD_COUNT="$IJ_SEEDVC_SHARD_COUNT" \
IJ_MAX_JOBS="$IJ_MAX_JOBS" \
IJ_SKIP_EXISTING="$IJ_SKIP_EXISTING" \
IJ_TIMBRE_PICK="$IJ_TIMBRE_PICK" \
IJ_SEED="$IJ_SEED" \
IJ_DIFFUSION_STEPS="$IJ_DIFFUSION_STEPS" \
IJ_INFERENCE_CFG_RATE="$IJ_INFERENCE_CFG_RATE" \
bash "$RUNNER_SCRIPT_BASENAME"'
EOF
}

mkdir -p \
  "$RUN_ROOT/vcdata/zh" \
  "$RUN_ROOT/vcdata/en" \
  "$RUN_ROOT/vcdata_job_runs/zh" \
  "$RUN_ROOT/vcdata_job_runs/en" \
  "$RUN_ROOT/pairs/zh" \
  "$RUN_ROOT/pairs/en" \
  "$RUN_ROOT/logs" \
  "$TRANSFORMERS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$SEEDVC_HUGGINGFACE_HUB_CACHE"

parse_compute_groups
parse_resubmit_group_tags

if discover_resume_batch; then
  load_existing_groups
else
  set +e
  collect_pending_jsonls
  PREPARE_STATUS=$?
  set -e

  if [ "$PREPARE_STATUS" -eq 2 ]; then
    exit 1
  fi
  if [ "$PREPARE_STATUS" -eq 3 ]; then
    echo "All splits are already complete. Nothing to submit."
    exit 0
  fi

  prepare_new_group_dirs
  assign_new_groups_balanced
fi

SUMMARY_PATH="$GROUP_ROOT/submitted_jobs.tsv"
touch "$SUMMARY_PATH"
load_submitted_group_tags

echo "=========================================="
echo "QZ combined batch submit: vcdata -> editx"
echo "  BATCH_ID=$BATCH_ID"
echo "  INPUT_DIR=$INPUT_DIR"
echo "  RUN_ROOT=$RUN_ROOT"
echo "  GROUP_ROOT=$GROUP_ROOT"
echo "  RUNNER_SCRIPT=$RUNNER_SCRIPT_BASENAME"
if [ "$RESUMING_BATCH" -eq 1 ]; then
  if [ "$HAS_RESUBMIT_GROUP_TAGS" -eq 1 ]; then
    echo "  MODE=resubmit_selected_groups"
  else
    echo "  MODE=resume_incomplete_batch"
  fi
else
  echo "  MODE=create_new_batch"
  echo "  TOTAL_SPLITS=$TOTAL_SPLITS"
  echo "  SKIPPED_COMPLETED_SPLITS=$SKIPPED_SPLITS"
  echo "  PENDING_SPLITS=$PENDING_COUNT"
fi
if [ "$HAS_RESUBMIT_GROUP_TAGS" -eq 1 ]; then
  echo "  RESUBMIT_GROUP_TAGS=$RESUBMIT_GROUP_TAGS"
fi
echo "  TASK_COUNT=$TASK_COUNT"
echo "  EFFECTIVE_GROUP_COUNT=$GROUP_COUNT"
echo "  COMPUTE_GROUPS=${COMPUTE_GROUPS[*]}"
echo "  QZCLI=$QZCLI"
echo "  ZH_RANGE=0000-$ZH_MAX_INDEX"
echo "  EN_RANGE=$((ZH_MAX_INDEX + 1))+"
echo "  FORCE_INPUT_LANG=${FORCE_INPUT_LANG:-auto}"
echo "  ZH_TEXT_FIELD=$ZH_TEXT_FIELD"
echo "  EN_TEXT_FIELD=$EN_TEXT_FIELD"
echo "  ZH_EDIT_PAIRS=$ZH_EDIT_PAIRS"
echo "  EN_EDIT_PAIRS=$EN_EDIT_PAIRS"
echo "  IMAGE=$IMAGE"
echo "  IMAGE_TYPE=$IMAGE_TYPE"
echo "  PRIORITY=$PRIORITY"
echo "  SPEC=$SPEC"
echo "  RUN_IJ_ON_QZ=$RUN_IJ_ON_QZ"
echo "  IJ_LIMIT=$IJ_LIMIT"
echo "  IJ_RUN_PAIR_AUDIO_METRICS=$IJ_RUN_PAIR_AUDIO_METRICS"
echo "  IJ_SPEED_TENSOR_PARALLEL_SIZE=$IJ_SPEED_TENSOR_PARALLEL_SIZE"
echo "  IJ_SPEED_GPU_MEMORY_UTILIZATION=$IJ_SPEED_GPU_MEMORY_UTILIZATION"
echo "  IJ_SEEDVC_SHARDED=$IJ_SEEDVC_SHARDED"
echo "  SEEDVC_HF_CACHE=$SEEDVC_HF_CACHE"
echo "  SEEDVC_HF_HUB_OFFLINE=$SEEDVC_HF_HUB_OFFLINE"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================="
echo "Platform logs are per-job. Local logs are also written to $RUN_ROOT/logs/<job_name>.log to avoid overwriting across groups/batches."
echo "=========================================="

for group_id in $(seq 0 $((GROUP_COUNT - 1))); do
  group_tag=$(printf '%02d' "$group_id")
  group_dir="${GROUP_DIRS[$group_id]}"
  split_count="${GROUP_SPLIT_COUNTS[$group_id]}"
  remaining_cases="${GROUP_REMAINING_CASES[$group_id]}"
  compute_group_name="${COMPUTE_GROUPS[$((group_id % ${#COMPUTE_GROUPS[@]}))]}"
  job_name="${JOB_NAME_PREFIX}-${BATCH_ID}-g${group_tag}"
  seed_for_group=$((SEED_BASE + group_id))
  job_command=$(job_command_for_group "$group_dir" "$group_tag" "$job_name" "$seed_for_group")

  echo "------------------------------------------"
  echo "Group $group_id/$GROUP_COUNT"
  echo "  JOB_NAME=$job_name"
  echo "  COMPUTE_GROUP=$compute_group_name"
  echo "  SPLIT_COUNT=$split_count"
  echo "  REMAINING_CASES=$remaining_cases"
  echo "  GROUP_DIR=$group_dir"
  echo "  PLAN_TSV=$group_dir/plan.tsv"

  if [ "$HAS_RESUBMIT_GROUP_TAGS" -eq 1 ] && [ -z "${RESUBMIT_GROUP_TAG_SET[$group_tag]:-}" ]; then
    echo "  Skipped: not selected in RESUBMIT_GROUP_TAGS"
    continue
  fi

  if [ "$split_count" -eq 0 ]; then
    echo "  Skipped: no assigned splits"
    continue
  fi

  if [ -n "${SUBMITTED_GROUP_TAGS[$group_tag]:-}" ] && [ "$HAS_RESUBMIT_GROUP_TAGS" -eq 0 ]; then
    echo "  Skipped: already submitted in $SUMMARY_PATH"
    continue
  fi
  if [ -n "${SUBMITTED_GROUP_TAGS[$group_tag]:-}" ] && [ "$HAS_RESUBMIT_GROUP_TAGS" -eq 1 ]; then
    echo "  Resubmitting: already present in $SUMMARY_PATH"
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  Dry run command:"
    printf '%s\n' "$job_command"
    continue
  fi

  TMP_OUTPUT=$(mktemp)

  set +e
  env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
    HF_HOME="$HF_HOME" \
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    "$QZCLI" create-job \
    --name "$job_name" \
    --workspace "$WORKSPACE" \
    --project "$PROJECT" \
    --compute-group "$compute_group_name" \
    --spec "$SPEC" \
    --framework "$FRAMEWORK" \
    --instances "$INSTANCES" \
    --shm "$SHM_GI" \
    --priority "$PRIORITY" \
    --image "$IMAGE" \
    --image-type "$IMAGE_TYPE" \
    --command "$job_command" >"$TMP_OUTPUT" 2>&1
  status=$?
  set -e

  cat "$TMP_OUTPUT"

  if [ "$status" -ne 0 ]; then
    echo "Submission failed for $job_name"
    if grep -q 'Cookie 已过期或无效' "$TMP_OUTPUT"; then
      echo "Fix: qzcli login"
    fi
    rm -f "$TMP_OUTPUT"
    exit "$status"
  fi

  job_id=$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$TMP_OUTPUT" | tail -n 1 || true)
  if [ -z "$job_id" ]; then
    job_uuid=$(grep -E '任务ID|job_id|Job ID' "$TMP_OUTPUT" | grep -Eo '[0-9a-fA-F-]{36}' | tail -n 1 || true)
    if [ -n "$job_uuid" ]; then
      job_id="job-$job_uuid"
    fi
  fi

  printf '%s\t%s\t%s\t%s\n' "$job_name" "${job_id:-UNKNOWN}" "$compute_group_name" "$group_dir" >> "$SUMMARY_PATH"
  SUBMITTED_GROUP_TAGS["$group_tag"]=1
  rm -f "$TMP_OUTPUT"
done

echo "=========================================="
echo "Combined batch submission completed."
echo "  SUMMARY_PATH=$SUMMARY_PATH"
echo "  GROUP_ROOT=$GROUP_ROOT"
echo "  RUN_ROOT=$RUN_ROOT"
echo "=========================================="
