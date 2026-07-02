#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

SOURCE_ROOT="${SOURCE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_data_temp/mtd_pass_nonmulti_primary_le_0p3_split_10k}"
PAIR_OUTPUTS_ROOT="${PAIR_OUTPUTS_ROOT:-${ROOT}/outputs/moss_tts_data_temp_zhen10k_qz_20260613_run03/pair_outputs}"
PROSODY_CONFIG="${PROSODY_CONFIG:-${ROOT}/configs/prosody_routes.yaml}"

EMOTION_PY_BIN="${EMOTION_PY_BIN:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/emotion/bin/python}"
WAVLMPY="${WAVLMPY:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/moss_ttsd_sglang/bin/python}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/contts-train/bin/python}"
SEED_VC_DIR="${SEED_VC_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes/third_party/seed-vc}"
DEPS_DIR="${DEPS_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes/.deps/seedvc}"
VC_EDIT_ROOT="${VC_EDIT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_edit}"
SEEDVC_HF_CACHE="${SEEDVC_HF_CACHE:-${SEED_VC_DIR}/checkpoints/hf_cache}"
SEEDVC_HF_HOME="${SEEDVC_HF_HOME:-$SEEDVC_HF_CACHE}"
SEEDVC_TRANSFORMERS_CACHE="${SEEDVC_TRANSFORMERS_CACHE:-$SEEDVC_HF_CACHE}"
SEEDVC_HUGGINGFACE_HUB_CACHE="${SEEDVC_HUGGINGFACE_HUB_CACHE:-$SEEDVC_HF_CACHE}"
SEEDVC_HF_HUB_OFFLINE="${SEEDVC_HF_HUB_OFFLINE:-1}"
SEEDVC_TRANSFORMERS_OFFLINE="${SEEDVC_TRANSFORMERS_OFFLINE:-1}"

export PYTHONPATH="$DEPS_DIR:$ROOT/scripts:$SEED_VC_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SPLITS="${SPLITS:-zh/zh_slim_0002 zh/zh_slim_0003 en/en_slim_0002 en/en_slim_0003}"
LIMIT="${LIMIT:-500}"
DEVICE="${DEVICE:-cuda:0}"
RUN_SPEED="${RUN_SPEED:-1}"
RUN_PROSODY="${RUN_PROSODY:-1}"
RUN_WAVLM="${RUN_WAVLM:-1}"
RUN_QC="${RUN_QC:-0}"
RUN_PAIR_AUDIO_METRICS="${RUN_PAIR_AUDIO_METRICS:-$RUN_QC}"
DRY_RUN="${DRY_RUN:-0}"

DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
SEED="${SEED:-42}"
TIMBRE_PICK="${TIMBRE_PICK:-random}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_JOBS="${MAX_JOBS:-0}"
SPEED_TAGS="${SPEED_TAGS:-speed_faster,speed_slower}"
SPEED_BATCH_SIZE="${SPEED_BATCH_SIZE:-${BATCH_SIZE:-1}}"
SPEED_MAX_NUM_SEQS="${SPEED_MAX_NUM_SEQS:-${MAX_NUM_SEQS:-$SPEED_BATCH_SIZE}}"
SPEED_TENSOR_PARALLEL_SIZE="${SPEED_TENSOR_PARALLEL_SIZE:-${TENSOR_PARALLEL_SIZE:-1}}"
SPEED_GPU_MEMORY_UTILIZATION="${SPEED_GPU_MEMORY_UTILIZATION:-${GPU_MEMORY_UTILIZATION:-0.35}}"
SPEED_PREPARE_WORKERS="${SPEED_PREPARE_WORKERS:-${PREPARE_WORKERS:-1}}"
SPEED_AUDIO_CONDITION_BUILD_WORKERS="${SPEED_AUDIO_CONDITION_BUILD_WORKERS:-${AUDIO_CONDITION_BUILD_WORKERS:-$SPEED_PREPARE_WORKERS}}"
SPEED_PREPROCESS_CACHE_SIZE="${SPEED_PREPROCESS_CACHE_SIZE:-${PREPROCESS_CACHE_SIZE:-512}}"
SPEED_MODEL_DIR="${SPEED_MODEL_DIR:-${VC_EDIT_ROOT}/models/Step-Audio-EditX}"
SPEED_TOKENIZER_DIR="${SPEED_TOKENIZER_DIR:-${VC_EDIT_ROOT}/models/Step-Audio-Tokenizer}"
SPEED_REPO_DIR="${SPEED_REPO_DIR:-${VC_EDIT_ROOT}/external/Step-Audio-EditX}"
SEEDVC_SHARDED="${SEEDVC_SHARDED:-0}"
SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-${GPU_IDS:-}}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-0}"
GPU_GUARD_ENABLE="${GPU_GUARD_ENABLE:-${PAIR_GPU_GUARD_ENABLE:-0}}"
GPU_GUARD_GPUS="${GPU_GUARD_GPUS:-${PAIR_GPU_GUARD_GPUS:-auto}}"
GPU_GUARD_PY="${GPU_GUARD_PY:-${PAIR_GPU_GUARD_PY:-$WAVLMPY}}"
GPU_GUARD_MATRIX_SIZE="${GPU_GUARD_MATRIX_SIZE:-${PAIR_GPU_GUARD_MATRIX_SIZE:-8192}}"
GPU_GUARD_ACTIVE_MS="${GPU_GUARD_ACTIVE_MS:-${PAIR_GPU_GUARD_ACTIVE_MS:-900}}"
GPU_GUARD_IDLE_MS="${GPU_GUARD_IDLE_MS:-${PAIR_GPU_GUARD_IDLE_MS:-150}}"
GPU_GUARD_DTYPE="${GPU_GUARD_DTYPE:-${PAIR_GPU_GUARD_DTYPE:-bfloat16}}"
GPU_GUARD_RESERVE_MIB="${GPU_GUARD_RESERVE_MIB:-${PAIR_GPU_GUARD_RESERVE_MIB:-0}}"

truthy() {
  case "$1" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

GPU_GUARD_PID=""

start_gpu_guard() {
  local pair_root="$1"
  local split="$2"
  truthy "$GPU_GUARD_ENABLE" || return 0
  if [ ! -x "$GPU_GUARD_PY" ]; then
    echo "[run03] [warn] gpu guard python not executable: $GPU_GUARD_PY" >&2
    return 0
  fi
  mkdir -p "$pair_root/logs"
  local guard_log="$pair_root/logs/gpu_util_guard_ij_${split}.log"
  echo "[run03] start gpu guard for I/J WavLM/QC: log=$guard_log gpus=$GPU_GUARD_GPUS exclude=$DEVICE"
  "$GPU_GUARD_PY" "$SCRIPT_DIR/gpu_util_guard.py" \
    --gpus "$GPU_GUARD_GPUS" \
    --exclude-device "$DEVICE" \
    --matrix-size "$GPU_GUARD_MATRIX_SIZE" \
    --active-ms "$GPU_GUARD_ACTIVE_MS" \
    --idle-ms "$GPU_GUARD_IDLE_MS" \
    --dtype "$GPU_GUARD_DTYPE" \
    --reserve-mib "$GPU_GUARD_RESERVE_MIB" \
    >"$guard_log" 2>&1 &
  GPU_GUARD_PID="$!"
  sleep 2
  if ! kill -0 "$GPU_GUARD_PID" 2>/dev/null; then
    echo "[run03] [warn] gpu guard exited early; see $guard_log" >&2
    GPU_GUARD_PID=""
  fi
}

stop_gpu_guard() {
  if [ -n "$GPU_GUARD_PID" ] && kill -0 "$GPU_GUARD_PID" 2>/dev/null; then
    echo "[run03] stop gpu guard pid=$GPU_GUARD_PID"
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

config_for_lang() {
  local lang="$1"
  if [ "$lang" = "en" ]; then
    echo "$ROOT/configs/default_en.yaml"
  else
    echo "$ROOT/configs/default.yaml"
  fi
}

run_speed_one() {
  local lang="$1"
  local split="$2"
  local source_jsonl="$3"
  local pair_root="$4"
  local run_name="${split}_J_speed_run03"

  echo "[run03-speed] ${lang}/${split} -> ${pair_root}"
  SOURCE_JSONL="$source_jsonl" \
  OUTPUT_ROOT="$pair_root" \
  RUN_NAME="$run_name" \
  LIMIT="$LIMIT" \
  SPEED_TAGS="$SPEED_TAGS" \
  DRY_RUN="$DRY_RUN" \
  PROSODY_CONFIG="$PROSODY_CONFIG" \
  MODEL_DIR="$SPEED_MODEL_DIR" \
  TOKENIZER_DIR="$SPEED_TOKENIZER_DIR" \
  REPO_DIR="$SPEED_REPO_DIR" \
  BATCH_SIZE="$SPEED_BATCH_SIZE" \
  MAX_NUM_SEQS="$SPEED_MAX_NUM_SEQS" \
  TENSOR_PARALLEL_SIZE="$SPEED_TENSOR_PARALLEL_SIZE" \
  GPU_MEMORY_UTILIZATION="$SPEED_GPU_MEMORY_UTILIZATION" \
  PREPARE_WORKERS="$SPEED_PREPARE_WORKERS" \
  AUDIO_CONDITION_BUILD_WORKERS="$SPEED_AUDIO_CONDITION_BUILD_WORKERS" \
  PREPROCESS_CACHE_SIZE="$SPEED_PREPROCESS_CACHE_SIZE" \
  bash "$SCRIPT_DIR/run_speed_pipeline.sh"
}

run_prosody_one() {
  local lang="$1"
  local split="$2"
  local source_jsonl="$3"
  local pair_root="$4"
  local run_name="${split}_I_run03"
  local jobs="$pair_root/jobs/i_seedvc_jobs.jsonl"
  local results="$pair_root/logs/i_seedvc_results.jsonl"
  local raw_pairs="$pair_root/pairs/I.jsonl"
  local scored_pairs="$pair_root/pairs/scored/I.jsonl"

  echo "[run03-prosody] ${lang}/${split} prepare -> ${jobs}"
  "$PY" "$SCRIPT_DIR/07_prepare_prosody_no_timbre_seedvc_jobs.py" \
    --source-jsonl "$source_jsonl" \
    --jobs-jsonl "$jobs" \
    --output-root "$pair_root" \
    --run-name "$run_name" \
    --limit "$LIMIT" \
    --timbre-pick "$TIMBRE_PICK" \
    --seed "$SEED"

  if truthy "$DRY_RUN"; then
    echo "[run03-prosody] dry run only; jobs=${jobs}"
    return 0
  fi

  local skip_args=()
  if truthy "$SKIP_EXISTING"; then
    skip_args+=(--skip-existing)
  fi

  echo "[run03-prosody] ${lang}/${split} Seed-VC -> ${results}"
  if truthy "$SEEDVC_SHARDED"; then
    JOBS_JSONL="$jobs" \
    RESULTS_JSONL="$results" \
    SEED_VC_DIR="$SEED_VC_DIR" \
    HF_HOME="$SEEDVC_HF_HOME" \
    TRANSFORMERS_CACHE="$SEEDVC_TRANSFORMERS_CACHE" \
    HUGGINGFACE_HUB_CACHE="$SEEDVC_HUGGINGFACE_HUB_CACHE" \
    HF_HUB_OFFLINE="$SEEDVC_HF_HUB_OFFLINE" \
    TRANSFORMERS_OFFLINE="$SEEDVC_TRANSFORMERS_OFFLINE" \
    PY="$PY" \
    DIFFUSION_STEPS="$DIFFUSION_STEPS" \
    INFERENCE_CFG_RATE="$INFERENCE_CFG_RATE" \
    LENGTH_ADJUST="1.0" \
    FP16="true" \
    MAX_JOBS="$MAX_JOBS" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    SEEDVC_GPU_IDS="$SEEDVC_GPU_IDS" \
    SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT" \
    bash "$SCRIPT_DIR/run_seedvc_jobs_sharded.sh"
  else
    HF_HOME="$SEEDVC_HF_HOME" \
    TRANSFORMERS_CACHE="$SEEDVC_TRANSFORMERS_CACHE" \
    HUGGINGFACE_HUB_CACHE="$SEEDVC_HUGGINGFACE_HUB_CACHE" \
    HF_HUB_OFFLINE="$SEEDVC_HF_HUB_OFFLINE" \
    TRANSFORMERS_OFFLINE="$SEEDVC_TRANSFORMERS_OFFLINE" \
    "$PY" "$SCRIPT_DIR/08_run_seedvc_jobs.py" \
      --jobs-jsonl "$jobs" \
      --results-jsonl "$results" \
      --seed-vc-dir "$SEED_VC_DIR" \
      --diffusion-steps "$DIFFUSION_STEPS" \
      --inference-cfg-rate "$INFERENCE_CFG_RATE" \
      --length-adjust 1.0 \
      --fp16 true \
      --max-jobs "$MAX_JOBS" \
      "${skip_args[@]}"
  fi

  echo "[run03-prosody] ${lang}/${split} collect -> ${raw_pairs}"
  "$PY" "$SCRIPT_DIR/09_collect_seedvc_prosody_no_timbre_pairs.py" \
    --jobs-jsonl "$jobs" \
    --results-jsonl "$results" \
    --output-jsonl "$raw_pairs" \
    --run-name "$run_name" \
    --require-existing-audio

  echo "[run03-prosody] ${lang}/${split} metrics -> ${scored_pairs}"
  "$PY" "$SCRIPT_DIR/03_add_prosody_metrics.py" \
    --config "$PROSODY_CONFIG" \
    --input-jsonl "$raw_pairs" \
    --output-jsonl "$scored_pairs" \
    --summary-json "$pair_root/metrics/I.metrics_summary.json" \
    --mode prosody_transfer
}

run_wavlm_one() {
  local lang="$1"
  local split="$2"
  local config="$3"
  echo "[run03-wavlm] ${lang}/${split} I,J_fast,J_slow"
  PAIR_OUTPUTS_ROOT="$PAIR_OUTPUTS_ROOT/$lang" \
  "$WAVLMPY" "$SCRIPT_DIR/11b_add_wavlm_sim.py" \
    --split "$split" \
    --config "$config" \
    --device "$DEVICE" \
    --pair-type I,J_fast,J_slow
}

run_pair_audio_metrics_one() {
  local pair_root="$1"
  local config="$2"
  echo "[run03-metrics] ${pair_root} I,J_fast,J_slow"
  "$EMOTION_PY_BIN" "$SCRIPT_DIR/04c_add_pair_audio_metrics.py" \
    --pair-root "$pair_root" \
    --config "$config" \
    --pair-type I,J_fast,J_slow \
    --sides both \
    --device "$DEVICE"
}

run_qc_one() {
  local pair_root="$1"
  local config="$2"
  echo "[run03-qc] ${pair_root}"
  QWEN_ASR_DEVICE="$DEVICE" "$EMOTION_PY_BIN" "$SCRIPT_DIR/qc_pairs.py" \
    --pair-root "$pair_root" \
    --config "$config" \
    --pair-type I,J_fast,J_slow \
    --merge-summary
}

echo "[run03] root=${ROOT}"
echo "[run03] source_root=${SOURCE_ROOT}"
echo "[run03] pair_outputs=${PAIR_OUTPUTS_ROOT}"
echo "[run03] splits=${SPLITS}"
echo "[run03] limit=${LIMIT} device=${DEVICE} dry_run=${DRY_RUN}"
echo "[run03] run_speed=${RUN_SPEED} run_prosody=${RUN_PROSODY} run_pair_audio_metrics=${RUN_PAIR_AUDIO_METRICS} run_wavlm=${RUN_WAVLM} run_qc=${RUN_QC}"
echo "[run03] speed_tp=${SPEED_TENSOR_PARALLEL_SIZE} speed_gpu_mem=${SPEED_GPU_MEMORY_UTILIZATION} speed_batch=${SPEED_BATCH_SIZE} speed_max_num_seqs=${SPEED_MAX_NUM_SEQS}"
echo "[run03] speed_model=${SPEED_MODEL_DIR}"
echo "[run03] seedvc_sharded=${SEEDVC_SHARDED} seedvc_gpu_ids=${SEEDVC_GPU_IDS:-<auto>} seedvc_shard_count=${SEEDVC_SHARD_COUNT}"
echo "[run03] seedvc_hf_cache=${SEEDVC_HUGGINGFACE_HUB_CACHE} seedvc_hf_offline=${SEEDVC_HF_HUB_OFFLINE}"

for item in $SPLITS; do
  lang="${item%%/*}"
  split="${item#*/}"
  source_jsonl="$SOURCE_ROOT/$lang/$split.jsonl"
  pair_root="$PAIR_OUTPUTS_ROOT/$lang/$split"
  config="$(config_for_lang "$lang")"

  if [ ! -f "$source_jsonl" ]; then
    echo "[run03] missing source jsonl: ${source_jsonl}" >&2
    exit 1
  fi
  mkdir -p "$pair_root/pairs/scored" "$pair_root/metrics"

  if truthy "$RUN_SPEED"; then
    run_speed_one "$lang" "$split" "$source_jsonl" "$pair_root"
  fi
  if truthy "$RUN_PROSODY"; then
    run_prosody_one "$lang" "$split" "$source_jsonl" "$pair_root"
  fi
  if truthy "$RUN_PAIR_AUDIO_METRICS" && ! truthy "$DRY_RUN"; then
    run_pair_audio_metrics_one "$pair_root" "$config"
  fi
  if ! truthy "$DRY_RUN" && { truthy "$RUN_WAVLM" || truthy "$RUN_QC"; }; then
    start_gpu_guard "$pair_root" "$split"
  fi
  if truthy "$RUN_WAVLM" && ! truthy "$DRY_RUN"; then
    run_wavlm_one "$lang" "$split" "$config"
  fi
  if truthy "$RUN_QC" && ! truthy "$DRY_RUN"; then
    run_qc_one "$pair_root" "$config"
  fi
  stop_gpu_guard
done

echo "[run03] done"
