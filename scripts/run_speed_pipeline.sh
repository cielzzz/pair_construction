#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
CONFIG="${PROSODY_CONFIG:-${PROJECT_ROOT}/configs/prosody_routes.yaml}"

ANALYSIS_PY="${ANALYSIS_PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/emotion/bin/python}"
CONDA_BIN="${CONDA_BIN:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/bin/conda}"
STEP_ENV="${STEP_ENV:-step_audio_editx}"
VC_EDIT_ROOT="${VC_EDIT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_edit}"
STEP_RUNNER="${STEP_RUNNER:-${SCRIPT_DIR}/run_step_editx_local.py}"
MODEL_DIR="${SPEED_MODEL_DIR:-${MODEL_DIR:-${VC_EDIT_ROOT}/models/Step-Audio-EditX}}"
TOKENIZER_DIR="${SPEED_TOKENIZER_DIR:-${TOKENIZER_DIR:-${VC_EDIT_ROOT}/models/Step-Audio-Tokenizer}}"
REPO_DIR="${SPEED_REPO_DIR:-${REPO_DIR:-${VC_EDIT_ROOT}/external/Step-Audio-EditX}}"

SOURCE_JSONL="${SOURCE_JSONL:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/outputs/delivery_filter_10k_langsplit_qz_20260604_run02/vcdata_job_runs/zh/zxy-delivery10k-vceditx-delivery10k-langsplit-0604-run02-g11/filtered_manifest_0009/manifest_shard0.jsonl}"
RUN_NAME="${RUN_NAME:-speed_pilot_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/${RUN_NAME}}"
LIMIT="${LIMIT:-16}"
START_INDEX="${START_INDEX:-0}"
SPEED_TAGS="${SPEED_TAGS:-speed_faster,speed_slower}"
DRY_RUN="${DRY_RUN:-0}"
MIN_DURATION="${MIN_DURATION:-}"
MAX_DURATION="${MAX_DURATION:-}"

BATCH_SIZE="${BATCH_SIZE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${BATCH_SIZE}}"
PREPARE_WORKERS="${PREPARE_WORKERS:-1}"
AUDIO_CONDITION_BUILD_WORKERS="${AUDIO_CONDITION_BUILD_WORKERS:-${PREPARE_WORKERS}}"
PREPROCESS_CACHE_SIZE="${PREPROCESS_CACHE_SIZE:-512}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"

JOBS_JSONL="${OUTPUT_ROOT}/jobs/step_speed_jobs.jsonl"
STEP_OUTPUT_DIR="${OUTPUT_ROOT}/stepaudio_speed"
REPORT_JSONL="${STEP_OUTPUT_DIR}/paired_report.jsonl"
PAIR_JSONL="${OUTPUT_ROOT}/intermediate/SpeedEdit.all.jsonl"
SCORED_DIR="${OUTPUT_ROOT}/pairs/scored"

mkdir -p "${OUTPUT_ROOT}/jobs" "${OUTPUT_ROOT}/intermediate" "${OUTPUT_ROOT}/pairs/scored" "${OUTPUT_ROOT}/metrics"

echo "[speed] project=${PROJECT_ROOT}"
echo "[speed] source=${SOURCE_JSONL}"
echo "[speed] run=${RUN_NAME}"
echo "[speed] output=${OUTPUT_ROOT}"
echo "[speed] tags=${SPEED_TAGS}"
echo "[speed] limit=${LIMIT} start=${START_INDEX} dry_run=${DRY_RUN}"
echo "[speed] min_duration=${MIN_DURATION:-<config default>} max_duration=${MAX_DURATION:-<config default>}"
echo "[speed] model_dir=${MODEL_DIR}"

PREPARE_DURATION_ARGS=()
if [ -n "${MIN_DURATION}" ]; then
  PREPARE_DURATION_ARGS+=(--min-duration "${MIN_DURATION}")
fi
if [ -n "${MAX_DURATION}" ]; then
  PREPARE_DURATION_ARGS+=(--max-duration "${MAX_DURATION}")
fi

"${ANALYSIS_PY}" "${SCRIPT_DIR}/01_prepare_step_speed_jobs.py" \
  --config "${CONFIG}" \
  --source-jsonl "${SOURCE_JSONL}" \
  --output-jsonl "${JOBS_JSONL}" \
  --run-name "${RUN_NAME}" \
  --limit "${LIMIT}" \
  --start-index "${START_INDEX}" \
  --speed-tags "${SPEED_TAGS}" \
  "${PREPARE_DURATION_ARGS[@]}"

if [ "${DRY_RUN}" = "1" ] || [ "${DRY_RUN}" = "true" ] || [ "${DRY_RUN}" = "TRUE" ]; then
  echo "[speed] dry run only; jobs=${JOBS_JSONL}"
  exit 0
fi

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate "${STEP_ENV}"
export VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}"

python "${STEP_RUNNER}" \
  --jobs-jsonl "${JOBS_JSONL}" \
  --model-dir "${MODEL_DIR}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --repo-dir "${REPO_DIR}" \
  --output-dir "${STEP_OUTPUT_DIR}" \
  --report-jsonl "${REPORT_JSONL}" \
  --batch-size "${BATCH_SIZE}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --prepare-workers "${PREPARE_WORKERS}" \
  --audio-condition-build-workers "${AUDIO_CONDITION_BUILD_WORKERS}" \
  --preprocess-cache-size "${PREPROCESS_CACHE_SIZE}" \
  --no-cosyvoice-cuda-graph \
  --batch-metrics-tsv "${STEP_OUTPUT_DIR}/batch_metrics.tsv" \
  --summary-json "${STEP_OUTPUT_DIR}/run_summary.json"

"${ANALYSIS_PY}" "${SCRIPT_DIR}/02_collect_step_speed_pairs.py" \
  --config "${CONFIG}" \
  --report-jsonl "${REPORT_JSONL}" \
  --output-jsonl "${PAIR_JSONL}" \
  --split-output-dir "${OUTPUT_ROOT}/pairs" \
  --run-name "${RUN_NAME}" \
  --require-existing-audio

for PAIR_TYPE in J_fast J_slow; do
  INPUT_JSONL="${OUTPUT_ROOT}/pairs/${PAIR_TYPE}.jsonl"
  if [ ! -s "${INPUT_JSONL}" ]; then
    echo "[speed] skip ${PAIR_TYPE}: missing or empty ${INPUT_JSONL}"
    continue
  fi
  "${ANALYSIS_PY}" "${SCRIPT_DIR}/03_add_prosody_metrics.py" \
    --config "${CONFIG}" \
    --input-jsonl "${INPUT_JSONL}" \
    --output-jsonl "${SCORED_DIR}/${PAIR_TYPE}.jsonl" \
    --summary-json "${OUTPUT_ROOT}/metrics/${PAIR_TYPE}.summary.json" \
    --mode speed_edit
done

echo "[speed] done"
echo "[speed] combined_pairs=${PAIR_JSONL}"
echo "[speed] scored_dir=${SCORED_DIR}"
