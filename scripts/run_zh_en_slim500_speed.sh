#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export PREPARE_WORKERS="${PREPARE_WORKERS:-1}"
export AUDIO_CONDITION_BUILD_WORKERS="${AUDIO_CONDITION_BUILD_WORKERS:-1}"
export LIMIT="${LIMIT:-500}"
export MIN_DURATION="${MIN_DURATION:-0}"
export MAX_DURATION="${MAX_DURATION:-999999}"

ZH_SOURCE_JSONL="${ZH_SOURCE_JSONL:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_data_temp/mtd_pass_nonmulti_primary_le_0p3_split_10k/zh/zh_slim_0001.jsonl}"
EN_SOURCE_JSONL="${EN_SOURCE_JSONL:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_data_temp/mtd_pass_nonmulti_primary_le_0p3_split_10k/en/en_slim_0001.jsonl}"

echo "[launcher] project=${PROJECT_ROOT}"
echo "[launcher] cuda=${CUDA_VISIBLE_DEVICES}"
echo "[launcher] limit=${LIMIT} min_duration=${MIN_DURATION} max_duration=${MAX_DURATION}"
echo "[launcher] zh=${ZH_SOURCE_JSONL}"
echo "[launcher] en=${EN_SOURCE_JSONL}"

cd "${PROJECT_ROOT}"

SOURCE_JSONL="${ZH_SOURCE_JSONL}" \
RUN_NAME="${ZH_RUN_NAME:-zh_slim500_speed_20260614_run02}" \
bash "${SCRIPT_DIR}/run_speed_pipeline.sh"

SOURCE_JSONL="${EN_SOURCE_JSONL}" \
RUN_NAME="${EN_RUN_NAME:-en_slim500_speed_20260614_run02}" \
bash "${SCRIPT_DIR}/run_speed_pipeline.sh"

echo "[launcher] done"
