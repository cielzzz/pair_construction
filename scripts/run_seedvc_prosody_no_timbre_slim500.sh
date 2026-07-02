#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/contts-train/bin/python}"
SEED_VC_DIR="${SEED_VC_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes/third_party/seed-vc}"
DEPS_DIR="${DEPS_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes/.deps/seedvc}"
export PYTHONPATH="$DEPS_DIR:$ROOT/scripts:$SEED_VC_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

ZH_SOURCE="${ZH_SOURCE:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_data_temp/mtd_pass_nonmulti_primary_le_0p3_split_10k/zh/zh_slim_0001.jsonl}"
EN_SOURCE="${EN_SOURCE:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_data_temp/mtd_pass_nonmulti_primary_le_0p3_split_10k/en/en_slim_0001.jsonl}"
LIMIT="${LIMIT:-500}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
SEED="${SEED:-42}"
TIMBRE_PICK="${TIMBRE_PICK:-random}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_JOBS="${MAX_JOBS:-0}"

run_one() {
  local lang="$1"
  local source_jsonl="$2"
  local run_name="${lang}_slim500_prosody_no_timbre_seedvc_v1_20260615_run01"
  local out_root="$ROOT/outputs/$run_name"
  local jobs="$out_root/jobs/prosody_no_timbre_seedvc_jobs.jsonl"
  local results="$out_root/logs/seedvc_results.jsonl"
  local raw_pairs="$out_root/pairs/I.jsonl"
  local pairs="$out_root/pairs/scored/I.jsonl"

  echo "[$lang] prepare jobs -> $jobs"
  "$PY" "$ROOT/scripts/07_prepare_prosody_no_timbre_seedvc_jobs.py" \
    --source-jsonl "$source_jsonl" \
    --jobs-jsonl "$jobs" \
    --output-root "$out_root" \
    --run-name "$run_name" \
    --limit "$LIMIT" \
    --timbre-pick "$TIMBRE_PICK" \
    --seed "$SEED"

  local skip_args=()
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    skip_args+=(--skip-existing)
  fi

  echo "[$lang] run Seed-VC -> $results"
  "$PY" "$ROOT/scripts/08_run_seedvc_jobs.py" \
    --jobs-jsonl "$jobs" \
    --results-jsonl "$results" \
    --seed-vc-dir "$SEED_VC_DIR" \
    --diffusion-steps "$DIFFUSION_STEPS" \
    --inference-cfg-rate "$INFERENCE_CFG_RATE" \
    --length-adjust 1.0 \
    --fp16 true \
    --max-jobs "$MAX_JOBS" \
    "${skip_args[@]}"

  echo "[$lang] collect pairs -> $raw_pairs"
  "$PY" "$ROOT/scripts/09_collect_seedvc_prosody_no_timbre_pairs.py" \
    --jobs-jsonl "$jobs" \
    --results-jsonl "$results" \
    --output-jsonl "$raw_pairs" \
    --run-name "$run_name" \
    --require-existing-audio

  echo "[$lang] add prosody metrics -> $pairs"
  "$PY" "$ROOT/scripts/03_add_prosody_metrics.py" \
    --input-jsonl "$raw_pairs" \
    --output-jsonl "$pairs" \
    --summary-json "$out_root/metrics/I.metrics_summary.json" \
    --mode prosody_transfer
}

cd "$ROOT"
run_one zh "$ZH_SOURCE"
run_one en "$EN_SOURCE"
