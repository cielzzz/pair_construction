#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

JOBS_JSONL="${JOBS_JSONL:?JOBS_JSONL is required}"
RESULTS_JSONL="${RESULTS_JSONL:?RESULTS_JSONL is required}"
SEED_VC_DIR="${SEED_VC_DIR:?SEED_VC_DIR is required}"

PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/contts-train/bin/python}"
GPU_IDS="${SEEDVC_GPU_IDS:-${GPU_IDS:-}}"
SHARD_COUNT="${SEEDVC_SHARD_COUNT:-0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
FP16="${FP16:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_JOBS="${MAX_JOBS:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
SHOW_MODEL_OUTPUT="${SHOW_MODEL_OUTPUT:-0}"

if [ -z "${GPU_IDS}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_IDS="${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
  else
    GPU_IDS="0"
  fi
fi

IFS=',' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
if [ "${#GPU_ID_LIST[@]}" -eq 0 ]; then
  echo "ERROR: no GPU ids resolved from GPU_IDS='${GPU_IDS}'" >&2
  exit 1
fi

if [ "${SHARD_COUNT}" -le 0 ] || [ "${SHARD_COUNT}" -gt "${#GPU_ID_LIST[@]}" ]; then
  SHARD_COUNT="${#GPU_ID_LIST[@]}"
fi

WORK_DIR=$(mktemp -d "$(dirname "${RESULTS_JSONL}")/.seedvc_shards.XXXXXX")
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "[seedvc-sharded] jobs=${JOBS_JSONL}"
echo "[seedvc-sharded] results=${RESULTS_JSONL}"
echo "[seedvc-sharded] gpu_ids=${GPU_IDS}"
echo "[seedvc-sharded] shard_count=${SHARD_COUNT}"

"${PY}" - "${JOBS_JSONL}" "${WORK_DIR}" "${SHARD_COUNT}" "${MAX_JOBS}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

jobs_path = Path(sys.argv[1])
work_dir = Path(sys.argv[2])
shard_count = int(sys.argv[3])
max_jobs = int(sys.argv[4])

handles = []
try:
    for idx in range(shard_count):
        handles.append((work_dir / f"jobs_shard_{idx:02d}.jsonl").open("w", encoding="utf-8"))

    written = [0] * shard_count
    with jobs_path.open("r", encoding="utf-8") as src:
        for job_idx, line in enumerate(src):
            if max_jobs > 0 and job_idx >= max_jobs:
                break
            line = line.strip()
            if not line:
                continue
            json.loads(line)
            shard_idx = job_idx % shard_count
            handles[shard_idx].write(line + "\n")
            written[shard_idx] += 1
finally:
    for handle in handles:
        handle.close()

for idx, count in enumerate(written):
    print(f"shard_{idx:02d}\t{count}")
PY

declare -a PIDS=()
declare -a SHARD_RESULT_PATHS=()
run_fail=0

for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_tag=$(printf "%02d" "${shard_idx}")
  shard_jobs="${WORK_DIR}/jobs_shard_${shard_tag}.jsonl"
  shard_results="${WORK_DIR}/results_shard_${shard_tag}.jsonl"
  shard_summary="${WORK_DIR}/summary_shard_${shard_tag}.json"
  gpu_id="${GPU_ID_LIST[$shard_idx]}"
  SHARD_RESULT_PATHS+=("${shard_results}")

  if [ ! -s "${shard_jobs}" ]; then
    : > "${shard_results}"
    echo "[seedvc-sharded] skip empty shard ${shard_tag}"
    continue
  fi

  echo "[seedvc-sharded] start shard=${shard_tag} gpu=${gpu_id}"
  shard_args=(
    "${SCRIPT_DIR}/08_run_seedvc_jobs.py"
    --jobs-jsonl "${shard_jobs}"
    --results-jsonl "${shard_results}"
    --seed-vc-dir "${SEED_VC_DIR}"
    --diffusion-steps "${DIFFUSION_STEPS}"
    --length-adjust "${LENGTH_ADJUST}"
    --inference-cfg-rate "${INFERENCE_CFG_RATE}"
    --fp16 "${FP16}"
    --summary-json "${shard_summary}"
  )
  if [ "${SKIP_EXISTING}" = "1" ] || [ "${SKIP_EXISTING}" = "true" ] || [ "${SKIP_EXISTING}" = "TRUE" ]; then
    shard_args+=(--skip-existing)
  fi
  if [ "${FAIL_FAST}" = "1" ] || [ "${FAIL_FAST}" = "true" ] || [ "${FAIL_FAST}" = "TRUE" ]; then
    shard_args+=(--fail-fast)
  fi
  if [ "${SHOW_MODEL_OUTPUT}" = "1" ] || [ "${SHOW_MODEL_OUTPUT}" = "true" ] || [ "${SHOW_MODEL_OUTPUT}" = "TRUE" ]; then
    shard_args+=(--show-model-output)
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY}" "${shard_args[@]}" > "${WORK_DIR}/seedvc_shard_${shard_tag}.log" 2>&1 &
  PIDS+=("$!:${shard_tag}:${WORK_DIR}/seedvc_shard_${shard_tag}.log")
done

for item in "${PIDS[@]}"; do
  pid="${item%%:*}"
  rest="${item#*:}"
  shard_tag="${rest%%:*}"
  log_path="${rest#*:}"
  if ! wait "${pid}"; then
    echo "[seedvc-sharded] shard ${shard_tag} failed; log=${log_path}" >&2
    tail -n 80 "${log_path}" >&2 || true
    run_fail=1
  else
    echo "[seedvc-sharded] shard ${shard_tag} done; log=${log_path}"
  fi
done

if [ "${run_fail}" -ne 0 ]; then
  exit 1
fi

mkdir -p "$(dirname "${RESULTS_JSONL}")"
: > "${RESULTS_JSONL}"
for shard_results in "${SHARD_RESULT_PATHS[@]}"; do
  [ -f "${shard_results}" ] || continue
  cat "${shard_results}" >> "${RESULTS_JSONL}"
done

"${PY}" - "${JOBS_JSONL}" "${RESULTS_JSONL}" "${WORK_DIR}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

jobs_path = Path(sys.argv[1])
results_path = Path(sys.argv[2])
work_dir = Path(sys.argv[3])

jobs = sum(1 for line in jobs_path.open("r", encoding="utf-8") if line.strip())
results = []
with results_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            results.append(json.loads(line))
ok = sum(1 for row in results if row.get("ok"))
failed = sum(1 for row in results if not row.get("ok"))
reused = sum(1 for row in results if row.get("reused"))
summary = {
    "jobs_jsonl": str(jobs_path.resolve()),
    "results_jsonl": str(results_path.resolve()),
    "jobs_total": jobs,
    "results_total": len(results),
    "ok": ok,
    "failed": failed,
    "reused": reused,
    "work_dir": str(work_dir),
}
summary_path = results_path.with_suffix(".sharded_summary.json")
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
if failed or len(results) != jobs:
    raise SystemExit("Seed-VC sharded run did not produce one successful result row per job.")
PY

echo "[seedvc-sharded] merged -> ${RESULTS_JSONL}"
