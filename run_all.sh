#!/usr/bin/env bash
# 一键流水线 —— 端到端从 vcdata + editx 输出到所有 pair jsonl
# 用法： bash run_all.sh <split> [--reuse-qzrun | --skip-emotion]
#  bash run_all.sh split_0000 --reuse-qzrun

set -euo pipefail
SPLIT="${1:?usage: run_all.sh <split> [--reuse-qzrun|--skip-emotion]}"
FLAG="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$SCRIPT_DIR/scripts"

echo "═══ 阶段一：中间表 ═══════════════════════"
python3 "$S/01_build_vcdata_base.py" --split "$SPLIT"
python3 "$S/02_build_editx_base.py"  --split "$SPLIT"
python3 "$S/03_join_editx_with_vcdata.py" --split "$SPLIT"

echo "═══ 阶段二：情感评估 ════════════════════"
case "$FLAG" in
  --skip-emotion)
    echo "[run_all] 跳过 emotion eval（依赖已有 per_file_dual.csv）" ;;
  --reuse-qzrun)
    bash "$S/04_run_emotion_eval.sh" "$SPLIT" --reuse-qzrun ;;
  *)
    bash "$S/04_run_emotion_eval.sh" "$SPLIT" ;;
esac

echo "═══ 阶段三：A / B / C ═══════"
python3 "$S/05_construct_A.py"       --split "$SPLIT"
python3 "$S/06_construct_B.py" --split "$SPLIT"
python3 "$S/07_construct_C.py" --split "$SPLIT"
python3 "$S/07b_construct_C_mixed.py" --split "$SPLIT"
python3 "$S/07c_construct_D.py" --split "$SPLIT"

echo "═══ 阶段四：H1 / H2 / H3 ════════════════"
python3 "$S/08_construct_H1.py" --split "$SPLIT"
python3 "$S/09_construct_H2.py" --split "$SPLIT"
python3 "$S/10_construct_H3.py" --split "$SPLIT"

echo "═══ DONE ════════════════════════════════"
python3 - <<PY
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open("$SCRIPT_DIR/configs/default.yaml"))
out = Path(cfg["paths"]["outputs_root"]) / "$SPLIT" / "pairs"
for p in sorted(out.glob("*.jsonl")):
    n = sum(1 for _ in p.open())
    print(f"  {p.name:25s} {n:>8d} rows")
PY
