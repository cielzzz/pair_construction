#!/usr/bin/env sh
# ═══════════════════════════════════════════════════════════════
# 英文 + 全阶段：原始 kxhuang jsonl 目录 → vcdata → editx → emotion → pair
# ═══════════════════════════════════════════════════════════════
# 用法：sh run_en_full.sh                  ← 默认 RUN_MODE=submit（启智批量）
#       RUN_MODE=local sh run_en_full.sh   ← 本地 demo 模式（HEAD_N 小批量）
#       RUN_MODE=submit DRY_RUN=1 sh run_en_full.sh
#
# 模式：full           （含 MOSS-TTS stage1）
# 语言：en             （PAIR_CONFIG=default_en.yaml）
#
# 处理阶段：
#   stage1     vcdata stage1_generate（启智批量 或 本地单 GPU）
#   stage1.5  merge_shards
#   stage2    editx（启智批量 或 本地）
#   stage3-5  pair_construction（本地，emotion eval + 6 类 pair）
#
# 改参数：编辑下方"参数区"

set -eu

# ───── 参数区 ─────────────────────────────────────────────────
# kxhuang 英文原始 jsonl 目录
INPUT_DIR="/inspire/hdd/project/embodied-multimodality/public/kxhuang/instructtts_data/instruction_0.1_enzh/en"

# 仅 local 模式有效：取头 N 行做 demo
LOCAL_INPUT_JSONL="$INPUT_DIR/split_0000.jsonl"
LOCAL_SPLIT_NAME="split_demo_en_full"
LOCAL_HEAD_N=100

# kxhuang 英文 jsonl 用 text 字段（与中文 sensevoice_small_clean 不同）
export TEXT_FIELD="text"

# 英文用 default_en.yaml
export PAIR_CONFIG="$(cd "$(dirname "$0")/.." && pwd)/configs/default_en.yaml"

# stage2 跑哪些 edit 模式
#   生产用 "style:chat"                                                  ← 英文最优单模式
#   并联用 "style:chat,style:remove,style:radio"                          ← 英文前三并联（推荐）
#   调研用 "style:radio,style:remove,style:news,style:chat,emotion:remove,emotion:coldness"
export EDIT_PAIRS="style:chat"   # 2026-05-29 收敛到单 style_chat（en pair_construction 已对齐到此 tag）

# 启智批量参数
export TASK_COUNT="${TASK_COUNT:-16}"
# BATCH_ID 按 lang 隔离
export BATCH_ID="${BATCH_ID:-en-$(date -u +%m%d-%H%M%S)}"
export AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-0}"
export FORCE_NEW_BATCH="${FORCE_NEW_BATCH:-0}"

RUN_MODE="${RUN_MODE:-submit}"
DRY_RUN_FLAG=""
[ "${DRY_RUN:-0}" = "1" ] && DRY_RUN_FLAG="--dry-run"

# ───── 路径常量 ───────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VCDATA_REPO="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction"

# ───── 执行 ───────────────────────────────────────────────────
echo "═══════════════════════════════════════════════"
echo " run_en_full  RUN_MODE=$RUN_MODE"
echo "═══════════════════════════════════════════════"

case "$RUN_MODE" in
    submit)
        echo "[stage 1 vcdata] 提交启智批量任务（INPUT_DIR=$INPUT_DIR）"
        echo "提交完后该脚本立即退出。等启智 vcdata 跑完，再跑 sh runs/run_en_from_vcdata.sh。"
        echo ""
        cd "$VCDATA_REPO"
        INPUT_DIR="$INPUT_DIR" \
        OUTPUT_ROOT="$VCDATA_REPO/outputs/instruction_0.1_enzh/en" \
        TEXT_FIELD="$TEXT_FIELD" \
        TASK_COUNT="$TASK_COUNT" \
        AUTO_RESUME_LATEST="$AUTO_RESUME_LATEST" \
        FORCE_NEW_BATCH="$FORCE_NEW_BATCH" \
        JOB_NAME_PREFIX="zxy-vcdata-en-splitbatch" \
        bash submit_split_batch_h200.sh $DRY_RUN_FLAG
        ;;
    local)
        echo "[local 模式] 本地单 GPU 跑 $LOCAL_HEAD_N 句 demo"
        exec bash "$PROJ_ROOT/run_e2e.sh" full "$LOCAL_INPUT_JSONL" "$LOCAL_SPLIT_NAME" "$LOCAL_HEAD_N"
        ;;
    *)
        echo "ERROR: RUN_MODE 必须是 submit 或 local（收到 $RUN_MODE）" >&2
        exit 1
        ;;
esac
