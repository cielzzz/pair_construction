#!/usr/bin/env sh
# ═══════════════════════════════════════════════════════════════
# 中文 + 全阶段：原始 jsonl 目录 → vcdata → editx → emotion → pair
# ═══════════════════════════════════════════════════════════════
# 用法：sh run_zh_full.sh                  ← 默认 RUN_MODE=submit（启智批量）
#       RUN_MODE=local sh run_zh_full.sh   ← 本地 demo 模式（HEAD_N 小批量）
#       RUN_MODE=submit DRY_RUN=1 sh run_zh_full.sh   ← 启智 dry-run（看分组不真提交）
#
# 模式：full           （含 MOSS-TTS stage1）
# 语言：zh             （PAIR_CONFIG=default.yaml）
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
# 中文原始 jsonl 目录（含 split_*.jsonl 多个文件）
INPUT_DIR="/inspire/hdd/project/embodied-multimodality/public/kxhuang/instructtts_data/instruction_0.1_enzh/zh"

# 仅 local 模式有效：取头 N 行做 demo
LOCAL_INPUT_JSONL="$INPUT_DIR/split_0000.jsonl"
LOCAL_SPLIT_NAME="split_demo"
LOCAL_HEAD_N=100

# 中文 jsonl 用 sensevoice_small_clean 字段做 TTS 文本
export TEXT_FIELD="sensevoice_small_clean"

# 中文用 default.yaml
export PAIR_CONFIG="$(cd "$(dirname "$0")/.." && pwd)/configs/default.yaml"

# stage2 跑哪些 edit 模式（type:info 逗号分隔）
#   生产用 "style:radio"                                                  ← 中文最优单模式
#   调研用 "style:radio,style:remove,style:news,style:chat,emotion:remove,emotion:coldness"
export EDIT_PAIRS="style:radio"

# 启智批量参数（RUN_MODE=submit 时生效）
export TASK_COUNT="${TASK_COUNT:-16}"       # 切多少组
# BATCH_ID 必须按 lang 隔离（否则 zh/en 会共享 .qz_split_batches/ 撞车）
export BATCH_ID="${BATCH_ID:-zh-$(date -u +%m%d-%H%M%S)}"
# 关掉 AUTO_RESUME_LATEST 避免续到别的 lang 的 batch；要 resume 显式传 RESUME_BATCH_ID
export AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-0}"
export FORCE_NEW_BATCH="${FORCE_NEW_BATCH:-0}"

# 模式：submit（启智批量）或 local（本地单 GPU 跑 HEAD_N 子集）
RUN_MODE="${RUN_MODE:-submit}"
DRY_RUN_FLAG=""
[ "${DRY_RUN:-0}" = "1" ] && DRY_RUN_FLAG="--dry-run"

# ───── 路径常量 ───────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VCDATA_REPO="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction"

# ───── 执行 ───────────────────────────────────────────────────
echo "═══════════════════════════════════════════════"
echo " run_zh_full  RUN_MODE=$RUN_MODE"
echo "═══════════════════════════════════════════════"

case "$RUN_MODE" in
    submit)
        echo "[stage 1 vcdata] 提交启智批量任务（INPUT_DIR=$INPUT_DIR）"
        echo "提交完后该脚本立即退出。等启智 vcdata 跑完，再跑 sh runs/run_zh_from_vcdata.sh 走 editx + emotion + pair。"
        echo ""
        cd "$VCDATA_REPO"
        INPUT_DIR="$INPUT_DIR" \
        OUTPUT_ROOT="$VCDATA_REPO/outputs/instruction_0.1_enzh/zh" \
        TEXT_FIELD="$TEXT_FIELD" \
        TASK_COUNT="$TASK_COUNT" \
        AUTO_RESUME_LATEST="$AUTO_RESUME_LATEST" \
        FORCE_NEW_BATCH="$FORCE_NEW_BATCH" \
        bash submit_split_batch_h200.sh $DRY_RUN_FLAG
        ;;
    local)
        echo "[local 模式] 本地单 GPU 跑 $LOCAL_HEAD_N 句 demo（$LOCAL_INPUT_JSONL → $LOCAL_SPLIT_NAME）"
        exec bash "$PROJ_ROOT/run_e2e.sh" full "$LOCAL_INPUT_JSONL" "$LOCAL_SPLIT_NAME" "$LOCAL_HEAD_N"
        ;;
    *)
        echo "ERROR: RUN_MODE 必须是 submit 或 local（收到 $RUN_MODE）" >&2
        exit 1
        ;;
esac
