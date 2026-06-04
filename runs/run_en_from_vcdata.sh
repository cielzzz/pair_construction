#!/usr/bin/env sh
# ═══════════════════════════════════════════════════════════════
# 英文 + 只跑 edit：vcdata 已就绪 → editx → emotion → pair
# ═══════════════════════════════════════════════════════════════
# 用法：
#   sh run_en_from_vcdata.sh                       ← 默认 RUN_MODE=submit（启智批量 editx）
#   RUN_MODE=after_editx sh run_en_from_vcdata.sh  ← editx 已完成，遍历所有 split 跑本地 emotion+pair
#   RUN_MODE=local sh run_en_from_vcdata.sh        ← 本地 demo
#   RUN_MODE=submit DRY_RUN=1 sh run_en_from_vcdata.sh
#
# 模式：from_vcdata    （跳过 stage1，复用上游 vcdata 输出）
# 语言：en             （PAIR_CONFIG=default_en.yaml）
#
# 处理阶段：
#   stage1.5  merge_shards（每个 split）
#   stage2    editx 跑指定 edit 模式（启智批量）
#   stage3-5  pair_construction（本地，emotion eval + 7 类 pair；RUN_MODE=after_editx 遍历全 split）
#
# ─── 与上游 vcdata submit 默认值的偏离（透明披露） ────────────
#   AUTO_RESUME_LATEST 默认 0（上游 vcdata 默认 1）  — 避免 zh/en batch 相互续跑
#   BATCH_ID 默认带 en- 前缀（上游无前缀）            — 同目的
#
# 改参数：编辑下方"参数区"

set -eu

# ───── 参数区 ─────────────────────────────────────────────────
# vcdata 上游输出根（含 split_*/）。
#   默认：对齐 run_en_full.sh 的 stage1 输出目录
#   备选：kxhuang 已跑好的 vc_mosstts 英文 vcdata 产物
#         "/inspire/qb-ilm/project/embodied-multimodality/public/kxhuang/vc_mosstts/instruction_0.1_enzh/en"
INPUT_VCDATA_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/outputs/instruction_0.1_enzh/en"

LOCAL_SPLIT_NAME="split_demo_en"

# 英文用 default_en.yaml
export PAIR_CONFIG="$(cd "$(dirname "$0")/.." && pwd)/configs/default_en.yaml"
export VCDATA_ROOT="$INPUT_VCDATA_ROOT"

# stage2 跑哪些 edit 模式
#   生产用 "style:chat"
#   并联用 "style:chat,style:remove,style:radio"
#   调研用 "style:radio,style:remove,style:news,style:chat,emotion:remove,emotion:coldness"
export EDIT_PAIRS="style:chat"   # 2026-05-29 收敛到单 style_chat（en pair_construction 已对齐到此 tag）

# 启智批量参数
export TASK_COUNT="${TASK_COUNT:-16}"
export BATCH_ID="${BATCH_ID:-en-$(date -u +%m%d-%H%M%S)}"
export AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-0}"
export FORCE_NEW_BATCH="${FORCE_NEW_BATCH:-0}"

RUN_MODE="${RUN_MODE:-submit}"
DRY_RUN_FLAG=""
[ "${DRY_RUN:-0}" = "1" ] && DRY_RUN_FLAG="--dry-run"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

echo "═══════════════════════════════════════════════"
echo " run_en_from_vcdata  RUN_MODE=$RUN_MODE"
echo "  INPUT_VCDATA_ROOT=$INPUT_VCDATA_ROOT"
echo "═══════════════════════════════════════════════"

case "$RUN_MODE" in
    submit)
        echo "[stage 2 editx] 提交启智批量任务"
        echo "提交完后该脚本立即退出。等启智 editx 跑完，再用 RUN_MODE=after_editx 跑本地 emotion+pair。"
        echo ""
        INPUT_VCDATA_ROOT="$INPUT_VCDATA_ROOT" \
        EDIT_PAIRS="$EDIT_PAIRS" \
        TASK_COUNT="$TASK_COUNT" \
        BATCH_ID="$BATCH_ID" \
        AUTO_RESUME_LATEST="$AUTO_RESUME_LATEST" \
        FORCE_NEW_BATCH="$FORCE_NEW_BATCH" \
        JOB_NAME_PREFIX="zxy-editx-en-batch" \
        bash "$PROJ_ROOT/submit_editx_batch_h200.sh" $DRY_RUN_FLAG
        ;;
    after_editx)
        echo "[after_editx] editx 已完成，遍历 $INPUT_VCDATA_ROOT 下所有 split_*/ 跑本地 emotion+pair"
        echo ""
        DONE=0; FAIL=0
        for split_dir in "$INPUT_VCDATA_ROOT"/split_*; do
            [ -d "$split_dir" ] || continue
            split_name=$(basename "$split_dir")
            echo "──────────── $split_name ────────────"
            if bash "$PROJ_ROOT/run_e2e.sh" from_vcdata "$split_name"; then
                DONE=$((DONE + 1))
            else
                FAIL=$((FAIL + 1))
                echo "  [warn] $split_name 失败，继续下一个" >&2
            fi
        done
        echo ""
        echo "═══ after_editx 完成：成功 $DONE / 失败 $FAIL ═══"
        ;;
    local)
        echo "[local] 单 split 本地跑：$LOCAL_SPLIT_NAME"
        exec bash "$PROJ_ROOT/run_e2e.sh" from_vcdata "$LOCAL_SPLIT_NAME"
        ;;
    *)
        echo "ERROR: RUN_MODE 必须是 submit / after_editx / local（收到 $RUN_MODE）" >&2
        exit 1
        ;;
esac
