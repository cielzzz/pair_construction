#!/usr/bin/env bash
# 端到端流水线 wrapper，支持两种入口：
#
#   场景 A (MODE=full)：输入是原始 kxhuang jsonl，需要先跑 MOSS-TTS 生 ref_audio
#     stages: 0 → 1 (vcdata) → 1.5 (merge) → 2 (editx) → 3 → 4 → 5
#
#   场景 B (MODE=from_vcdata)：vcdata 已就绪（已有 manifest_shard*.jsonl + ref_audio/）
#     stages: 1.5 (merge) → 2 (editx) → 3 → 4 → 5
#
# 用法：
#   bash run_e2e.sh full <input_jsonl> <split_name> [head_n]
#   bash run_e2e.sh from_vcdata <split_name>
#
# 推荐通过 runs/ 下的 wrapper 调用（参数都写死）：
#   sh runs/run_demo_full.sh
#   sh runs/run_demo_from_vcdata.sh

set -uo pipefail

MODE="${1:?usage: run_e2e.sh <full|from_vcdata> ...}"
case "$MODE" in
    full)
        INPUT_JSONL="${2:?usage: full <input_jsonl> <split_name> [head_n]}"
        SPLIT="${3:?usage: full <input_jsonl> <split_name> [head_n]}"
        HEAD_N="${4:-0}"
        ;;
    from_vcdata)
        SPLIT="${2:?usage: from_vcdata <split_name>}"
        INPUT_JSONL=""   # 不需要
        HEAD_N=0
        ;;
    *)
        echo "ERROR: MODE 必须是 full 或 from_vcdata（收到 '$MODE'）" >&2
        exit 1
        ;;
esac

# ─── 可通过环境变量定制 ──────────────────────────
# stage1 文本字段名（中文 kxhuang 用 sensevoice_small_clean；英文 kxhuang 用 text）
TEXT_FIELD="${TEXT_FIELD:-sensevoice_small_clean}"
# stage2 跑哪些 edit 模式，逗号分隔 type:info，例：
#   "style:radio"
#   "style:radio,style:remove,style:news,style:chat,emotion:remove,emotion:coldness"
EDIT_PAIRS="${EDIT_PAIRS:-style:radio}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$SCRIPT_DIR"
S="$PROJ_ROOT/scripts"

VCDATA_REPO="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction"
VC_EDIT_REPO="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_edit"
EMOTION_ACTIVATE="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/emotion_eval/activate_emotion_eval.sh"
EMOTION_PY() { bash "$EMOTION_ACTIVATE" python "$@"; }

CONFIG_FILE="${PAIR_CONFIG:-$PROJ_ROOT/configs/default.yaml}"
read_yaml() {
    EMOTION_PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])" \
        "$CONFIG_FILE" "$@"
}

VCDATA_OUTPUTS_DIR="${VCDATA_ROOT:-$(read_yaml paths vcdata_root)}"   # env 优先
EVAL_ROOT="$(read_yaml paths emotion_eval_root)"
PAIR_OUTPUTS_ROOT="$(read_yaml paths outputs_root)"
SPLIT_DIR="$VCDATA_OUTPUTS_DIR/$SPLIT"

echo "═══════════════════════════════════════════════"
echo " pair_construction E2E"
echo "  MODE           = $MODE"
echo "  SPLIT          = $SPLIT"
[ "$MODE" = "full" ] && echo "  INPUT_JSONL    = $INPUT_JSONL"
[ "$MODE" = "full" ] && echo "  HEAD_N         = $HEAD_N  (0=全量)"
echo "  SPLIT_DIR      = $SPLIT_DIR"
echo "  PAIR_OUTPUTS   = $PAIR_OUTPUTS_ROOT"
echo "═══════════════════════════════════════════════"

# ─── stage 0 + 1：vcdata（只 full 模式） ─────────
if [ "$MODE" = "full" ]; then
    # stage 0：取 head_n 子集
    PENDING_INPUT_DIR="$PROJ_ROOT/outputs/$SPLIT/_e2e_input"
    mkdir -p "$PENDING_INPUT_DIR"
    SOURCE_JSONL="$PENDING_INPUT_DIR/$SPLIT.jsonl"
    if [ "$HEAD_N" -gt 0 ]; then
        head -n "$HEAD_N" "$INPUT_JSONL" > "$SOURCE_JSONL"
        echo "[stage0] 取头 $HEAD_N 行 → $SOURCE_JSONL"
    else
        ln -sf "$INPUT_JSONL" "$SOURCE_JSONL"
        echo "[stage0] symlink 全量 → $SOURCE_JSONL"
    fi

    # stage 1：vcdata stage1_generate
    if [ -d "$SPLIT_DIR" ] && [ -f "$SPLIT_DIR/.stage1_generate_state.json" ]; then
        echo "[stage1] $SPLIT_DIR 已存在且有 state.json，跳过 vcdata stage1（如需重跑请先删）"
    else
        echo "[stage1] 跑 vcdata stage1_generate.py"
        bash "$VCDATA_REPO/activate_moss_ttsd_vc.sh" python "$VCDATA_REPO/stage1_generate.py" \
            --input-dir "$PENDING_INPUT_DIR" \
            --output-dir "$VCDATA_OUTPUTS_DIR" \
            --model "$VCDATA_REPO/MOSS-TTS" \
            --device cuda:0 \
            --shard-id 0 \
            --num-shards 1 \
            --num-candidates 16 \
            --batch-size 16 \
            --text-field "$TEXT_FIELD" \
            --audio-path-field local_path \
            --similarity-threshold 0.85 \
            --seed 42 \
            --resume \
            --allow-resume-shard-change
        [ $? -ne 0 ] && { echo "[stage1] FAILED"; exit 1; }
    fi
else
    # from_vcdata 模式：校验 vcdata 输出已存在
    if [ ! -d "$SPLIT_DIR" ]; then
        echo "ERROR: vcdata 目录不存在: $SPLIT_DIR" >&2
        echo "       请先用 MODE=full 跑过 vcdata，或检查 SPLIT 名" >&2
        exit 1
    fi
    if ! ls "$SPLIT_DIR"/manifest_shard*.jsonl >/dev/null 2>&1; then
        echo "ERROR: $SPLIT_DIR 下找不到 manifest_shard*.jsonl" >&2
        exit 1
    fi
    echo "[from_vcdata] 用已有 vcdata 输出: $SPLIT_DIR"
fi

# ─── stage 1.5：merge_shards（两种模式都跑） ─────
MERGED="$SPLIT_DIR/merged.stepaudio_input.all.jsonl"
if [ ! -f "$MERGED" ]; then
    echo "[stage1.5] merge_shards → $MERGED"
    bash "$VCDATA_REPO/activate_moss_ttsd_vc.sh" python "$VCDATA_REPO/merge_shards.py" \
        --input-dir "$SPLIT_DIR" \
        --output "$MERGED"
    [ $? -ne 0 ] && { echo "[stage1.5] FAILED"; exit 1; }
else
    echo "[stage1.5] $MERGED 已存在，跳过"
fi

# ─── stage 2：editx 各模式（按 EDIT_PAIRS 循环） ─────
STEPX_PY="/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/step_audio_editx/bin/python"
IFS=',' read -ra _EDIT_LIST <<< "$EDIT_PAIRS"
for pair in "${_EDIT_LIST[@]}"; do
    e_type="${pair%%:*}"
    e_info="${pair##*:}"
    e_tag="${e_type}_${e_info}"
    existing="$(ls "$SPLIT_DIR"/stepaudio_${e_tag}_${SPLIT}_*/paired_report.jsonl 2>/dev/null | head -1)"
    if [ -n "$existing" ] && [ -f "$existing" ]; then
        echo "[stage2] $e_tag 已存在 → $existing"
        continue
    fi
    echo "[stage2] 跑 editx $e_tag（约 15-20 分钟 100 句）"
    PYTHON_BIN="$STEPX_PY" EDIT_TYPE="$e_type" EDIT_INFO="$e_info" \
        bash "$VC_EDIT_REPO/vc_edit_framework/scripts/run_step_editx_split.sh" "$SPLIT_DIR"
    [ $? -ne 0 ] && { echo "[stage2] $e_tag FAILED"; exit 1; }
done

# ─── stage 3：pair_construction 中间表 ─────────
cd "$PROJ_ROOT"
echo "═══ stage 3：构建中间表 ═══════════════════"
EMOTION_PY "$S/01_build_vcdata_base.py" --split "$SPLIT" || exit 1
EMOTION_PY "$S/02_build_editx_base.py"  --split "$SPLIT" || exit 1
EMOTION_PY "$S/03_join_editx_with_vcdata.py" --split "$SPLIT" || exit 1

# ─── stage 4：emotion eval ─────────────────────
echo "═══ stage 4：emotion eval ════════════════"
bash "$S/04_run_emotion_eval.sh" "$SPLIT" cuda:0
[ $? -ne 0 ] && { echo "[stage4] FAILED"; exit 1; }

# ─── stage 5：构造 6 类 pair ───────────────────
echo "═══ stage 5：6 类 pair ════════════════════"
EMOTION_PY "$S/05_construct_A.py"       --split "$SPLIT" || exit 1
EMOTION_PY "$S/06_construct_B.py" --split "$SPLIT" || exit 1
EMOTION_PY "$S/07_construct_C.py" --split "$SPLIT" || exit 1
EMOTION_PY "$S/07b_construct_C_mixed.py" --split "$SPLIT" || exit 1
EMOTION_PY "$S/07c_construct_D.py" --split "$SPLIT" || exit 1
EMOTION_PY "$S/08_construct_H1.py"      --split "$SPLIT" || exit 1
EMOTION_PY "$S/09_construct_H2.py"      --split "$SPLIT" || exit 1
EMOTION_PY "$S/10_construct_H3.py"      --split "$SPLIT" || exit 1

echo "═══ DONE ════════════════════════════════"
PAIR_DIR="$PAIR_OUTPUTS_ROOT/$SPLIT/pairs"
for f in "$PAIR_DIR"/*.jsonl; do
    [ -f "$f" ] || continue
    printf '  %-20s %8d rows\n' "$(basename "$f")" "$(wc -l < "$f")"
done
