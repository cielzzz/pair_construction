#!/usr/bin/env bash
# 启智平台 editx 批量提交（仿 vcdata_construction/submit_split_batch_h200.sh）
#
# 工作流：
#   1. 扫 INPUT_VCDATA_ROOT 下所有 split_*/ 目录（要求已有 manifest_shard*.jsonl）
#   2. 按"剩余 cases 数"负载均衡分到 TASK_COUNT 个 group_NN/
#   3. 每组 group_NN/ 里 symlink 所有该组负责的 split 目录
#   4. 每组提交一个启智任务，任务内循环对每个 split 跑 EDIT_PAIRS 指定的 edit 模式
#
# 关键环境变量（覆盖默认值）：
#   INPUT_VCDATA_ROOT   含 split_*/ 的 vcdata 输出根（如 kxhuang/vc_mosstts/.../zh）
#   EDIT_PAIRS          逗号分隔 type:info，默认 "style:radio"
#                        例 "style:radio,style:remove,style:news,style:chat,emotion:remove,emotion:coldness"
#   TASK_COUNT          切多少组（默认 16）
#   JOB_NAME_PREFIX     启智任务名前缀（默认 zxy-editx-batch）
#   AUTO_RESUME_LATEST  默认 1（续上次 batch）
#   FORCE_NEW_BATCH     默认 0（设为 1 强制开新 batch）
#   DRY_RUN             加 --dry-run 仅打印分组+任务命令，不真提交

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
JOB_ROOT="$SCRIPT_DIR"

# 启智 / qzcli 参数（与 vcdata submit 一致；按需覆盖）
QZCLI="${QZCLI:-qzcli}"
WORKSPACE="${WORKSPACE:-CI-情境智能}"
PROJECT="${PROJECT:-CI-情境智能}"
COMPUTE_GROUP="${COMPUTE_GROUP:-lcg-8bc04380-24f8-4eb4-89c0-a7a3f6ca29b4}"
DEFAULT_COMPUTE_GROUPS_CSV="lcg-4202c8f7-8308-412c-92b9-77daccab3c7f,lcg-4202c8f7-8308-412c-92b9-77daccab3c7f,lcg-4202c8f7-8308-412c-92b9-77daccab3c7f,lcg-4202c8f7-8308-412c-92b9-77daccab3c7f,lcg-73efc9b6-8d94-406c-b150-50c91fda377f,lcg-73efc9b6-8d94-406c-b150-50c91fda377f,lcg-8bc04380-24f8-4eb4-89c0-a7a3f6ca29b4,lcg-8bc04380-24f8-4eb4-89c0-a7a3f6ca29b4,lcg-8bc04380-24f8-4eb4-89c0-a7a3f6ca29b4,lcg-8bc04380-24f8-4eb4-89c0-a7a3f6ca29b4,lcg-8bc04380-24f8-4eb4-89c0-a7a3f6ca29b4,lcg-8bc04380-24f8-4eb4-89c0-a7a3f6ca29b4,lcg-8bc04380-24f8-4eb4-89c0-a7a3f6ca29b4,lcg-8bc04380-24f8-4eb4-89c0-a7a3f6ca29b4,lcg-8bc04380-24f8-4eb4-89c0-a7a3f6ca29b4,lcg-8bc04380-24f8-4eb4-89c0-a7a3f6ca29b4"
COMPUTE_GROUPS_CSV="${COMPUTE_GROUPS_CSV:-$DEFAULT_COMPUTE_GROUPS_CSV}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
INSTANCES=1

# editx 业务参数
INPUT_VCDATA_ROOT="${INPUT_VCDATA_ROOT:?需要 INPUT_VCDATA_ROOT，例 /inspire/qb-ilm/.../kxhuang/vc_mosstts/instruction_0.1_enzh/zh}"
EDIT_PAIRS="${EDIT_PAIRS:-style:radio}"

# editx 入口脚本（不修改上游）
VC_EDIT_REPO="${VC_EDIT_REPO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_edit}"
EDITX_SCRIPT="$VC_EDIT_REPO/vc_edit_framework/scripts/run_step_editx_split.sh"
STEPX_PY="${STEPX_PY:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/step_audio_editx/bin/python}"

TASK_COUNT="${TASK_COUNT:-16}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-zxy-editx-batch}"
BATCH_ID="${BATCH_ID:-$(date -u +%m%d-%H%M%S)}"
GROUP_ROOT="${GROUP_ROOT:-$JOB_ROOT/.qz_editx_batches/$BATCH_ID}"
AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-1}"
FORCE_NEW_BATCH="${FORCE_NEW_BATCH:-0}"
RESUME_BATCH_ID="${RESUME_BATCH_ID:-}"
DRY_RUN=0
for arg in "$@"; do
  [ "$arg" = "--dry-run" ] && DRY_RUN=1
done

declare -a PENDING_SPLITS=()
declare -a PENDING_REMAINING=()
declare -a COMPUTE_GROUPS=()
declare -a GROUP_DIRS=()
declare -a GROUP_SPLIT_COUNTS=()
declare -a GROUP_REMAINING_CASES=()
declare -A SUBMITTED_GROUP_TAGS=()

TOTAL_SPLITS=0
SKIPPED_SPLITS=0
PENDING_COUNT=0
GROUP_COUNT=0
RESUMING_BATCH=0

trim_spaces() { printf '%s' "$1" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'; }

# 一个 split 跑完所有 EDIT_PAIRS 模式所需 cases 数
manifest_total() {
    local split_dir="$1"
    local mp=("$split_dir"/manifest_shard*.jsonl)
    [ -e "${mp[0]}" ] || { echo 0; return; }
    wc -l "${mp[@]}" | awk 'END { print $1 + 0 }'
}

# 一个 split 已完成的 cases 数（所有 edit_tag paired_report.jsonl 行数之和）
editx_done() {
    local split_dir="$1"
    local done=0
    local pair type info tag report
    IFS=',' read -ra _PAIRS <<< "$EDIT_PAIRS"
    for pair in "${_PAIRS[@]}"; do
        type="${pair%%:*}"; info="${pair##*:}"
        tag="${type}_${info}"
        for report in "$split_dir"/stepaudio_${tag}_*/paired_report.jsonl; do
            [ -f "$report" ] || continue
            done=$((done + $(wc -l < "$report")))
            break
        done
    done
    echo "$done"
}

parse_compute_groups() {
    COMPUTE_GROUPS=()
    if [ -n "$COMPUTE_GROUPS_CSV" ]; then
        IFS=',' read -r -a raw_groups <<< "$COMPUTE_GROUPS_CSV"
        for g in "${raw_groups[@]}"; do
            g=$(trim_spaces "$g")
            if [ -n "$g" ]; then
                COMPUTE_GROUPS+=("$g")
            fi
        done
    fi
    if [ "${#COMPUTE_GROUPS[@]}" -eq 0 ]; then
        COMPUTE_GROUPS=("$COMPUTE_GROUP")
    fi
    return 0
}

collect_pending_splits() {
    PENDING_SPLITS=()
    PENDING_REMAINING=()
    TOTAL_SPLITS=0
    SKIPPED_SPLITS=0

    IFS=',' read -ra _PAIRS <<< "$EDIT_PAIRS"
    local num_modes="${#_PAIRS[@]}"

    local split_dir total done remaining
    for split_dir in "$INPUT_VCDATA_ROOT"/split_*; do
        [ -d "$split_dir" ] || continue
        TOTAL_SPLITS=$((TOTAL_SPLITS + 1))
        total=$(manifest_total "$split_dir")
        # 期望完成数 = manifest 行数 × 模式数
        local expected=$((total * num_modes))
        done=$(editx_done "$split_dir")
        remaining=$((expected - done))
        [ "$remaining" -le 0 ] && { SKIPPED_SPLITS=$((SKIPPED_SPLITS + 1)); continue; }

        PENDING_SPLITS+=("$split_dir")
        PENDING_REMAINING+=("$remaining")
    done

    [ "$TOTAL_SPLITS" -eq 0 ] && { echo "ERROR: $INPUT_VCDATA_ROOT 下无 split_*/" >&2; return 2; }
    PENDING_COUNT="${#PENDING_SPLITS[@]}"
    [ "$PENDING_COUNT" -eq 0 ] && return 3
    return 0
}

latest_batch_dir() {
    ls -dt "$JOB_ROOT"/.qz_editx_batches/* 2>/dev/null | head -n 1 || true
}

discover_resume_batch() {
    local candidate=""
    [ "$FORCE_NEW_BATCH" -eq 1 ] && return 1
    if [ -n "$RESUME_BATCH_ID" ]; then
        candidate="$JOB_ROOT/.qz_editx_batches/$RESUME_BATCH_ID"
    elif [ "$AUTO_RESUME_LATEST" -eq 1 ]; then
        candidate=$(latest_batch_dir)
    fi
    [ -z "$candidate" ] || [ ! -d "$candidate" ] && return 1
    local g=$(find "$candidate" -maxdepth 1 -type d -name 'group_*' | wc -l)
    [ "$g" -le 0 ] && return 1
    local s=0
    [ -f "$candidate/submitted_jobs.tsv" ] && s=$(awk 'END{print NR+0}' "$candidate/submitted_jobs.tsv")
    [ "$s" -ge "$g" ] && return 1
    BATCH_ID=$(basename "$candidate")
    GROUP_ROOT="$candidate"
    RESUMING_BATCH=1
    return 0
}

prepare_new_groups() {
    GROUP_COUNT="$TASK_COUNT"
    [ "$GROUP_COUNT" -gt "$PENDING_COUNT" ] && GROUP_COUNT="$PENDING_COUNT"
    [ "$GROUP_COUNT" -le 0 ] && { echo "ERROR: TASK_COUNT 必须 >= 1"; return 1; }
    mkdir -p "$GROUP_ROOT"
    GROUP_DIRS=(); GROUP_SPLIT_COUNTS=(); GROUP_REMAINING_CASES=()
    for i in $(seq 0 $((GROUP_COUNT - 1))); do
        local d="$GROUP_ROOT/group_$(printf '%02d' "$i")"
        mkdir -p "$d"
        : > "$d/splits.txt"
        : > "$d/plan.tsv"
        GROUP_DIRS+=("$d")
        GROUP_SPLIT_COUNTS+=(0)
        GROUP_REMAINING_CASES+=(0)
    done
}

assign_balanced() {
    local tmpf=$(mktemp)
    for idx in "${!PENDING_SPLITS[@]}"; do
        printf '%d\t%s\n' "${PENDING_REMAINING[$idx]}" "${PENDING_SPLITS[$idx]}" >> "$tmpf"
    done
    while IFS=$'\t' read -r rem split_path; do
        local best=0; local bl="${GROUP_REMAINING_CASES[0]}"
        for g in $(seq 1 $((GROUP_COUNT - 1))); do
            if [ "${GROUP_REMAINING_CASES[$g]}" -lt "$bl" ]; then
                best="$g"; bl="${GROUP_REMAINING_CASES[$g]}"
            fi
        done
        local basename=$(basename "$split_path")
        ln -sf "$split_path" "${GROUP_DIRS[$best]}/$basename"
        printf '%s\n' "$basename" >> "${GROUP_DIRS[$best]}/splits.txt"
        printf '%s\t%s\n' "$basename" "$rem" >> "${GROUP_DIRS[$best]}/plan.tsv"
        GROUP_SPLIT_COUNTS[$best]=$((GROUP_SPLIT_COUNTS[$best] + 1))
        GROUP_REMAINING_CASES[$best]=$((GROUP_REMAINING_CASES[$best] + rem))
    done < <(sort -rn -k1,1 "$tmpf")
    rm -f "$tmpf"
}

load_existing_groups() {
    GROUP_DIRS=(); GROUP_SPLIT_COUNTS=(); GROUP_REMAINING_CASES=()
    mapfile -t GROUP_DIRS < <(find "$GROUP_ROOT" -maxdepth 1 -type d -name 'group_*' | sort)
    GROUP_COUNT="${#GROUP_DIRS[@]}"
    [ "$GROUP_COUNT" -le 0 ] && { echo "ERROR: 无 group_*"; return 1; }
    for d in "${GROUP_DIRS[@]}"; do
        local sc=0; local rc=0
        [ -f "$d/splits.txt" ] && sc=$(awk 'END{print NR+0}' "$d/splits.txt")
        [ -f "$d/plan.tsv" ] && rc=$(awk -F'\t' '{s+=$2}END{print s+0}' "$d/plan.tsv")
        GROUP_SPLIT_COUNTS+=("$sc")
        GROUP_REMAINING_CASES+=("$rc")
    done
}

load_submitted_tags() {
    SUBMITTED_GROUP_TAGS=()
    [ -f "$SUMMARY_PATH" ] || return
    while IFS=$'\t' read -r jname _rest; do
        [[ "$jname" =~ -g([0-9]{2})$ ]] && SUBMITTED_GROUP_TAGS["${BASH_REMATCH[1]}"]=1
    done < "$SUMMARY_PATH"
}

# 一个 group 的启智任务命令：遍历该组所有 split + 所有 EDIT_PAIRS
job_command_for_group() {
    local group_dir="$1"
    cat <<EOF
bash -lc 'set -e; \
for split_link in "$group_dir"/split_*; do \
  [ -d "\$split_link" ] || continue; \
  for pair in \$(echo "$EDIT_PAIRS" | tr "," " "); do \
    t=\${pair%%:*}; i=\${pair##*:}; \
    echo ">>> editx \$t:\$i on \$(basename \$split_link)"; \
    PYTHON_BIN="$STEPX_PY" EDIT_TYPE="\$t" EDIT_INFO="\$i" \
      bash "$EDITX_SCRIPT" "\$split_link"; \
  done; \
done'
EOF
}

parse_compute_groups

if discover_resume_batch; then
    load_existing_groups
else
    set +e
    collect_pending_splits
    rc=$?
    set -e
    [ "$rc" -eq 2 ] && exit 1
    [ "$rc" -eq 3 ] && { echo "所有 split 的 editx 都已完成，无可提交。"; exit 0; }
    prepare_new_groups
    assign_balanced
fi

SUMMARY_PATH="$GROUP_ROOT/submitted_jobs.tsv"
touch "$SUMMARY_PATH"
load_submitted_tags

echo "=========================================="
echo "editx batch submit (启智 H200)"
echo "  BATCH_ID=$BATCH_ID"
echo "  INPUT_VCDATA_ROOT=$INPUT_VCDATA_ROOT"
echo "  EDIT_PAIRS=$EDIT_PAIRS"
echo "  GROUP_ROOT=$GROUP_ROOT"
if [ "$RESUMING_BATCH" -eq 1 ]; then
    echo "  MODE=resume_incomplete_batch"
else
    echo "  MODE=create_new_batch"
    echo "  TOTAL_SPLITS=$TOTAL_SPLITS, SKIPPED=$SKIPPED_SPLITS, PENDING=$PENDING_COUNT"
fi
echo "  TASK_COUNT=$TASK_COUNT  EFFECTIVE_GROUP_COUNT=$GROUP_COUNT"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================="

for gid in $(seq 0 $((GROUP_COUNT - 1))); do
    tag=$(printf '%02d' "$gid")
    gdir="${GROUP_DIRS[$gid]}"
    sc="${GROUP_SPLIT_COUNTS[$gid]}"
    rc="${GROUP_REMAINING_CASES[$gid]}"
    cg="${COMPUTE_GROUPS[$((gid % ${#COMPUTE_GROUPS[@]}))]}"
    jname="${JOB_NAME_PREFIX}-${BATCH_ID}-g${tag}"
    jcmd=$(job_command_for_group "$gdir")

    echo "------------------------------------------"
    echo "Group $gid/$GROUP_COUNT  JOB=$jname  CG=$cg  SPLITS=$sc  REMAINING_CASES=$rc"
    echo "  GROUP_DIR=$gdir"
    [ "$sc" -eq 0 ] && { echo "  Skipped: 该组无 split"; continue; }
    [ -n "${SUBMITTED_GROUP_TAGS[$tag]:-}" ] && { echo "  Skipped: 已提交过"; continue; }

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY-RUN] 命令:"
        printf '%s\n' "$jcmd"
        continue
    fi

    TMP=$(mktemp)
    set +e
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
        "$QZCLI" create-job \
        --name "$jname" \
        --workspace "$WORKSPACE" \
        --project "$PROJECT" \
        --compute-group "$cg" \
        --spec "$SPEC" \
        --framework "$FRAMEWORK" \
        --instances "$INSTANCES" \
        --shm "$SHM_GI" \
        --priority "$PRIORITY" \
        --image "$IMAGE" \
        --image-type "$IMAGE_TYPE" \
        --command "$jcmd" >"$TMP" 2>&1
    status=$?
    set -e
    cat "$TMP"
    if [ "$status" -ne 0 ]; then
        echo "Submission failed for $jname"
        grep -q 'Cookie 已过期或无效' "$TMP" && echo "Fix: qzcli login"
        rm -f "$TMP"; exit "$status"
    fi
    jid=$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$TMP" | tail -n 1 || true)
    printf '%s\t%s\t%s\t%s\n' "$jname" "${jid:-UNKNOWN}" "$cg" "$gdir" >> "$SUMMARY_PATH"
    SUBMITTED_GROUP_TAGS["$tag"]=1
    rm -f "$TMP"
done

echo "=========================================="
echo "editx 批量提交完成。SUMMARY=$SUMMARY_PATH"
echo "=========================================="
