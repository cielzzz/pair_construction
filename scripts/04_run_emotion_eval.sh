#!/usr/bin/env bash
# 04: 对一个 split 下的 original / ref / 各 edit_tag 音频跑 emotion2vec + SenseVoice
#
# 复用 emotion_eval/scripts/{score_neutrality.py, sensevoice_score.py}（不修改上游）。
# 自动把 score_neutrality 的多个 per-name per_file.csv 合并成单一 per_file.csv（加 group 列），
# 然后 sensevoice_score 给该合并文件追加 sv_label。
#
# 用法： bash 04_run_emotion_eval.sh <split_name> [device]
#   bash 04_run_emotion_eval.sh split_0000 cuda:0
#
# 已有 split_0000 的 qzrun 结果可直接复用：
#   bash 04_run_emotion_eval.sh split_0000 --reuse-qzrun

set -euo pipefail
SPLIT="${1:?usage: 04_run_emotion_eval.sh <split_name> [device|--reuse-qzrun]}"
ARG2="${2:-cuda:0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# read_yaml 必须用 emotion env 的 python（有 pyyaml）。
EMOTION_PY_BIN="${EMOTION_PY_BIN:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/emotion/bin/python}"
# 支持通过 PAIR_CONFIG 环境变量切换配置文件（与 _utils.load_config 一致）
CONFIG_FILE="${PAIR_CONFIG:-$PROJ_ROOT/configs/default.yaml}"
read_yaml() {
    "$EMOTION_PY_BIN" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])" \
        "$CONFIG_FILE" "$@"
}
VCDATA_ROOT="${VCDATA_ROOT:-$(read_yaml paths vcdata_root)}"
EVAL_ROOT="${EMOTION_EVAL_ROOT:-$(read_yaml paths emotion_eval_root)}"
OUT_ROOT="${PAIR_OUTPUTS_ROOT:-$(read_yaml paths outputs_root)}"

SPLIT_DIR="$VCDATA_ROOT/$SPLIT"
OUT="$OUT_ROOT/$SPLIT/emotion"
mkdir -p "$OUT"
INTERMEDIATE_ROOT="$OUT_ROOT/$SPLIT/intermediate"
AUDIO_INDEX_JSON="$INTERMEDIATE_ROOT/audio_index.json"

# ─── 复用 qzrun 已有结果（仅 split_0000 适用） ───────────
if [ "$ARG2" = "--reuse-qzrun" ]; then
    QZRUN_OUT="$EVAL_ROOT/outputs/zh_${SPLIT}_qzrun"
    if [ ! -d "$QZRUN_OUT" ]; then
        QZRUN_OUT="$EVAL_ROOT/outputs/zh_split_${SPLIT#split_}_qzrun"
    fi
    if [ ! -f "$QZRUN_OUT/per_file_dual.csv" ]; then
        echo "[04] 找不到 $QZRUN_OUT/per_file_dual.csv" >&2
        exit 1
    fi
    # 直接 symlink 关键产物
    ln -sf "$QZRUN_OUT/per_file_dual.csv" "$OUT/per_file_dual.csv"
    ln -sf "$QZRUN_OUT/per_pair.csv"      "$OUT/per_pair.csv"
    ln -sf "$QZRUN_OUT/per_row.csv"       "$OUT/per_row.csv"
    echo "[04] reused: $QZRUN_OUT  → $OUT"
    echo "[04] 注意：reuse-qzrun 模式下未对 original_audio 评分，A 类与 H1 的 original-side emotion 将为空"
    exit 0
fi

DEVICE="$ARG2"

# 不 source emotion env（它 set -e 会污染当前 shell）；用绝对 python 路径即可
# 但需要 MODELSCOPE_CACHE / HF_HOME 让 funasr 找到已下模型
export MODELSCOPE_CACHE="$EVAL_ROOT/model_cache/modelscope"
export HF_HOME="$EVAL_ROOT/model_cache/hf"

# ─── 0) 预建 split 音频索引（供 02/04 共享） ─────────────
"$EMOTION_PY_BIN" - <<PY
import sys

sys.path.insert(0, "$SCRIPT_DIR")
from _utils import load_config, load_or_build_audio_index

cfg = load_config("$CONFIG_FILE")
idx = load_or_build_audio_index(cfg, "$SPLIT", force_rebuild=True)
print(f"[04] audio index ready: ref={len(idx.get('ref_audio_by_row', {}))}, "
      f"edit_tags={sorted(idx.get('edit_audio_by_tag_row', {}).keys())}")
PY

# ─── 1) 给 original_audio 建 symlink 目录 ─────────────
ORIG_DIR="$OUT/_links_original"
mkdir -p "$ORIG_DIR"
"$EMOTION_PY_BIN" - <<PY
import json, os
from pathlib import Path
base = Path("$INTERMEDIATE_ROOT/vcdata_base.jsonl")
linkdir = Path("$ORIG_DIR")
if not base.exists():
    raise SystemExit(f"先跑 01: 缺 {base}")
seen = set()

def ensure_symlink(src: str, dst: Path):
    if os.path.lexists(dst):
        try:
            if os.path.realpath(dst) == os.path.realpath(src):
                return
        except OSError:
            pass
        os.unlink(dst)
    os.symlink(src, dst)

with open(base) as f:
    for ln in f:
        r = json.loads(ln)
        src = r["original_audio"]
        if src in seen:
            continue
        seen.add(src)
        dst = linkdir / f"{r['original_idx']:06d}{Path(src).suffix}"
        ensure_symlink(src, dst)
print(f"linked {len(seen)} original audios → {linkdir}")
PY

# ─── 2) 组装 --input 列表 ─────────────────────────────
INPUTS=()
INPUTS+=( "-i" "original=$ORIG_DIR" )
INPUTS+=( "-i" "ref=$SPLIT_DIR/ref_audio" )
mapfile -t EDIT_INPUTS < <("$EMOTION_PY_BIN" - <<PY
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, "$SCRIPT_DIR")
from _utils import load_config, load_or_build_audio_index

cfg = load_config("$CONFIG_FILE")
idx = load_or_build_audio_index(cfg, "$SPLIT")
out_root = Path("$OUT")

def ensure_symlink(src: str, dst: Path):
    if os.path.lexists(dst):
        try:
            if os.path.realpath(dst) == os.path.realpath(src):
                return
        except OSError:
            pass
        os.unlink(dst)
    os.symlink(src, dst)

for tag, row_map in sorted(idx.get("edit_audio_by_tag_row", {}).items()):
    linkdir = out_root / f"_links_{tag}"
    linkdir.mkdir(parents=True, exist_ok=True)
    mapping = []
    for row, src in sorted(row_map.items(), key=lambda item: int(item[0])):
        src_path = Path(src)
        if not src_path.exists():
            continue
        dst = linkdir / f"{int(row):06d}{src_path.suffix}"
        ensure_symlink(str(src_path), dst)
        mapping.append((str(dst), str(src_path)))
    if not mapping:
        continue
    with open(linkdir / "_mapping.csv", "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["link_wav", "edited_audio"])
        writer.writerows(mapping)
    print(f"{tag}={linkdir}")
PY
)
for item in "${EDIT_INPUTS[@]}"; do
    [ -n "$item" ] || continue
    INPUTS+=( "-i" "$item" )
done
echo "[04] inputs: ${INPUTS[*]}"

# ─── 3) emotion2vec ───────────────────────────────────
# .wav 仅；score_neutrality 用 rglob('*.wav')，对 .flac 不识别。
# 对 original 池中的 .flac，先转码为 wav。
export ORIG_DIR
"$EMOTION_PY_BIN" - <<PY
import os, csv
from pathlib import Path
import soundfile as sf
linkdir = Path(os.environ["ORIG_DIR"])
flacs = list(linkdir.glob("*.flac"))
print(f"[04] 转码 {len(flacs)} 个 flac → wav (soundfile, 已存在跳过)")
ok = skipped = 0
mapping = []   # (wav_path, original_audio_path)
for f in flacs:
    w = f.with_suffix(".wav")
    real = os.path.realpath(str(f))  # symlink target == 真实 original_audio
    mapping.append((str(w), real))
    if w.exists():
        skipped += 1
        continue
    try:
        data, sr = sf.read(str(f))
        sf.write(str(w), data, sr)
        ok += 1
    except Exception as e:
        print(f"  [warn] {f.name}: {e}")
# 写 mapping 让 _emotion_lookup 把 link_wav 的 emotion record 别名到 original_audio
with open(linkdir / "_mapping.csv", "w", encoding="utf-8", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["link_wav", "original_audio"])
    w.writerows(mapping)
print(f"[04] 转码完成：新建 {ok}，已存在 {skipped}；mapping={len(mapping)}")
PY

"$EMOTION_PY_BIN" "$EVAL_ROOT/scripts/score_neutrality.py" \
    "${INPUTS[@]}" \
    --out "$OUT" \
    --device "$DEVICE"

# ─── 4) 合并 per-name per_file.csv → 单一 per_file.csv（加 group 列） ───
"$EMOTION_PY_BIN" - <<PY
import pandas as pd
from pathlib import Path
out_root = Path("$OUT")
frames = []
for sub in out_root.iterdir():
    pf = sub / "per_file.csv"
    if not pf.exists():
        continue
    df = pd.read_csv(pf)
    df["group"] = sub.name
    df["wav"] = df["path"]   # sensevoice_score 需要 wav 列
    frames.append(df)
if not frames:
    raise SystemExit("[04] 没找到任何 per_file.csv")
all_df = pd.concat(frames, ignore_index=True)
out_csv = out_root / "per_file.csv"
all_df.to_csv(out_csv, index=False)
print(f"[04] 合并 {len(frames)} 个 group，共 {len(all_df)} 行 → {out_csv}")
PY

# ─── 5) SenseVoice 双分类器 ─────────────────────────
"$EMOTION_PY_BIN" "$EVAL_ROOT/scripts/sensevoice_score.py" \
    --run-dir "$OUT" \
    --device "$DEVICE" || echo "[04] [warn] sensevoice 失败，emotion2vec 已就绪"

echo "[04] emotion eval done. outputs → $OUT"

# ─── 6) DNSMOS（Microsoft P.835 raw 分数，与 vcdata 原始 dnsmos 字段同口径） ─
# 用 moss_ttsd_sglang env 跑（emotion env 没装 onnxruntime；该 env 有 onnxruntime+librosa）
DNSMOS_PY="${DNSMOS_PY:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/moss_ttsd_sglang/bin/python}"
if [ -x "$DNSMOS_PY" ]; then
    echo "[04b] 跑 DNSMOS（给 per_file_dual.csv 加 dnsmos_ovrl/sig/bak 三列）..."
    PAIR_CONFIG="$CONFIG_FILE" VCDATA_ROOT="${VCDATA_ROOT:-}" \
        "$DNSMOS_PY" "$SCRIPT_DIR/04b_add_dnsmos.py" \
        --split "$SPLIT" \
        --flush-every "${DNSMOS_FLUSH_EVERY:-500}" || echo "[04b] [warn] DNSMOS 失败，继续"
else
    echo "[04b] [skip] 找不到 moss_ttsd_sglang env，跳过 DNSMOS"
fi

echo "[04] 关键产物：$OUT/per_file_dual.csv  (供 05-10 脚本查询，含 dnsmos_ovrl)"
