"""扫 outputs/<split>/pairs/*.jsonl，构建聚合 index.parquet。

用法：
    python app/index_builder.py [--outputs-dir <path>] [--duration-cache <path>]

每行索引粒度：(split, pair_type, is_filtered)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader import (
    DATA_DIR, INDEX_PARQUET, OUTPUTS_DIR,
    infer_lang_from_split, infer_source_from_path,
    display_split, load_duration_cache,
)


# 从文件名解析 pair_type + is_filtered
def parse_jsonl_name(name: str):
    stem = name[:-len(".jsonl")] if name.endswith(".jsonl") else name
    if stem.endswith("_qc"):
        return stem[:-len("_qc")], True
    if stem.endswith("_filtered"):
        return stem[:-len("_filtered")], True
    if stem.endswith("_bakfilt"):
        return stem[:-len("_bakfilt")], True
    return stem, False


def aggregate_one(jsonl_path: Path, dur_map: dict, row_filter=None) -> dict:
    """聚合单个 pair jsonl 的统计。
    row_filter: 可选 callable(row_dict) -> bool。返回 False 的行不计入 n_pairs/统计。
                用于 quality_gate 的 *_qc.jsonl 按 qc_pass==1 真实计数。
    """
    ref_emos, tgt_emos, tags = Counter(), Counter(), Counter()
    sims, timbre_sims, ref_baks, tgt_baks = [], [], [], []
    ref_hours = tgt_hours = 0.0
    n = 0
    n_raw = 0
    sample_row = None
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n_raw += 1
            if sample_row is None:
                sample_row = r
            if row_filter is not None and not row_filter(r):
                continue
            n += 1
            re_ = r.get("ref_emotion") or {}
            te = r.get("tgt_emotion") or {}
            ref_emos[re_.get("top1_label") or "—"] += 1
            tgt_emos[te.get("top1_label") or "—"] += 1
            tag = r.get("source_edit_tag") or r.get("source_edit") or "—"
            tags[tag] += 1
            sim = r.get("ref_vs_tgt_speaker_sim_wavlm") or r.get("ref_speaker_sim_wavlm")
            if sim is not None:
                sims.append(sim)
            # I 类专用：target ↔ timbre_ref 的音色相似度
            ts = r.get("timbre_ref_vs_tgt_speaker_sim_wavlm")
            if ts is not None:
                timbre_sims.append(ts)
            rb = r.get("ref_dnsmos_bak")
            if rb is not None:
                ref_baks.append(rb)
            tb = r.get("tgt_dnsmos_bak")
            if tb is not None:
                tgt_baks.append(tb)
            if dur_map:
                ref_dur = dur_map.get(r.get("reference_audio"))
                tgt_dur = dur_map.get(r.get("target_audio"))
                if ref_dur is not None:
                    ref_hours += ref_dur
                if tgt_dur is not None:
                    tgt_hours += tgt_dur
    ref_hours /= 3600.0
    tgt_hours /= 3600.0
    sims_s = pd.Series(sims) if sims else pd.Series([], dtype=float)
    ts_s = pd.Series(timbre_sims) if timbre_sims else pd.Series([], dtype=float)
    return {
        "n_pairs": n,
        "n_pairs_raw": n_raw,
        "ref_hours": ref_hours,
        "tgt_hours": tgt_hours,
        "ref_emo_dist": json.dumps(dict(ref_emos), ensure_ascii=False),
        "tgt_emo_dist": json.dumps(dict(tgt_emos), ensure_ascii=False),
        "source_edit_tag_dist": json.dumps(dict(tags), ensure_ascii=False),
        "sim_wavlm_mean": float(sims_s.mean()) if len(sims_s) else None,
        "sim_wavlm_p25": float(sims_s.quantile(0.25)) if len(sims_s) else None,
        "sim_wavlm_p50": float(sims_s.quantile(0.50)) if len(sims_s) else None,
        "sim_wavlm_p75": float(sims_s.quantile(0.75)) if len(sims_s) else None,
        # I 类专用：timbre_ref ↔ target 音色相似度（其它类此处为 None）
        "timbre_sim_wavlm_mean": float(ts_s.mean()) if len(ts_s) else None,
        "timbre_sim_wavlm_p25": float(ts_s.quantile(0.25)) if len(ts_s) else None,
        "timbre_sim_wavlm_p50": float(ts_s.quantile(0.50)) if len(ts_s) else None,
        "timbre_sim_wavlm_p75": float(ts_s.quantile(0.75)) if len(ts_s) else None,
        "ref_dnsmos_mean": float(pd.Series(ref_baks).mean()) if ref_baks else None,
        "tgt_dnsmos_mean": float(pd.Series(tgt_baks).mean()) if tgt_baks else None,
        "_sample_row": sample_row,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default=str(OUTPUTS_DIR),
                    help=f"扫描根目录，默认 {OUTPUTS_DIR}")
    ap.add_argument("--out", default=str(INDEX_PARQUET),
                    help=f"输出 parquet 路径，默认 {INDEX_PARQUET}")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dur_map = load_duration_cache()
    if dur_map:
        print(f"[index] 用 duration_cache: {len(dur_map)} 条")
    else:
        print(f"[index] 无 duration_cache（ref_hours / tgt_hours 将全 0）")

    def iter_split_dirs(root: Path):
        """支持两种布局：
           A) <root>/<split>/pairs/...                — 单语言根
           B) <root>/<group>/<split>/pairs/...        — 多语言根 (group=zh/en/...)
           只下钻一层，避免无限递归。
        """
        for top in sorted(root.iterdir()):
            if not top.is_dir():
                continue
            if (top / "pairs").is_dir():
                yield top                               # 布局 A
                continue
            for sub in sorted(top.iterdir()):           # 布局 B
                if sub.is_dir() and (sub / "pairs").is_dir():
                    yield sub

    def iter_pair_jsonls(split_dir: Path):
        """对一个 split_dir 产出 (jsonl_path, is_filtered_override)。
        infra run03+ 之后约定：
            <split>/pairs/X.jsonl              → orig    (is_filtered=False)
            <split>/quality_gate/X_qc.jsonl    → filtered(is_filtered=True)
        pairs/scored/ 只是 dnsmos 注解版（无截断），不读。
        任一档缺失另一档照常输出；都缺则什么都不产。
        """
        pairs_dir = split_dir / "pairs"
        qg_dir = split_dir / "quality_gate"
        # orig：pairs/ 下直接的 jsonl（非递归，自动跳过 scored/ 子目录）
        if pairs_dir.is_dir():
            for j in sorted(pairs_dir.glob("*.jsonl")):
                yield j, False
        # filtered：quality_gate/*.jsonl（glob 自动忽略 _progress.json / csv/ / *.log）
        if qg_dir.is_dir():
            for j in sorted(qg_dir.glob("*.jsonl")):
                yield j, True

    # quality_gate 真实过滤判据：qc_pass==1
    def _qc_pass(r):
        return r.get("qc_pass") == 1

    rows = []
    for split_dir in iter_split_dirs(outputs_dir):
        split = split_dir.name
        for jsonl, is_filtered_override in iter_pair_jsonls(split_dir):
            pair_type, name_filtered = parse_jsonl_name(jsonl.name)
            is_filtered = is_filtered_override or name_filtered
            # filtered 档（来自 quality_gate/）按 qc_pass 真实计数
            row_filter = _qc_pass if is_filtered_override else None
            agg = aggregate_one(jsonl, dur_map, row_filter=row_filter)
            sample = agg.pop("_sample_row")
            lang = infer_lang_from_split(split, sample)
            ref_path = (sample or {}).get("reference_audio", "") or ""
            tgt_path = (sample or {}).get("target_audio", "") or ""
            source = infer_source_from_path(ref_path) or infer_source_from_path(tgt_path)
            rows.append({
                "split": split,
                "display_split": display_split(split),
                "source": source,
                "language": lang,
                "pair_type": pair_type,
                "is_filtered": is_filtered,
                "jsonl_path": str(jsonl),
                **agg,
            })
            tag = "scored" if is_filtered else "orig  "
            print(f"  [{tag}] {split}/{jsonl.name}: n={agg['n_pairs']:>5}  "
                  f"ref_h={agg['ref_hours']:.2f}  tgt_h={agg['tgt_hours']:.2f}")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"\n[index] 写 {len(df)} 行 → {out_path}")
    print(f"  splits = {df['split'].nunique()}")
    print(f"  sources = {df['source'].unique().tolist()}")
    print(f"  languages = {df['language'].unique().tolist()}")
    print(f"  total pairs (含 filtered 重复): {df['n_pairs'].sum():,}")


if __name__ == "__main__":
    main()
