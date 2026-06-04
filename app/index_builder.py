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
    load_duration_cache,
)


# 从文件名解析 pair_type + is_filtered
def parse_jsonl_name(name: str):
    stem = name[:-len(".jsonl")] if name.endswith(".jsonl") else name
    if stem.endswith("_filtered"):
        return stem[:-len("_filtered")], True
    if stem.endswith("_bakfilt"):
        return stem[:-len("_bakfilt")], True
    return stem, False


def aggregate_one(jsonl_path: Path, dur_map: dict) -> dict:
    """聚合单个 pair jsonl 的统计。"""
    ref_emos, tgt_emos, tags = Counter(), Counter(), Counter()
    sims, ref_baks, tgt_baks = [], [], []
    ref_hours = tgt_hours = 0.0
    n = 0
    sample_row = None
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if sample_row is None:
                sample_row = r
            n += 1
            re_ = r.get("ref_emotion") or {}
            te = r.get("tgt_emotion") or {}
            ref_emos[re_.get("top1_label") or "—"] += 1
            tgt_emos[te.get("top1_label") or "—"] += 1
            tag = r.get("source_edit_tag") or r.get("source_edit") or "—"
            tags[tag] += 1
            sim = r.get("ref_vs_tgt_speaker_sim_wavlm")
            if sim is not None:
                sims.append(sim)
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
    return {
        "n_pairs": n,
        "ref_hours": ref_hours,
        "tgt_hours": tgt_hours,
        "ref_emo_dist": json.dumps(dict(ref_emos), ensure_ascii=False),
        "tgt_emo_dist": json.dumps(dict(tgt_emos), ensure_ascii=False),
        "source_edit_tag_dist": json.dumps(dict(tags), ensure_ascii=False),
        "sim_wavlm_mean": float(sims_s.mean()) if len(sims_s) else None,
        "sim_wavlm_p25": float(sims_s.quantile(0.25)) if len(sims_s) else None,
        "sim_wavlm_p50": float(sims_s.quantile(0.50)) if len(sims_s) else None,
        "sim_wavlm_p75": float(sims_s.quantile(0.75)) if len(sims_s) else None,
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

    rows = []
    for split_dir in sorted(outputs_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        pairs_dir = split_dir / "pairs"
        if not pairs_dir.is_dir():
            continue
        split = split_dir.name
        for jsonl in sorted(pairs_dir.glob("*.jsonl")):
            pair_type, is_filtered = parse_jsonl_name(jsonl.name)
            agg = aggregate_one(jsonl, dur_map)
            sample = agg.pop("_sample_row")
            lang = infer_lang_from_split(split, sample)
            ref_path = (sample or {}).get("reference_audio", "") or ""
            tgt_path = (sample or {}).get("target_audio", "") or ""
            source = infer_source_from_path(ref_path) or infer_source_from_path(tgt_path)
            rows.append({
                "split": split,
                "source": source,
                "language": lang,
                "pair_type": pair_type,
                "is_filtered": is_filtered,
                "jsonl_path": str(jsonl),
                **agg,
            })
            print(f"  {split}/{jsonl.name}: n={agg['n_pairs']:>5}  "
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
