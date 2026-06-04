#!/usr/bin/env python
"""10: H3.jsonl —— 跨 speaker 负样本

对 vcdata_base 每一行，从其它 row 随机 sample 一条音频作为 target，
显式标记 `is_negative: true`，instruction 写为"不应学习此 pair"。

可选强约束：caption_result.gender 不同（更稳的跨 speaker）。
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (
    load_config, iter_jsonl, write_jsonl,
    intermediate_path, pair_path, make_pair_id,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    h3 = cfg["h3"]

    vc_path = intermediate_path(cfg, args.split, "vcdata_base.jsonl")
    if not vc_path.exists():
        sys.exit(f"[10] 先跑 01。缺 {vc_path}")

    rows_all = list(iter_jsonl(vc_path))
    # 复用 A 类同款基础过滤
    sim_min = cfg["a"]["sim_min"]
    pool = [r for r in rows_all if (r.get("speaker_similarity") or 0.0) >= sim_min]

    rng = random.Random(h3["random_seed"])
    n_per = h3["negatives_per_anchor"]
    gender_mismatch = h3.get("require_gender_mismatch", False)

    out_rows, idx = [], 0
    for anchor in pool:
        cand_pool = pool
        for _ in range(n_per):
            for _try in range(20):
                cand = rng.choice(cand_pool)
                if cand["original_idx"] == anchor["original_idx"]:
                    continue
                if gender_mismatch:
                    ga = anchor.get("caption_gender")
                    gb = cand.get("caption_gender")
                    if ga and gb and ga == gb:
                        continue
                break
            else:
                continue
            out_rows.append({
                "pair_id": make_pair_id(args.split, "H3", idx),
                "pair_type": "H3",
                "reference_audio": anchor["ref_audio"],
                "reference_text": anchor["ref_text"],
                "target_audio": cand["ref_audio"],
                "target_text": cand["ref_text"],
                "instruction": h3["instruction"],
                "source_edit": None,
                "speaker_similarity": None,
                "ref_emotion": None,
                "tgt_emotion": None,
                "is_negative": True,
                "meta": {
                    "split": args.split,
                    "anchor_source_row_index": anchor["original_idx"],
                    "neg_source_row_index": cand["original_idx"],
                    "anchor_gender": anchor.get("caption_gender"),
                    "neg_gender": cand.get("caption_gender"),
                },
            })
            idx += 1

    out = pair_path(cfg, args.split, "H3.jsonl")
    write_jsonl(out, out_rows)
    print(f"[10] H3={len(out_rows)}  → {out}")


if __name__ == "__main__":
    main()
