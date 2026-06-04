#!/usr/bin/env python
"""05: A.jsonl —— 只换文本、表达基本不变

A = vcdata 全量（仅 sim_min 与 flag 可选门槛）。
口径上 A 实质就是 vcdata_construction 的设计产出：真实 reference → 合成 target，
所以不再二次做 emotion 过滤；如需更严的"低变化"子集，看 H1。

instruction 是后标注（不参与生成）。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (
    load_config, iter_jsonl, write_jsonl,
    intermediate_path, emotion_path, pair_path, make_pair_id,
)
from _emotion_lookup import EmotionTable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    vc_path = intermediate_path(cfg, args.split, "vcdata_base.jsonl")
    if not vc_path.exists():
        sys.exit(f"[05] 先跑 01。缺 {vc_path}")

    emo = EmotionTable()
    emo.load_csv(emotion_path(cfg, args.split, "per_file_dual.csv"))
    emo.load_per_pair_for_src(emotion_path(cfg, args.split, "per_pair.csv"))
    emo.load_link_mapping(emotion_path(cfg, args.split, "_links_original/_mapping.csv"))

    a_cfg = cfg["a"]
    out_path = pair_path(cfg, args.split, "A.jsonl")

    rows = []
    for v in iter_jsonl(vc_path):
        sim = v.get("speaker_similarity") or 0.0
        if sim < a_cfg["sim_min"]:
            continue
        if a_cfg.get("require_flag_ok") and v.get("flag") != "OK":
            continue
        rows.append({
            "pair_id": make_pair_id(args.split, "A", len(rows)),
            "pair_type": "A",
            "reference_audio": v["original_audio"],
            "reference_text": v["original_text"],
            "target_audio": v["ref_audio"],
            "target_text": v["ref_text"],
            "instruction": a_cfg["instruction"],
            "source_edit": None,
            "speaker_similarity": sim,
            "ref_emotion": emo.emotion_summary(v["original_audio"]),
            "tgt_emotion": emo.emotion_summary(v["ref_audio"]),
            "meta": {
                "split": args.split,
                "source_row_index": v["original_idx"],
                "flag": v.get("flag"),
                "caption_summary": v.get("caption_summary"),
            },
        })

    write_jsonl(out_path, rows)
    print(f"[05] A={len(rows)}  → {out_path}")


if __name__ == "__main__":
    main()
