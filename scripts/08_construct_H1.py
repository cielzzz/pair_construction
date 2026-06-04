#!/usr/bin/env python
"""08: H1.jsonl —— 零变化对照

A 全量中筛"表达几乎不变"：
- emotion top1 一致
- emotion 9 维 cosine >= configs.h1.cosine_min（默认 0.97）

reference = original_audio
target    = ref_audio
instruction = "保持原样 / 不要改变表达方式"
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
    ap.add_argument("--cosine-min", type=float, default=None,
                    help="覆盖 config.h1.cosine_min")
    ap.add_argument("--suffix", default="",
                    help="输出文件名后缀，如 _cos90 → H1_cos90.jsonl")
    args = ap.parse_args()
    cfg = load_config(args.config)

    vc_path = intermediate_path(cfg, args.split, "vcdata_base.jsonl")
    if not vc_path.exists():
        sys.exit(f"[08] 先跑 01。缺 {vc_path}")

    emo = EmotionTable()
    emo.load_csv(emotion_path(cfg, args.split, "per_file_dual.csv"))
    emo.load_per_pair_for_src(emotion_path(cfg, args.split, "per_pair.csv"))
    emo.load_link_mapping(emotion_path(cfg, args.split, "_links_original/_mapping.csv"))

    h1_cfg = dict(cfg["h1"])
    if args.cosine_min is not None:
        h1_cfg["cosine_min"] = args.cosine_min

    rows = []
    skip_no_emotion = 0
    for v in iter_jsonl(vc_path):
        sim = v.get("speaker_similarity") or 0.0
        if sim < h1_cfg["sim_min"]:
            continue
        e_orig = emo.emotion_summary(v["original_audio"])
        e_ref = emo.emotion_summary(v["ref_audio"])
        if e_orig["top1_label"] is None or e_ref["top1_label"] is None:
            skip_no_emotion += 1
            continue
        if h1_cfg["same_top1_label"] and e_orig["top1_label"] != e_ref["top1_label"]:
            continue
        cos = EmotionTable.cosine9(emo.get(v["original_audio"]), emo.get(v["ref_audio"]))
        if cos is None or cos < h1_cfg["cosine_min"]:
            continue

        rows.append({
            "pair_id": make_pair_id(args.split, "H1", len(rows)),
            "pair_type": "H1",
            "reference_audio": v["original_audio"],
            "reference_text": v["original_text"],
            "target_audio": v["ref_audio"],
            "target_text": v["ref_text"],
            "instruction": h1_cfg["instruction"],
            "source_edit": None,
            "speaker_similarity": sim,
            "ref_emotion": e_orig,
            "tgt_emotion": e_ref,
            "meta": {
                "split": args.split,
                "source_row_index": v["original_idx"],
                "emotion_cosine": cos,
            },
        })

    out_name = f"H1{args.suffix}.jsonl"
    out = pair_path(cfg, args.split, out_name)
    write_jsonl(out, rows)
    msg = f"[08] H1 (cosine_min={h1_cfg['cosine_min']:.2f})={len(rows)}  → {out_name}"
    if skip_no_emotion:
        msg += f"  (缺 emotion 跳过 {skip_no_emotion} 条)"
    print(msg)


if __name__ == "__main__":
    main()
