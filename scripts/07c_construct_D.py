#!/usr/bin/env python
"""07c: D.jsonl —— 高表现 → 高表现 同情绪换文本

利用 MOSS-TTS（vcdata 阶段）已天然保留情绪的样本筛出来：
- reference = vcdata ref_audio (MOSS-TTS 合成，text = ref_text)
- target    = original_audio   (真实音频，text = original_text，跨文本)
- 两边同 speaker（vcdata 设计保证）
- 两边都"高表现"且 emotion 一致

实质：H1（零变化对照）的"双高表现"子集 —— 不只要表达不变，还要求两边都不中性。
"""
from __future__ import annotations
import argparse
import random
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
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)

    vc_path = intermediate_path(cfg, args.split, "vcdata_base.jsonl")
    if not vc_path.exists():
        sys.exit(f"[07c] 先跑 01。缺 {vc_path}")

    emo = EmotionTable()
    emo.load_csv(emotion_path(cfg, args.split, "per_file_dual.csv"))
    emo.load_per_pair_for_src(emotion_path(cfg, args.split, "per_pair.csv"))
    emo.load_link_mapping(emotion_path(cfg, args.split, "_links_original/_mapping.csv"))

    d = cfg["d"]
    forbidden_emo_labels = set(cfg.get("emotion_filter", {}).get("forbidden_top1_labels", []))
    rng = random.Random(args.seed)

    rows = []
    cnt_no_emo = cnt_top1_mismatch = cnt_cos_low = cnt_ref_too_neu = cnt_tgt_too_neu = 0
    cnt_ref_top1_neu = cnt_tgt_top1_neu = cnt_forbidden_emo = 0
    for v in iter_jsonl(vc_path):
        e_orig = emo.emotion_summary(v["original_audio"])
        e_ref = emo.emotion_summary(v["ref_audio"])
        if e_orig["top1_label"] is None or e_ref["top1_label"] is None:
            cnt_no_emo += 1; continue
        if forbidden_emo_labels and (
            e_orig["top1_label"] in forbidden_emo_labels
            or e_ref["top1_label"] in forbidden_emo_labels
        ):
            cnt_forbidden_emo += 1; continue
        if d["same_top1_label"] and e_orig["top1_label"] != e_ref["top1_label"]:
            cnt_top1_mismatch += 1; continue
        if d.get("ref_top1_not_neutral", False) and e_ref["top1_label"] == "neutral":
            cnt_ref_top1_neu += 1; continue
        if d.get("tgt_top1_not_neutral", False) and e_orig["top1_label"] == "neutral":
            cnt_tgt_top1_neu += 1; continue
        cos = EmotionTable.cosine9(emo.get(v["original_audio"]), emo.get(v["ref_audio"]))
        if cos is None or cos < d["cosine_min"]:
            cnt_cos_low += 1; continue
        if e_ref["P_neutral"] is not None and e_ref["P_neutral"] > d["ref_neutral_max"]:
            cnt_ref_too_neu += 1; continue
        if e_orig["P_neutral"] is not None and e_orig["P_neutral"] > d["tgt_neutral_max"]:
            cnt_tgt_too_neu += 1; continue

        rows.append({
            "pair_id": make_pair_id(args.split, "D", len(rows)),
            "pair_type": "D",
            "reference_audio": v["ref_audio"],
            "reference_text": v["ref_text"],
            "target_audio": v["original_audio"],
            "target_text": v["original_text"],
            "instruction": rng.choice(d["instruction_pool"]),
            "source_edit": None,
            "speaker_similarity": v.get("speaker_similarity"),
            "ref_emotion": e_ref,
            "tgt_emotion": e_orig,
            "meta": {
                "split": args.split,
                "source_row_index": v["original_idx"],
                "emotion_cosine": cos,
                "cross_text": True,
            },
        })

    out = pair_path(cfg, args.split, "D.jsonl")
    write_jsonl(out, rows)
    print(f"[07c] D={len(rows)}  (top1≠ {cnt_top1_mismatch}, ref-top1=neu {cnt_ref_top1_neu}, tgt-top1=neu {cnt_tgt_top1_neu}, "
          f"cos< {cnt_cos_low}, ref太中性 {cnt_ref_too_neu}, tgt太中性 {cnt_tgt_too_neu}, 缺emotion {cnt_no_emo}, "
          f"forbidden emo {cnt_forbidden_emo})  → {out}")


if __name__ == "__main__":
    main()
