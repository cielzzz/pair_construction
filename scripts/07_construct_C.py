#!/usr/bin/env python
"""07: C.jsonl —— 高表现 ref_audio → 中性 edited_audio（同文本）

reference = ref_audio (P_neutral 足够低 == "高表现")
target    = edited_audio (P_neutral 足够高 == 中性化)
两边 text == ref_text
instruction = "平静一点" / "不要这么夸张" / "去掉情绪起伏"
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

    jo_path = intermediate_path(cfg, args.split, "joined_editx.jsonl")
    if not jo_path.exists():
        sys.exit(f"[07] 先跑 03。缺 {jo_path}")

    emo = EmotionTable()
    emo.load_csv(emotion_path(cfg, args.split, "per_file_dual.csv"))
    emo.load_per_pair_for_src(emotion_path(cfg, args.split, "per_pair.csv"))

    bc = cfg["bc"]
    c_cfg = cfg["c"]
    whitelist = set(bc["edit_whitelist"])
    p_edited_min = bc["edited_neutral_min"]
    p_ref_max = bc.get("ref_neutral_max", 1.0)
    sv_must = bc["edited_sv_must_be_neutral"]
    rng = random.Random(args.seed)

    rows = []
    dropped_ref_neutral = 0
    for r in iter_jsonl(jo_path):
        if r["edit_tag"] not in whitelist:
            continue
        e_edited = emo.emotion_summary(r["edited_audio"])
        e_ref = emo.emotion_summary(r["ref_audio"])
        if e_edited["P_neutral"] is None or e_edited["P_neutral"] < p_edited_min:
            continue
        if bc.get("edited_top1_must_be_neutral", False) and e_edited["top1_label"] != "neutral":
            continue
        if sv_must and e_edited["sv_label"] != "neutral":
            continue
        # reference (= ref_audio) 必须"高表现"：top1 ≠ neutral 硬条件
        if bc.get("ref_top1_not_neutral", False) and e_ref["top1_label"] == "neutral":
            dropped_ref_neutral += 1
            continue
        # 软兜底
        if e_ref["P_neutral"] is not None and e_ref["P_neutral"] > p_ref_max:
            dropped_ref_neutral += 1
            continue
        rows.append({
            "pair_id": make_pair_id(args.split, "C", len(rows)),
            "pair_type": "C",
            "reference_audio": r["ref_audio"],
            "reference_text": r["ref_text"],
            "target_audio": r["edited_audio"],
            "target_text": r["ref_text"],
            "instruction": rng.choice(c_cfg["instruction_pool"]),
            "source_edit": r["edit_tag"],
            "speaker_similarity": r.get("speaker_similarity"),
            "ref_emotion": e_ref,
            "tgt_emotion": e_edited,
            "meta": {
                "split": args.split,
                "source_row_index": r["original_idx"],
                "editx_instruction": r["instruction"],
            },
        })

    out = pair_path(cfg, args.split, "C.jsonl")
    write_jsonl(out, rows)
    print(f"[07] C={len(rows)}  (ref 太中性丢弃 {dropped_ref_neutral} 条)  → {out}")


if __name__ == "__main__":
    main()
