#!/usr/bin/env python
"""07b: C_mixed.jsonl —— 跨文本版高表现 → 中性

与 C_clean 的区别：
- reference 不再是 vcdata 合成的 ref_audio，而是真实 original_audio
- 两边文本不同（original_text vs ref_text）
- 同 speaker（vcdata 保证 ref_audio/edited 与 original 音色一致）

reference = original_audio   (text = original_text，真实高表现)
target    = edited_audio      (text = ref_text，style_radio 中性化)
instruction 必须显式提示"按新文本说"，因为文本变了
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
        sys.exit(f"[07b] 先跑 03。缺 {jo_path}")

    emo = EmotionTable()
    emo.load_csv(emotion_path(cfg, args.split, "per_file_dual.csv"))
    emo.load_per_pair_for_src(emotion_path(cfg, args.split, "per_pair.csv"))
    emo.load_all_link_mappings(emotion_path(cfg, args.split, ""))

    bc = cfg["bc"]
    cm = cfg["c_mixed"]
    whitelist = set(bc["edit_whitelist"])
    p_edited_min = bc["edited_neutral_min"]
    p_ref_max = bc.get("ref_neutral_max", 1.0)
    sv_must = bc["edited_sv_must_be_neutral"]
    forbidden_emo_labels = set(cfg.get("emotion_filter", {}).get("forbidden_top1_labels", []))
    rng = random.Random(args.seed)

    rows = []
    dropped_orig_too_neutral = 0
    dropped_orig_top1_neu = 0
    dropped_forbidden_emo = 0
    skip_no_emotion = 0
    require_top1_nonneu = cm.get("original_top1_not_neutral", False)
    for r in iter_jsonl(jo_path):
        if r["edit_tag"] not in whitelist:
            continue
        e_edited = emo.emotion_summary(r["edited_audio"])
        e_orig = emo.emotion_summary(r["original_audio"])
        if forbidden_emo_labels and (
            e_edited["top1_label"] in forbidden_emo_labels
            or e_orig["top1_label"] in forbidden_emo_labels
        ):
            dropped_forbidden_emo += 1
            continue
        if e_edited["P_neutral"] is None or e_edited["P_neutral"] < p_edited_min:
            continue
        if bc.get("edited_top1_must_be_neutral", False) and e_edited["top1_label"] != "neutral":
            continue
        if sv_must and e_edited["sv_label"] != "neutral":
            continue
        # reference (= original) 必须够高表现：P(neutral) 不能太高
        if e_orig["P_neutral"] is None:
            skip_no_emotion += 1
            continue
        if require_top1_nonneu and e_orig["top1_label"] == "neutral":
            dropped_orig_top1_neu += 1
            continue
        if e_orig["P_neutral"] > p_ref_max:
            dropped_orig_too_neutral += 1
            continue
        rows.append({
            "pair_id": make_pair_id(args.split, "C-mixed", len(rows)),
            "pair_type": "C-mixed",
            "reference_audio": r["original_audio"],
            "reference_text": r["original_text"],
            "target_audio": r["edited_audio"],
            "target_text": r["ref_text"],
            "instruction": rng.choice(cm["instruction_pool"]),
            "source_edit": r["edit_tag"],
            "speaker_similarity": r.get("speaker_similarity"),
            "ref_emotion": e_orig,
            "tgt_emotion": e_edited,
            "meta": {
                "split": args.split,
                "source_row_index": r["original_idx"],
                "editx_instruction": r["instruction"],
                "cross_text": True,
            },
        })

    out = pair_path(cfg, args.split, "C_mixed.jsonl")
    write_jsonl(out, rows)
    print(f"[07b] C_mixed={len(rows)}  (orig top1=neu {dropped_orig_top1_neu}, "
          f"orig 太中性 {dropped_orig_too_neutral}, 缺 emotion {skip_no_emotion}, "
          f"forbidden emo {dropped_forbidden_emo})  → {out}")


if __name__ == "__main__":
    main()
