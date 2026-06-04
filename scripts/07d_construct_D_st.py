#!/usr/bin/env python
"""07d: D_st.jsonl —— 高表现 → 高表现 同情绪 同文本

利用 stepfun-editx 偶尔保留情绪的样本：
- reference = ref_audio    (MOSS-TTS 合成, text = ref_text)
- target    = edited_audio (stepfun-editx 改造, text = ref_text)
- 两者 text 都是 ref_text → 同文本
- 同 top1（非 neutral）+ 9 维 emotion cosine >= 阈值 → 同情绪 + 双高表现

数据量预期很少：editx 大多会改变情绪，这里筛的是"漏网"样本。
与 D 类的区别：D 跨文本（ref vs original_audio），D_st 同文本（ref vs edited_audio）。
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
        sys.exit(f"[07d] 先跑 03。缺 {jo_path}")

    emo = EmotionTable()
    emo.load_csv(emotion_path(cfg, args.split, "per_file_dual.csv"))
    emo.load_per_pair_for_src(emotion_path(cfg, args.split, "per_pair.csv"))
    emo.load_link_mapping(emotion_path(cfg, args.split, "_links_original/_mapping.csv"))

    dst = cfg.get("d_st", {
        "cosine_min": 0.95,
        "same_top1_label": True,
        "ref_top1_not_neutral": True,
        "tgt_top1_not_neutral": True,
        "ref_neutral_max": 0.95,
        "tgt_neutral_max": 0.95,
        "instruction_pool": [
            "保持当前情绪不变，换一种说话风格",
            "情绪和表达强度保持一致，换个语气说",
        ],
    })
    forbidden_emo_labels = set(cfg.get("emotion_filter", {}).get("forbidden_top1_labels", []))
    rng = random.Random(args.seed)

    # 与 B/C/H2 对齐：D_st 也只看 bc.edit_whitelist 里的 edit_tag
    whitelist = set(cfg.get("bc", {}).get("edit_whitelist", []))

    rows = []
    cnt_no_emo = cnt_top1_mismatch = cnt_cos_low = 0
    cnt_ref_top1_neu = cnt_tgt_top1_neu = 0
    cnt_ref_too_neu = cnt_tgt_too_neu = 0
    cnt_tag_skip = cnt_forbidden_emo = 0

    for r in iter_jsonl(jo_path):
        if whitelist and r.get("edit_tag") not in whitelist:
            cnt_tag_skip += 1; continue
        e_ref    = emo.emotion_summary(r["ref_audio"])
        e_edited = emo.emotion_summary(r["edited_audio"])
        if e_ref["top1_label"] is None or e_edited["top1_label"] is None:
            cnt_no_emo += 1; continue
        if forbidden_emo_labels and (
            e_ref["top1_label"] in forbidden_emo_labels
            or e_edited["top1_label"] in forbidden_emo_labels
        ):
            cnt_forbidden_emo += 1; continue
        if dst["same_top1_label"] and e_ref["top1_label"] != e_edited["top1_label"]:
            cnt_top1_mismatch += 1; continue
        if dst.get("ref_top1_not_neutral", False) and e_ref["top1_label"] == "neutral":
            cnt_ref_top1_neu += 1; continue
        if dst.get("tgt_top1_not_neutral", False) and e_edited["top1_label"] == "neutral":
            cnt_tgt_top1_neu += 1; continue
        cos = EmotionTable.cosine9(emo.get(r["ref_audio"]), emo.get(r["edited_audio"]))
        if cos is None or cos < dst["cosine_min"]:
            cnt_cos_low += 1; continue
        if e_ref["P_neutral"] is not None and e_ref["P_neutral"] > dst["ref_neutral_max"]:
            cnt_ref_too_neu += 1; continue
        if e_edited["P_neutral"] is not None and e_edited["P_neutral"] > dst["tgt_neutral_max"]:
            cnt_tgt_too_neu += 1; continue

        rows.append({
            "pair_id": make_pair_id(args.split, "D-st", len(rows)),
            "pair_type": "D-st",
            "reference_audio": r["ref_audio"],
            "reference_text": r["ref_text"],
            "target_audio": r["edited_audio"],
            "target_text": r["ref_text"],
            "instruction": rng.choice(dst["instruction_pool"]),
            "source_edit": r["edit_tag"],
            "speaker_similarity": r.get("speaker_similarity"),
            "ref_emotion": e_ref,
            "tgt_emotion": e_edited,
            "meta": {
                "split": args.split,
                "source_row_index": r["original_idx"],
                "emotion_cosine": cos,
                "cross_text": False,
                "editx_instruction": r.get("instruction"),
            },
        })

    out = pair_path(cfg, args.split, "D_st.jsonl")
    write_jsonl(out, rows)
    print(f"[07d] D_st={len(rows)}  (top1≠ {cnt_top1_mismatch}, ref-top1=neu {cnt_ref_top1_neu}, "
          f"tgt-top1=neu {cnt_tgt_top1_neu}, cos< {cnt_cos_low}, "
          f"ref太中性 {cnt_ref_too_neu}, tgt太中性 {cnt_tgt_too_neu}, "
          f"缺emotion {cnt_no_emo}, tag-非白名单 {cnt_tag_skip}, "
          f"forbidden emo {cnt_forbidden_emo})  → {out}")


if __name__ == "__main__":
    main()
