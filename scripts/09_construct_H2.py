#!/usr/bin/env python
"""09: H2.jsonl —— 中性 -> 更中性

当前默认语义：
- reference = ref_audio
- target    = edited_audio
- 两边文本一致，target 的中性度应高于 reference

保留 legacy edited-self / edited-neighbor 两种模式以兼容旧实验，
但默认配置应使用 ``ref_to_edited``。
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
    ap.add_argument(
        "--mode",
        default=None,
        choices=["ref_to_edited", "self", "neighbor"],
        help="覆盖 config.h2.mode；默认推荐 ref_to_edited。",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)

    jo_path = intermediate_path(cfg, args.split, "joined_editx.jsonl")
    if not jo_path.exists():
        sys.exit(f"[09] 先跑 03。缺 {jo_path}")

    emo = EmotionTable()
    emo.load_csv(emotion_path(cfg, args.split, "per_file_dual.csv"))
    emo.load_all_link_mappings(emotion_path(cfg, args.split, ""))

    h2 = cfg["h2"]
    target_edit_tag = h2["source_edit_tag"]
    edited_p_min = h2["p_neutral_min"]
    ref_p_min = h2.get("ref_neutral_min", 0.5)
    sv_req = h2["require_sv_neutral"]
    mode = args.mode or h2.get("mode", "ref_to_edited")
    ref_top1_must_be_neutral = h2.get("ref_top1_must_be_neutral", False)
    edited_top1_must_be_neutral = h2.get("edited_top1_must_be_neutral", False)
    require_target_more_neutral = h2.get("require_target_more_neutral", mode == "ref_to_edited")
    target_more_neutral_margin = float(h2.get("target_more_neutral_margin", 0.0))
    rng = random.Random(args.seed)

    pool = []
    for r in iter_jsonl(jo_path):
        if r["edit_tag"] != target_edit_tag:
            continue
        e_ref = emo.emotion_summary(r["ref_audio"])
        e_edited = emo.emotion_summary(r["edited_audio"])

        if e_edited["P_neutral"] is None or e_edited["P_neutral"] < edited_p_min:
            continue
        if edited_top1_must_be_neutral and e_edited["top1_label"] != "neutral":
            continue
        if sv_req and e_edited["sv_label"] != "neutral":
            continue
        if mode == "ref_to_edited":
            if e_ref["P_neutral"] is None or e_ref["P_neutral"] < ref_p_min:
                continue
            if ref_top1_must_be_neutral and e_ref["top1_label"] != "neutral":
                continue
            if require_target_more_neutral and e_edited["P_neutral"] <= e_ref["P_neutral"] + target_more_neutral_margin:
                continue
        pool.append((r, e_ref, e_edited))

    print(f"[09] H2 候选池：{len(pool)} (edit_tag={target_edit_tag}, mode={mode})")
    if not pool:
        write_jsonl(pair_path(cfg, args.split, "H2.jsonl"), [])
        return

    rows = []
    for r, e_ref, e_edited in pool:
        if mode == "ref_to_edited":
            ref_audio = r["ref_audio"]
            ref_text = r["ref_text"]
            tgt_audio = r["edited_audio"]
            tgt_text = r["ref_text"]
            e_tgt = e_edited
            legacy_similarity = None
        elif mode == "self":
            ref_audio = r["edited_audio"]
            ref_text = r["ref_text"]
            tgt_audio = r["edited_audio"]
            tgt_text = r["ref_text"]
            e_ref = e_edited
            e_tgt = e_edited
            legacy_similarity = r.get("speaker_similarity")
        else:
            cand = rng.choice(pool)
            while cand[0]["original_idx"] == r["original_idx"] and len(pool) > 1:
                cand = rng.choice(pool)
            ref_audio = r["edited_audio"]
            ref_text = r["ref_text"]
            tgt_audio = cand[0]["edited_audio"]
            tgt_text = cand[0]["ref_text"]
            e_ref = e_edited
            e_tgt = cand[2]
            legacy_similarity = None
        rows.append({
            "pair_id": make_pair_id(args.split, "H2", len(rows)),
            "pair_type": "H2",
            "reference_audio": ref_audio,
            "reference_text": ref_text,
            "target_audio": tgt_audio,
            "target_text": tgt_text,
            "instruction": rng.choice(h2["instruction_pool"]),
            "source_edit": target_edit_tag,
            "speaker_similarity": legacy_similarity,
            "ref_emotion": e_ref,
            "tgt_emotion": e_tgt,
            "meta": {
                "split": args.split,
                "source_row_index": r["original_idx"],
                "mode": mode,
                "editx_instruction": r["instruction"],
                "vcdata_best_similarity": r.get("speaker_similarity"),
            },
        })

    out = pair_path(cfg, args.split, "H2.jsonl")
    write_jsonl(out, rows)
    print(f"[09] H2={len(rows)}  → {out}")


if __name__ == "__main__":
    main()
