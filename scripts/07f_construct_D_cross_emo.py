#!/usr/bin/env python
"""07f: D_cross_emo.jsonl —— A1.3.4.2 Affect Intensity Matching (跨情绪强度对齐)

物理：reference = ref_audio (vcdata 合成 高表现), target = original_audio (真人 跨情绪高表现)
- 同 speaker（vcdata clone 出来的 ref_audio 跟 original 同 speaker）
- 跨文本（ref_text ≠ original_text）
- 跨情绪类别（ref.top1 != tgt.top1）
- 双侧都高强度（P_neutral 都 < ref_neutral_max / tgt_neutral_max，如 0.5）
- 不卡 emo_cos（不要求同情绪）
- sim 可选过滤（默认 0.5，跟 D 同档）

instruction: "保持音色，从 {ref_emotion} 转换为 {tgt_emotion} 的情绪"
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (
    load_config, iter_jsonl, write_jsonl,
    intermediate_path, emotion_path, pair_path, make_pair_id,
)
from _emotion_lookup import EmotionTable


EMO_CN = {
    "angry": "愤怒", "sad": "悲伤", "happy": "开心", "surprised": "惊讶",
    "fearful": "恐惧", "disgusted": "厌恶", "neutral": "中性", "other": "其他", "unk": "未知",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)

    vc_path = intermediate_path(cfg, args.split, "vcdata_base.jsonl")
    if not vc_path.exists():
        sys.exit(f"[07f] 缺 {vc_path}（先跑 01）")

    emo = EmotionTable()
    emo.load_csv(emotion_path(cfg, args.split, "per_file_dual.csv"))
    emo.load_per_pair_for_src(emotion_path(cfg, args.split, "per_pair.csv"))
    emo.load_all_link_mappings(emotion_path(cfg, args.split, ""))

    dce = cfg["d_cross_emo"]
    forbidden_emo_labels = set(cfg.get("emotion_filter", {}).get("forbidden_top1_labels", []))
    rng = random.Random(args.seed)

    rows = []
    cnt_no_emo = cnt_same_top1 = cnt_ref_neu = cnt_tgt_neu = 0
    cnt_ref_top1_neu = cnt_tgt_top1_neu = cnt_forbidden_emo = 0
    for v in iter_jsonl(vc_path):
        e_orig = emo.emotion_summary(v["original_audio"])
        e_ref  = emo.emotion_summary(v["ref_audio"])
        if e_orig["top1_label"] is None or e_ref["top1_label"] is None:
            cnt_no_emo += 1; continue
        if forbidden_emo_labels and (
            e_orig["top1_label"] in forbidden_emo_labels
            or e_ref["top1_label"] in forbidden_emo_labels
        ):
            cnt_forbidden_emo += 1; continue
        if dce.get("ref_top1_not_neutral", False) and e_ref["top1_label"] == "neutral":
            cnt_ref_top1_neu += 1; continue
        if dce.get("tgt_top1_not_neutral", False) and e_orig["top1_label"] == "neutral":
            cnt_tgt_top1_neu += 1; continue
        # 关键：top1 必须不同（跨情绪类别）
        if not dce.get("same_top1_label", False) and e_ref["top1_label"] == e_orig["top1_label"]:
            cnt_same_top1 += 1; continue
        # 双侧高强度
        if e_ref["P_neutral"] is not None and e_ref["P_neutral"] > dce.get("ref_neutral_max", 0.5):
            cnt_ref_neu += 1; continue
        if e_orig["P_neutral"] is not None and e_orig["P_neutral"] > dce.get("tgt_neutral_max", 0.5):
            cnt_tgt_neu += 1; continue

        instr = rng.choice(dce.get("instruction_pool", [
            "保持音色，从{ref_emotion}转换为{tgt_emotion}的情绪",
        ])).format(
            ref_emotion=EMO_CN.get(e_ref["top1_label"], e_ref["top1_label"]),
            tgt_emotion=EMO_CN.get(e_orig["top1_label"], e_orig["top1_label"]),
        )
        rows.append({
            "pair_id": make_pair_id(args.split, "Dxe", len(rows)),
            "pair_type": "D_cross_emo",
            "reference_audio": v["ref_audio"],
            "reference_text":  v["ref_text"],
            "target_audio":    v["original_audio"],
            "target_text":     v["original_text"],
            "instruction":     instr,
            "ref_emotion":     e_ref,
            "tgt_emotion":     e_orig,
            "meta": {
                "split": args.split,
                "source_row_index": v["original_idx"],
                "cross_text": True,
                "cross_emotion": True,
                "ref_top1":      e_ref["top1_label"],
                "tgt_top1":      e_orig["top1_label"],
            },
        })

    out = pair_path(cfg, args.split, "D_cross_emo.jsonl")
    write_jsonl(out, rows)
    from collections import Counter
    pair_dist = Counter((r["ref_emotion"]["top1_label"], r["tgt_emotion"]["top1_label"]) for r in rows)
    print(f"[07f] D_cross_emo={len(rows)}  (无emo {cnt_no_emo}, ref-top1=neu {cnt_ref_top1_neu}, "
          f"tgt-top1=neu {cnt_tgt_top1_neu}, top1 相同 {cnt_same_top1}, ref P_neu>{dce['ref_neutral_max']} {cnt_ref_neu}, "
          f"tgt P_neu>{dce['tgt_neutral_max']} {cnt_tgt_neu}, "
          f"forbidden emo {cnt_forbidden_emo})  → {out}")
    print(f"\n  (ref_top1 → tgt_top1) 分布 top 10:")
    for k, c in pair_dist.most_common(10):
        print(f"    {k[0]} → {k[1]}: {c}")


if __name__ == "__main__":
    main()
