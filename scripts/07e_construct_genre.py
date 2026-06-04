#!/usr/bin/env python
"""07e: Genre.jsonl —— B2.3.2 Genre Conversion (v2: ref_audio 同源派生)

物理：reference = ref_audio (vcdata clone), target = editx_{tag}_edited
- 同 ref_audio 源（ref_audio 喂 editx 改造）
- 同文本（都是 ref_text）
- editx tag 白名单：zh = [style_news, style_chat, style_remove]（去掉 style_radio = 中性化用）
- 不卡 emotion（genre 改造 emotion 会偏中性）
- sim ≥ 0.75（同 vcdata 源派生，跟 B/C/D_st 一档）

instruction: "保持音色，从当前风格转换为 {tgt_genre} 风格"
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (
    load_config, iter_jsonl, write_jsonl,
    intermediate_path, pair_path, make_pair_id,
)

TAG_HUMAN = {
    "style_news":       "新闻播报",
    "style_chat":       "客服聊天",
    "style_remove":     "去除风格",
    "style_radio":      "电台播报",
    "emotion_coldness": "冷漠疏离",
    "emotion_remove":   "去除情绪",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)

    jo_path = intermediate_path(cfg, args.split, "joined_editx.jsonl")
    if not jo_path.exists():
        sys.exit(f"[07e] 先跑 03。缺 {jo_path}")

    gc = cfg.get("genre", {})
    whitelist = set(gc.get("edit_tag_whitelist", ["style_news", "style_chat", "style_remove"]))
    rng = random.Random(args.seed)

    rows = []
    n_total = 0; n_skip_tag = 0
    for r in iter_jsonl(jo_path):
        n_total += 1
        if r["edit_tag"] not in whitelist:
            n_skip_tag += 1; continue
        instr = rng.choice(gc.get("instruction_pool", [
            "保持音色，从当前风格转换为{tgt_genre}风格",
        ])).format(tgt_genre=TAG_HUMAN.get(r["edit_tag"], r["edit_tag"]))
        rows.append({
            "pair_id": make_pair_id(args.split, "Genre", len(rows)),
            "pair_type": "Genre",
            "reference_audio": r["ref_audio"],         # 同源派生：reference = ref_audio
            "reference_text":  r["ref_text"],
            "target_audio":    r["edited_audio"],
            "target_text":     r["ref_text"],          # 同文本
            "instruction":     instr,
            "source_edit_tag": r["edit_tag"],
            "ref_emotion":     None,
            "tgt_emotion":     None,
            "meta": {
                "split": args.split,
                "source_row_index": r.get("original_idx"),
                "cross_text": False,                    # 同文本
            },
        })

    out = pair_path(cfg, args.split, "Genre.jsonl")
    write_jsonl(out, rows)
    from collections import Counter
    tag_dist = dict(Counter(r["source_edit_tag"] for r in rows))
    print(f"[07e] Genre={len(rows)}  (跳过非白名单 tag {n_skip_tag}/{n_total})  → {out}")
    print(f"      tag 分布: {tag_dist}")


if __name__ == "__main__":
    main()
