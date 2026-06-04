#!/usr/bin/env python
"""07e: Genre_conv.jsonl —— B2.3.2 Genre Conversion (genre A ↔ B 互转 同文本)

物理：从 joined_editx.jsonl 中按 ref_audio 分组，每组内取两个不同 edit_tag 的 edited_audio 配对：
- reference = edited_audio (tag_A)
- target    = edited_audio (tag_B)
- 同 ref_audio 源，同文本 (ref_text)，同 speaker
- WavLM-L sim 过滤在 11b 步骤做（这里不卡）
- 不卡 emotion（两个 genre 都偏平淡）

instruction: "保持音色，从 {tag_A} 风格转换为 {tag_B} 风格"
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (
    load_config, iter_jsonl, write_jsonl,
    intermediate_path, pair_path, make_pair_id,
)


# zh genre tag 的人类可读名（用于 instruction 填充）
TAG_HUMAN = {
    "style_radio":      "电台播报",
    "style_news":       "新闻播报",
    "style_chat":       "客服聊天",
    "style_remove":     "去除风格",
    "emotion_coldness": "冷漠疏离",
    "emotion_remove":   "去除情绪",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-ref", type=int, default=0, help="限制处理的 ref_audio 个数（debug 用）")
    args = ap.parse_args()
    cfg = load_config(args.config)

    jo_path = intermediate_path(cfg, args.split, "joined_editx.jsonl")
    if not jo_path.exists():
        sys.exit(f"[07e] 先跑 03。缺 {jo_path}")

    gc = cfg.get("genre_conv", {
        "cross_pairs": [
            ["style_radio", "style_news"],
            ["style_radio", "style_chat"],
            ["style_news",  "style_chat"],
            # 其他 tag 涉及"去除"/"冷漠"等语义，作为 genre pair 不太合适，先不配
        ],
        "bidirectional": True,
        "instruction_pool": [
            "保持音色，从{ref_genre}风格转换为{tgt_genre}风格",
            "把这段话的播报风格从{ref_genre}变成{tgt_genre}，保持说话人不变",
            "保持当前说话人音色，把语调从{ref_genre}调整为{tgt_genre}",
        ],
    })

    rng = random.Random(args.seed)

    # 按 ref_audio 分组
    by_ref = defaultdict(dict)
    for r in iter_jsonl(jo_path):
        tag = r["edit_tag"]
        by_ref[r["ref_audio"]][tag] = r
    ref_audios = sorted(by_ref.keys())
    if args.limit_ref > 0:
        ref_audios = ref_audios[:args.limit_ref]
    print(f"[07e] joined: {sum(len(v) for v in by_ref.values())} rows, 独立 ref_audio: {len(ref_audios)}")

    cross_pairs = [tuple(p) for p in gc["cross_pairs"]]
    if gc.get("bidirectional", True):
        cross_pairs = list(set(cross_pairs + [(b, a) for a, b in cross_pairs]))

    rows = []
    cnt_missing = 0
    for ref_audio in ref_audios:
        group = by_ref[ref_audio]
        for tag_a, tag_b in cross_pairs:
            if tag_a not in group or tag_b not in group:
                cnt_missing += 1; continue
            ra, rb = group[tag_a], group[tag_b]
            instr = rng.choice(gc["instruction_pool"]).format(
                ref_genre=TAG_HUMAN.get(tag_a, tag_a),
                tgt_genre=TAG_HUMAN.get(tag_b, tag_b),
            )
            rows.append({
                "pair_id": make_pair_id(args.split, "Genre", len(rows)),
                "pair_type": "Genre_conv",
                "reference_audio": ra["edited_audio"],
                "reference_text":  ra["ref_text"],
                "target_audio":    rb["edited_audio"],
                "target_text":     rb["ref_text"],   # 同文本
                "instruction":     instr,
                "source_edit_pair": [tag_a, tag_b],
                "ref_emotion":     None,
                "tgt_emotion":     None,
                "meta": {
                    "split": args.split,
                    "source_row_index": ra.get("original_idx"),
                    "ref_genre_tag":    tag_a,
                    "tgt_genre_tag":    tag_b,
                    "cross_text":       False,
                },
            })

    out = pair_path(cfg, args.split, "Genre_conv.jsonl")
    write_jsonl(out, rows)
    print(f"[07e] Genre_conv={len(rows)}  (missing tag pair: {cnt_missing})  → {out}")

    # 按 (tag_a, tag_b) 分组统计
    from collections import Counter
    by_pair = Counter((r["meta"]["ref_genre_tag"], r["meta"]["tgt_genre_tag"]) for r in rows)
    print(f"\n  各 genre pair 分布：")
    for (a, b), c in by_pair.most_common():
        print(f"    {a} → {b}: {c}")


if __name__ == "__main__":
    main()
