#!/usr/bin/env python
"""09: H2.jsonl —— 已满足指令对照

当前 editx 只覆盖中性化方向，所以 H2 实质是"已中性 reference + 中性 instruction"。
未来扩展到 happy/angry 时，可让 source_edit_tag 支持多组配置。

reference = edited_audio (P_neutral >= 0.9 且 sv == neutral)
target    = edited_audio (mode=self) | 同属性近邻 (mode=neighbor)
instruction = "更中性" / "去掉情绪起伏" / "更像新闻播报"
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
        choices=["self", "neighbor"],
        help="覆盖 config.h2.mode",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)

    jo_path = intermediate_path(cfg, args.split, "joined_editx.jsonl")
    if not jo_path.exists():
        sys.exit(f"[09] 先跑 03。缺 {jo_path}")

    emo = EmotionTable()
    emo.load_csv(emotion_path(cfg, args.split, "per_file_dual.csv"))

    h2 = cfg["h2"]
    target_edit_tag = h2["source_edit_tag"]
    p_min = h2["p_neutral_min"]
    sv_req = h2["require_sv_neutral"]
    mode = args.mode or h2["mode"]
    rng = random.Random(args.seed)

    pool = []
    for r in iter_jsonl(jo_path):
        if r["edit_tag"] != target_edit_tag:
            continue
        e = emo.emotion_summary(r["edited_audio"])
        if e["P_neutral"] is None or e["P_neutral"] < p_min:
            continue
        if sv_req and e["sv_label"] != "neutral":
            continue
        pool.append((r, e))

    print(f"[09] 高置信中性 pool 大小：{len(pool)} (edit_tag={target_edit_tag}, mode={mode})")
    if not pool:
        write_jsonl(pair_path(cfg, args.split, "H2.jsonl"), [])
        return

    rows = []
    for r, e_ref in pool:
        if mode == "self":
            tgt_audio = r["edited_audio"]
            e_tgt = e_ref
        else:
            cand = rng.choice(pool)
            while cand[0]["original_idx"] == r["original_idx"] and len(pool) > 1:
                cand = rng.choice(pool)
            tgt_audio = cand[0]["edited_audio"]
            e_tgt = cand[1]
        rows.append({
            "pair_id": make_pair_id(args.split, "H2", len(rows)),
            "pair_type": "H2",
            "reference_audio": r["edited_audio"],
            "reference_text": r["ref_text"],
            "target_audio": tgt_audio,
            "target_text": r["ref_text"],
            "instruction": rng.choice(h2["instruction_pool"]),
            "source_edit": target_edit_tag,
            "speaker_similarity": r.get("speaker_similarity"),
            "ref_emotion": e_ref,
            "tgt_emotion": e_tgt,
            "meta": {
                "split": args.split,
                "source_row_index": r["original_idx"],
                "mode": mode,
            },
        })

    out = pair_path(cfg, args.split, "H2.jsonl")
    write_jsonl(out, rows)
    print(f"[09] H2={len(rows)}  → {out}")


if __name__ == "__main__":
    main()
