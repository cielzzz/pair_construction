#!/usr/bin/env python
"""01: 把 vcdata merged.stepaudio_input.all.jsonl 标准化为 vcdata_base.jsonl

不修改上游。只读、只标准化字段。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (
    load_config, iter_jsonl, write_jsonl,
    split_dir, intermediate_path, make_sample_id_vc,
)


def normalize(row: dict, split: str) -> dict:
    cap = row.get("caption_result") or {}
    return {
        "sample_id": make_sample_id_vc(split, row["original_idx"]),
        "split": split,
        "original_idx": row["original_idx"],
        "original_audio": row["original_audio_path"],
        "original_text": row["original_text"],
        "ref_audio": row["ref_audio_path"],
        "ref_text": row["ref_text"],
        "speaker_similarity": row.get("best_similarity"),
        "flag": row.get("flag"),
        "duration": row.get("duration"),
        "language": row.get("language"),
        "dnsmos": row.get("dnsmos"),
        "caption_summary": cap.get("summary"),
        "caption_gender": cap.get("gender"),
        "caption_emotion": cap.get("emotion"),
        "caption_tone": cap.get("tone"),
        "zh_summary": row.get("zh_summary"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="e.g. split_0000")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--source-jsonl",
        default=None,
        help="覆盖默认的 merged.stepaudio_input.all.jsonl 路径",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    sd = split_dir(cfg, args.split)
    src = Path(args.source_jsonl) if args.source_jsonl else (sd / "merged.stepaudio_input.all.jsonl")
    if not src.exists():
        sys.exit(f"[01] 找不到 vcdata 合并 manifest: {src}\n  提示：split 是否已合并 shards？")

    out = intermediate_path(cfg, args.split, "vcdata_base.jsonl")
    n = write_jsonl(out, (normalize(r, args.split) for r in iter_jsonl(src)))
    print(f"[01] vcdata_base.jsonl ← {src.name}  rows={n}  → {out}")


if __name__ == "__main__":
    main()
