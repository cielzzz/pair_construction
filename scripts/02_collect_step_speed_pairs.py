#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
import random
import re
from pathlib import Path

from _common import iter_jsonl, load_config, make_pair_id, write_jsonl


PAIR_TYPE_BY_TAG = {
    "speed_faster": "J_fast",
    "speed_slower": "J_slow",
    "speed_more_faster": "J_fast",
    "speed_more_slower": "J_slow",
}


def infer_language(text: str | None) -> str:
    value = text or ""
    return "zh" if re.search(r"[\u4e00-\u9fff]", value) else "en"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Step-Audio-EditX speed reports into pair jsonl.")
    parser.add_argument("--report-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--split-output-dir", default="", help="Optional directory for one <pair_type>.jsonl per speed class.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default="speed_pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-existing-audio", action="store_true")
    return parser.parse_args()


def one_stage_audio(row: dict) -> str | None:
    block = row.get("one_stage") or {}
    return block.get("audio2")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    speed_cfg = cfg.get("speed_edit", {})
    rng = random.Random(args.seed)

    rows = []
    missing_audio = 0
    for row in iter_jsonl(args.report_jsonl):
        metadata = row.get("metadata") or {}
        tag = metadata.get("edit_tag") or metadata.get("delta_tag")
        if tag not in PAIR_TYPE_BY_TAG:
            continue
        target_audio = one_stage_audio(row)
        if not target_audio:
            missing_audio += 1
            continue
        if args.require_existing_audio and not Path(target_audio).exists():
            missing_audio += 1
            continue
        language = infer_language(row.get("text1") or row.get("text2"))
        language_pool = (((speed_cfg.get("instruction_pool_by_language") or {}).get(language) or {}).get(tag)) or []
        instruction_pool = language_pool or speed_cfg.get("instruction_pool", {}).get(tag) or [row.get("instruction") or f"speed:{tag}"]
        pair_type = PAIR_TYPE_BY_TAG[tag]
        rows.append(
            {
                "pair_id": make_pair_id(args.run_name, pair_type, len(rows)),
                "pair_type": pair_type,
                "reference_audio": row.get("audio1"),
                "reference_text": row.get("text1"),
                "target_audio": target_audio,
                "target_text": row.get("text2") or row.get("text1"),
                "instruction": rng.choice(instruction_pool),
                "source_edit_tag": tag,
                "source_edit": tag,
                "taxonomy_nodes": [
                    "A1.2.1.1 Speech Rate Control",
                    "B2.2.1 Speech-Rate Conversion",
                    "B3.1/B3.2 Duration Compression/Expansion",
                ],
                "meta": {
                    "route": "speed_edit",
                    "run_name": args.run_name,
                    "language": language,
                    "editx_instruction": row.get("instruction"),
                    "editx_job_id": row.get("job_id"),
                    "expected_duration_ratio": speed_cfg.get("expected_duration_ratio", {}).get(tag),
                    "source_metadata": metadata,
                },
            }
        )

    n = write_jsonl(args.output_jsonl, rows)
    print(f"collected {n} speed pairs -> {Path(args.output_jsonl).resolve()}  missing_audio={missing_audio}")
    if args.split_output_dir:
        by_pair_type = defaultdict(list)
        for row in rows:
            by_pair_type[row["pair_type"]].append(row)
        out_dir = Path(args.split_output_dir).expanduser().resolve()
        for pair_type, pair_rows in sorted(by_pair_type.items()):
            out_path = out_dir / f"{pair_type}.jsonl"
            wrote = write_jsonl(out_path, pair_rows)
            print(f"collected {wrote} {pair_type} pairs -> {out_path}")


if __name__ == "__main__":
    main()
