#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _common import iter_jsonl, load_config, normalize_source_row, write_jsonl


SPEED_INFO = {
    "speed_faster": "faster",
    "speed_slower": "slower",
    "speed_more_faster": "more faster",
    "speed_more_slower": "more slower",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Step-Audio-EditX speed edit jobs.")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default="speed_pilot")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--speed-tags", default="")
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    return parser.parse_args()


def duration_ok(duration: object, min_duration: float | None, max_duration: float | None) -> bool:
    if duration in (None, ""):
        return True
    try:
        value = float(duration)
    except (TypeError, ValueError):
        return True
    if min_duration is not None and value < min_duration:
        return False
    if max_duration is not None and value > max_duration:
        return False
    return True


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    speed_cfg = cfg.get("speed_edit", {})
    min_duration = args.min_duration if args.min_duration is not None else speed_cfg.get("min_duration_sec")
    max_duration = args.max_duration if args.max_duration is not None else speed_cfg.get("max_duration_sec")
    tags = [t.strip() for t in args.speed_tags.split(",") if t.strip()]
    if not tags:
        tags = list(cfg["step_audio_editx"].get("speed_tags") or SPEED_INFO)

    rows = []
    selected = 0
    start = max(args.start_index, 0)
    stop = None if args.limit <= 0 else start + args.limit
    for source_index, row in enumerate(iter_jsonl(args.source_jsonl)):
        if source_index < start:
            continue
        if stop is not None and source_index >= stop:
            break
        src = normalize_source_row(row, source_index)
        if not src:
            continue
        if not duration_ok(src["duration"], min_duration, max_duration):
            continue
        for tag in tags:
            edit_info = SPEED_INFO.get(tag)
            if edit_info is None:
                raise ValueError(f"unsupported speed tag: {tag}")
            rows.append(
                {
                    "job_id": f"{args.run_name}_{tag}_{selected:06d}",
                    "model": "step_audio_editx",
                    "mode": "inplace_edit",
                    "language": src["language"],
                    "audio1": src["audio"],
                    "text1": src["text"],
                    "generated_text": "",
                    "output_relpath": f"{tag}/row_{selected:06d}",
                    "metadata": {
                        "route": "speed_edit",
                        "run_name": args.run_name,
                        "source_index": src["source_index"],
                        "source_row_index": src["source_row_index"],
                        "source_speaker_id": src["speaker_id"],
                        "source_duration": src["duration"],
                        "edit_tag": tag,
                        "edit_type": "speed",
                        "edit_info": edit_info,
                    },
                }
            )
        selected += 1

    n = write_jsonl(args.output_jsonl, rows)
    print(f"prepared {n} speed jobs from {Path(args.source_jsonl).resolve()} -> {Path(args.output_jsonl).resolve()}")


if __name__ == "__main__":
    main()
