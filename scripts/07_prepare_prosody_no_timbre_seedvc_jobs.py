#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Any

from _common import iter_jsonl, normalize_source_row, write_json, write_jsonl


TAXONOMY_NODES = [
    "A1.3.2.1 Rhythm Imitation",
    "A1.3.2.2 Pause-Pattern Imitation",
    "A1.3.2.3 Intonation Imitation",
    "A1.3.2.5 Source-Like Pacing without Content Copying",
    "A1.3.8.1 Timbre from One Reference, Prosody from Another",
    "B2.5.3 Preserve Rhythm but Change Voice",
]

INSTRUCTION_BY_LANGUAGE = {
    "zh": "保留 prosody_ref_audio 的语速、停顿、节奏和重音；不要模仿其音色，使用 timbre_ref_audio 的说话人音色。",
    "en": "Keep the speaking rate, pauses, rhythm, and emphasis of prosody_ref_audio; do not imitate its timbre, and instead use the speaker timbre from timbre_ref_audio.",
}

PAIR_TYPE = "I"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Seed-VC I jobs.")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--jobs-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--timbre-pick", choices=("random", "next"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio-subdir", default="audio/seedvc_v1")
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def select_sources(source_jsonl: str, start_index: int, limit: int) -> list[dict[str, Any]]:
    selected = []
    stop = None if limit <= 0 else start_index + limit
    for row_index, row in enumerate(iter_jsonl(source_jsonl)):
        if row_index < start_index:
            continue
        if stop is not None and row_index >= stop:
            break
        src = normalize_source_row(row, row_index)
        if src:
            selected.append(src)
    return selected


def pick_timbre_indices(sources: list[dict[str, Any]], mode: str, seed: int) -> list[int]:
    if len(sources) <= 1:
        raise ValueError("I with batch timbre references needs at least two valid rows.")
    by_index = list(range(len(sources)))
    if mode == "next":
        picks = []
        for idx, src in enumerate(sources):
            candidates = [j for j in by_index if j != idx and sources[j]["audio"] != src["audio"]]
            picks.append(candidates[0] if candidates else (idx + 1) % len(sources))
        return picks

    rng = random.Random(seed)
    picks = []
    for idx, src in enumerate(sources):
        candidates = [j for j in by_index if j != idx and sources[j]["audio"] != src["audio"]]
        if not candidates:
            candidates = [j for j in by_index if j != idx]
        picks.append(rng.choice(candidates))
    return picks


def make_job(
    run_name: str,
    idx: int,
    src: dict[str, Any],
    timbre_src: dict[str, Any],
    output_audio: Path,
    seed: int,
) -> dict[str, Any]:
    return {
        "job_id": f"{run_name}:seedvc_v1:{idx:06d}",
        "pair_type": PAIR_TYPE,
        "reference_audio": src["audio"],
        "reference_text": src["text"],
        "prosody_ref_audio": src["audio"],
        "prosody_ref_text": src["text"],
        "timbre_ref_audio": timbre_src["audio"],
        "timbre_ref_text": timbre_src["text"],
        "target_audio": str(output_audio),
        "output_audio": str(output_audio),
        "target_text": src["text"],
        "instruction": INSTRUCTION_BY_LANGUAGE.get(src.get("language"), INSTRUCTION_BY_LANGUAGE["zh"]),
        "source_edit_tag": "seedvc_v1_batch_timbre",
        "source_edit": "seedvc_v1_batch_timbre",
        "target_voice_profile": "batch_timbre_ref",
        "taxonomy_nodes": TAXONOMY_NODES,
        "metadata": {
            "route": "prosody_no_timbre_seedvc_v1",
            "run_name": run_name,
            "source_index": src["source_index"],
            "source_row_index": src["source_row_index"],
            "source_speaker_id": src.get("speaker_id"),
            "timbre_source_index": timbre_src["source_index"],
            "timbre_source_row_index": timbre_src["source_row_index"],
            "timbre_speaker_id": timbre_src.get("speaker_id"),
            "language": src.get("language"),
            "backend": "seed_vc_v1_zero_shot_voice_conversion",
            "seed": seed,
            "note": "Source audio supplies content, timing, pauses and local rhythm; a different in-batch utterance supplies timbre reference. This route avoids DSP pitch/EQ anonymization.",
        },
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root).expanduser().resolve()
    jobs_path = Path(args.jobs_jsonl).expanduser().resolve()
    audio_root = out_root / args.audio_subdir
    sources = select_sources(args.source_jsonl, args.start_index, args.limit)
    timbre_indices = pick_timbre_indices(sources, args.timbre_pick, args.seed)

    jobs = []
    for idx, src in enumerate(sources):
        timbre_src = sources[timbre_indices[idx]]
        output_audio = audio_root / f"{idx:06d}.wav"
        jobs.append(make_job(args.run_name, idx, src, timbre_src, output_audio, args.seed))

    n = write_jsonl(jobs_path, jobs)
    summary = {
        "run_name": args.run_name,
        "source_jsonl": str(Path(args.source_jsonl).resolve()),
        "output_root": str(out_root),
        "jobs_jsonl": str(jobs_path),
        "sources_selected": len(sources),
        "jobs_written": n,
        "timbre_pick": args.timbre_pick,
        "seed": args.seed,
        "backend": "seed_vc_v1_zero_shot_voice_conversion",
        "schema_note": "I uses reference_audio/reference_text as aliases for prosody_ref_audio/prosody_ref_text.",
    }
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else out_root / "metrics" / "seedvc_prepare_summary.json"
    write_json(summary_path, summary)
    print(f"[prepare-seedvc] wrote {n} jobs -> {jobs_path}")
    print(f"[prepare-seedvc] summary -> {summary_path}")


if __name__ == "__main__":
    main()
