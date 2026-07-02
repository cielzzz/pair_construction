#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _common import iter_jsonl, make_pair_id, write_json, write_jsonl


PAIR_TYPE = "I"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Seed-VC outputs into I pair jsonl.")
    parser.add_argument("--jobs-jsonl", required=True)
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--require-existing-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {row.get("job_id"): row for row in iter_jsonl(args.results_jsonl)}
    rows = []
    missing = 0
    failed = 0

    for job in iter_jsonl(args.jobs_jsonl):
        result = results.get(job.get("job_id"))
        if not result:
            missing += 1
            continue
        if not result.get("ok"):
            failed += 1
            continue
        target_audio = result.get("audio") or job.get("output_audio")
        if args.require_existing_audio and not Path(str(target_audio)).exists():
            missing += 1
            continue
        rows.append(
            {
                "pair_id": make_pair_id(args.run_name, PAIR_TYPE, len(rows)),
                "pair_type": PAIR_TYPE,
                "reference_audio": job.get("prosody_ref_audio"),
                "reference_text": job.get("prosody_ref_text"),
                "prosody_ref_audio": job.get("prosody_ref_audio"),
                "prosody_ref_text": job.get("prosody_ref_text"),
                "timbre_ref_audio": job.get("timbre_ref_audio"),
                "timbre_ref_text": job.get("timbre_ref_text"),
                "target_audio": target_audio,
                "target_text": job.get("target_text") or job.get("prosody_ref_text"),
                "instruction": job.get("instruction"),
                "source_edit_tag": job.get("source_edit_tag"),
                "source_edit": job.get("source_edit"),
                "target_voice_profile": job.get("target_voice_profile"),
                "taxonomy_nodes": job.get("taxonomy_nodes") or [],
                "meta": {
                    **(job.get("metadata") or {}),
                    "run_name": args.run_name,
                    "backend": "seed_vc_v1_zero_shot_voice_conversion",
                    "backend_job_id": job.get("job_id"),
                    "backend_result": result,
                    "schema_note": "reference_audio/reference_text are aliases for prosody_ref_audio/prosody_ref_text for I.",
                },
            }
        )

    n = write_jsonl(args.output_jsonl, rows)
    summary = {
        "run_name": args.run_name,
        "jobs_jsonl": str(Path(args.jobs_jsonl).resolve()),
        "results_jsonl": str(Path(args.results_jsonl).resolve()),
        "output_jsonl": str(Path(args.output_jsonl).resolve()),
        "pairs_written": n,
        "missing_results_or_audio": missing,
        "failed_results": failed,
        "schema_note": "I rows include reference_audio/reference_text aliases for QC compatibility.",
    }
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else Path(args.output_jsonl).with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(f"[collect-seedvc] wrote {n} pairs -> {Path(args.output_jsonl).resolve()}")
    print(f"[collect-seedvc] summary -> {summary_path}")


if __name__ == "__main__":
    main()
