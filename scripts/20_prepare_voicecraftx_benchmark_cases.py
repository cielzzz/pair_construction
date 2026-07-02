#!/usr/bin/env python3
"""Prepare VoiceCraft-X benchmark cases and MFA corpora."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    base_dir = Path.cwd().resolve()
    cases_path = resolve_path(args.cases, base_dir)
    output_root = resolve_path(args.output_root, base_dir)
    corpus_root = output_root / "mfa_corpus"
    align_root = output_root / "mfa_alignments"

    out_rows = []
    for case in iter_jsonl(cases_path):
        lang = case.get("language")
        if lang not in {"zh", "en"}:
            continue
        case_id = case["case_id"]
        audio_src = resolve_path(case["audio"], base_dir)
        corpus_dir = corpus_root / lang
        corpus_dir.mkdir(parents=True, exist_ok=True)
        audio_dst = corpus_dir / f"{case_id}.wav"
        text_dst = corpus_dir / f"{case_id}.txt"
        shutil.copy2(audio_src, audio_dst)
        text_dst.write_text(str(case.get("source_text") or "").strip() + "\n", encoding="utf-8")

        row = {
            "case_id": case_id,
            "task": case.get("task"),
            "official_task": case.get("official_task"),
            "language": lang,
            "file_name": case.get("file_name") or case_id,
            "audio": str(audio_dst),
            "alignment_path": str(align_root / lang / f"{case_id}.csv"),
            "source_text": case.get("source_text"),
            "target_text": case.get("target_text"),
            "instruction": case.get("instruction"),
        }
        out_rows.append(row)

    write_jsonl(output_root / "voicecraftx_cases.jsonl", out_rows)
    print(f"wrote {len(out_rows)} cases -> {output_root / 'voicecraftx_cases.jsonl'}")
    print(f"corpus_root={corpus_root}")
    print(f"align_root={align_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
