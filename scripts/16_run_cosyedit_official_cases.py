#!/usr/bin/env python3
"""Run CosyEdit over the official semantic benchmark cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    base_dir = Path.cwd().resolve()
    cases_path = resolve_path(args.cases, base_dir)
    repo_dir = resolve_path(args.repo_dir, base_dir)
    model_dir = resolve_path(args.model_dir, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)
    summary_path = resolve_path(args.summary_jsonl, base_dir)

    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "third_party" / "Matcha-TTS"))
    os.chdir(repo_dir)

    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: PLC0415

    load_start = time.time()
    model = AutoModel(model_dir=str(model_dir))
    load_elapsed = time.time() - load_start

    cases = list(iter_jsonl(cases_path))
    if args.limit:
        cases = cases[: args.limit]

    rows: list[dict] = []
    for idx, case in enumerate(cases, 1):
        source_audio = resolve_path(case["audio"], base_dir)
        task = str(case.get("task") or "task")
        lang = str(case.get("language") or "unk")
        file_name = str(case.get("file_name") or case["case_id"])
        output_wav = output_dir / "outputs" / task / lang / f"{file_name}_{task}_cosyedit.wav"
        metrics_json = output_dir / "metrics" / task / lang / f"{file_name}_{task}_cosyedit_run.json"
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.parent.mkdir(parents=True, exist_ok=True)

        ok = True
        error = None
        elapsed = 0.0
        num_chunks = 0
        output_size = 0
        try:
            run_start = time.time()
            chunks = list(model.inference_edit(case["target_text"], case["source_text"], str(source_audio)))
            elapsed = time.time() - run_start
            num_chunks = len(chunks)
            if not chunks:
                raise RuntimeError("CosyEdit returned no chunks")
            speech = chunks[-1]["tts_speech"]
            if hasattr(speech, "detach"):
                speech = speech.detach().cpu().numpy()
            sf.write(str(output_wav), speech.squeeze(), int(getattr(model, "sample_rate", 22050)))
            output_size = output_wav.stat().st_size
        except Exception as exc:  # keep batch progress.
            ok = False
            error = f"{type(exc).__name__}: {exc}"

        row = {
            "ok": ok,
            "case_id": case["case_id"],
            "task": task,
            "language": lang,
            "file_name": file_name,
            "source_audio": str(source_audio),
            "source_text": case.get("source_text"),
            "target_text": case.get("target_text"),
            "output_wav": str(output_wav),
            "metrics_json": str(metrics_json),
            "model_dir": str(model_dir),
            "repo_dir": str(repo_dir),
            "python": sys.executable,
            "load_elapsed_s": round(load_elapsed, 3),
            "elapsed_s": round(elapsed, 3),
            "sample_rate": int(getattr(model, "sample_rate", 22050)),
            "num_chunks": num_chunks,
            "output_exists": output_wav.exists(),
            "output_size": output_size,
            "error": error,
        }
        metrics_json.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        print(f"[{idx}/{len(cases)}] {case['case_id']} ok={ok} elapsed={elapsed:.1f}s", flush=True)

    write_jsonl(summary_path, rows)
    print(f"wrote {len(rows)} rows -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
