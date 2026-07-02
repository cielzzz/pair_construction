#!/usr/bin/env python3
"""Run VoiceCraft-X speech editing cases from JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from omegaconf import OmegaConf


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


def build_chinese_char_alignment(alignment_path: Path, output_dir: Path) -> Path:
    """VoiceCraft-X's diff helper indexes Chinese alignments per character."""
    char_dir = output_dir / "mfa_alignments_char" / alignment_path.parent.name
    char_dir.mkdir(parents=True, exist_ok=True)
    char_path = char_dir / alignment_path.name

    with alignment_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if (row.get("Type") or "").strip() == "words"]

    out_rows = []
    for row in rows:
        label = (row.get("Label") or "").strip()
        if not label:
            continue
        begin = float(row["Begin"])
        end = float(row["End"])
        chars = list(label)
        step = (end - begin) / max(len(chars), 1)
        for idx, ch in enumerate(chars):
            out_rows.append(
                {
                    "Begin": begin + step * idx,
                    "End": begin + step * (idx + 1),
                    "Label": ch,
                    "Type": "words",
                    "Speaker": row.get("Speaker") or alignment_path.parent.name,
                }
            )

    with char_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Begin", "End", "Label", "Type", "Speaker"])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    return char_path


def language_name(value: str | None) -> str:
    value = (value or "english").lower()
    if value in {"zh", "zho", "chinese", "mandarin"}:
        return "chinese"
    if value in {"en", "eng", "english"}:
        return "english"
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--summary-jsonl", required=True)
    ap.add_argument("--config", default="src/config/inference/edit.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_dir = Path.cwd().resolve()
    cases_path = resolve_path(args.cases, base_dir)
    repo_dir = resolve_path(args.repo_dir, base_dir)
    model_dir = resolve_path(args.model_dir, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)
    summary_path = resolve_path(args.summary_jsonl, base_dir)

    sys.path.insert(0, str(repo_dir / "src"))
    os.environ.setdefault("AUDIOCRAFT_CLUSTER", "default")
    os.chdir(repo_dir / "src")

    from helper import generate, load_speaker_model, load_tokenizer, load_voicecraftx  # noqa: PLC0415

    config = OmegaConf.load(str(resolve_path(args.config, repo_dir)))
    config.pretrained_models = str(model_dir)
    config.voicecraftx_path = str(model_dir / "voicecraftx.ckpt")
    config.model.config_path = str(model_dir)
    config.MAX_LENGTH = args.max_length

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    load_start = time.time()
    speaker_model = load_speaker_model(config)
    text_tokenizer, audio_tokenizer = load_tokenizer(config)
    audio_tokenizer = audio_tokenizer.to(device)
    model = load_voicecraftx(config).to(device)
    load_elapsed = time.time() - load_start

    cases = list(iter_jsonl(cases_path))
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    for idx, case in enumerate(cases, 1):
        case_id = case.get("case_id") or f"case_{idx:04d}"
        lang = language_name(case.get("language"))
        task = case.get("task") or "editing"
        prompt_audio = resolve_path(case["audio"], base_dir)
        alignment_path = resolve_path(case["alignment_path"], base_dir)
        runtime_alignment_path = (
            build_chinese_char_alignment(alignment_path, output_dir)
            if lang == "chinese"
            else alignment_path
        )
        file_stem = case.get("file_name") or case_id
        case_dir = output_dir / "outputs" / task / lang
        case_dir.mkdir(parents=True, exist_ok=True)

        ok = True
        error = None
        elapsed = 0.0
        output_wavs: list[str] = []
        try:
            run_start = time.time()
            outputs = generate(
                config=config,
                device=device,
                language=lang,
                prompt_audio=str(prompt_audio),
                prompt_text=case["source_text"],
                target_text=case["target_text"],
                model=model,
                speaker_model=speaker_model,
                text_tokenizer=text_tokenizer,
                audio_tokenizer=audio_tokenizer,
                task="editing",
                alignment_path=str(runtime_alignment_path),
                n_samples=args.n_samples,
            )
            elapsed = time.time() - run_start
            for sample_idx, tokens in enumerate(outputs):
                audio = audio_tokenizer.decode(tokens)
                wav = audio[0].detach().cpu().numpy()
                if wav.ndim == 2:
                    wav = wav.squeeze(0)
                output_wav = case_dir / f"{file_stem}_{task}_voicecraftx_s{sample_idx}.wav"
                sf.write(str(output_wav), wav, int(config.SAMPLE_RATE))
                output_wavs.append(str(output_wav))
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"

        row = {
            "ok": ok,
            "case_id": case_id,
            "task": task,
            "language": lang,
            "file_name": file_stem,
            "source_audio": str(prompt_audio),
            "alignment_path": str(runtime_alignment_path),
            "original_alignment_path": str(alignment_path),
            "source_text": case.get("source_text"),
            "target_text": case.get("target_text"),
            "output_wav": output_wavs[0] if output_wavs else None,
            "output_wavs": output_wavs,
            "model_dir": str(model_dir),
            "repo_dir": str(repo_dir),
            "python": sys.executable,
            "device": str(device),
            "n_samples": args.n_samples,
            "load_elapsed_s": round(load_elapsed, 3),
            "elapsed_s": round(elapsed, 3),
            "sample_rate": int(config.SAMPLE_RATE),
            "error": error,
        }
        rows.append(row)
        print(f"[{idx}/{len(cases)}] {case_id} ok={ok} elapsed={elapsed:.1f}s", flush=True)

    write_jsonl(summary_path, rows)
    print(f"wrote {len(rows)} rows -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
