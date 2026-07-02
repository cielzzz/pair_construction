#!/usr/bin/env python3
"""Evaluate generated speech-editing benchmark cases.

The script is backend-agnostic: it builds an ASR manifest from a generation
summary, then merges ASR, text metrics, audio stats, and optional WavLM speaker
similarity into report files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _speaker_sim_cache import CachedSpeakerSimilarity  # noqa: E402

VCDATA_CODE_ROOT = os.environ.get(
    "VCDATA_CODE_ROOT",
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/code/vcdata_construction",
)


THRESHOLDS = {
    "strict_text_error_max": 0.2,
    "tail_silence_sec_warn": 0.8,
    "tail_silence_ratio_warn": 0.25,
    "duration_ratio_warn_gt": 1.7,
    "duration_ratio_warn_lt": 0.45,
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(path: str | Path | None, base_dir: Path) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def normalize_language(value: str | None) -> str:
    value = (value or "auto").lower()
    if value in {"zh", "zho", "chinese", "mandarin"}:
        return "zh"
    if value in {"en", "eng", "english"}:
        return "en"
    return value


def rel_path(path: Path | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base_dir))
    except ValueError:
        return str(path)


def normalize_zh(text: str | None) -> list[str]:
    chars: list[str] = []
    for ch in text or "":
        cat = unicodedata.category(ch)
        if ch.isspace() or cat.startswith("P") or cat.startswith("S"):
            continue
        chars.append(ch.lower())
    return chars


def normalize_en(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_metric(language: str, target_text: str | None, asr_text: str | None) -> tuple[str, float | None, bool]:
    if language == "zh":
        ref = normalize_zh(target_text)
        hyp = normalize_zh(asr_text)
        name = "CER"
    else:
        ref = normalize_en(target_text)
        hyp = normalize_en(asr_text)
        name = "WER"
    if not ref:
        value = 0.0 if not hyp else 1.0
    else:
        value = edit_distance(ref, hyp) / len(ref)
    return name, round(float(value), 4), ref == hyp


def audio_stats(path: Path | None) -> dict[str, float | None]:
    if path is None or not path.exists():
        return {
            "duration_sec": None,
            "sample_rate": None,
            "rms": None,
            "peak": None,
            "tail_silence_sec": None,
            "tail_silence_ratio": None,
        }
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.size else np.zeros(0, dtype=np.float32)
    duration = float(len(mono) / sr) if sr else 0.0
    abs_mono = np.abs(mono)
    peak = float(abs_mono.max()) if abs_mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    silence_threshold = max(0.01, peak * 0.01)
    active = np.flatnonzero(abs_mono > silence_threshold)
    if active.size:
        tail_samples = max(0, len(mono) - int(active[-1]) - 1)
    else:
        tail_samples = len(mono)
    tail_sec = float(tail_samples / sr) if sr else 0.0
    tail_ratio = float(tail_sec / duration) if duration > 0 else None
    return {
        "duration_sec": round(duration, 4),
        "sample_rate": int(sr),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "tail_silence_sec": round(tail_sec, 4),
        "tail_silence_ratio": round(tail_ratio, 4) if tail_ratio is not None else None,
    }


def mean(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return round(float(sum(nums) / len(nums)), 4) if nums else None


def median(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return round(float(statistics.median(nums)), 4) if nums else None


def load_speaker_similarity(device: str, cache_path: Path) -> CachedSpeakerSimilarity:
    sys.path.insert(0, VCDATA_CODE_ROOT)
    from speaker_similarity import SpeakerSimilarity  # noqa: PLC0415

    models = os.environ.get("VCDATA_MODELS_DIR", f"{VCDATA_CODE_ROOT}/models")
    ss = SpeakerSimilarity(
        device=device,
        checkpoint=f"{models}/wavlm_large_finetune.pth",
        seed_tts_eval_root=f"{models}/seed-tts-eval",
        wavlm_dir=f"{models}/wavlm-large",
    )
    return CachedSpeakerSimilarity(ss, cache_path)


def build_asr_manifest(generation_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    manifest_path = output_dir / "asr_manifest.jsonl"
    rows = []
    for row in generation_rows:
        if not row.get("ok") or not row.get("output_wav"):
            continue
        rows.append(
            {
                "uid": row["case_id"],
                "audio": row["output_wav"],
                "language": normalize_language(row.get("language") or "auto"),
            }
        )
    write_jsonl(manifest_path, rows)
    return manifest_path


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(row.get("task") or "task", row.get("language") or "unk")].append(row)
    summary = []
    for (task, lang), rows in sorted(grouped.items()):
        summary.append(
            {
                "task": task,
                "language": lang,
                "n": len(rows),
                "gen_ok": sum(1 for r in rows if r.get("gen_ok")),
                "strict_text_pass": sum(1 for r in rows if r.get("strict_text_pass")),
                "text_exact": sum(1 for r in rows if r.get("text_exact")),
                "audio_warn": sum(1 for r in rows if r.get("audio_warn")),
                "avg_text_error": mean([r.get("text_error") for r in rows]),
                "median_text_error": median([r.get("text_error") for r in rows]),
                "avg_speaker_sim_wavlm": mean([r.get("speaker_sim_wavlm") for r in rows]),
                "min_speaker_sim_wavlm": min(
                    [float(r["speaker_sim_wavlm"]) for r in rows if r.get("speaker_sim_wavlm") is not None],
                    default=None,
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "case_id",
        "task",
        "language",
        "file_name",
        "gen_ok",
        "strict_text_pass",
        "text_metric",
        "text_error",
        "text_exact",
        "speaker_sim_wavlm",
        "audio_warn",
        "duration_ratio",
        "tail_silence_sec",
        "tail_silence_ratio",
        "source_text",
        "target_text",
        "asr_text",
        "source_wav",
        "output_wav",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["tail_silence_sec"] = (row.get("target_audio_stats") or {}).get("tail_silence_sec")
            flat["tail_silence_ratio"] = (row.get("target_audio_stats") or {}).get("tail_silence_ratio")
            writer.writerow({key: flat.get(key) for key in fields})


def write_report(path: Path, backend_name: str, summary: list[dict[str, Any]], totals: dict[str, Any]) -> None:
    lines = [
        f"# {backend_name} Semantic Editing Benchmark",
        "",
        f"- total: `{totals['n']}`",
        f"- generated: `{totals['gen_ok']}`",
        f"- strict text pass (CER/WER <= 0.2): `{totals['strict_text_pass']}`",
        f"- audio_warn: `{totals['audio_warn']}`",
        "",
        "| task | lang | n | gen | strict pass | exact | audio warn | avg CER/WER | median CER/WER | avg WavLM | min WavLM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {task} | {language} | {n} | {gen_ok} | {strict_text_pass} | {text_exact} | {audio_warn} | "
            "{avg_text_error} | {median_text_error} | {avg_speaker_sim_wavlm} | {min_speaker_sim_wavlm} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--generation-summary", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--backend-name", default="backend")
    ap.add_argument("--asr-results", default=None)
    ap.add_argument("--prepare-asr-only", action="store_true")
    ap.add_argument("--compute-wavlm", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    base_dir = Path.cwd().resolve()
    cases_path = resolve_path(args.cases, base_dir)
    generation_path = resolve_path(args.generation_summary, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)
    assert cases_path is not None and generation_path is not None and output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = {row["case_id"]: row for row in iter_jsonl(cases_path)}
    generation_rows = list(iter_jsonl(generation_path))
    manifest_path = build_asr_manifest(generation_rows, output_dir)
    if args.prepare_asr_only:
        print(f"wrote ASR manifest: {manifest_path}")
        return 0

    if not args.asr_results:
        raise SystemExit("--asr-results is required unless --prepare-asr-only is set")
    asr_path = resolve_path(args.asr_results, base_dir)
    assert asr_path is not None
    asr = {row["uid"]: row for row in iter_jsonl(asr_path)}

    wavlm = None
    wavlm_rows = []
    if args.compute_wavlm:
        wavlm = load_speaker_similarity(args.device, output_dir / "metrics" / "wavlm_embeddings.pkl")
        print("[eval] WavLM speaker similarity loaded", flush=True)

    results = []
    for idx, gen in enumerate(generation_rows, 1):
        case = cases.get(gen["case_id"], {})
        lang = normalize_language(str(gen.get("language") or case.get("language") or "unk"))
        source_path = resolve_path(gen.get("source_audio") or case.get("audio"), base_dir)
        target_path = resolve_path(gen.get("output_wav"), base_dir)
        asr_row = asr.get(gen["case_id"], {})
        asr_text = asr_row.get("text") if asr_row.get("ok") else None
        metric_name, metric_value, exact = text_metric(lang, gen.get("target_text") or case.get("target_text"), asr_text)
        source_stats = audio_stats(source_path)
        target_stats = audio_stats(target_path if gen.get("ok") else None)
        duration_ratio = None
        if source_stats.get("duration_sec") and target_stats.get("duration_sec"):
            duration_ratio = round(float(target_stats["duration_sec"]) / float(source_stats["duration_sec"]), 4)
        tail_silence_warn = bool(
            target_stats.get("tail_silence_sec") is not None
            and (
                float(target_stats["tail_silence_sec"]) > THRESHOLDS["tail_silence_sec_warn"]
                or float(target_stats.get("tail_silence_ratio") or 0.0) > THRESHOLDS["tail_silence_ratio_warn"]
            )
        )
        duration_warn = bool(
            duration_ratio is not None
            and (
                duration_ratio > THRESHOLDS["duration_ratio_warn_gt"]
                or duration_ratio < THRESHOLDS["duration_ratio_warn_lt"]
            )
        )
        speaker_sim = None
        wavlm_error = None
        if wavlm is not None and gen.get("ok") and source_path and target_path and source_path.exists() and target_path.exists():
            try:
                speaker_sim = round(float(wavlm.compute_similarity_files(str(source_path), str(target_path))), 4)
            except Exception as exc:
                wavlm_error = f"{type(exc).__name__}: {exc}"
            wavlm_rows.append(
                {
                    "case_id": gen["case_id"],
                    "source_wav": str(source_path),
                    "output_wav": str(target_path),
                    "speaker_sim_wavlm": speaker_sim,
                    "error": wavlm_error,
                }
            )
        row = {
            "index": idx,
            "backend": args.backend_name,
            "case_id": gen["case_id"],
            "task": gen.get("task") or case.get("task"),
            "language": lang,
            "file_name": gen.get("file_name") or case.get("file_name"),
            "instruction": case.get("instruction"),
            "source_text": gen.get("source_text") or case.get("source_text"),
            "target_text": gen.get("target_text") or case.get("target_text"),
            "asr_text": asr_text,
            "asr_ok": bool(asr_row.get("ok")),
            "asr_error": asr_row.get("error"),
            "text_metric": metric_name,
            "text_error": metric_value,
            "strict_text_pass": metric_value is not None and metric_value <= THRESHOLDS["strict_text_error_max"],
            "text_exact": exact,
            "speaker_sim_wavlm": speaker_sim,
            "wavlm_error": wavlm_error,
            "gen_ok": bool(gen.get("ok")),
            "gen_error": gen.get("error"),
            "elapsed_s": gen.get("elapsed_s"),
            "source_wav": rel_path(source_path, base_dir),
            "output_wav": rel_path(target_path, base_dir),
            "source_audio_stats": source_stats,
            "target_audio_stats": target_stats,
            "duration_ratio": duration_ratio,
            "tail_silence_warn": tail_silence_warn,
            "duration_warn": duration_warn,
            "audio_warn": tail_silence_warn or duration_warn,
            "run_metrics_json": rel_path(resolve_path(gen.get("metrics_json"), base_dir), base_dir),
        }
        results.append(row)

    if wavlm is not None:
        wavlm.save()

    summary = summarize(results)
    totals = {
        "n": len(results),
        "gen_ok": sum(1 for r in results if r.get("gen_ok")),
        "strict_text_pass": sum(1 for r in results if r.get("strict_text_pass")),
        "text_exact": sum(1 for r in results if r.get("text_exact")),
        "audio_warn": sum(1 for r in results if r.get("audio_warn")),
        "avg_text_error": mean([r.get("text_error") for r in results]),
        "avg_speaker_sim_wavlm": mean([r.get("speaker_sim_wavlm") for r in results]),
    }
    write_jsonl(output_dir / "results.jsonl", results)
    write_csv(output_dir / "results.csv", results)
    write_jsonl(output_dir / "wavlm_sim.jsonl", wavlm_rows)
    summary_obj = {
        "backend": args.backend_name,
        "thresholds": THRESHOLDS,
        "totals": totals,
        "summary": summary,
        "paths": {
            "cases": str(cases_path),
            "generation_summary": str(generation_path),
            "asr_manifest": str(manifest_path),
            "asr_results": str(asr_path),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "report.md", args.backend_name, summary, totals)
    print(f"wrote results: {output_dir / 'results.jsonl'}")
    print(json.dumps(totals, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
