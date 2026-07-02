#!/usr/bin/env python
"""Batch ASR helper for pair QC.

Reads a JSONL manifest:
  {"uid": "...", "audio": "/abs/path.wav", "language": "zh|en|auto"}

Writes a JSONL result file:
  {"uid": "...", "ok": 1, "text": "...", "language": "Chinese"}
or
  {"uid": "...", "ok": 0, "error": "..."}

Notes:
- This script is intentionally isolated so the main QC script can call it via a
  dedicated Python interpreter / environment.
- The current user-provided environment may have a broken sklearn binary.
  We install a minimal stub before importing qwen_asr so import-time generation
  utilities do not crash on sklearn import.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import math
import sys
import types
from pathlib import Path


def install_sklearn_stub() -> None:
    sklearn = types.ModuleType("sklearn")
    sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    metrics = types.ModuleType("sklearn.metrics")
    metrics.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)

    def roc_curve(*args, **kwargs):
        raise NotImplementedError("sklearn.metrics.roc_curve stub should not be used for ASR inference")

    metrics.roc_curve = roc_curve
    sklearn.metrics = metrics
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.metrics", metrics)


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_language(lang: str | None) -> str | None:
    if not lang:
        return None
    lang = lang.strip().lower()
    if lang in ("auto", "none", ""):
        return None
    if lang == "zh":
        return "Chinese"
    if lang == "en":
        return "English"
    return lang


def batched(seq, size: int):
    if size <= 0:
        size = len(seq) or 1
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL manifest")
    ap.add_argument("--output", required=True, help="JSONL output")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    records = list(iter_jsonl(Path(args.input)))
    if not records:
        write_jsonl(Path(args.output), [])
        return 0

    install_sklearn_stub()

    try:
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]

        model = Qwen3ASRModel.from_pretrained(
            args.model,
            dtype=dtype,
            device_map=args.device,
            max_inference_batch_size=max(1, args.batch_size),
            max_new_tokens=max(64, args.max_new_tokens),
        )
    except Exception as exc:
        err = f"ASR backend init failed: {type(exc).__name__}: {exc}"
        write_jsonl(
            Path(args.output),
            [{"uid": rec["uid"], "ok": 0, "error": err} for rec in records],
        )
        return 0

    out_rows = []
    for chunk in batched(records, args.batch_size):
        audios = [rec["audio"] for rec in chunk]
        langs = [normalize_language(rec.get("language")) for rec in chunk]
        try:
            results = model.transcribe(audio=audios, language=langs)
            if len(results) != len(chunk):
                raise RuntimeError(f"result count mismatch: {len(results)} != {len(chunk)}")
            for rec, result in zip(chunk, results):
                out_rows.append(
                    {
                        "uid": rec["uid"],
                        "ok": 1,
                        "text": getattr(result, "text", ""),
                        "language": getattr(result, "language", None),
                    }
                )
        except Exception as exc:
            batch_err = f"ASR batch failed: {type(exc).__name__}: {exc}"
            # Fallback one-by-one to isolate bad cases.
            for rec in chunk:
                try:
                    result = model.transcribe(
                        audio=rec["audio"],
                        language=normalize_language(rec.get("language")),
                    )[0]
                    out_rows.append(
                        {
                            "uid": rec["uid"],
                            "ok": 1,
                            "text": getattr(result, "text", ""),
                            "language": getattr(result, "language", None),
                        }
                    )
                except Exception as inner_exc:
                    out_rows.append(
                        {
                            "uid": rec["uid"],
                            "ok": 0,
                            "error": f"{batch_err}; single={type(inner_exc).__name__}: {inner_exc}",
                        }
                    )

    write_jsonl(Path(args.output), out_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
