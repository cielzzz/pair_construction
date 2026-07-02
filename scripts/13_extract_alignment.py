#!/usr/bin/env python
"""Extract word-level alignment for B1 local edit tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _alignment_backends import extract_alignment


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--backend", choices=["paraformer", "qwen", "whisperx", "fallback", "mock"], default="paraformer")
    ap.add_argument("--output", required=True)
    ap.add_argument("--pause-threshold-ms", type=int, default=100)
    args = ap.parse_args()

    out = extract_alignment(
        args.audio,
        args.ref_text,
        backend=args.backend,
        pause_threshold_ms=args.pause_threshold_ms,
    )
    write_json(Path(args.output), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
