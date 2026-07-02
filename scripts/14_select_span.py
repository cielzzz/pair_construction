#!/usr/bin/env python
"""Select edit spans from B1 alignment JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from _alignment_backends import clean_text


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def word_char_ranges(words: list[dict[str, Any]]) -> tuple[str, list[tuple[int, int]]]:
    cursor = 0
    ranges = []
    text_parts = []
    for word in words:
        text = clean_text(str(word.get("text") or ""))
        start = cursor
        cursor += len(text)
        ranges.append((start, cursor))
        text_parts.append(text)
    return "".join(text_parts), ranges


def span_from_indices(words: list[dict[str, Any]], indices: list[int]) -> dict[str, Any] | None:
    if not indices:
        return None
    selected = [words[i] for i in indices]
    return {
        "start_ms": int(min(w["start_ms"] for w in selected)),
        "end_ms": int(max(w["end_ms"] for w in selected)),
        "anchor_words": [str(w.get("text") or "") for w in selected],
        "anchor_word_indices": indices,
    }


def select_spans(alignment: dict[str, Any], target_spec: dict[str, Any]) -> dict[str, Any]:
    words = alignment.get("words") or []
    mode = target_spec.get("mode")
    params = target_spec.get("params") or {}
    spans: list[dict[str, Any]] = []

    if mode == "manual_span":
        spans.append(
            {
                "start_ms": int(params["start_ms"]),
                "end_ms": int(params["end_ms"]),
                "anchor_words": [],
                "anchor_word_indices": [],
            }
        )
    elif mode == "anchor_word":
        anchor = clean_text(params.get("anchor_word"))
        for idx, word in enumerate(words):
            if clean_text(word.get("text")) == anchor:
                spans.append(span_from_indices(words, [idx]))
                break
    elif mode == "filler_words":
        fillers = {clean_text(x) for x in params.get("filler_word_list", []) if clean_text(x)}
        for idx, word in enumerate(words):
            if clean_text(word.get("text")) in fillers:
                span = span_from_indices(words, [idx])
                if span:
                    spans.append(span)
    elif mode == "regex":
        pattern = re.compile(params["regex"])
        flat_text, ranges = word_char_ranges(words)
        for match in pattern.finditer(flat_text):
            indices = [
                idx
                for idx, (start, end) in enumerate(ranges)
                if start < match.end() and end > match.start()
            ]
            span = span_from_indices(words, indices)
            if span:
                spans.append(span)
    else:
        raise ValueError(f"unsupported selection mode: {mode}")

    return {
        "audio_id": alignment.get("audio_id"),
        "edit_spans": [s for s in spans if s],
        "selection_mode": mode,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment", required=True)
    ap.add_argument("--target-spec", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = select_spans(load_json(Path(args.alignment)), load_json(Path(args.target_spec)))
    write_json(Path(args.output), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
