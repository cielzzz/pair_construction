from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("PROSODY_CONFIG") or PROJECT_ROOT / "configs" / "prosody_routes.yaml")
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return cfg


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "default"


def first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def normalize_source_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    audio = first_present(
        row,
        (
            "ref_audio",
            "ref_audio_path",
            "local_path",
            "audio",
            "audio_path",
            "original_audio",
            "original_audio_path",
        ),
    )
    text = first_present(row, ("ref_text", "text", "mtd_transcript", "original_text", "transcript"))
    if not audio or not text:
        return None
    return {
        "source_index": index,
        "source_row_index": row.get("original_idx", row.get("source_row_index", index)),
        "audio": str(audio),
        "text": str(text),
        "language": row.get("language") or ("zh" if re.search(r"[\u4e00-\u9fff]", str(text)) else "en"),
        "duration": first_present(row, ("duration", "duration_sec", "audio_duration", "seconds")),
        "speaker_id": first_present(row, ("speaker_id", "spk_id", "speaker", "speaker_name")),
        "raw": row,
    }


def make_pair_id(run_name: str, pair_type: str, idx: int) -> str:
    return f"{run_name}:{pair_type}:{idx:06d}"


def text_units(text: str | None) -> int:
    if not text:
        return 0
    if re.search(r"[\u4e00-\u9fff]", text):
        return len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))
    return len(re.findall(r"[A-Za-z0-9']+", text))
