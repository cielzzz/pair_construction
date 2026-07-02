#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


MANIFEST_RE = re.compile(r"^(filtered_manifest_)(\d{4})(?:_(zh|en))?\.jsonl$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare zh/en language-specific views for delivery_filter_10k."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--zh-dir", required=True, type=Path)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--combined-dir", type=Path)
    parser.add_argument("--mixed-index", type=int, default=197)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_unlink(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        raise IsADirectoryError(f"refuse to remove directory: {path}")
    path.unlink()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_split_manifest(
    src_path: Path, zh_path: Path, en_path: Path, force: bool
) -> Counter[str]:
    counts: Counter[str] = Counter()
    if force:
        safe_unlink(zh_path)
        safe_unlink(en_path)
    ensure_parent(zh_path)
    ensure_parent(en_path)

    with src_path.open("r", encoding="utf-8") as src, \
        zh_path.open("w", encoding="utf-8") as zh_out, \
        en_path.open("w", encoding="utf-8") as en_out:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            lang = item.get("language")
            if lang == "zh":
                zh_out.write(line)
            elif lang == "en":
                en_out.write(line)
            else:
                raise ValueError(
                    f"{src_path}:{line_no} unexpected language={lang!r}; expected 'zh' or 'en'"
                )
            counts[lang] += 1
    return counts


def link_manifest(src_path: Path, dst_path: Path, force: bool) -> None:
    if force:
        safe_unlink(dst_path)
    ensure_parent(dst_path)
    if dst_path.exists() or dst_path.is_symlink():
        return
    os.symlink(src_path, dst_path)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    zh_dir = args.zh_dir.resolve()
    en_dir = args.en_dir.resolve()
    combined_dir = args.combined_dir.resolve() if args.combined_dir else None

    zh_dir.mkdir(parents=True, exist_ok=True)
    en_dir.mkdir(parents=True, exist_ok=True)
    if combined_dir:
        combined_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, object] = {
        "mixed_index": args.mixed_index,
        "zh_linked": 0,
        "en_linked": 0,
        "combined_linked": 0,
        "split_counts": {},
    }

    manifest_paths = sorted(input_dir.glob("filtered_manifest_*.jsonl"))
    if not manifest_paths:
        raise FileNotFoundError(f"no manifests found under {input_dir}")

    for src_path in manifest_paths:
        match = MANIFEST_RE.match(src_path.name)
        if not match:
            continue
        prefix, idx_str, explicit_lang = match.groups()
        idx = int(idx_str)

        if explicit_lang == "zh":
            link_manifest(src_path, zh_dir / src_path.name, args.force)
            if combined_dir:
                link_manifest(src_path, combined_dir / src_path.name, args.force)
                stats["combined_linked"] = int(stats["combined_linked"]) + 1
            stats["zh_linked"] = int(stats["zh_linked"]) + 1
            continue
        if explicit_lang == "en":
            link_manifest(src_path, en_dir / src_path.name, args.force)
            if combined_dir:
                link_manifest(src_path, combined_dir / src_path.name, args.force)
                stats["combined_linked"] = int(stats["combined_linked"]) + 1
            stats["en_linked"] = int(stats["en_linked"]) + 1
            continue

        if idx < args.mixed_index:
            link_manifest(src_path, zh_dir / src_path.name, args.force)
            if combined_dir:
                link_manifest(src_path, combined_dir / src_path.name, args.force)
                stats["combined_linked"] = int(stats["combined_linked"]) + 1
            stats["zh_linked"] = int(stats["zh_linked"]) + 1
            continue
        if idx > args.mixed_index:
            link_manifest(src_path, en_dir / src_path.name, args.force)
            if combined_dir:
                link_manifest(src_path, combined_dir / src_path.name, args.force)
                stats["combined_linked"] = int(stats["combined_linked"]) + 1
            stats["en_linked"] = int(stats["en_linked"]) + 1
            continue

        zh_path = zh_dir / f"{prefix}{idx_str}_zh.jsonl"
        en_path = en_dir / f"{prefix}{idx_str}_en.jsonl"
        counts = write_split_manifest(src_path, zh_path, en_path, args.force)
        if combined_dir:
            link_manifest(zh_path, combined_dir / zh_path.name, args.force)
            link_manifest(en_path, combined_dir / en_path.name, args.force)
            stats["combined_linked"] = int(stats["combined_linked"]) + 2
        stats["split_counts"] = {
            "source": src_path.name,
            "zh_output": zh_path.name,
            "en_output": en_path.name,
            "zh": counts.get("zh", 0),
            "en": counts.get("en", 0),
        }

    summary_path = input_dir / "language_split_summary.json"
    if args.force:
        safe_unlink(summary_path)
    summary_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
