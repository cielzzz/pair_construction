#!/usr/bin/env python
"""Export only QC-rejected rows for inspection.

Expected usage:
- run qc_pairs.py on raw/scored pair inputs
- export only the rows that QC rejects

Outputs are written to:
  <pair_root>/quality_gate/rejections/
    summary.json
    <type>_qc_rejected.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def discover_pair_types(qc_dir: Path, requested: str) -> list[str]:
    if requested != "all":
        return [requested]
    pair_types: set[str] = set()
    for path in qc_dir.glob("*_qc.jsonl"):
        pair_types.add(path.stem[:-len("_qc")])
    if not pair_types:
        for path in qc_dir.glob("*__qc.jsonl"):
            pair_types.add(path.stem[:-len("__qc")])
    return sorted(pair_types)


def normalize_qc_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("hard_pass", 1)
    out.setdefault("qc_pass", out["hard_pass"])
    out.setdefault("hard_fail_flags", [])
    out.setdefault("soft_fail_flags", [])
    out.setdefault("flags", out["hard_fail_flags"] or out["soft_fail_flags"])
    return out


def cleanup_old_exports(out_dir: Path, pair_type: str) -> None:
    stale_names = [
        f"{pair_type}__dropped_by_filter.jsonl",
        f"{pair_type}__hard_fail.jsonl",
        f"{pair_type}__qc_fail.jsonl",
        f"{pair_type}_dropped_by_filter.jsonl",
        f"{pair_type}_hard_fail.jsonl",
        f"{pair_type}_qc_fail.jsonl",
        f"{pair_type}_filtered_out.jsonl",
        f"{pair_type}_qc_rejected.jsonl",
    ]
    for name in stale_names:
        path = out_dir / name
        if path.exists():
            path.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-root", required=True, help=".../pair_outputs/<lang>/<split>")
    ap.add_argument("--pair-type", default="all", help="A/B/.../H2/all")
    args = ap.parse_args()

    pair_root = Path(args.pair_root)
    qc_dir = pair_root / "quality_gate"
    out_dir = qc_dir / "rejections"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_types = discover_pair_types(qc_dir, args.pair_type)
    if not pair_types:
        raise SystemExit(f"no qc jsonl found under {qc_dir}")

    summary: dict[str, Any] = {
        "pair_root": str(pair_root),
        "pair_type": args.pair_type,
        "files": {},
    }

    for pair_type in pair_types:
        qc_path = qc_dir / f"{pair_type}_qc.jsonl"
        if not qc_path.exists():
            legacy_qc_path = qc_dir / f"{pair_type}__qc.jsonl"
            qc_path = legacy_qc_path if legacy_qc_path.exists() else qc_path
        if not qc_path.exists():
            continue
        cleanup_old_exports(out_dir, pair_type)

        qc_rows = [normalize_qc_row(row) for row in iter_jsonl(qc_path)]
        qc_rejected_rows: list[dict[str, Any]] = []
        for qc_row in qc_rows:
            if int(qc_row["qc_pass"]) != 1:
                qc_rejected_rows.append({
                    **qc_row,
                    "_rejection_stage": "quality_gate",
                    "_rejection_reason": "qc_rejected",
                    "_qc_reject_basis": "hard_fail" if int(qc_row["hard_pass"]) != 1 else "soft_gate",
                })

        qc_rejected_path = out_dir / f"{pair_type}_qc_rejected.jsonl"
        write_jsonl(qc_rejected_path, qc_rejected_rows)

        summary["files"][pair_type] = {
            "qc_input_count": len(qc_rows),
            "qc_kept": len(qc_rows) - len(qc_rejected_rows),
            "qc_rejected": len(qc_rejected_rows),
            "qc_rejected_jsonl": str(qc_rejected_path),
        }
        print(f"{pair_type}: qc_input={len(qc_rows)} qc_rejected={len(qc_rejected_rows)}")

    write_json(out_dir / "summary.json", summary)
    print(f"[export] wrote rejection views to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
