#!/usr/bin/env python
"""Add emotion/SenseVoice/DNSMOS metrics for pair-generated audio.

The regular 04 step evaluates original/ref/editx audio. Optional pair types
such as I/J create new target audio later, so QC cannot populate target-side
emotion and DNSMOS fields unless those files are evaluated too.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _emotion_lookup import EmotionTable
from _utils import load_config


DEFAULT_EMOTION_PY = "/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/emotion/bin/python"
DEFAULT_DNSMOS_PY = "/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/moss_ttsd_sglang/bin/python"
REQUIRED_SUMMARY_FIELDS = (
    "top1_label",
    "top1_prob",
    "P_neutral",
    "sv_label",
    "dnsmos_ovrl",
    "dnsmos_sig",
    "dnsmos_bak",
)


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "pair_audio"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def union_fieldnames(*parts: list[str], rows: list[dict[str, Any]] | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for fields in parts:
        for field in fields:
            if field not in seen:
                seen.add(field)
                out.append(field)
    for row in rows or []:
        for field in row.keys():
            if field not in seen:
                seen.add(field)
                out.append(field)
    return out


def load_emotion_table(emotion_dir: Path) -> EmotionTable:
    table = EmotionTable()
    table.load_csv(emotion_dir / "per_file_dual.csv")
    table.load_per_pair_for_src(emotion_dir / "per_pair.csv")
    table.load_all_link_mappings(emotion_dir)
    return table


def has_complete_metrics(table: EmotionTable, audio_path: str) -> bool:
    summary = table.emotion_summary(audio_path)
    return all(summary.get(key) not in (None, "") for key in REQUIRED_SUMMARY_FIELDS)


def resolve_pair_files(pair_root: Path, pair_types: list[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    if len(pair_types) == 1 and pair_types[0].lower() == "all":
        candidates = sorted((pair_root / "pairs" / "scored").glob("*.jsonl"))
        if not candidates:
            candidates = sorted((pair_root / "pairs").glob("*.jsonl"))
        for path in candidates:
            if path.name.endswith("_bakfilt.jsonl"):
                continue
            files.append((path.stem, path))
        return files

    for pair_type in pair_types:
        candidates = [
            pair_root / "pairs" / "scored" / f"{pair_type}.jsonl",
            pair_root / "pairs" / "filtered" / f"{pair_type}.jsonl",
            pair_root / "pairs" / f"{pair_type}.jsonl",
        ]
        for path in candidates:
            if path.exists() and path not in seen:
                seen.add(path)
                files.append((pair_type, path))
                break
        else:
            print(f"[04c] [warn] missing pair file for {pair_type}", file=sys.stderr)
    return files


def collect_missing_audio(
    pair_files: list[tuple[str, Path]],
    table: EmotionTable | None,
    sides: set[str],
    force: bool,
    limit: int,
) -> dict[str, dict[str, Any]]:
    wanted: dict[str, dict[str, Any]] = {}
    side_keys = []
    if "reference" in sides:
        side_keys.append(("reference", "reference_audio"))
    if "target" in sides:
        side_keys.append(("target", "target_audio"))

    for pair_type, path in pair_files:
        for row in iter_jsonl(path):
            for side, key in side_keys:
                audio = row.get(key)
                if not audio:
                    continue
                if not force and table is not None and has_complete_metrics(table, audio):
                    continue
                rec = wanted.setdefault(
                    str(audio),
                    {"audio": str(audio), "pair_types": set(), "sides": set(), "pair_ids": []},
                )
                rec["pair_types"].add(pair_type)
                rec["sides"].add(side)
                if row.get("pair_id") and len(rec["pair_ids"]) < 8:
                    rec["pair_ids"].append(row.get("pair_id"))
                if limit > 0 and len(wanted) >= limit:
                    return wanted
    return wanted


def ensure_audio_for_scoring(source: str, link_dir: Path) -> tuple[Path | None, str | None]:
    src = Path(source)
    if not src.exists():
        return None, f"missing_source:{source}"
    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:20]
    dst = link_dir / f"{digest}.wav"
    if src.suffix.lower() == ".wav":
        if os.path.lexists(dst):
            try:
                if os.path.realpath(dst) == os.path.realpath(src):
                    return dst, None
            except OSError:
                pass
            dst.unlink()
        os.symlink(str(src), dst)
        return dst, None

    if dst.exists():
        return dst, None
    try:
        data, sr = sf.read(str(src))
        sf.write(str(dst), data, sr)
    except Exception as exc:
        return None, f"transcode_failed:{source}:{exc}"
    return dst, None


def run_command(cmd: list[str], env: dict[str, str], label: str) -> None:
    print(f"[04c] {label}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def build_per_file_for_sensevoice(run_dir: Path, group: str, link_to_source: dict[str, str]) -> Path:
    src = run_dir / group / "per_file.csv"
    rows, fields = read_csv(src)
    if not rows:
        raise SystemExit(f"[04c] score_neutrality produced no rows: {src}")
    for row in rows:
        wav = row.get("wav") or row.get("path")
        row["group"] = group
        row["wav"] = wav
        row["source_audio"] = link_to_source.get(wav, "")
        row["metric_source"] = "pair_audio_metrics"
    out = run_dir / "per_file.csv"
    fieldnames = union_fieldnames(fields, ["group", "wav", "source_audio", "metric_source"], rows=rows)
    write_csv(out, rows, fieldnames)
    return out


def add_source_columns(dual_csv: Path, link_to_source: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows, fields = read_csv(dual_csv)
    for row in rows:
        wav = row.get("wav") or row.get("path")
        row["source_audio"] = row.get("source_audio") or link_to_source.get(wav, "")
        row["metric_source"] = row.get("metric_source") or "pair_audio_metrics"
    fields = union_fieldnames(fields, ["source_audio", "metric_source"], rows=rows)
    write_csv(dual_csv, rows, fields)
    return rows, fields


def audio_keys(row: dict[str, Any], resolve_realpaths: bool = False) -> set[str]:
    keys: set[str] = set()
    for name in ("source_audio", "audio_path", "target_audio", "wav", "path"):
        value = row.get(name)
        if not value:
            continue
        keys.add(str(value))
        if resolve_realpaths:
            try:
                keys.add(os.path.realpath(str(value)))
            except OSError:
                pass
    return keys


def merge_dual_into_main(
    main_csv: Path,
    extra_rows: list[dict[str, Any]],
    extra_fields: list[str],
    resolve_realpaths: bool = False,
) -> tuple[int, int]:
    main_rows, main_fields = read_csv(main_csv)
    extra_keys: set[str] = set()
    for row in extra_rows:
        extra_keys.update(audio_keys(row, resolve_realpaths=resolve_realpaths))

    kept = [row for row in main_rows if audio_keys(row, resolve_realpaths=resolve_realpaths).isdisjoint(extra_keys)]
    merged = kept + extra_rows
    fields = union_fieldnames(main_fields, extra_fields, rows=merged)
    write_csv(main_csv, merged, fields)
    return len(main_rows) - len(kept), len(merged)


def write_mapping(link_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["link_wav", "audio_path", "pair_types", "sides", "pair_ids"]
    write_csv(link_dir / "_mapping.csv", rows, fields)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True, help=".../pair_outputs/<lang>/<split>")
    parser.add_argument("--config", default=None)
    parser.add_argument("--pair-type", default="I,J_fast,J_slow")
    parser.add_argument("--sides", default="both", choices=["both", "reference", "target"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--emotion-py", default=os.environ.get("EMOTION_PY_BIN", DEFAULT_EMOTION_PY))
    parser.add_argument("--dnsmos-py", default=os.environ.get("DNSMOS_PY", DEFAULT_DNSMOS_PY))
    parser.add_argument("--emotion-eval-root", default=os.environ.get("EMOTION_EVAL_ROOT"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--skip-dnsmos", action="store_true")
    parser.add_argument("--dnsmos-workers", type=int, default=int(os.environ.get("DNSMOS_WORKERS", "1")))
    parser.add_argument("--onnx", default=None)
    args = parser.parse_args()

    pair_root = Path(args.pair_root).expanduser().resolve()
    emotion_dir = pair_root / "emotion"
    emotion_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config) if args.config else {"paths": {}}
    eval_root = Path(
        args.emotion_eval_root
        or os.environ.get("EMOTION_EVAL_ROOT", "")
        or cfg.get("paths", {}).get("emotion_eval_root", "")
        or "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/code/emotion_eval"
    )
    score_script = eval_root / "scripts" / "score_neutrality.py"
    sensevoice_script = eval_root / "scripts" / "sensevoice_score.py"
    if not score_script.exists() or not sensevoice_script.exists():
        raise SystemExit(f"[04c] missing emotion_eval scripts under {eval_root}")

    pair_types = parse_csv_list(args.pair_type)
    side_set = {"reference", "target"} if args.sides == "both" else {args.sides}
    table = None if args.force else load_emotion_table(emotion_dir)
    pair_files = resolve_pair_files(pair_root, pair_types)
    wanted = collect_missing_audio(pair_files, table, side_set, args.force, args.limit)

    print(f"[04c] pair_root={pair_root}")
    print(f"[04c] pair_types={','.join(pair_types)} sides={','.join(sorted(side_set))} force={int(args.force)}")
    if args.force:
        print("[04c] force=1, skip loading existing emotion table and rescore selected pair audio")
    print(f"[04c] missing_or_incomplete_audio={len(wanted)}")
    if not wanted:
        return 0
    for sample in list(wanted)[:5]:
        print(f"[04c] sample_missing={sample}")
    if args.dry_run:
        return 0

    slug = slugify(args.run_name or "_".join(pair_types))
    group = f"pair_qc_{slug}"
    link_dir = emotion_dir / f"_links_{group}"
    run_dir = emotion_dir / "_pair_metric_runs" / group
    link_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    mapping_rows: list[dict[str, Any]] = []
    link_to_source: dict[str, str] = {}
    materialize_errors: list[str] = []
    for source, rec in sorted(wanted.items()):
        link, error = ensure_audio_for_scoring(source, link_dir)
        if error:
            materialize_errors.append(error)
            continue
        assert link is not None
        link_s = str(link)
        link_to_source[link_s] = source
        mapping_rows.append(
            {
                "link_wav": link_s,
                "audio_path": source,
                "pair_types": "|".join(sorted(rec["pair_types"])),
                "sides": "|".join(sorted(rec["sides"])),
                "pair_ids": "|".join(rec["pair_ids"]),
            }
        )
    write_mapping(link_dir, mapping_rows)
    if materialize_errors:
        for err in materialize_errors[:20]:
            print(f"[04c] [warn] {err}", file=sys.stderr)
    if not mapping_rows:
        raise SystemExit("[04c] no scorable audio after materialization")

    env = os.environ.copy()
    env.setdefault("MODELSCOPE_CACHE", str(eval_root / "model_cache" / "modelscope"))
    env.setdefault("HF_HOME", str(eval_root / "model_cache" / "hf"))
    run_command(
        [
            args.emotion_py,
            str(score_script),
            "-i",
            f"{group}={link_dir}",
            "--out",
            str(run_dir),
            "--device",
            args.device,
        ],
        env,
        "emotion2vec",
    )
    build_per_file_for_sensevoice(run_dir, group, link_to_source)
    run_command(
        [args.emotion_py, str(sensevoice_script), "--run-dir", str(run_dir), "--device", args.device],
        env,
        "sensevoice",
    )

    dual_csv = run_dir / "per_file_dual.csv"
    if not dual_csv.exists():
        raise SystemExit(f"[04c] missing {dual_csv}")
    if not args.skip_dnsmos:
        dnsmos_py = Path(args.dnsmos_py)
        if not dnsmos_py.exists():
            raise SystemExit(f"[04c] missing DNSMOS python: {dnsmos_py}")
        cmd = [str(dnsmos_py), str(Path(__file__).resolve().parent / "04b_add_dnsmos.py"), "--csv", str(dual_csv)]
        if args.dnsmos_workers > 1:
            cmd.extend(["--workers", str(args.dnsmos_workers)])
        if args.onnx:
            cmd.extend(["--onnx", args.onnx])
        run_command(cmd, env, "dnsmos")

    extra_rows, extra_fields = add_source_columns(dual_csv, link_to_source)
    replaced, total = merge_dual_into_main(emotion_dir / "per_file_dual.csv", extra_rows, extra_fields)

    summary = {
        "pair_root": str(pair_root),
        "pair_types": pair_types,
        "sides": sorted(side_set),
        "missing_or_incomplete_audio": len(wanted),
        "scored_audio": len(extra_rows),
        "materialize_errors": materialize_errors,
        "replaced_existing_rows": replaced,
        "main_per_file_dual_rows": total,
        "link_dir": str(link_dir),
        "run_dir": str(run_dir),
    }
    with (run_dir / "pair_audio_metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[04c] done scored={len(extra_rows)} replaced={replaced} main_rows={total}")
    print(f"[04c] summary={run_dir / 'pair_audio_metrics_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
