#!/usr/bin/env python3
"""Summarize vcdata baseline metrics from pair_construction outputs.

Example:
  python scripts/summarize_vcdata_baseline.py \
    --run-root outputs/delivery_filter_10k_langsplit_qz_20260604_run02 \
    --batch-root .qz_vcdata_editx_batches/delivery10k-langsplit-0604-run02
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


START_RE = re.compile(r"\[(.*?)\] (zh|en): start vcdata")
PUBLISH_RE = re.compile(r"\[(.*?)\] (zh|en): vcdata verified and published")
GROUP_RE = re.compile(r"-g(\d+)$")


@dataclass
class GpuStats:
    sample_count: int
    avg_util_pct: float | None
    p90_util_pct: float | None
    peak_mem_gib: float | None
    avg_power_w: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="pair_construction run root")
    parser.add_argument("--batch-root", required=True, help="matching .qz_vcdata_editx_batches/<batch_id> root")
    parser.add_argument("--job", action="append", default=[], help="optional job name filter; repeatable")
    parser.add_argument("--format", choices=("markdown", "tsv"), default="markdown")
    return parser.parse_args()


def parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def count_manifest_records(split_dir: Path) -> int:
    total = 0
    for manifest_path in split_dir.glob("manifest_shard*.jsonl"):
        with manifest_path.open("r", encoding="utf-8") as handle:
            total += sum(1 for _ in handle)
    return total


def load_plan_rows(plan_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with plan_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                rows.append(parts)
    return rows


def load_gpu_stats(csv_path: Path) -> GpuStats | None:
    if not csv_path.exists():
        return None
    util_values: list[float] = []
    mem_values_gib: list[float] = []
    power_values: list[float] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                util_values.append(float(row["utilization_gpu_pct"]))
                mem_values_gib.append(float(row["memory_used_mib"]) / 1024.0)
                power_raw = row.get("power_draw_w", "").strip()
                if power_raw:
                    power_values.append(float(power_raw))
            except (KeyError, ValueError):
                continue
    if not util_values:
        return GpuStats(0, None, None, None, None)
    return GpuStats(
        sample_count=len(util_values),
        avg_util_pct=sum(util_values) / len(util_values),
        p90_util_pct=percentile(util_values, 0.90),
        peak_mem_gib=max(mem_values_gib) if mem_values_gib else None,
        avg_power_w=(sum(power_values) / len(power_values)) if power_values else None,
    )


def load_attempt_completion(tsv_path: Path) -> dict[str, str] | None:
    if not tsv_path.exists():
        return None
    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if not rows:
        return None
    for row in reversed(rows):
        if row.get("pending_cases_after") == "0":
            return row
    return rows[-1]


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def summarize(run_root: Path, batch_root: Path, job_filters: set[str]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    logs_dir = run_root / "logs"
    for log_path in sorted(logs_dir.glob("*.log")):
        job_name = log_path.stem
        if job_filters and job_name not in job_filters:
            continue
        group_match = GROUP_RE.search(job_name)
        if not group_match:
            continue
        group_id = group_match.group(1)
        plan_path = batch_root / f"group_{group_id}" / "plan.tsv"
        if not plan_path.exists():
            continue
        plan_rows = load_plan_rows(plan_path)
        lines = log_path.read_text(errors="ignore").splitlines()
        starts: list[tuple[int, str, str]] = []
        publishes: list[tuple[int, str, str]] = []
        for idx, line in enumerate(lines):
            start_match = START_RE.search(line)
            if start_match:
                starts.append((idx, start_match.group(2), start_match.group(1)))
            publish_match = PUBLISH_RE.search(line)
            if publish_match:
                publishes.append((idx, publish_match.group(2), publish_match.group(1)))
        for publish_idx, lang, publish_ts in publishes:
            start_candidates = [
                ts for start_idx, start_lang, ts in starts
                if start_lang == lang and start_idx < publish_idx
            ]
            if not start_candidates:
                continue
            start_ts = start_candidates[-1]
            expected_cases = 0
            actual_cases = 0
            split_count = 0
            for parts in plan_rows:
                if parts[0] != lang:
                    continue
                split_name = parts[1].removesuffix(".jsonl")
                expected_cases += int(parts[3])
                split_count += 1
                actual_cases += count_manifest_records(run_root / "vcdata" / lang / split_name)
            attempt_completion = load_attempt_completion(
                run_root / "vcdata_job_runs" / lang / job_name / "vcdata_attempt_metrics.tsv"
            )
            case_source = "published_manifests"
            if attempt_completion is not None:
                start_ts = attempt_completion["start_ts"]
                publish_ts = attempt_completion["end_ts"]
                actual_cases = int(attempt_completion["pending_cases_before"])
                case_source = "attempt_pending_cases_before"
            elapsed_hours = (parse_utc(publish_ts) - parse_utc(start_ts)).total_seconds() / 3600.0
            gpu_stats = load_gpu_stats(run_root / "vcdata_job_runs" / lang / job_name / "gpu_metrics.csv")
            rows.append(
                {
                    "job": job_name,
                    "lang": lang,
                    "group": group_id,
                    "split_count": split_count,
                    "expected_cases": expected_cases,
                    "actual_cases": actual_cases,
                    "case_source": case_source,
                    "start_ts": start_ts,
                    "publish_ts": publish_ts,
                    "elapsed_hours": elapsed_hours,
                    "cases_per_hour": (actual_cases / elapsed_hours) if elapsed_hours > 0 else 0.0,
                    "cases_per_gpu_hour": (actual_cases / elapsed_hours / 8.0) if elapsed_hours > 0 else 0.0,
                    "gpu_samples": gpu_stats.sample_count if gpu_stats else 0,
                    "avg_gpu_util_pct": gpu_stats.avg_util_pct if gpu_stats else None,
                    "p90_gpu_util_pct": gpu_stats.p90_util_pct if gpu_stats else None,
                    "peak_gpu_mem_gib": gpu_stats.peak_mem_gib if gpu_stats else None,
                    "avg_power_w": gpu_stats.avg_power_w if gpu_stats else None,
                }
            )
    return rows


def render_markdown(rows: list[dict[str, str | int | float]]) -> str:
    header = [
        "job", "lang", "splits", "actual_cases", "hours", "cases/hour",
        "cases/gpu_hour", "case_source", "gpu_samples", "avg_gpu_util", "p90_gpu_util", "peak_gpu_mem_gib",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["job"]),
                    str(row["lang"]),
                    str(row["split_count"]),
                    str(row["actual_cases"]),
                    fmt_float(float(row["elapsed_hours"]), 2),
                    fmt_float(float(row["cases_per_hour"]), 2),
                    fmt_float(float(row["cases_per_gpu_hour"]), 2),
                    str(row["case_source"]),
                    str(row["gpu_samples"]),
                    fmt_float(row["avg_gpu_util_pct"]),
                    fmt_float(row["p90_gpu_util_pct"]),
                    fmt_float(row["peak_gpu_mem_gib"]),
                ]
            )
            + " |"
        )
    if rows:
        total_cases = sum(int(row["actual_cases"]) for row in rows)
        total_hours = sum(float(row["elapsed_hours"]) for row in rows)
        weighted_cases_per_hour = total_cases / total_hours if total_hours else 0.0
        weighted_cases_per_gpu_hour = weighted_cases_per_hour / 8.0 if total_hours else 0.0
        util_rows = [row for row in rows if row["avg_gpu_util_pct"] is not None]
        avg_util = (
            sum(float(row["avg_gpu_util_pct"]) for row in util_rows) / len(util_rows)
            if util_rows else None
        )
        p90_util = (
            sum(float(row["p90_gpu_util_pct"]) for row in util_rows) / len(util_rows)
            if util_rows else None
        )
        peak_mem = (
            max(float(row["peak_gpu_mem_gib"]) for row in util_rows)
            if util_rows else None
        )
        lines.append(
            "| TOTAL | - | "
            + " | ".join(
                [
                    str(sum(int(row["split_count"]) for row in rows)),
                    str(total_cases),
                    fmt_float(total_hours, 2),
                    fmt_float(weighted_cases_per_hour, 2),
                    fmt_float(weighted_cases_per_gpu_hour, 2),
                    "-",
                    str(sum(int(row["gpu_samples"]) for row in rows)),
                    fmt_float(avg_util),
                    fmt_float(p90_util),
                    fmt_float(peak_mem),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_tsv(rows: list[dict[str, str | int | float]]) -> str:
    header = [
        "job", "lang", "group", "split_count", "expected_cases", "actual_cases",
        "case_source", "start_ts", "publish_ts", "elapsed_hours", "cases_per_hour",
        "cases_per_gpu_hour", "gpu_samples", "avg_gpu_util_pct",
        "p90_gpu_util_pct", "peak_gpu_mem_gib", "avg_power_w",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row[key]) for key in header))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    batch_root = Path(args.batch_root).resolve()
    rows = summarize(run_root, batch_root, set(args.job))
    if args.format == "markdown":
        print(render_markdown(rows))
    else:
        print(render_tsv(rows))


if __name__ == "__main__":
    main()
