"""扫上游 raw jsonl 源（kxhuang 等）→ raw_source.parquet + duration_cache.parquet。

用法：
    python app/raw_scanner.py \\
        --add instruction_0.1_enzh:zh:/inspire/hdd/.../kxhuang/instructtts_data/instruction_0.1_enzh/zh \\
        --add instruction_0.1_enzh:en:/inspire/hdd/.../kxhuang/instructtts_data/instruction_0.1_enzh/en
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader import DATA_DIR, RAW_SOURCE_PARQUET, DURATION_CACHE_PARQUET


def scan_one_dir(source: str, lang: str, root: Path):
    """扫一个目录下所有 split_*.jsonl，返回 (rows_summary, durations)"""
    rows_summary = []
    durations = []  # list of (path, duration)
    splits = sorted(root.glob("split_*.jsonl"))
    print(f"[raw] {source}/{lang}: 扫 {len(splits)} 个 split_*.jsonl from {root}")
    for jsonl in splits:
        n = 0
        total_dur = 0.0
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                n += 1
                d = r.get("duration")
                if d is not None:
                    total_dur += float(d)
                lp = r.get("local_path")
                if lp and d is not None:
                    durations.append((lp, float(d)))
        rows_summary.append({
            "source": source,
            "language": lang,
            "split_jsonl": jsonl.name,
            "n_rows": n,
            "total_hours": total_dur / 3600.0,
        })
        print(f"  {jsonl.name}: {n:>6} rows  {total_dur/3600:.2f} hr")
    return rows_summary, durations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--add", action="append", default=[],
        help="格式：<source>:<language>:<dir>。可重复传多次。"
    )
    ap.add_argument("--raw-out", default=str(RAW_SOURCE_PARQUET),
                    help=f"raw source 输出，默认 {RAW_SOURCE_PARQUET}")
    ap.add_argument("--dur-out", default=str(DURATION_CACHE_PARQUET),
                    help=f"duration_cache 输出，默认 {DURATION_CACHE_PARQUET}")
    args = ap.parse_args()

    if not args.add:
        print("用 --add <source>:<lang>:<dir> 添加源。例：")
        print("  --add instruction_0.1_enzh:zh:/inspire/hdd/.../kxhuang/.../instruction_0.1_enzh/zh")
        sys.exit(1)

    raw_rows, all_dur = [], []
    for spec in args.add:
        try:
            source, lang, d = spec.split(":", 2)
        except ValueError:
            print(f"[raw] 跳过无效格式: {spec}", file=sys.stderr)
            continue
        root = Path(d)
        if not root.is_dir():
            print(f"[raw] {root} 不存在，跳过", file=sys.stderr)
            continue
        rows, durs = scan_one_dir(source, lang, root)
        raw_rows.extend(rows)
        all_dur.extend(durs)

    raw_path = Path(args.raw_out)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_parquet(raw_path, index=False)
    print(f"\n[raw] 写 raw source 索引 {len(raw_rows)} 行 → {raw_path}")

    dur_df = pd.DataFrame(all_dur, columns=["path", "duration"]).drop_duplicates("path")
    dur_path = Path(args.dur_out)
    dur_df.to_parquet(dur_path, index=False)
    print(f"[raw] 写 duration_cache {len(dur_df):,} 条 → {dur_path}")


if __name__ == "__main__":
    main()
