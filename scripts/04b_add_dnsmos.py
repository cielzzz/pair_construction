#!/usr/bin/env python
"""04b: 给 emotion/per_file_dual.csv 加 dnsmos_ovrl / dnsmos_sig / dnsmos_bak 三列

读所有 wav 行，跑 Microsoft DNSMOS_v4 (sig_bak_ovr.onnx)，更新 csv。

为了对齐 vcdata 原始 jsonl 的 dnsmos 字段（同一 Microsoft 模型 raw 分数），
不做 polynomial calibration。

用法：
  python 04b_add_dnsmos.py --split split_demo
"""
from __future__ import annotations
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, emotion_path
from _dnsmos import compute_dnsmos


def _compute_one(args: tuple[int, str, str | None]) -> tuple[int, dict | None]:
    idx, wav, onnx_path = args
    return idx, compute_dnsmos(wav, onnx_path=onnx_path)


def write_csv_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--csv", default=None, help="直接给指定 per_file_dual.csv 补 DNSMOS")
    ap.add_argument("--onnx", default=None, help="覆盖默认 sig_bak_ovr.onnx 路径")
    ap.add_argument("--flush-every", type=int, default=int(os.environ.get("DNSMOS_FLUSH_EVERY", "500")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("DNSMOS_WORKERS", "1")))
    ap.add_argument("--chunksize", type=int, default=int(os.environ.get("DNSMOS_CHUNKSIZE", "8")))
    args = ap.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        if not args.split:
            sys.exit("[04b] 需要 --split 或 --csv")
        cfg = load_config(args.config)
        csv_path = emotion_path(cfg, args.split, "per_file_dual.csv")
    if not csv_path.exists():
        sys.exit(f"[04b] 缺 {csv_path}（先跑 04）")

    # 读现有
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"[04b] {csv_path} 为空")

    fieldnames = list(rows[0].keys())
    for new in ("dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak"):
        if new not in fieldnames:
            fieldnames.append(new)

    # 已加 dnsmos 跳过
    if "dnsmos_ovrl" in rows[0]:
        already = sum(1 for r in rows if r.get("dnsmos_ovrl") not in ("", None))
        if already >= len(rows):
            print(f"[04b] 所有 {len(rows)} 行已有 dnsmos，跳过")
            return
        print(f"[04b] {already}/{len(rows)} 已有 dnsmos，续算剩余 {len(rows)-already}")

    tasks: list[tuple[int, str, str | None]] = []
    missing_wav = 0
    for i, r in enumerate(rows):
        wav = r.get("wav") or r.get("path")
        if not wav:
            missing_wav += 1
            continue
        if r.get("dnsmos_ovrl") not in ("", None):
            continue
        tasks.append((i, wav, args.onnx))
    if missing_wav:
        print(f"[04b] {missing_wav} rows missing wav/path")
    if not tasks:
        write_csv_atomic(csv_path, rows, fieldnames)
        print(f"[04b] no missing dnsmos rows: {csv_path}")
        return

    workers = max(1, args.workers)
    chunksize = max(1, args.chunksize)
    if workers > 1:
        os.environ.setdefault("DNSMOS_ORT_THREADS", "1")
    print(f"[04b] scoring {len(tasks)} rows with workers={workers} chunksize={chunksize}")

    def apply_result(idx: int, d: dict | None) -> bool:
        if d is None:
            rows[idx]["dnsmos_ovrl"] = ""
            rows[idx]["dnsmos_sig"] = ""
            rows[idx]["dnsmos_bak"] = ""
            return False
        rows[idx]["dnsmos_ovrl"] = d["OVRL"]
        rows[idx]["dnsmos_sig"] = d["SIG"]
        rows[idx]["dnsmos_bak"] = d["BAK"]
        return True

    t0 = time.time()
    ok = fail = 0
    if workers == 1:
        iterator = map(_compute_one, tasks)
        for done, (idx, d) in enumerate(iterator, 1):
            if apply_result(idx, d):
                ok += 1
            else:
                fail += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}]  ok={ok} fail={fail}  elapsed={time.time()-t0:.1f}s")
            if args.flush_every > 0 and done % args.flush_every == 0:
                write_csv_atomic(csv_path, rows, fieldnames)
                print(f"  [flush] {done}/{len(tasks)} -> {csv_path}")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            iterator = pool.imap_unordered(_compute_one, tasks, chunksize=chunksize)
            for done, (idx, d) in enumerate(iterator, 1):
                if apply_result(idx, d):
                    ok += 1
                else:
                    fail += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}]  ok={ok} fail={fail}  elapsed={time.time()-t0:.1f}s")
                if args.flush_every > 0 and done % args.flush_every == 0:
                    write_csv_atomic(csv_path, rows, fieldnames)
                    print(f"  [flush] {done}/{len(tasks)} -> {csv_path}")

    # 写回（包含新列）
    write_csv_atomic(csv_path, rows, fieldnames)
    print(f"[04b] done: ok={ok} fail={fail}  elapsed={time.time()-t0:.1f}s  → {csv_path}")


if __name__ == "__main__":
    main()
