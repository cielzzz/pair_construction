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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, emotion_path
from _dnsmos import compute_dnsmos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--onnx", default=None, help="覆盖默认 sig_bak_ovr.onnx 路径")
    args = ap.parse_args()
    cfg = load_config(args.config)

    csv_path = emotion_path(cfg, args.split, "per_file_dual.csv")
    if not csv_path.exists():
        sys.exit(f"[04b] 缺 {csv_path}（先跑 04）")

    # 读现有
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"[04b] {csv_path} 为空")

    # 已加 dnsmos 跳过
    if "dnsmos_ovrl" in rows[0]:
        already = sum(1 for r in rows if r.get("dnsmos_ovrl") not in ("", None))
        if already >= len(rows):
            print(f"[04b] 所有 {len(rows)} 行已有 dnsmos，跳过")
            return
        print(f"[04b] {already}/{len(rows)} 已有 dnsmos，续算剩余 {len(rows)-already}")

    t0 = time.time()
    ok = fail = 0
    for i, r in enumerate(rows):
        wav = r.get("wav") or r.get("path")
        if not wav:
            fail += 1; continue
        if r.get("dnsmos_ovrl") not in ("", None):
            continue
        d = compute_dnsmos(wav, onnx_path=args.onnx)
        if d is None:
            r["dnsmos_ovrl"] = ""
            r["dnsmos_sig"] = ""
            r["dnsmos_bak"] = ""
            fail += 1
        else:
            r["dnsmos_ovrl"] = d["OVRL"]
            r["dnsmos_sig"]  = d["SIG"]
            r["dnsmos_bak"]  = d["BAK"]
            ok += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}]  ok={ok} fail={fail}  elapsed={time.time()-t0:.1f}s")

    # 写回（包含新列）
    fieldnames = list(rows[0].keys())
    for new in ("dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak"):
        if new not in fieldnames:
            fieldnames.append(new)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[04b] done: ok={ok} fail={fail}  elapsed={time.time()-t0:.1f}s  → {csv_path}")


if __name__ == "__main__":
    main()
