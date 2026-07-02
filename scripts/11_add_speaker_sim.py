#!/usr/bin/env python
"""11: legacy helper for CAM++ speaker similarity + optional *_filtered.jsonl output

注意：
- 当前主链与 `_infra` 现行流程已经切到 `11b_add_wavlm_sim.py` + `qc_pairs.py`
- speaker similarity 的正式 gate 现在统一在 `qc_pairs.py` 内完成
- 本脚本仅保留给旧实验 / 旧产物兼容，不再是 `run_pairs_local.sh` 的默认步骤

策略（2026-05-29 定）：
  bc.speaker_sim_min  -> 应用于 B + C 类（同 vcdata 源派生，可强求音色一致）
  d_st.speaker_sim_min -> 应用于 D_st 类（同 vcdata 源派生）
  C_mixed / D 不过滤（跨真人↔合成 by design）

优先复用 outputs/<split>/quality/all_class_ref_vs_tgt_sim.csv 缓存；缺时调 CAM++ 实时算。
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, iter_jsonl, write_jsonl, pair_path

SIM_KEY = {
    "B":       "bc",
    "C":       "bc",
    "C_mixed": None,
    "D":       None,
    "D_st":    "d_st",
    "H2":      "h2",
}


def load_cache(split: str, cfg: dict) -> dict:
    p = Path(cfg["paths"]["outputs_root"]) / split / "quality" / "all_class_ref_vs_tgt_sim.csv"
    if not p.exists():
        return {}
    cache = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            cache[(row["pair_type"], int(row["nn"]))] = float(row["sim"])
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = load_config(args.config)

    cache = load_cache(args.split, cfg)
    print(f"[11] sim cache: {len(cache)} entries")

    cam_loaded = [False]
    def live_sim(ref, tgt):
        from qc_pairs import compute_sim, get_sim_model
        if not cam_loaded[0]:
            get_sim_model(args.device); cam_loaded[0] = True
        return compute_sim(ref, tgt, args.device)

    summary = []
    for pt, sk in SIM_KEY.items():
        pp = pair_path(cfg, args.split, f"{pt}.jsonl")
        if not pp.exists():
            print(f"[skip] {pp.name}")
            continue
        sim_min = cfg.get(sk, {}).get("speaker_sim_min", 0.0) if sk else 0.0
        rows = list(iter_jsonl(pp))
        kept = []
        for i, r in enumerate(rows, 1):
            sim = cache.get((pt, i))
            if sim is None:
                sim = live_sim(r["reference_audio"], r["target_audio"])
            r["ref_vs_tgt_speaker_sim"] = sim
            if sim is not None and sim >= sim_min:
                kept.append(r)
        write_jsonl(pp, rows)
        fp = None
        if sim_min > 0:
            fp = pp.with_name(f"{pt}_filtered.jsonl")
            write_jsonl(fp, kept)
        pct = f"{100*len(kept)/len(rows):.0f}%" if rows else "—"
        print(f"[11] {pt:<10} orig={len(rows):<4} sim_min={sim_min:<5} kept={len(kept):<4} ({pct})  → {fp.name if fp else '(no filter)'}")
        summary.append((pt, len(rows), len(kept), sim_min, fp))

    print()
    print(f"=== {args.split} 汇总 ===")
    print(f"{'type':<10} {'orig':<6} {'sim_min':<8} {'kept':<6} {'kept%':<6}")
    for pt, n, k, sm, _ in summary:
        kp = f"{100*k/n:.0f}%" if n else "—"
        print(f"{pt:<10} {n:<6} {sm:<8} {k:<6} {kp:<6}")


if __name__ == "__main__":
    main()
