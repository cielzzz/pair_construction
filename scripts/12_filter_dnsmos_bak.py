#!/usr/bin/env python
"""12: 可选 DNSMOS-BAK 防电音过滤

应用于所有 pair jsonl（B/C/C_mixed/D/D_st/D_cross_emo/Genre），
给每条 pair 加 ref_dnsmos_bak / tgt_dnsmos_bak 字段，
如果 cfg.dnsmos_bak_filter.apply==True 或命令行 --apply，则过滤产 *_bakfilt.jsonl

依赖 emotion/per_file_dual.csv 里已有 dnsmos_bak 列（04b 已加）
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (
    load_config, iter_jsonl, write_jsonl,
    emotion_path, pair_path, scored_pair_path, filtered_pair_path,
    base_pair_jsonl_paths, preferred_pair_input_path,
)
from _emotion_lookup import EmotionTable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--apply", action="store_true", help="开启过滤（默认只写字段不过滤）")
    ap.add_argument("--ref-bak-min", type=float, default=None)
    ap.add_argument("--tgt-bak-min", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg_dbf = cfg.get("dnsmos_bak_filter", {"apply": False, "ref_bak_min": 3.5, "tgt_bak_min": 3.5})

    apply_filter = args.apply or cfg_dbf.get("apply", False)
    ref_min = args.ref_bak_min if args.ref_bak_min is not None else cfg_dbf.get("ref_bak_min", 3.5)
    tgt_min = args.tgt_bak_min if args.tgt_bak_min is not None else cfg_dbf.get("tgt_bak_min", 3.5)

    # 加载 emotion table 读 dnsmos_bak
    emo = EmotionTable()
    emo.load_csv(emotion_path(cfg, args.split, "per_file_dual.csv"))
    emo.load_per_pair_for_src(emotion_path(cfg, args.split, "per_pair.csv"))
    emo.load_all_link_mappings(emotion_path(cfg, args.split, ""))

    def dnsmos_bak(audio_path):
        rec = emo.get(audio_path)
        if rec is None: return None
        v = rec.get("dnsmos_bak")
        if v in ("", None): return None
        try: return float(v)
        except (TypeError, ValueError): return None

    summary = []
    for raw_pair_path in base_pair_jsonl_paths(cfg, args.split):
        source_path = preferred_pair_input_path(cfg, args.split, raw_pair_path.name)
        rows = list(iter_jsonl(source_path))
        kept = []
        for r in rows:
            r["ref_dnsmos_bak"] = dnsmos_bak(r.get("reference_audio", ""))
            r["tgt_dnsmos_bak"] = dnsmos_bak(r.get("target_audio", ""))
            if not apply_filter:
                continue
            rb, tb = r["ref_dnsmos_bak"], r["tgt_dnsmos_bak"]
            if rb is None or tb is None:
                continue
            if rb >= ref_min and tb >= tgt_min:
                kept.append(r)

        scored_path = scored_pair_path(cfg, args.split, raw_pair_path.name)
        write_jsonl(scored_path, rows)
        if apply_filter:
            legacy_out = pair_path(cfg, args.split, raw_pair_path.stem + "_bakfilt.jsonl")
            layered_out = filtered_pair_path(cfg, args.split, raw_pair_path.stem + "_bakfilt.jsonl")
            write_jsonl(legacy_out, kept)
            write_jsonl(layered_out, kept)
            summary.append((raw_pair_path.name, len(rows), len(kept), legacy_out.name, scored_path.name))
        else:
            summary.append((raw_pair_path.name, len(rows), len(rows), "(只写字段)", scored_path.name))

    print(f"\n=== {args.split} dnsmos_bak {'过滤' if apply_filter else '记录'} 汇总 ===")
    print(f"{'jsonl':<28} {'orig':<6} {'kept':<6} {'output':<30} {'scored':<24}")
    for n, o, k, out, scored_name in summary:
        kp = f"{100*k/o:.0f}%" if o else "—"
        print(f"{n:<28} {o:<6} {k:<6} ({kp:<5}) {out:<30} {scored_name:<24}")


if __name__ == "__main__":
    main()
