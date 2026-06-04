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
from _utils import load_config, iter_jsonl, write_jsonl, emotion_path, pair_path
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
    emo.load_link_mapping(emotion_path(cfg, args.split, "_links_original/_mapping.csv"))

    def dnsmos_bak(audio_path):
        rec = emo.get(audio_path)
        if rec is None: return None
        v = rec.get("dnsmos_bak")
        if v in ("", None): return None
        try: return float(v)
        except (TypeError, ValueError): return None

    pair_dir = Path(cfg["paths"]["outputs_root"]) / args.split / "pairs"
    summary = []
    for jsonl_name in sorted(pair_dir.glob("*.jsonl")):
        if "_filtered" in jsonl_name.name or "_bakfilt" in jsonl_name.name:
            continue
        rows = list(iter_jsonl(jsonl_name))
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

        write_jsonl(jsonl_name, rows)  # in-place 加字段
        if apply_filter:
            out = jsonl_name.with_name(jsonl_name.stem + "_bakfilt.jsonl")
            write_jsonl(out, kept)
            summary.append((jsonl_name.name, len(rows), len(kept), out.name))
        else:
            summary.append((jsonl_name.name, len(rows), len(rows), "(只写字段)"))

    print(f"\n=== {args.split} dnsmos_bak {'过滤' if apply_filter else '记录'} 汇总 ===")
    print(f"{'jsonl':<28} {'orig':<6} {'kept':<6} {'output':<30}")
    for n, o, k, out in summary:
        kp = f"{100*k/o:.0f}%" if o else "—"
        print(f"{n:<28} {o:<6} {k:<6} ({kp:<5}) {out}")


if __name__ == "__main__":
    main()
