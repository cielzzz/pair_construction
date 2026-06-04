#!/usr/bin/env python
"""03: editx_base ⋈ vcdata_base on (input_audio == ref_audio)

用绝对路径做 key 才是真正的天然 join：editx 的 input_audio 就是 vcdata 产的 ref_audio。
旧版用 source_row_index ⋈ original_idx 只在 zh demo（idx 顺序连号）下凑巧能命中，
在英文 split_demo_en（vcdata original_idx 稀疏跳跃如 [0,1,2,3,48,49,...,1442]）只命中 12/100。

输出 joined_editx.jsonl，是 B/C/H2/D_st 类构造的统一输入。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, iter_jsonl, write_jsonl, intermediate_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    vc_path = intermediate_path(cfg, args.split, "vcdata_base.jsonl")
    ed_path = intermediate_path(cfg, args.split, "editx_base.jsonl")
    out_path = intermediate_path(cfg, args.split, "joined_editx.jsonl")

    if not vc_path.exists():
        sys.exit(f"[03] 缺 vcdata_base.jsonl: {vc_path}（先跑 01）")
    if not ed_path.exists():
        sys.exit(f"[03] 缺 editx_base.jsonl: {ed_path}（先跑 02）")

    vc_index: dict[str, dict] = {r["ref_audio"]: r for r in iter_jsonl(vc_path)}
    print(f"[03] vcdata rows: {len(vc_index)}")

    n_miss = 0
    def joined_rows():
        nonlocal n_miss
        for e in iter_jsonl(ed_path):
            v = vc_index.get(e["input_audio"])
            if v is None:
                n_miss += 1; continue
            yield {
                "sample_id": e["sample_id"],
                "split": args.split,
                "original_idx": v["original_idx"],
                "original_audio": v["original_audio"],
                "original_text": v["original_text"],
                "ref_audio": v["ref_audio"],
                "ref_text": v["ref_text"],
                "edited_audio": e["edited_audio"],
                "edit_tag": e["edit_tag"],
                "edit_type": e["edit_type"],
                "edit_info": e["edit_info"],
                "instruction": e["instruction"],
                "speaker_similarity": v["speaker_similarity"],
                "flag": v["flag"],
                "duration": v["duration"],
                "caption_summary": v["caption_summary"],
                "caption_gender": v["caption_gender"],
                "zh_summary": v["zh_summary"],
            }

    n = write_jsonl(out_path, joined_rows())
    print(f"[03] joined_editx.jsonl rows={n} (editx 无匹配 ref_audio 跳过 {n_miss})  → {out_path}")


if __name__ == "__main__":
    main()
