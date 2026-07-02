#!/usr/bin/env python
"""02: 把每个 edit 模式的 paired_report.jsonl 标准化合并为 editx_base.jsonl

不修改上游。只读、只标准化字段。

读取路径模板（来自 configs.paths + configs.editx）：
  <vcdata_root>/<split>/stepaudio_<edit_tag>_split_<NNNN>_all_qzrun/paired_report.jsonl
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (
    load_config, iter_jsonl, write_jsonl,
    split_dir, split_idx_of, intermediate_path, make_sample_id_editx,
    prefer_local_ref_audio, prefer_local_edit_audio,
)


def normalize(row: dict, split: str, edit_tag: str, sd: Path) -> dict:
    md = row.get("metadata") or {}
    os_blk = row.get("one_stage") or {}
    edit_row_index = md["source_row_index"]
    ref_row_index = md.get("original_idx", edit_row_index)
    actual_edit_tag = md.get("edit_tag", edit_tag)
    ref_audio = prefer_local_ref_audio(sd, ref_row_index, row.get("audio1") or md.get("source_ref_audio_path"))
    return {
        "sample_id": make_sample_id_editx(split, edit_tag, edit_row_index),
        "split": split,
        "source_row_index": edit_row_index,
        "original_idx": ref_row_index,
        "job_id": row.get("job_id"),
        "edit_tag": actual_edit_tag,
        "edit_type": md.get("edit_type"),
        "edit_info": md.get("edit_info"),
        "instruction": row.get("instruction") or os_blk.get("prompt"),
        "input_audio": ref_audio,
        "input_text": row.get("text1"),
        "edited_audio": prefer_local_edit_audio(sd, split, actual_edit_tag, edit_row_index, os_blk.get("audio2")),
        "edited_text": row.get("text2"),
        "original_audio_from_meta": md.get("original_audio_path"),
        "best_similarity_from_meta": md.get("best_similarity"),
        "model": md.get("model"),
        "ref_audio": ref_audio,
        "ref_text": row.get("text1"),
    }


def iter_one_tag(cfg: dict, split: str, edit_tag: str) -> Iterator[dict]:
    sd = split_dir(cfg, split)
    # 先尝试模板路径（要求 split_NNNN 数字格式）
    pr = None
    try:
        idx = split_idx_of(split)
        rel = cfg["editx"]["paired_report_template"].format(edit_tag=edit_tag, split_idx=idx)
        cand = sd / rel
        if cand.exists():
            pr = cand
    except ValueError:
        pass
    # fallback：glob 任何 stepaudio_<edit_tag>_<split>_*/paired_report.jsonl
    if pr is None:
        matches = sorted(sd.glob(f"stepaudio_{edit_tag}_{split}_*/paired_report.jsonl"))
        if matches:
            pr = matches[0]
    if pr is None or not pr.exists():
        print(f"[02] [skip] {edit_tag}: 在 {sd} 下找不到 paired_report.jsonl", file=sys.stderr)
        return
    cnt = 0
    for r in iter_jsonl(pr):
        try:
            yield normalize(r, split, edit_tag, sd)
            cnt += 1
        except KeyError as e:
            print(f"[02] [warn] {edit_tag} row missing {e}: skip", file=sys.stderr)
    print(f"[02] {edit_tag}: {cnt} rows ← {pr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--edit-tags",
        nargs="+",
        default=None,
        help="覆盖 configs.editx.edit_tags",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    tags = args.edit_tags or cfg["editx"]["edit_tags"]

    out = intermediate_path(cfg, args.split, "editx_base.jsonl")

    def all_rows():
        for tag in tags:
            yield from iter_one_tag(cfg, args.split, tag)

    n = write_jsonl(out, all_rows())
    print(f"[02] editx_base.jsonl  rows={n}  tags={tags}  → {out}")


if __name__ == "__main__":
    main()
