#!/usr/bin/env python
"""为 D-clone TTS baseline 选 jobs：从 split_demo / split_demo_en 各 5 句高表现力 ref。

要求每条 job 提供：
- ref_audio (P_neutral 最低的，保证 ref 是高表现)
- ref_text (ref_audio 对应原文，是 voice clone 必需的)
- target_text (用 original_text，跨文本，符合 D-clone 语义)
- target_audio_ref (原 vcdata 的 original_audio，用于参考；非必需)
- target_emotion_topk_ref (ref 的 top1 情绪，用于 D-clone 评估时检查"保留情绪")

输出统一 jobs.jsonl，供下面任意一个 baseline 子项目消费。
"""
import json, csv, re, argparse
from pathlib import Path

PC_ROOT = Path("/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction")
OUT = PC_ROOT / "tts_baselines" / "jobs.jsonl"


def collect(lang, language, split, n=5):
    join_p = PC_ROOT / f"outputs/{split}/intermediate/joined_editx.jsonl"
    emo_p = PC_ROOT / f"outputs/{split}/emotion/per_file_dual.csv"
    rows = [json.loads(l) for l in join_p.open()]
    # 用 ref group 的 emotion CSV 路径反推 idx
    idx2 = {}
    with emo_p.open() as f:
        for r in csv.DictReader(f):
            p = r.get("wav") or r.get("path")
            if not p or "/ref_audio/" not in p:
                continue
            m = re.search(r"ref_audio/(\d+)_ref\.wav$", p)
            if not m:
                continue
            idx2[int(m.group(1))] = {
                "path": p,
                "neutral": float(r.get("neutral") or 0.0),
                "top1": r.get("top1_label"),
                "sv": r.get("sv_label"),
            }
    cand = []
    for r in rows:
        idx = r["original_idx"]
        if idx not in idx2:
            continue
        cand.append((idx2[idx]["neutral"], r, idx2[idx]))
    cand.sort(key=lambda x: x[0])  # ref P_neutral 从低到高（最有表现力的先）
    picked = []
    seen = set()
    for _, r, e in cand:
        if r["original_idx"] in seen:
            continue
        seen.add(r["original_idx"])
        picked.append({
            "tag": f"{lang}_{r['original_idx']:06d}",
            "lang": lang,
            "language": language,
            "split": split,
            "original_idx": r["original_idx"],
            "ref_audio": e["path"],
            "ref_text": r["ref_text"],
            "target_text": r["original_text"],
            "ref_emotion_top1": e["top1"],
            "ref_p_neutral": e["neutral"],
            "ref_sv_label": e["sv"],
        })
        if len(picked) >= n:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh-split", default="split_demo")
    ap.add_argument("--en-split", default="split_demo_en")
    ap.add_argument("--n-per-lang", type=int, default=5)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    zh = collect("zh", "Chinese", args.zh_split, args.n_per_lang)
    en = collect("en", "English", args.en_split, args.n_per_lang)
    jobs = zh + en
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(f"写 {len(jobs)} jobs → {args.out}")
    for j in jobs:
        print(f"  {j['tag']} ref_top1={j['ref_emotion_top1']} P_neu={j['ref_p_neutral']:.3f}  ref:{j['ref_text'][:25]}  tgt:{j['target_text'][:25]}")


if __name__ == "__main__":
    main()
