#!/usr/bin/env python
"""quality_check.py —— 对一个 pairs/X.jsonl 做三维度质量验收

三个维度（按 pair_type 自适应判定）：
  1. emotion   —— ref_emotion / tgt_emotion 的 top1 / P_neutral / sv_label 满足该类语义
                  （C 类要求 tgt 中性，D 类要求两边都高表现 + 一致 等）
  2. speaker_sim —— ref_audio vs target_audio 的 WavLM-Large+ECAPA-TDNN cosine
                    复用 vcdata_construction/speaker_similarity.py（同一套打分器）
  3. dnsmos    —— 两边音频的 DNSMOS 质量分（OVRL ≥ 阈值）

输出：
  outputs/<split>/quality/<pair_type>__report.md  人类可读
  outputs/<split>/quality/<pair_type>__per_pair.csv  逐 pair 指标

用法：
  python quality_check.py --split split_demo --pair-type C
  python quality_check.py --split split_demo --pair-type all   # 所有 pair_*.jsonl
"""
from __future__ import annotations
import argparse
import csv
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, iter_jsonl, pair_path, emotion_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 各 pair_type 的 emotion / speaker_sim / dnsmos 阈值 ────────
# emotion_ok 的判定函数：(ref_emo, tgt_emo) → bool
# speaker_sim 期望区间：(min, None=不卡上限)
# dnsmos 期望：min OVRL
DEFAULT_THRESHOLDS = {
    "A": {
        # A 是 vcdata 全量，只要求音色相近，emotion 不卡
        "emotion_ok": lambda r, t: True,
        "speaker_sim_min": 0.30,
        "dnsmos_min": 2.0,
    },
    "B": {
        # 中性→高表现：ref 中性，tgt 高表现
        "emotion_ok": lambda r, t: _p_neu(r) >= 0.5 and _p_neu(t) <= 0.5,
        "speaker_sim_min": 0.30,
        "dnsmos_min": 2.0,
    },
    "C": {
        # 高表现→中性
        "emotion_ok": lambda r, t: _p_neu(r) <= 0.95 and _p_neu(t) >= 0.5,
        "speaker_sim_min": 0.30,
        "dnsmos_min": 2.0,
    },
    "C-mixed": {
        "emotion_ok": lambda r, t: _p_neu(r) <= 0.95 and _p_neu(t) >= 0.5,
        "speaker_sim_min": 0.30,
        "dnsmos_min": 2.0,
    },
    "D": {
        # 高表现→高表现 同情绪
        "emotion_ok": lambda r, t: (_top1(r) == _top1(t) and _p_neu(r) <= 0.95 and _p_neu(t) <= 0.95),
        "speaker_sim_min": 0.40,
        "dnsmos_min": 2.0,
    },
    "H1": {
        "emotion_ok": lambda r, t: _top1(r) == _top1(t),
        "speaker_sim_min": 0.30,
        "dnsmos_min": 2.0,
    },
    "H2": {
        # H2: ref 已中性，target 更中性；legacy 脚本里只做宽松 neutral gate
        "emotion_ok": lambda r, t: _p_neu(r) >= 0.5 and _p_neu(t) >= 0.5,
        "speaker_sim_min": 0.30,
        "dnsmos_min": 2.0,
    },
    "H3": {
        # 跨 speaker 负样本：sim 应该低
        "emotion_ok": lambda r, t: True,
        "speaker_sim_min": 0.0,    # 不卡
        "dnsmos_min": 2.0,
    },
}


def _p_neu(e: Optional[dict]) -> float:
    if e is None: return 0.0
    v = e.get("P_neutral")
    return float(v) if v else 0.0

def _top1(e: Optional[dict]) -> Optional[str]:
    if e is None: return None
    return e.get("top1_label")


# ── speaker similarity：funasr CAM++（国内主流，emotion env 自带，无需 transformers） ──
_sim_model = None
def get_sim_model(device: str = "cuda:0"):
    global _sim_model
    if _sim_model is not None: return _sim_model
    from funasr import AutoModel
    print(f"Loading CAM++ speaker verification on {device} ...")
    _sim_model = AutoModel(
        model="iic/speech_campplus_sv_zh-cn_16k-common",
        hub="ms", device=device, disable_update=True,
    )
    print("CAM++ loaded.")
    return _sim_model


def _emb_to_np(x):
    import numpy as np
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().flatten()
    return np.asarray(x).flatten()


def compute_sim(ref_path: str, tgt_path: str, device: str = "cuda:0") -> Optional[float]:
    """CAM++ 推 ref 和 tgt 各一次，取 spk_embedding 算 cosine。"""
    try:
        m = get_sim_model(device)
        r1 = m.generate(input=ref_path, disable_pbar=True)
        r2 = m.generate(input=tgt_path, disable_pbar=True)
        if not (r1 and r2 and "spk_embedding" in r1[0] and "spk_embedding" in r2[0]):
            return None
        import numpy as np
        e1 = _emb_to_np(r1[0]["spk_embedding"])
        e2 = _emb_to_np(r2[0]["spk_embedding"])
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))
    except Exception as ex:
        print(f"  [sim warn] {Path(ref_path).name} ↔ {Path(tgt_path).name}: {ex}", file=sys.stderr)
        return None


# ── DNSMOS：优先从 emotion/per_file_dual.csv 的 dnsmos_ovrl 列读（由 04b 预算）─
# 设置 _emotion_table 后 compute_dnsmos 即生效；否则回退 None
_emotion_table = None
def set_emotion_table(t):
    global _emotion_table
    _emotion_table = t


def compute_dnsmos(audio_path: str) -> Optional[float]:
    if _emotion_table is None:
        return None
    return _emotion_table.dnsmos_ovrl(audio_path)


def evaluate_pair_file(split: str, pair_type: str, cfg: dict, sample_limit: int, device: str):
    pp = pair_path(cfg, split, f"{pair_type}.jsonl")
    if not pp.exists():
        print(f"[skip] 不存在 {pp}")
        return None
    rows = list(iter_jsonl(pp))
    if sample_limit and sample_limit < len(rows):
        rows = rows[:sample_limit]

    # 简单从 pair_type 推 base type 找阈值
    base_type = pair_type.replace("_", "-")
    if base_type not in DEFAULT_THRESHOLDS:
        # fuzzy: A.jsonl → A，C.jsonl → C
        for k in DEFAULT_THRESHOLDS:
            if pair_type.replace("_", "-").startswith(k):
                base_type = k; break
    th = DEFAULT_THRESHOLDS.get(base_type, DEFAULT_THRESHOLDS["A"])

    per_pair = []
    n_emo_ok = n_sim_ok = n_dns_ok = n_all_ok = 0
    for r in rows:
        re_, te_ = r.get("ref_emotion"), r.get("tgt_emotion")
        # 1. emotion
        try:
            emo_ok = th["emotion_ok"](re_, te_)
        except Exception:
            emo_ok = False
        # 2. speaker sim
        sim = compute_sim(r["reference_audio"], r["target_audio"], device)
        sim_ok = sim is not None and sim >= th["speaker_sim_min"]
        # 3. dnsmos
        dns_ref = compute_dnsmos(r["reference_audio"])
        dns_tgt = compute_dnsmos(r["target_audio"])
        dns_ok = (dns_ref is not None and dns_ref >= th["dnsmos_min"]
                  and dns_tgt is not None and dns_tgt >= th["dnsmos_min"])
        all_ok = emo_ok and sim_ok and dns_ok
        n_emo_ok += emo_ok; n_sim_ok += sim_ok; n_dns_ok += dns_ok; n_all_ok += all_ok
        per_pair.append({
            "pair_id": r["pair_id"],
            "emotion_ok": int(emo_ok),
            "speaker_sim": sim if sim is not None else "",
            "speaker_sim_ok": int(sim_ok),
            "dnsmos_ref": dns_ref if dns_ref is not None else "",
            "dnsmos_tgt": dns_tgt if dns_tgt is not None else "",
            "dnsmos_ok": int(dns_ok),
            "all_ok": int(all_ok),
        })

    n = len(rows)
    summary = {
        "pair_type": pair_type,
        "n": n,
        "emotion_pass": f"{n_emo_ok}/{n} ({100*n_emo_ok/n:.1f}%)",
        "speaker_sim_pass": f"{n_sim_ok}/{n} ({100*n_sim_ok/n:.1f}%)",
        "dnsmos_pass": f"{n_dns_ok}/{n} ({100*n_dns_ok/n:.1f}%)",
        "all_pass": f"{n_all_ok}/{n} ({100*n_all_ok/n:.1f}%)",
        "thresholds": th,
    }
    return summary, per_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pair-type", default="all",
                    help="A | B | C | C_mixed | D | H1 | H2 | H3 | all")
    ap.add_argument("--limit", type=int, default=0, help="每类抽前 N 条做评估（0 = 全量；调试用 50）")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = load_config(args.config)

    pair_dir = Path(cfg["paths"]["outputs_root"]) / args.split / "pairs"
    if args.pair_type == "all":
        types = sorted([p.stem for p in pair_dir.glob("*.jsonl")])
    else:
        types = [args.pair_type]

    # 加载 per_file_dual.csv 让 compute_dnsmos 能查 dnsmos_ovrl
    from _emotion_lookup import EmotionTable
    et = EmotionTable()
    et.load_csv(emotion_path(cfg, args.split, "per_file_dual.csv"))
    et.load_per_pair_for_src(emotion_path(cfg, args.split, "per_pair.csv"))
    et.load_all_link_mappings(emotion_path(cfg, args.split, ""))
    set_emotion_table(et)

    out_dir = Path(cfg["paths"]["outputs_root"]) / args.split / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [f"# {args.split} 质量验收报告", "",
                "三维度：emotion 类别符合 / speaker_sim (WavLM-L + ECAPA-TDNN cosine) / DNSMOS",
                ""]
    for t in types:
        print(f"\n=== {t} ===")
        result = evaluate_pair_file(args.split, t, cfg, args.limit, args.device)
        if result is None: continue
        summary, per_pair = result
        # 写 per_pair csv
        csv_path = out_dir / f"{t}__per_pair.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=per_pair[0].keys())
            w.writeheader(); w.writerows(per_pair)
        print(f"  → {csv_path}")
        # 摘要
        for k, v in summary.items():
            if k == "thresholds": continue
            print(f"  {k}: {v}")
        md_lines += [f"## {t}",
                     f"- n={summary['n']}",
                     f"- emotion pass: **{summary['emotion_pass']}**",
                     f"- speaker_sim pass (>= {summary['thresholds']['speaker_sim_min']}): **{summary['speaker_sim_pass']}**",
                     f"- dnsmos pass (>= {summary['thresholds']['dnsmos_min']}): **{summary['dnsmos_pass']}**",
                     f"- 三维度都过: **{summary['all_pass']}**", ""]

    rpt = out_dir / "report.md"
    rpt.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n=== 总报告 → {rpt} ===")


if __name__ == "__main__":
    main()
