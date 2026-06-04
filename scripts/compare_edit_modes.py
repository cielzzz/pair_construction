#!/usr/bin/env python
"""比较各 edit 模式的中性化效果（按 split）。

读 emotion/per_file_dual.csv（含所有 edit_tag 的评分），按 group 分组算：
  - n: 样本数
  - P_neutral mean / median
  - top1=neutral 比例
  - sv=neutral 比例
  - BOTH agree (e2v top1=neutral AND sv=neutral) 比例
  - delta_neu vs ref（如果 per_pair.csv 在）

排名 = mean(P_neutral) 降序。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, emotion_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="输出 markdown 报告路径；默认 emotion/edit_modes_compare.md")
    args = ap.parse_args()
    cfg = load_config(args.config)

    csv_path = emotion_path(cfg, args.split, "per_file_dual.csv")
    if not csv_path.exists():
        sys.exit(f"[compare] 缺 {csv_path}（先跑 04）")

    df = pd.read_csv(csv_path)
    # 兼容字段差异：sv_label 可能为 NaN
    df["sv_is_neutral"] = (df["sv_label"] == "neutral").astype(int)
    df["e2v_is_neutral"] = (df["top1_label"] == "neutral").astype(int)
    df["both_neutral"] = ((df["sv_is_neutral"] == 1) & (df["e2v_is_neutral"] == 1)).astype(int)

    # 按 group 聚合
    rows = []
    for grp, g in df.groupby("group"):
        rows.append({
            "group": grp,
            "n": len(g),
            "P_neu_mean": g["neutral"].mean(),
            "P_neu_median": g["neutral"].median(),
            "frac_e2v_neutral": g["e2v_is_neutral"].mean(),
            "frac_sv_neutral": g["sv_is_neutral"].mean(),
            "frac_both_neutral": g["both_neutral"].mean(),
        })
    summary = pd.DataFrame(rows)
    # 按 P_neu_mean 降序（不包括 original / ref baseline）
    edit_groups = [g for g in summary["group"] if g not in {"original", "ref"}]
    baseline = summary[summary["group"].isin({"original", "ref"})]
    edits = summary[summary["group"].isin(edit_groups)].sort_values("P_neu_mean", ascending=False)

    # 算 delta_neu vs ref（如果 ref 有数据）
    ref_mean = baseline.set_index("group").get("P_neu_mean", pd.Series()).get("ref", None)

    # 输出
    out_path = Path(args.out) if args.out else emotion_path(cfg, args.split, "edit_modes_compare.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {args.split} 各 edit 模式中性化对比",
        "",
        f"数据源：`{csv_path}`",
        f"样本量：每 group n={int(edits['n'].iloc[0]) if len(edits) else 0}（{len(df)} 行总计）",
        "",
        "## 基线（未编辑）",
        "",
        "| group | n | P_neu mean | P_neu median | frac e2v=neutral | frac sv=neutral | frac BOTH=neutral |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in baseline.iterrows():
        lines.append(f"| **{r['group']}** | {int(r['n'])} | {r['P_neu_mean']:.3f} | {r['P_neu_median']:.3f} | "
                     f"{r['frac_e2v_neutral']:.3f} | {r['frac_sv_neutral']:.3f} | {r['frac_both_neutral']:.3f} |")
    lines += [
        "",
        "## 各 edit 模式（按 P_neu mean 降序）",
        "",
        "| 排名 | edit_tag | n | P_neu mean ↑ | P_neu median | Δ vs ref | frac e2v=neutral | frac sv=neutral | **frac BOTH=neutral** |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, r) in enumerate(edits.iterrows(), 1):
        delta = f"{r['P_neu_mean'] - ref_mean:+.3f}" if ref_mean is not None else "-"
        marker = " ⭐" if rank == 1 else ""
        lines.append(
            f"| {rank}{marker} | **{r['group']}** | {int(r['n'])} | "
            f"{r['P_neu_mean']:.3f} | {r['P_neu_median']:.3f} | {delta} | "
            f"{r['frac_e2v_neutral']:.3f} | {r['frac_sv_neutral']:.3f} | **{r['frac_both_neutral']:.3f}** |"
        )
    lines += [
        "",
        "## 解读",
        "",
        "- **P_neu mean** 越大 → 平均中性化越强",
        "- **frac BOTH=neutral** 越大 → emotion2vec + SenseVoice 双分类器共识越高（最稳的中性化指标）",
        "- **Δ vs ref** 是 edited 比 ref 多出多少中性度；越正越说明 edit 真的把音频变中性了",
        "- 排第 1 的（带 ⭐）就是该 split 推荐的中性化模式",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare] 报告 → {out_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
