"""Source compare：多数据源横向对比 + 转化漏斗 + donut"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loader import load_index, load_raw_source
from _theme import (
    ACCENT, ACCENT_2, BG_DARK, TEXT_PRIMARY, TEXT_MUTED,
    pair_color_seq, kpi_style,
)


st.set_page_config(page_title="Source compare · pair_construction", layout="wide", page_icon="📈")
st.markdown(kpi_style(), unsafe_allow_html=True)
st.title("📈 数据源横向对比")
st.caption("加新数据源时，这一页会自动新增对应列 / 列对应的视图")

idx = load_index()
raw = load_raw_source()

if idx.empty and raw.empty:
    st.warning("索引为空。先跑 `python app/index_builder.py` 和 `python app/raw_scanner.py --add ...`。")
    st.stop()


# ════════ 各源 raw 概览 KPI ════════
if not raw.empty:
    st.subheader("🎙️ 各源 raw 概览")
    by_src = (raw.groupby(["source", "language"], as_index=False)
                 .agg(n_rows=("n_rows", "sum"),
                      total_hours=("total_hours", "sum"),
                      n_split=("split_jsonl", "nunique")))

    # 每个 (source, language) 横排成卡
    sources_langs = list(by_src.itertuples(index=False))
    cols = st.columns(min(4, len(sources_langs)) or 1)
    for i, r in enumerate(sources_langs):
        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div style='background:linear-gradient(135deg,#141a2b,#0f1422);
                            border:1px solid #1f2738; border-radius:10px;
                            padding:14px 16px; margin-bottom:8px;
                            box-shadow:0 0 0 1px rgba(0,212,255,0.04);'>
                  <div style='color:#94a3b8;font-size:12px'>{r.source} · <b style='color:#00d4ff'>{r.language}</b></div>
                  <div style='color:#e2e8f0;font-family:monospace;font-size:26px;line-height:1.2;'>
                    {r.n_rows:,}
                    <span style='color:#94a3b8;font-size:13px'>rows</span>
                  </div>
                  <div style='color:#94a3b8;font-size:11px;margin-top:4px'>
                    {r.total_hours:,.1f} hours · {r.n_split} jsonl
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()


# ════════ 转化漏斗：raw → orig pair → filtered pair ════════
if not idx.empty and not raw.empty:
    st.subheader("🔻 raw → pair 转化漏斗")
    orig = idx[idx["is_filtered"] == False]
    filt = idx[idx["is_filtered"] == True]
    raw_by_sl = raw.groupby(["source", "language"], as_index=False).agg(
        raw_rows=("n_rows", "sum"), raw_hours=("total_hours", "sum"))
    orig_by_sl = orig.groupby(["source", "language"], as_index=False).agg(
        pair_orig=("n_pairs", "sum"))
    filt_by_sl = filt.groupby(["source", "language"], as_index=False).agg(
        pair_filtered=("n_pairs", "sum"))
    funnel = (raw_by_sl
              .merge(orig_by_sl, on=["source", "language"], how="left")
              .merge(filt_by_sl, on=["source", "language"], how="left")
              .fillna(0))
    funnel["pair_per_raw"] = (funnel["pair_orig"] / funnel["raw_rows"].replace(0, pd.NA)).round(2)
    funnel["retain_pct"] = (funnel["pair_filtered"] / funnel["pair_orig"].replace(0, pd.NA) * 100).round(1)

    # 每 (source, language) 一条横向漏斗
    for _, r in funnel.iterrows():
        col_title, col_funnel = st.columns([1, 3])
        with col_title:
            st.markdown(
                f"""
                <div style='padding:24px 8px;'>
                  <div style='color:#94a3b8;font-size:12px'>{r['source']}</div>
                  <div style='font-size:22px;color:#00d4ff;font-weight:600;'>{r['language']}</div>
                  <div style='margin-top:10px;font-size:11px;color:#94a3b8'>
                    pair/raw: <b style='color:#e2e8f0'>{r['pair_per_raw']}</b>
                  </div>
                  <div style='font-size:11px;color:#94a3b8'>
                    sim filter 留存: <b style='color:#e2e8f0'>{r['retain_pct']}%</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_funnel:
            fig = go.Figure(go.Funnel(
                y=["raw rows", "pair (orig)", "pair (filtered)"],
                x=[int(r["raw_rows"]), int(r["pair_orig"]), int(r["pair_filtered"])],
                textinfo="value+percent initial",
                marker=dict(color=[ACCENT_2, ACCENT, "#a78bfa"],
                            line=dict(color=BG_DARK, width=1)),
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.dataframe(funnel, use_container_width=True, hide_index=True)
    st.divider()


# ════════ 各源 donut 横排（pair_type 占比） ════════
if not idx.empty:
    st.subheader("🍩 各 source × language 的 pair_type 占比")
    orig = idx[idx["is_filtered"] == False]
    combos = (orig.groupby(["source", "language"], as_index=False)["n_pairs"].sum()
                  .sort_values("n_pairs", ascending=False))

    cols_per_row = 3
    for row_start in range(0, len(combos), cols_per_row):
        row_combos = combos.iloc[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, (_, c) in enumerate(row_combos.iterrows()):
            sub = orig[(orig["source"] == c["source"]) & (orig["language"] == c["language"])]
            tdist = (sub.groupby("pair_type", as_index=False)["n_pairs"].sum()
                        .sort_values("n_pairs", ascending=False))
            with cols[i]:
                fig = go.Figure(go.Pie(
                    labels=tdist["pair_type"], values=tdist["n_pairs"],
                    hole=0.55,
                    marker=dict(colors=pair_color_seq(tdist["pair_type"].tolist())),
                    textinfo="label+percent",
                    textfont=dict(size=10),
                    sort=False,
                ))
                fig.update_layout(
                    title=f"<b>{c['source']}</b> · {c['language']}",
                    showlegend=False,
                    annotations=[dict(
                        text=f"<b>{int(c['n_pairs']):,}</b><br><span style='color:#94a3b8;font-size:9px'>pairs</span>",
                        font=dict(color=TEXT_PRIMARY, size=14),
                        showarrow=False, x=0.5, y=0.5,
                    )],
                    height=300, margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
    st.divider()


# ════════ 堆叠柱：各源各类型 ════════
if not idx.empty:
    st.subheader("📊 source 维度堆叠图")
    orig = idx[idx["is_filtered"] == False]
    melt = (orig.groupby(["source", "language", "pair_type"], as_index=False)
                .agg(n_pairs=("n_pairs", "sum")))
    pair_types_unique = melt["pair_type"].unique().tolist()
    fig = px.bar(
        melt, x="source", y="n_pairs", color="pair_type",
        facet_col="language",
        color_discrete_map={k: v for k, v in zip(
            pair_types_unique, pair_color_seq(pair_types_unique))},
        title="各 source 各 pair_type（按语言分面）",
    )
    fig.update_layout(height=420, xaxis_title=None, yaxis_title="条数",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0))
    st.plotly_chart(fig, use_container_width=True)
