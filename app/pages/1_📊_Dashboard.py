"""Dashboard：总览 KPI + 类型 / 源 / 语言分布 + 留存率"""
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
    ACCENT, ACCENT_2, BG_CARD, BG_DARK, TEXT_MUTED, TEXT_PRIMARY,
    pair_color_seq, kpi_style,
)


st.set_page_config(page_title="Dashboard · pair_construction", layout="wide", page_icon="📊")
st.markdown(kpi_style(), unsafe_allow_html=True)
st.title("📊 数据看板 · 总览")
st.caption("KPI、类型分布、源 × 语言 × 类型 三维分析、过滤留存")

idx = load_index()
raw = load_raw_source()

if idx.empty and raw.empty:
    st.warning("索引为空。先跑 `python app/index_builder.py` 和 `python app/raw_scanner.py --add ...`。")
    st.stop()


# ════════ KPI ════════
st.subheader("📈 KPI")

total_raw_rows = int(raw["n_rows"].sum()) if not raw.empty else 0
total_raw_hours = float(raw["total_hours"].sum()) if not raw.empty else 0.0
n_sources = raw["source"].nunique() if not raw.empty else 0

if not idx.empty:
    orig_idx = idx[idx["is_filtered"] == False]
    total_pairs = int(orig_idx["n_pairs"].sum())
    total_filtered = int(idx[idx["is_filtered"] == True]["n_pairs"].sum())
    total_pair_hours = float(orig_idx["ref_hours"].sum() + orig_idx["tgt_hours"].sum())
    n_splits = orig_idx["split"].nunique()
    n_pair_types = orig_idx["pair_type"].nunique()
else:
    total_pairs = total_filtered = 0
    total_pair_hours = 0.0
    n_splits = n_pair_types = 0

cols = st.columns(5)
with cols[0]:
    st.metric("🎙️ 上游 raw 条数", f"{total_raw_rows:,}",
              help=f"覆盖 {n_sources} 数据源")
with cols[1]:
    st.metric("⏱️ 上游小时数", f"{total_raw_hours:,.1f} h")
with cols[2]:
    st.metric("🔗 pair 总数（orig）", f"{total_pairs:,}",
              help=f"{n_splits} splits × {n_pair_types} 类")
with cols[3]:
    if total_pairs:
        retain = 100 * total_filtered / total_pairs
        st.metric("✅ filtered 留存", f"{total_filtered:,}",
                  f"{retain:.1f}% of orig")
    else:
        st.metric("✅ filtered 留存", "—")
with cols[4]:
    st.metric("⏱️ pair 时长（ref+tgt）", f"{total_pair_hours:,.1f} h")

st.divider()


# ════════ 类型占比 / 类型 × 源 三维 ════════
if not idx.empty:
    orig = idx[idx["is_filtered"] == False].copy()
    by_type = (orig.groupby("pair_type", as_index=False)
                  .agg(n_pairs=("n_pairs", "sum"),
                       ref_hours=("ref_hours", "sum"),
                       tgt_hours=("tgt_hours", "sum")))
    by_type["total_hours"] = by_type["ref_hours"] + by_type["tgt_hours"]
    by_type = by_type.sort_values("n_pairs", ascending=False)

    st.subheader("🧩 pair_type 占比 / 三维分布")

    c1, c2 = st.columns([1, 1])
    with c1:
        # 总占比 donut
        fig = go.Figure(go.Pie(
            labels=by_type["pair_type"],
            values=by_type["n_pairs"],
            hole=0.55,
            marker=dict(colors=pair_color_seq(by_type["pair_type"].tolist())),
            textinfo="label+percent",
            textfont=dict(size=11),
            sort=False,
        ))
        fig.update_layout(
            title="全部 pair 按类型占比",
            showlegend=False,
            annotations=[dict(
                text=f"<b>{int(by_type['n_pairs'].sum()):,}</b><br><span style='color:#94a3b8;font-size:11px'>pairs</span>",
                font=dict(color=TEXT_PRIMARY, size=20),
                showarrow=False, x=0.5, y=0.5,
            )],
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # treemap：source > language > pair_type
        tree = (orig.groupby(["source", "language", "pair_type"], as_index=False)
                    .agg(n_pairs=("n_pairs", "sum")))
        fig = px.treemap(
            tree,
            path=["source", "language", "pair_type"],
            values="n_pairs",
            color="pair_type",
            color_discrete_map={k: v for k, v in zip(
                tree["pair_type"].unique(),
                pair_color_seq(tree["pair_type"].unique().tolist()),
            )},
        )
        fig.update_traces(textinfo="label+percent parent",
                          marker=dict(line=dict(color=BG_DARK, width=2)))
        fig.update_layout(title="source → language → pair_type 层次",
                          height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()


# ════════ source × language 下的 pair_type 横排 donut ════════
if not idx.empty:
    st.subheader("🍩 各 source × language 下的 pair_type 分布")

    combos = orig.groupby(["source", "language"], as_index=False)["n_pairs"].sum()
    combos = combos.sort_values("n_pairs", ascending=False)

    # 最多展示 6 个组合，其余折叠到表格
    top_n = min(6, len(combos))
    cols_per_row = 3
    for row_start in range(0, top_n, cols_per_row):
        row_combos = combos.iloc[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, (_, c) in enumerate(row_combos.iterrows()):
            sub = orig[(orig["source"] == c["source"]) & (orig["language"] == c["language"])]
            tdist = sub.groupby("pair_type", as_index=False)["n_pairs"].sum().sort_values("n_pairs", ascending=False)
            with cols[i]:
                fig = go.Figure(go.Pie(
                    labels=tdist["pair_type"],
                    values=tdist["n_pairs"],
                    hole=0.55,
                    marker=dict(colors=pair_color_seq(tdist["pair_type"].tolist())),
                    textinfo="percent",
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
                    height=280, margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

    if len(combos) > top_n:
        st.caption(f"显示前 {top_n} 个组合，其余 {len(combos) - top_n} 个见下表")

    # 补全表
    pivot = (orig.groupby(["source", "language", "pair_type"], as_index=False)["n_pairs"].sum()
                 .pivot_table(index=["source", "language"],
                              columns="pair_type", values="n_pairs", fill_value=0))
    pivot["__total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("__total", ascending=False).drop(columns="__total")
    st.dataframe(pivot, use_container_width=True)

    st.divider()


# ════════ 类型柱状（pair 数 + 小时数） ════════
if not idx.empty:
    st.subheader("📊 pair_type 横向对比")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(by_type, x="pair_type", y="n_pairs",
                     color="pair_type",
                     color_discrete_map={k: v for k, v in zip(
                         by_type["pair_type"], pair_color_seq(by_type["pair_type"].tolist()))},
                     title="条数")
        fig.update_layout(showlegend=False, height=350,
                          xaxis_title=None, yaxis_title="条数")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        melt = by_type[["pair_type", "ref_hours", "tgt_hours"]].melt(
            id_vars="pair_type", var_name="side", value_name="hours")
        fig = px.bar(melt, x="pair_type", y="hours", color="side",
                     color_discrete_sequence=[ACCENT, ACCENT_2],
                     title="小时数（ref + tgt）", barmode="stack")
        fig.update_layout(height=350, xaxis_title=None, yaxis_title="小时")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()


# ════════ raw source 概览（如果有）════════
if not raw.empty:
    st.subheader("🎙️ 上游数据源")
    by_src = (raw.groupby(["source", "language"], as_index=False)
                 .agg(n_rows=("n_rows", "sum"),
                      total_hours=("total_hours", "sum"),
                      n_split_jsonl=("split_jsonl", "nunique")))

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(by_src, x="source", y="n_rows", color="language",
                     barmode="group", color_discrete_sequence=[ACCENT, ACCENT_2],
                     title="raw 条数 by source × language")
        fig.update_layout(height=320, xaxis_title=None, yaxis_title="条数")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(by_src, x="source", y="total_hours", color="language",
                     barmode="group", color_discrete_sequence=[ACCENT, ACCENT_2],
                     title="raw 小时数")
        fig.update_layout(height=320, xaxis_title=None, yaxis_title="小时")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(by_src, use_container_width=True, hide_index=True)
    st.divider()


# ════════ 过滤留存（11b WavLM sim） ════════
if not idx.empty:
    st.subheader("🎯 过滤留存率（11b WavLM-L sim）")
    filt = idx.pivot_table(index=["split", "pair_type"],
                           columns="is_filtered", values="n_pairs", fill_value=0).reset_index()
    filt.columns = ["split", "pair_type", "orig", "filtered"]
    filt = filt[filt["filtered"] > 0].copy()
    filt["retain_pct"] = (filt["filtered"] / filt["orig"] * 100).round(1)

    # 按类型聚合的留存率
    type_retain = (filt.groupby("pair_type", as_index=False)
                       .agg(orig=("orig", "sum"), filtered=("filtered", "sum")))
    type_retain["retain_pct"] = (type_retain["filtered"] / type_retain["orig"] * 100).round(1)
    type_retain = type_retain.sort_values("retain_pct", ascending=False)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(type_retain, x="pair_type", y="retain_pct",
                     color="retain_pct",
                     color_continuous_scale=[[0, "#f87171"], [0.5, "#fbbf24"], [1, ACCENT]],
                     title="各 pair_type 留存率（%）")
        fig.update_layout(height=340, xaxis_title=None, yaxis_title="留存率 %",
                          coloraxis_showscale=False)
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**各 (split, pair_type) 留存详表**")
        st.dataframe(filt.sort_values(["split", "pair_type"]),
                     use_container_width=True, hide_index=True, height=340)
