"""Source compare：多数据源横向对比"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loader import load_index, load_raw_source


st.set_page_config(page_title="Source compare", layout="wide")
st.title("📈 数据源横向对比")

idx = load_index()
raw = load_raw_source()

if idx.empty and raw.empty:
    st.warning("索引为空。先跑 `python app/index_builder.py` 和 `python app/raw_scanner.py --add ...`。")
    st.stop()


# ─── raw 源对比 ──────────────────────────────────────────
if not raw.empty:
    st.subheader("上游 raw 数据源")
    raw_by_src = (raw
                  .groupby(["source", "language"], as_index=False)
                  .agg(n_rows=("n_rows", "sum"),
                       total_hours=("total_hours", "sum"),
                       n_split_jsonl=("split_jsonl", "nunique")))
    st.dataframe(raw_by_src, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(raw_by_src, x="source", y="n_rows", color="language",
                     barmode="group", title="raw 条数对比")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(raw_by_src, x="source", y="total_hours", color="language",
                     barmode="group", title="raw 小时数对比")
        st.plotly_chart(fig, use_container_width=True)
    st.divider()


# ─── pair 转化率 ─────────────────────────────────────────
if not idx.empty and not raw.empty:
    st.subheader("pair / raw 转化漏斗（每 raw 行产几条 pair）")
    orig = idx[idx["is_filtered"] == False]
    pair_by_src = (orig
                   .groupby(["source", "language"], as_index=False)
                   .agg(total_pairs=("n_pairs", "sum"),
                        pair_ref_h=("ref_hours", "sum"),
                        pair_tgt_h=("tgt_hours", "sum")))
    raw_by_src2 = (raw
                   .groupby(["source", "language"], as_index=False)
                   .agg(raw_rows=("n_rows", "sum"),
                        raw_hours=("total_hours", "sum")))
    funnel = raw_by_src2.merge(pair_by_src, on=["source", "language"], how="left").fillna(0)
    funnel["pair_per_raw_row"] = (funnel["total_pairs"] / funnel["raw_rows"].replace(0, pd.NA)).round(2)
    st.dataframe(funnel, use_container_width=True, hide_index=True)

    fig = px.bar(funnel, x="source", y="pair_per_raw_row", color="language",
                 barmode="group",
                 title="转化率：每条 raw 平均产几条 pair",
                 labels={"pair_per_raw_row": "pair / raw 行"})
    st.plotly_chart(fig, use_container_width=True)
    st.divider()


# ─── 各源各 pair_type 数 ─────────────────────────────────
if not idx.empty:
    st.subheader("各 source × language 下的 pair_type 分布")
    orig = idx[idx["is_filtered"] == False]
    pivot = (orig
             .groupby(["source", "language", "pair_type"], as_index=False)
             .agg(n_pairs=("n_pairs", "sum"))
             .pivot_table(index=["source", "language"],
                          columns="pair_type",
                          values="n_pairs",
                          fill_value=0))
    st.dataframe(pivot, use_container_width=True)

    melt = (orig
            .groupby(["source", "language", "pair_type"], as_index=False)
            .agg(n_pairs=("n_pairs", "sum")))
    fig = px.bar(melt, x="source", y="n_pairs", color="pair_type",
                 facet_col="language", title="各源各类型 pair 数（堆叠）",
                 labels={"n_pairs": "条数"})
    st.plotly_chart(fig, use_container_width=True)
