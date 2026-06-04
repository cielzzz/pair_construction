"""Dashboard：总览 KPI + 增长 + 分布饼图 / 堆叠柱"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loader import load_index, load_raw_source


st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 数据看板：总览")

idx = load_index()
raw = load_raw_source()

if idx.empty and raw.empty:
    st.warning("没有索引。先跑 `python app/index_builder.py` 和 `python app/raw_scanner.py --add ...`。")
    st.stop()


# ─── 顶部 KPI ─────────────────────────────────────────────
st.subheader("KPI 速览")
k1, k2, k3, k4, k5 = st.columns(5)

# raw source 总数
total_raw_rows = int(raw["n_rows"].sum()) if not raw.empty else 0
total_raw_hours = float(raw["total_hours"].sum()) if not raw.empty else 0.0
with k1:
    st.metric("🎙️ 上游 raw 总条数", f"{total_raw_rows:,}")
with k2:
    st.metric("⏱️ 上游 raw 总小时", f"{total_raw_hours:,.1f} h")

# pair 总数 (取 is_filtered=False 避免重复)
if not idx.empty:
    orig_idx = idx[idx["is_filtered"] == False]
    total_pairs = int(orig_idx["n_pairs"].sum())
    total_filtered = int(idx[idx["is_filtered"] == True]["n_pairs"].sum())
    total_ref_h = float(orig_idx["ref_hours"].sum())
    total_tgt_h = float(orig_idx["tgt_hours"].sum())
else:
    total_pairs = total_filtered = 0
    total_ref_h = total_tgt_h = 0.0

with k3:
    st.metric("🔗 pair 总数（orig）", f"{total_pairs:,}")
with k4:
    st.metric("✅ pair 总数（filtered）", f"{total_filtered:,}")
with k5:
    st.metric("⏱️ pair ref+tgt 小时", f"{(total_ref_h + total_tgt_h):,.1f} h",
              help=f"ref={total_ref_h:,.1f} h, tgt={total_tgt_h:,.1f} h")

st.divider()


# ─── raw source 分布 ──────────────────────────────────────
if not raw.empty:
    st.subheader("上游数据源 / 语言分布")
    by_src_lang = (raw
                   .groupby(["source", "language"], as_index=False)
                   .agg(n_rows=("n_rows", "sum"), total_hours=("total_hours", "sum")))
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(by_src_lang, x="source", y="n_rows", color="language",
                     title="raw 条数 by source × language",
                     labels={"n_rows": "条数"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(by_src_lang, x="source", y="total_hours", color="language",
                     title="raw 小时数 by source × language",
                     labels={"total_hours": "小时"})
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(by_src_lang, use_container_width=True, hide_index=True)
    st.divider()


# ─── pair 维度 ────────────────────────────────────────────
if not idx.empty:
    st.subheader("pair 维度分析（按未过滤的 *.jsonl 计）")
    orig = idx[idx["is_filtered"] == False].copy()

    # 按 pair_type 聚合
    by_type = (orig
               .groupby("pair_type", as_index=False)
               .agg(n_pairs=("n_pairs", "sum"),
                    ref_hours=("ref_hours", "sum"),
                    tgt_hours=("tgt_hours", "sum")))
    by_type["total_hours"] = by_type["ref_hours"] + by_type["tgt_hours"]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(by_type.sort_values("n_pairs", ascending=False),
                     x="pair_type", y="n_pairs",
                     title="pair 数 by pair_type",
                     labels={"n_pairs": "条数"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(by_type.sort_values("total_hours", ascending=False),
                     x="pair_type", y=["ref_hours", "tgt_hours"],
                     title="pair 小时数 by pair_type（ref + tgt）",
                     labels={"value": "小时", "variable": "side"})
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(by_type, use_container_width=True, hide_index=True)

    st.divider()

    # 按 source × language × pair_type
    st.subheader("source × language × pair_type 热力图")
    pivot = (orig
             .groupby(["source", "language", "pair_type"], as_index=False)
             .agg(n_pairs=("n_pairs", "sum"))
             .pivot_table(index=["source", "language"],
                          columns="pair_type",
                          values="n_pairs",
                          fill_value=0))
    st.dataframe(pivot, use_container_width=True)

    # 按 split 增长（按 jsonl_path 推断时间）
    st.subheader("各 split 总览")
    by_split = (orig
                .groupby(["source", "language", "split"], as_index=False)
                .agg(n_pair_types=("pair_type", "nunique"),
                     n_pairs=("n_pairs", "sum"),
                     ref_hours=("ref_hours", "sum"),
                     tgt_hours=("tgt_hours", "sum")))
    by_split["total_hours"] = by_split["ref_hours"] + by_split["tgt_hours"]
    st.dataframe(by_split.sort_values(["source", "language", "split"]),
                 use_container_width=True, hide_index=True)

    # orig vs filtered 留存
    st.subheader("过滤留存率（11b WavLM sim）")
    filt = idx.pivot_table(index=["split", "pair_type"],
                           columns="is_filtered",
                           values="n_pairs",
                           fill_value=0).reset_index()
    filt.columns = ["split", "pair_type", "orig", "filtered"]
    filt = filt[filt["filtered"] > 0].copy()
    filt["retain_pct"] = (filt["filtered"] / filt["orig"] * 100).round(1)
    st.dataframe(filt.sort_values(["split", "pair_type"]),
                 use_container_width=True, hide_index=True)
