"""Browser：多维交叉过滤 + 单 pair 详情 + 音频对比"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loader import (
    load_index, load_pair_jsonl,
    filter_index, filter_pairs,
)


st.set_page_config(page_title="Browser", layout="wide")
st.title("🔍 pair 浏览器")

idx = load_index()
if idx.empty:
    st.warning("索引为空。先跑 `python app/index_builder.py`。")
    st.stop()


# ─── 侧栏：聚合层过滤（选哪些 jsonl 进来）───────────────────
st.sidebar.header("第一层：聚合过滤")

sources = sorted(idx["source"].dropna().unique().tolist())
languages = sorted(idx["language"].dropna().unique().tolist())
splits_all = sorted(idx["display_split"].unique().tolist())
pair_types_all = sorted(idx["pair_type"].unique().tolist())

sel_sources = st.sidebar.multiselect("source", sources, default=sources)
sel_langs = st.sidebar.multiselect("language", languages, default=languages)
sel_splits = st.sidebar.multiselect("split", splits_all, default=splits_all)
sel_types = st.sidebar.multiselect("pair_type", pair_types_all, default=pair_types_all)
filtered_mode = st.sidebar.radio(
    "用 orig 还是 filtered jsonl？",
    options=["orig（未过滤）", "filtered（已过 11b sim）"],
    index=0,
)
is_filtered = filtered_mode.startswith("filtered")

# 用 display_split 列做过滤（sel_splits 里装的就是 display 值）
sub_idx = idx.copy()
if sel_sources: sub_idx = sub_idx[sub_idx["source"].isin(sel_sources)]
if sel_langs:   sub_idx = sub_idx[sub_idx["language"].isin(sel_langs)]
if sel_splits:  sub_idx = sub_idx[sub_idx["display_split"].isin(sel_splits)]
if sel_types:   sub_idx = sub_idx[sub_idx["pair_type"].isin(sel_types)]
sub_idx = sub_idx[sub_idx["is_filtered"] == is_filtered]

if sub_idx.empty:
    st.warning("聚合层过滤后没数据。")
    st.stop()

st.markdown(f"**第一层选中 {len(sub_idx)} 个 jsonl，"
            f"合计 {int(sub_idx['n_pairs'].sum()):,} 条 pair**")

with st.expander("详细：各 jsonl 行数"):
    st.dataframe(sub_idx[["display_split", "language", "pair_type", "n_pairs",
                          "jsonl_path"]].rename(columns={"display_split": "split"}),
                 use_container_width=True, hide_index=True)


# ─── 加载实际 pair（取选中 jsonl 的合并）───────────────────
LOAD_CAP = 10_000
st.sidebar.header("第二层：pair 级过滤")
load_cap = st.sidebar.number_input(
    "单 jsonl 最大加载行数",
    min_value=100, max_value=200_000, value=LOAD_CAP, step=1000,
    help="防止 OOM。超过时随机采样取前 N 条。"
)

dfs = []
for _, row in sub_idx.iterrows():
    df = load_pair_jsonl(row["jsonl_path"], max_rows=int(load_cap))
    if not df.empty:
        df["__split"] = row["display_split"]
        df["__source"] = row["source"]
        df["__language"] = row["language"]
        df["__jsonl"] = row["jsonl_path"]
        dfs.append(df)

if not dfs:
    st.warning("加载的 jsonl 都是空的。")
    st.stop()

pairs = pd.concat(dfs, ignore_index=True)
st.markdown(f"实际加载 **{len(pairs):,}** 条")


# ─── pair 级过滤 ─────────────────────────────────────────
ref_emos = sorted([x for x in pairs["ref_emo_top1"].dropna().unique().tolist()])
tgt_emos = sorted([x for x in pairs["tgt_emo_top1"].dropna().unique().tolist()])
tags_all = sorted([x for x in pairs["source_edit_tag"].dropna().unique().tolist()])
has_none_tag = pairs["source_edit_tag"].isna().any()
if has_none_tag:
    tags_all_disp = tags_all + ["(none)"]
else:
    tags_all_disp = tags_all

sel_ref_emo = st.sidebar.multiselect("ref_emotion.top1", ref_emos, default=ref_emos)
sel_tgt_emo = st.sidebar.multiselect("tgt_emotion.top1", tgt_emos, default=tgt_emos)
sel_tags_disp = st.sidebar.multiselect("source_edit_tag", tags_all_disp, default=tags_all_disp)
sel_tags = [None if t == "(none)" else t for t in sel_tags_disp]

# sim / dnsmos 范围
sim_min, sim_max = float(pairs["sim_wavlm"].min(skipna=True) or 0.0), float(pairs["sim_wavlm"].max(skipna=True) or 1.0)
if sim_min == sim_max:
    sim_min, sim_max = 0.0, 1.0
sim_range = st.sidebar.slider("sim_wavlm 范围", 0.0, 1.0,
                              (max(0.0, sim_min), min(1.0, sim_max)), 0.01)

ref_b_min = float(pairs["ref_dnsmos_bak"].min(skipna=True) or 0.0)
ref_b_max = float(pairs["ref_dnsmos_bak"].max(skipna=True) or 5.0)
if ref_b_min == ref_b_max:
    ref_b_min, ref_b_max = 0.0, 5.0
ref_dns_range = st.sidebar.slider("ref_dnsmos_bak 范围", 0.0, 5.0,
                                  (max(0.0, ref_b_min), min(5.0, ref_b_max)), 0.05)

tgt_b_min = float(pairs["tgt_dnsmos_bak"].min(skipna=True) or 0.0)
tgt_b_max = float(pairs["tgt_dnsmos_bak"].max(skipna=True) or 5.0)
if tgt_b_min == tgt_b_max:
    tgt_b_min, tgt_b_max = 0.0, 5.0
tgt_dns_range = st.sidebar.slider("tgt_dnsmos_bak 范围", 0.0, 5.0,
                                  (max(0.0, tgt_b_min), min(5.0, tgt_b_max)), 0.05)

instr_q = st.sidebar.text_input("instruction 关键词", "")
text_q = st.sidebar.text_input("ref/tgt 文本关键词", "")

filtered_pairs = filter_pairs(
    pairs,
    pair_types=None,  # 一层已过
    source_edit_tags=sel_tags if sel_tags != tags_all_disp else None,
    ref_emos=sel_ref_emo if sel_ref_emo != ref_emos else None,
    tgt_emos=sel_tgt_emo if sel_tgt_emo != tgt_emos else None,
    sim_range=sim_range,
    ref_dnsmos_range=ref_dns_range,
    tgt_dnsmos_range=tgt_dns_range,
    instruction_query=instr_q,
    text_query=text_q,
)

st.subheader(f"过滤结果：{len(filtered_pairs):,} 条")

if filtered_pairs.empty:
    st.info("当前过滤条件下没匹配数据。")
    st.stop()

# ─── 表格 + 单条详情 ─────────────────────────────────────
show_cols = ["pair_id", "pair_type", "__split", "__language",
             "ref_emo_top1", "tgt_emo_top1", "source_edit_tag",
             "sim_wavlm", "ref_dnsmos_bak", "tgt_dnsmos_bak",
             "instruction", "reference_text", "target_text"]
st.dataframe(filtered_pairs[show_cols], use_container_width=True, hide_index=True,
             height=350)

st.divider()
st.subheader("单条详情 + 音频对比")

selected_idx = st.number_input(
    "选第几条（0-based）",
    min_value=0,
    max_value=max(0, len(filtered_pairs) - 1),
    value=0,
    step=1,
)
row = filtered_pairs.iloc[int(selected_idx)]

st.markdown(f"**pair_id**: `{row['pair_id']}`  •  "
            f"**type**: `{row['pair_type']}`  •  "
            f"**split**: `{row['__split']}` ({row['__language']})")

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### reference")
    st.markdown(f"**text**: {row['reference_text']}")
    if row.get("reference_audio"):
        try:
            st.audio(row["reference_audio"])
        except Exception as e:
            st.warning(f"音频加载失败: {e}\n路径: `{row['reference_audio']}`")
    st.json({
        "top1": row.get("ref_emo_top1"),
        "p_neutral": row.get("ref_emo_p_neu"),
        "sv_label": row.get("ref_sv_label"),
        "dnsmos_ovrl": row.get("ref_dnsmos_ovrl"),
        "dnsmos_bak": row.get("ref_dnsmos_bak"),
    })
with c2:
    st.markdown("#### target")
    st.markdown(f"**text**: {row['target_text']}")
    if row.get("target_audio"):
        try:
            st.audio(row["target_audio"])
        except Exception as e:
            st.warning(f"音频加载失败: {e}\n路径: `{row['target_audio']}`")
    st.json({
        "top1": row.get("tgt_emo_top1"),
        "p_neutral": row.get("tgt_emo_p_neu"),
        "sv_label": row.get("tgt_sv_label"),
        "dnsmos_ovrl": row.get("tgt_dnsmos_ovrl"),
        "dnsmos_bak": row.get("tgt_dnsmos_bak"),
    })

st.markdown(f"**instruction**: {row['instruction']}")
st.markdown(f"**sim_wavlm**: {row.get('sim_wavlm')}  •  "
            f"**source_edit_tag**: {row.get('source_edit_tag')}")

with st.expander("完整 JSON"):
    raw = row.get("_raw")
    if isinstance(raw, dict):
        st.json(raw)
