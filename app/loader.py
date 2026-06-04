"""共享 loader：从 parquet 索引和 jsonl 加载数据，带缓存。

约定：
- outputs/<split>/pairs/*.jsonl 是 pair 数据
- app/data/index.parquet 是聚合索引（由 index_builder.py 生成）
- app/data/raw_source.parquet 是上游源 jsonl 聚合（由 raw_scanner.py 生成）
- app/data/duration_cache.parquet 是 path → duration 缓存（可选）
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# streamlit 是可选导入：保证 module 在非 streamlit 上下文也能用（index_builder 也调用此处）
try:
    import streamlit as st
    _CACHE = st.cache_data
except ImportError:
    def _CACHE(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUTS_DIR = PROJ_ROOT / "outputs"

INDEX_PARQUET = DATA_DIR / "index.parquet"
RAW_SOURCE_PARQUET = DATA_DIR / "raw_source.parquet"
DURATION_CACHE_PARQUET = DATA_DIR / "duration_cache.parquet"


# ─── 路径→语言/源 推断 ──────────────────────────────────────

def infer_lang_from_split(split: str, sample_row: Optional[dict] = None) -> str:
    """推断 split 对应语言。先看 split 名，再回退 sample row 的 audio 路径。"""
    s = split.lower()
    if "zh" in s or "_cn" in s or "chinese" in s:
        return "zh"
    if "en" in s or "english" in s:
        return "en"
    # 兜底：看 sample audio 路径
    if sample_row:
        for k in ("reference_audio", "target_audio"):
            v = sample_row.get(k, "") or ""
            if "/zh/" in v:
                return "zh"
            if "/en/" in v:
                return "en"
    return "unknown"


def infer_source_from_path(audio_path: str) -> str:
    """从音频路径里推断数据源名。例：
    /inspire/hdd/.../instruction_0.1_enzh/zh/... → 'instruction_0.1_enzh'
    /inspire/.../vcdata_construction/outputs/instruction_0.1_enzh/... → 'instruction_0.1_enzh'
    """
    if not audio_path:
        return "unknown"
    parts = audio_path.split("/")
    for p in parts:
        if p.startswith("instruction_") and ("_enzh" in p or "_zh" in p or "_en" in p):
            return p
    return "unknown"


# ─── 索引加载 ───────────────────────────────────────────────

@_CACHE(show_spinner=False)
def load_index() -> pd.DataFrame:
    """加载 pair 聚合索引（index_builder 生成）。"""
    if not INDEX_PARQUET.exists():
        return pd.DataFrame(columns=[
            "split", "source", "language", "pair_type", "is_filtered",
            "n_pairs", "jsonl_path",
            "ref_hours", "tgt_hours",
            "ref_emo_dist", "tgt_emo_dist", "source_edit_tag_dist",
            "sim_wavlm_p25", "sim_wavlm_p50", "sim_wavlm_p75", "sim_wavlm_mean",
            "ref_dnsmos_mean", "tgt_dnsmos_mean",
        ])
    return pd.read_parquet(INDEX_PARQUET)


@_CACHE(show_spinner=False)
def load_raw_source() -> pd.DataFrame:
    """加载上游 raw jsonl 源聚合（raw_scanner 生成）。"""
    if not RAW_SOURCE_PARQUET.exists():
        return pd.DataFrame(columns=[
            "source", "language", "split_jsonl",
            "n_rows", "total_hours",
        ])
    return pd.read_parquet(RAW_SOURCE_PARQUET)


@_CACHE(show_spinner=False)
def load_duration_cache() -> dict:
    """audio_path → duration 映射，用于 pair hours 计算。"""
    if not DURATION_CACHE_PARQUET.exists():
        return {}
    df = pd.read_parquet(DURATION_CACHE_PARQUET)
    return dict(zip(df["path"], df["duration"]))


# ─── jsonl 加载（按需，limit） ──────────────────────────────

@_CACHE(show_spinner="加载 pair jsonl ...")
def load_pair_jsonl(jsonl_path: str, max_rows: int = 5000) -> pd.DataFrame:
    """加载一个 pair jsonl 到 DataFrame，平铺 emotion + meta。"""
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            r = json.loads(line)
            ref_emo = r.get("ref_emotion") or {}
            tgt_emo = r.get("tgt_emotion") or {}
            rows.append({
                "pair_id": r.get("pair_id"),
                "pair_type": r.get("pair_type"),
                "reference_audio": r.get("reference_audio"),
                "reference_text": r.get("reference_text", ""),
                "target_audio": r.get("target_audio"),
                "target_text": r.get("target_text", ""),
                "instruction": r.get("instruction", ""),
                "source_edit_tag": r.get("source_edit_tag") or r.get("source_edit"),
                "ref_emo_top1": ref_emo.get("top1_label"),
                "ref_emo_p_neu": ref_emo.get("P_neutral"),
                "ref_sv_label": ref_emo.get("sv_label"),
                "ref_dnsmos_ovrl": ref_emo.get("dnsmos_ovrl"),
                "tgt_emo_top1": tgt_emo.get("top1_label"),
                "tgt_emo_p_neu": tgt_emo.get("P_neutral"),
                "tgt_sv_label": tgt_emo.get("sv_label"),
                "tgt_dnsmos_ovrl": tgt_emo.get("dnsmos_ovrl"),
                "sim_wavlm": r.get("ref_vs_tgt_speaker_sim_wavlm"),
                "ref_dnsmos_bak": r.get("ref_dnsmos_bak"),
                "tgt_dnsmos_bak": r.get("tgt_dnsmos_bak"),
                "meta_split": (r.get("meta") or {}).get("split"),
                "_raw": r,  # 保留完整原始 dict 供 detail view
            })
    return pd.DataFrame(rows)


# ─── 多维过滤辅助 ───────────────────────────────────────────

def filter_index(
    idx: pd.DataFrame,
    sources: Optional[list] = None,
    languages: Optional[list] = None,
    splits: Optional[list] = None,
    pair_types: Optional[list] = None,
    is_filtered: Optional[bool] = None,
) -> pd.DataFrame:
    df = idx.copy()
    if sources:
        df = df[df["source"].isin(sources)]
    if languages:
        df = df[df["language"].isin(languages)]
    if splits:
        df = df[df["split"].isin(splits)]
    if pair_types:
        df = df[df["pair_type"].isin(pair_types)]
    if is_filtered is not None:
        df = df[df["is_filtered"] == is_filtered]
    return df


def filter_pairs(
    df: pd.DataFrame,
    pair_types: Optional[list] = None,
    source_edit_tags: Optional[list] = None,
    ref_emos: Optional[list] = None,
    tgt_emos: Optional[list] = None,
    sim_range: Optional[tuple] = None,
    ref_dnsmos_range: Optional[tuple] = None,
    tgt_dnsmos_range: Optional[tuple] = None,
    instruction_query: str = "",
    text_query: str = "",
) -> pd.DataFrame:
    q = df.copy()
    if pair_types:
        q = q[q["pair_type"].isin(pair_types)]
    if source_edit_tags:
        # None 也算一个 tag
        keep_none = None in source_edit_tags
        tags = [t for t in source_edit_tags if t is not None]
        if keep_none and tags:
            q = q[q["source_edit_tag"].isin(tags) | q["source_edit_tag"].isna()]
        elif keep_none:
            q = q[q["source_edit_tag"].isna()]
        elif tags:
            q = q[q["source_edit_tag"].isin(tags)]
    if ref_emos:
        q = q[q["ref_emo_top1"].isin(ref_emos)]
    if tgt_emos:
        q = q[q["tgt_emo_top1"].isin(tgt_emos)]
    if sim_range and "sim_wavlm" in q.columns:
        lo, hi = sim_range
        q = q[(q["sim_wavlm"].isna()) | ((q["sim_wavlm"] >= lo) & (q["sim_wavlm"] <= hi))]
    if ref_dnsmos_range and "ref_dnsmos_bak" in q.columns:
        lo, hi = ref_dnsmos_range
        q = q[(q["ref_dnsmos_bak"].isna()) | ((q["ref_dnsmos_bak"] >= lo) & (q["ref_dnsmos_bak"] <= hi))]
    if tgt_dnsmos_range and "tgt_dnsmos_bak" in q.columns:
        lo, hi = tgt_dnsmos_range
        q = q[(q["tgt_dnsmos_bak"].isna()) | ((q["tgt_dnsmos_bak"] >= lo) & (q["tgt_dnsmos_bak"] <= hi))]
    if instruction_query:
        q = q[q["instruction"].str.contains(instruction_query, case=False, na=False)]
    if text_query:
        m_r = q["reference_text"].str.contains(text_query, case=False, na=False)
        m_t = q["target_text"].str.contains(text_query, case=False, na=False)
        q = q[m_r | m_t]
    return q
