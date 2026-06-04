"""pair_construction 数据看板入口

启动：
    streamlit run app/app.py --server.port 8501 --server.headless true

侧栏内置 3 页（streamlit 自动从 pages/ 目录扫描）：
- Dashboard：总览 KPI + 增长 + 数据源 / 语言 / 类型分布
- Browser：单 pair 浏览器 + 多维交叉过滤 + 音频对比
- Source compare：多数据源横向对比
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader import (
    INDEX_PARQUET, RAW_SOURCE_PARQUET, DURATION_CACHE_PARQUET,
    load_index, load_raw_source, load_duration_cache,
)


st.set_page_config(
    page_title="pair_construction 数据看板",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("pair_construction 数据看板")
st.caption("从 vcdata + editx 输出构造的 11 类 TTS 训练 pair —— 多数据源 / 多语言 / 多 split 聚合视图")

# ─── 索引状态 ─────────────────────────────────────────────
idx = load_index()
raw = load_raw_source()
dur = load_duration_cache()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("pair 索引行数", f"{len(idx):,}")
    if not idx.empty:
        st.caption(f"覆盖 {idx['split'].nunique()} 个 split × "
                   f"{idx['pair_type'].nunique()} 类")
with c2:
    st.metric("raw source 索引行数", f"{len(raw):,}")
    if not raw.empty:
        n_sources = raw["source"].nunique()
        n_langs = raw["language"].nunique()
        st.caption(f"{n_sources} 数据源 × {n_langs} 语言")
with c3:
    st.metric("duration cache 条数", f"{len(dur):,}")
    st.caption("audio path → duration（影响 pair hours 计算）")

st.divider()

st.markdown("### 索引状态 / 重建命令")

with st.expander("如何更新索引？"):
    st.markdown(f"""
**pair 索引**（聚合所有 `outputs/<split>/pairs/*.jsonl`）：
```bash
python app/index_builder.py
# 当前 → {INDEX_PARQUET}
```

**raw source 索引** + **duration cache**（用上游 manifest 的 duration 字段）：
```bash
python app/raw_scanner.py \\
    --add instruction_0.1_enzh:zh:/inspire/hdd/.../kxhuang/instructtts_data/instruction_0.1_enzh/zh \\
    --add instruction_0.1_enzh:en:/inspire/hdd/.../kxhuang/instructtts_data/instruction_0.1_enzh/en
# 当前 → {RAW_SOURCE_PARQUET}
#       {DURATION_CACHE_PARQUET}
```

`duration_cache` 建好后再跑一次 `index_builder.py`，pair 的 ref_hours / tgt_hours 才会有值。
""")

st.markdown("### 三个子页")
st.markdown("""
左侧 sidebar 进入：

- **📊 Dashboard** — 总览 KPI、数据源 / 语言 / 类型分布、增长曲线
- **🔍 Browser** — 多维过滤 + 单 pair 详情 + 音频对比
- **📈 Source compare** — 多数据源横向对比（每加一个新源就有一列）
""")

if idx.empty and raw.empty:
    st.warning("两个索引都为空。请先运行 `python app/index_builder.py` 和 `python app/raw_scanner.py --add ...`，刷新页面。")
