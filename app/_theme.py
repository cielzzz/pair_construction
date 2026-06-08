"""集中化的 plotly 暗色主题 + 颜色板。其他 page 共用，确保视觉一致。"""
from __future__ import annotations

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go


# ─── 颜色板 ──────────────────────────────────────────────
ACCENT = "#00d4ff"            # 主青蓝
ACCENT_2 = "#7ce8ff"
BG_DARK = "#0a0e1a"
BG_CARD = "#141a2b"
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED = "#94a3b8"

# 11 类 pair_type 用色（赛博风谱）
PAIR_TYPE_COLORS = {
    "A":            "#60a5fa",   # 蓝
    "B":            "#f472b6",   # 粉
    "C":            "#fb923c",   # 橙
    "C-mixed":      "#fbbf24",   # 黄橙
    "C_mixed":      "#fbbf24",
    "D":            "#34d399",   # 青绿
    "D-st":         "#22d3ee",   # 青
    "D_st":         "#22d3ee",
    "D_cross_emo":  "#a78bfa",   # 紫
    "Genre":        "#f87171",   # 红
    "H1":           "#94a3b8",   # 灰
    "H2":           "#64748b",
    "H3":           "#475569",
}
# 兜底色序列（pair_type 外的情况）
COLOR_SEQ = ["#00d4ff", "#7ce8ff", "#a78bfa", "#fbbf24", "#34d399", "#f472b6", "#60a5fa", "#fb923c", "#f87171"]


# ─── plotly template 注册 ───────────────────────────────
_DARK_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
                  color=TEXT_PRIMARY, size=12),
        title=dict(font=dict(size=14, color=TEXT_PRIMARY), x=0.02, xanchor="left"),
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        colorway=COLOR_SEQ,
        xaxis=dict(gridcolor="#1f2738", zerolinecolor="#1f2738", linecolor="#1f2738",
                   tickcolor="#1f2738", tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(gridcolor="#1f2738", zerolinecolor="#1f2738", linecolor="#1f2738",
                   tickcolor="#1f2738", tickfont=dict(color=TEXT_MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_PRIMARY)),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=BG_DARK, font_color=TEXT_PRIMARY,
                        bordercolor=ACCENT, font_size=12),
    )
)
pio.templates["pair_dark"] = _DARK_TEMPLATE
pio.templates.default = "pair_dark"


def pair_color_seq(pair_types):
    """根据 pair_type 列表返回对应颜色（不在表里的用默认 cycle）。"""
    out = []
    cycle_i = 0
    for t in pair_types:
        c = PAIR_TYPE_COLORS.get(t)
        if c is None:
            c = COLOR_SEQ[cycle_i % len(COLOR_SEQ)]
            cycle_i += 1
        out.append(c)
    return out


def kpi_style():
    """统一的 KPI 数字 CSS。给 st.markdown(... unsafe_allow_html=True) 用。"""
    return """
    <style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #141a2b 0%, #0f1422 100%);
        border: 1px solid #1f2738;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 0 0 1px rgba(0,212,255,0.03), 0 4px 12px rgba(0,0,0,0.2);
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-family: ui-monospace, monospace !important;
    }
    [data-testid="stMetricDelta"] { color: #00d4ff !important; }
    h1, h2, h3 { letter-spacing: 0.02em; }
    hr { border-color: #1f2738 !important; }
    </style>
    """
