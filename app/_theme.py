"""集中化的 plotly 浅色主题 + 颜色板。其他 page 共用，确保视觉一致。
   命名沿用 BG_DARK / BG_CARD 等以兼容现有 page 代码，但实际值是浅色。
"""
from __future__ import annotations

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go


# ─── 颜色板（light 主题）──────────────────────────────────
ACCENT = "#0ea5e9"            # 主蓝青（比深色版稍深，保证白底对比）
ACCENT_2 = "#38bdf8"
BG_DARK = "#ffffff"           # 历史命名，浅色版本下是页面底色
BG_CARD = "#f8fafc"           # 卡片 / 图表底
TEXT_PRIMARY = "#1e293b"      # 深 slate 主文字
TEXT_MUTED = "#64748b"        # 灰
BORDER_COLOR = "#e2e8f0"      # 卡片/分隔线
GRID_COLOR = "#e2e8f0"

# 15 类 pair_type 用色（赛博风谱）
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
    "Genre_conv":   "#fb7185",   # 浅红粉
    "H1":           "#94a3b8",   # 灰
    "H2":           "#64748b",
    "H3":           "#475569",
    "I":            "#14b8a6",   # 蓝绿（teal）
    "J_fast":       "#eab308",   # 鲜黄
    "J_slow":       "#ca8a04",   # 暗黄
}
# 兜底色序列（pair_type 外的情况）
COLOR_SEQ = ["#00d4ff", "#7ce8ff", "#a78bfa", "#fbbf24", "#34d399", "#f472b6", "#60a5fa", "#fb923c", "#f87171"]


# ─── plotly template 注册（light）─────────────────────────
_LIGHT_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family="-apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
                  color=TEXT_PRIMARY, size=12),
        title=dict(font=dict(size=14, color=TEXT_PRIMARY), x=0.02, xanchor="left"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        colorway=COLOR_SEQ,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR,
                   tickcolor=GRID_COLOR, tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR,
                   tickcolor=GRID_COLOR, tickfont=dict(color=TEXT_MUTED)),
        legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(color=TEXT_PRIMARY)),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor="#ffffff", font_color=TEXT_PRIMARY,
                        bordercolor=ACCENT, font_size=12),
    )
)
# 历史名 pair_dark 留作别名，避免引用方报错
pio.templates["pair_light"] = _LIGHT_TEMPLATE
pio.templates["pair_dark"] = _LIGHT_TEMPLATE
pio.templates.default = "pair_light"


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
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 0 0 1px rgba(14,165,233,0.04), 0 4px 12px rgba(15,23,42,0.04);
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-family: -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        font-weight: 600;
    }
    [data-testid="stMetricDelta"] { color: #0ea5e9 !important; }
    h1, h2, h3 { letter-spacing: 0.01em; color: #0f172a; }
    hr { border-color: #e2e8f0 !important; }
    </style>
    """
