"""
Institutional TradingView-Style Interactive Candlestick & Order Flow Engine.
===========================================================================
Generates responsive Plotly dual-pane charts featuring:
1. Candlestick Price Action (Goldman Slate palette).
2. Volume Profile (VPVR) & Point of Control (PoC) high-volume institutional node.
3. Value Area High (VAH) & Value Area Low (VAL) 70% volume distribution.
4. Fair Value Gaps (FVG) and Liquidity Imbalance Zones.
5. Opening Range Breakout (ORB) High & Low boundary references.
6. Sub-pane Volume Histogram with 20-period Moving Average.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import get_logger

logger = get_logger(__name__)


def calculate_volume_profile(
    df: pd.DataFrame, n_bins: int = 24
) -> Tuple[List[Dict[str, Any]], float, float, float]:
    """
    Computes horizontal Volume Profile across price levels.

    Returns:
        (bins_list, poc_price, vah_price, val_price)
    """
    if df.empty or "Close" not in df.columns:
        return [], 0.0, 0.0, 0.0

    high_col = "High" if "High" in df.columns else "Close"
    low_col = "Low" if "Low" in df.columns else "Close"
    vol_col = "Volume" if "Volume" in df.columns else None

    min_p = float(df[low_col].min())
    max_p = float(df[high_col].max())

    if min_p >= max_p or min_p <= 0:
        return [], float(df["Close"].iloc[-1]), min_p, max_p

    bin_edges = np.linspace(min_p, max_p, n_bins + 1)
    bin_volumes = np.zeros(n_bins)

    for _, row in df.iterrows():
        p = float(row["Close"])
        v = float(row[vol_col]) if vol_col else 1000.0
        # Determine bin index
        idx = int(np.digitize(p, bin_edges)) - 1
        idx = max(0, min(n_bins - 1, idx))
        bin_volumes[idx] += v

    total_vol = float(np.sum(bin_volumes))
    poc_idx = int(np.argmax(bin_volumes))
    poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0)

    # Calculate Value Area (70% of total volume around PoC)
    target_va_vol = total_vol * 0.70
    accum_vol = bin_volumes[poc_idx]
    up_idx = poc_idx
    down_idx = poc_idx

    while accum_vol < target_va_vol and (up_idx < n_bins - 1 or down_idx > 0):
        next_up_vol = bin_volumes[up_idx + 1] if up_idx < n_bins - 1 else 0
        next_down_vol = bin_volumes[down_idx - 1] if down_idx > 0 else 0

        if next_up_vol >= next_down_vol and up_idx < n_bins - 1:
            up_idx += 1
            accum_vol += next_up_vol
        elif down_idx > 0:
            down_idx -= 1
            accum_vol += next_down_vol
        else:
            break

    val_price = float(bin_edges[down_idx])
    vah_price = float(bin_edges[up_idx + 1])

    bins_list = []
    max_bin_vol = float(np.max(bin_volumes)) if np.max(bin_volumes) > 0 else 1.0
    for i in range(n_bins):
        mid = float((bin_edges[i] + bin_edges[i + 1]) / 2.0)
        bins_list.append(
            {
                "price": mid,
                "volume": float(bin_volumes[i]),
                "rel_volume": float(bin_volumes[i] / max_bin_vol),
                "is_poc": (i == poc_idx),
                "in_value_area": (down_idx <= i <= up_idx),
            }
        )

    return bins_list, poc_price, vah_price, val_price


def detect_fair_value_gaps(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects 3-candle Fair Value Gaps (FVG) / institutional liquidity imbalances.
    """
    if len(df) < 5 or "High" not in df.columns or "Low" not in df.columns:
        return []

    fvgs = []
    for i in range(2, len(df)):
        c1_high = float(df["High"].iloc[i - 2])
        c1_low = float(df["Low"].iloc[i - 2])
        c3_high = float(df["High"].iloc[i])
        c3_low = float(df["Low"].iloc[i])

        # Bullish FVG: Low of candle 3 is above High of candle 1
        if c3_low > c1_high:
            fvgs.append(
                {
                    "type": "BULLISH_FVG",
                    "top": c3_low,
                    "bottom": c1_high,
                    "date": df.index[i - 1],
                    "end_date": df.index[-1],
                }
            )
        # Bearish FVG: High of candle 3 is below Low of candle 1
        elif c3_high < c1_low:
            fvgs.append(
                {
                    "type": "BEARISH_FVG",
                    "top": c1_low,
                    "bottom": c3_high,
                    "date": df.index[i - 1],
                    "end_date": df.index[-1],
                }
            )

    # Return the most recent 4 active FVGs
    return fvgs[-4:]


def build_orderflow_candlestick_chart(
    price_df: pd.DataFrame,
    ticker: str = "ASSET",
    sm_data: Optional[Dict[str, Any]] = None,
    orb_levels: Optional[Dict[str, float]] = None,
    height: int = 680,
) -> go.Figure:
    """
    Builds an institutional TradingView-style dual-pane interactive chart.

    Panes:
        Row 1 (75%): Candlestick price action, PoC, Value Area, FVGs, and ORB levels.
        Row 2 (25%): Color-coded volume bars and 20-period volume SMA.
    """
    if price_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No price data available for chart generation",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#94A3B8"),
        )
        fig.update_layout(template="plotly_dark", height=height)
        return fig

    # Ensure required columns exist
    df = price_df.copy()
    if "Close" not in df.columns:
        return go.Figure()
    for col in ["Open", "High", "Low"]:
        if col not in df.columns:
            df[col] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 1000.0

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    # 1. Candlestick Price Chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#10B981",  # Emerald Green
            increasing_fillcolor="#10B981",
            decreasing_line_color="#EF4444",  # Crimson Red
            decreasing_fillcolor="#EF4444",
            line_width=1.2,
        ),
        row=1,
        col=1,
    )

    # 2. Add Moving Averages if present
    if "sma50" in df.columns or "ma21" in df.columns:
        ma_col = "sma50" if "sma50" in df.columns else "ma21"
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[ma_col],
                mode="lines",
                name=f"{ma_col.upper()}",
                line=dict(color="#38BDF8", width=1.3),
            ),
            row=1,
            col=1,
        )
    if "sma200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sma200"],
                mode="lines",
                name="SMA 200",
                line=dict(color="#F59E0B", width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # 3. Volume Profile & PoC Calculation
    bins, poc_price, vah_price, val_price = calculate_volume_profile(df, n_bins=20)

    # Point of Control (PoC) Line
    if poc_price > 0:
        fig.add_hline(
            y=poc_price,
            line_width=1.5,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"PoC: ${poc_price:.2f}",
            annotation_position="top right",
            annotation_font=dict(color="#F59E0B", size=10),
            row=1,
            col=1,
        )

    # Value Area Shading (70% Volume Node)
    if vah_price > val_price > 0:
        fig.add_hrect(
            y0=val_price,
            y1=vah_price,
            fillcolor="rgba(56, 189, 248, 0.05)",
            line_width=1,
            line_color="rgba(56, 189, 248, 0.25)",
            annotation_text="Value Area (70%)",
            annotation_position="bottom right",
            annotation_font=dict(color="#38BDF8", size=9),
            row=1,
            col=1,
        )

    # 4. Fair Value Gaps (FVG)
    fvgs = detect_fair_value_gaps(df)
    for fvg in fvgs:
        is_bull = fvg["type"] == "BULLISH_FVG"
        fill_color = (
            "rgba(16, 185, 129, 0.12)" if is_bull else "rgba(239, 68, 68, 0.12)"
        )
        border_color = (
            "rgba(16, 185, 129, 0.4)" if is_bull else "rgba(239, 68, 68, 0.4)"
        )

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=fvg["date"],
            x1=fvg["end_date"],
            y0=fvg["bottom"],
            y1=fvg["top"],
            fillcolor=fill_color,
            line=dict(color=border_color, width=1, dash="dot"),
            row=1,
            col=1,
        )

    # 5. Opening Range Breakout (ORB) Levels
    if orb_levels:
        orb_h = orb_levels.get("orb_high")
        orb_l = orb_levels.get("orb_low")
        if orb_h and orb_h > 0:
            fig.add_hline(
                y=orb_h,
                line_width=1.2,
                line_dash="dot",
                line_color="#10B981",
                annotation_text=f"ORB High: ${orb_h:.2f}",
                annotation_position="top left",
                annotation_font=dict(color="#10B981", size=10),
                row=1,
                col=1,
            )
        if orb_l and orb_l > 0:
            fig.add_hline(
                y=orb_l,
                line_width=1.2,
                line_dash="dot",
                line_color="#EF4444",
                annotation_text=f"ORB Low: ${orb_l:.2f}",
                annotation_position="bottom left",
                annotation_font=dict(color="#EF4444", size=10),
                row=1,
                col=1,
            )

    # 6. Volume Histogram Pane
    vol_colors = [
        "#10B981" if c >= o else "#EF4444" for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.75,
        ),
        row=2,
        col=1,
    )

    # Volume 20-period Moving Average
    vol_ma20 = df["Volume"].rolling(20, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=vol_ma20,
            mode="lines",
            name="Vol MA20",
            line=dict(color="#E2E8F0", width=1.2),
        ),
        row=2,
        col=1,
    )

    # 7. Institutional Dark Layout Styling
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0A0E17",
        plot_bgcolor="#0F172A",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10, color="#94A3B8"),
            bgcolor="rgba(15, 23, 42, 0.6)",
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.05)",
        linecolor="rgba(255, 255, 255, 0.1)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.05)",
        linecolor="rgba(255, 255, 255, 0.1)",
        row=1,
        col=1,
        title_text="Price ($)",
        title_font=dict(size=11, color="#94A3B8"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.05)",
        linecolor="rgba(255, 255, 255, 0.1)",
        row=2,
        col=1,
        title_text="Volume",
        title_font=dict(size=11, color="#94A3B8"),
    )

    return fig
