"""
Automated Dynamic Pivot Support & Resistance Charting Component for Streamlit.

Functions:
- Renders responsive Plotly multi-timeframe candlestick charts with volume.
- Automatically computes dynamic Pivot Support / Resistance corridors.
- Overlays +2.5 ATR TP1, +4.5 ATR TP2, and Stop Loss brackets with entry execution flags.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick_sr_chart(
    ticker: str,
    price_df: pd.DataFrame,
    spot_price: float,
    tp1: Optional[float] = None,
    tp2: Optional[float] = None,
    sl: Optional[float] = None,
    entry_price: Optional[float] = None,
) -> go.Figure:
    """
    Constructs an institutional-grade interactive Plotly chart with automated S/R levels.
    """
    if price_df.empty or len(price_df) < 10:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Insufficient price history to render chart for {ticker}",
            showarrow=False,
            font=dict(size=14, color="#8b949e"),
        )
        fig.update_layout(template="plotly_dark", height=450)
        return fig

    # Work with the most recent 120 trading days for clean visual layout
    df = price_df.tail(120).copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    # 1. Candlestick Main Chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=f"{ticker} Price",
            increasing_line_color="#00D4AA",
            decreasing_line_color="#FF4B4B",
        ),
        row=1,
        col=1,
    )

    # 2. 21-day EMA & 200-day SMA Overlay
    if len(df) >= 21:
        ema21 = df["Close"].ewm(span=21, adjust=False).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ema21,
                name="21 EMA",
                line=dict(color="#FFD700", width=1.5),
            ),
            row=1,
            col=1,
        )

    if "sma200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sma200"],
                name="200 SMA",
                line=dict(color="#3B82F6", width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )

    # 3. Dynamic Support & Resistance Pivots
    rolling_max = df["High"].rolling(20, center=True).max().dropna()
    rolling_min = df["Low"].rolling(20, center=True).min().dropna()

    if not rolling_max.empty:
        res_level = float(rolling_max.iloc[-1])
        fig.add_hline(
            y=res_level,
            line=dict(color="#FF4B4B", width=1, dash="dash"),
            annotation_text=f"Resistance ${res_level:.2f}",
            annotation_position="top right",
            row=1,
            col=1,
        )

    if not rolling_min.empty:
        sup_level = float(rolling_min.iloc[-1])
        fig.add_hline(
            y=sup_level,
            line=dict(color="#00D4AA", width=1, dash="dash"),
            annotation_text=f"Support ${sup_level:.2f}",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

    # 4. Target & Stop Loss Levels
    if tp1 and tp1 > 0:
        fig.add_hline(
            y=tp1,
            line=dict(color="#10B981", width=1.5, dash="dot"),
            annotation_text=f"TP1 (+2.5 ATR) ${tp1:.2f}",
            annotation_position="top left",
            row=1,
            col=1,
        )

    if tp2 and tp2 > 0:
        fig.add_hline(
            y=tp2,
            line=dict(color="#34D399", width=2.0),
            annotation_text=f"TP2 (+4.5 ATR Runner) ${tp2:.2f}",
            annotation_position="top left",
            row=1,
            col=1,
        )

    if sl and sl > 0:
        fig.add_hline(
            y=sl,
            line=dict(color="#EF4444", width=2.0),
            annotation_text=f"Stop Loss ${sl:.2f}",
            annotation_position="bottom left",
            row=1,
            col=1,
        )

    # 5. Volume Sub-Chart
    if "Volume" in df.columns:
        vol_colors = np.where(df["Close"] >= df["Open"], "#00D4AA", "#FF4B4B")
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11),
        ),
    )

    return fig
