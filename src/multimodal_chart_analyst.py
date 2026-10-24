"""Multimodal Chart Analyst & Wyckoff / Liquidity Sweep Radar.

This module blends institutional Smart Money Concepts (SMC) and Wyckoff
Methodology with multimodal visual intelligence. It renders high-fidelity
candlestick charts with volume profile and liquidity overlays, then
extracts market structure, liquidity sweeps, Fair Value Gaps (FVGs),
order blocks, and Wyckoff cycle phases.

Supports both:
1. Multimodal Gemini Vision API analysis (via Google Gemini REST API).
2. Pure deterministic algorithmic SMC/Wyckoff engine for offline, CI/CD,
   and zero-latency local fallback.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("Sentilyze.MultimodalChartAnalyst")
_MATPLOTLIB_LOCK = threading.Lock()


@dataclass
class FVGZone:
    """Fair Value Gap zone specification."""

    top: float
    bottom: float
    fvg_type: str  # 'BULLISH' or 'BEARISH'
    bar_index: int
    mitigated: bool = False


@dataclass
class WyckoffAnalysisResult:
    """Institutional market structure and Wyckoff analysis result."""

    ticker: str
    wyckoff_phase: str  # e.g., 'ACCUMULATION_SPRING', 'MARKUP', 'DISTRIBUTION_UTAD', 'MARKDOWN', 'RANGE_CONSOLIDATION'
    market_bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float  # [0.0, 1.0]
    liquidity_sweep_detected: bool
    sweep_type: str  # 'SELL_SIDE_LIQUIDITY_RUN', 'BUY_SIDE_LIQUIDITY_RUN', 'NONE'
    sweep_price_level: Optional[float] = None
    order_block_level: Optional[float] = None
    order_block_type: Optional[str] = None  # 'BULLISH_OB', 'BEARISH_OB', 'NONE'
    fair_value_gaps: List[Dict[str, Any]] = field(default_factory=list)
    key_support: Optional[float] = None
    key_resistance: Optional[float] = None
    analyst_source: str = "ALGORITHMIC"  # 'GEMINI_MULTIMODAL' or 'ALGORITHMIC'
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


class MultimodalChartAnalyst:
    """Institutional Multimodal & Algorithmic Wyckoff Chart Analyst."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        timeout: int = 15,
    ) -> None:
        """Initialize analyst with optional Gemini credentials and model configuration."""
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        )
        self.model_name = model_name
        self.timeout = timeout

    # ------------------------------------------------------------------
    # 1. High-Resolution Visual Chart Rendering
    # ------------------------------------------------------------------
    def render_chart_image(
        self,
        df: pd.DataFrame,
        ticker: str = "TICKER",
        lookback_bars: int = 60,
        dpi: int = 120,
    ) -> bytes:
        """Render institutional candlestick chart with volume and liquidity markers to PNG bytes.

        Thread-safe matplotlib rendering using _MATPLOTLIB_LOCK.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        if df.empty or len(df) < 5:
            # Generate minimal blank chart for edge cases
            with _MATPLOTLIB_LOCK:
                fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
                ax.text(
                    0.5,
                    0.5,
                    f"Insufficient Data for {ticker}",
                    ha="center",
                    va="center",
                )
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                plt.close(fig)
                return buf.getvalue()

        # Slice recent lookback window
        window = df.tail(lookback_bars).copy().reset_index(drop=True)

        # Standardize column names (lowercase)
        col_map = {c: c.lower() for c in window.columns}
        window = window.rename(columns=col_map)

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in window.columns:
                raise ValueError(f"Required column '{col}' missing from DataFrame.")

        # Calculate fast indicators for visual cues
        window["sma_20"] = window["close"].rolling(20, min_periods=1).mean().bfill()
        window["rolling_std"] = (
            window["close"].rolling(20, min_periods=1).std().fillna(0.0)
        )
        window["bb_upper"] = window["sma_20"] + (2.0 * window["rolling_std"])
        window["bb_lower"] = window["sma_20"] - (2.0 * window["rolling_std"])

        with _MATPLOTLIB_LOCK:
            fig, (ax_main, ax_vol) = plt.subplots(
                2,
                1,
                figsize=(10, 6),
                dpi=dpi,
                gridspec_kw={"height_ratios": [3.5, 1.0], "hspace": 0.08},
                facecolor="#0E1117",
            )

            # Background styling
            ax_main.set_facecolor("#161B22")
            ax_vol.set_facecolor("#161B22")
            for ax in (ax_main, ax_vol):
                ax.tick_params(colors="#8B949E", labelsize=8)
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, color="#30363D")
                for spine in ax.spines.values():
                    spine.set_color("#30363D")

            # Draw Candlesticks
            for idx, row in window.iterrows():
                o, h, l, c = (
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                )
                is_bull = c >= o
                color = "#238636" if is_bull else "#DA3633"

                # High-Low Wick
                ax_main.plot([idx, idx], [l, h], color=color, linewidth=1.0, zorder=2)

                # Open-Close Body
                body_bottom = min(o, c)
                body_height = max(abs(c - o), 0.001 * c)  # ensure non-zero
                rect = patches.Rectangle(
                    (idx - 0.35, body_bottom),
                    0.7,
                    body_height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=3,
                )
                ax_main.add_patch(rect)

                # Volume Bar
                vol_color = "#238636" if is_bull else "#DA3633"
                ax_vol.bar(
                    idx,
                    row["volume"],
                    width=0.7,
                    color=vol_color,
                    alpha=0.75,
                    zorder=2,
                )

            # Overlays: SMA 20 & Bollinger Bands
            ax_main.plot(
                window.index,
                window["sma_20"],
                color="#58A6FF",
                linewidth=1.2,
                label="SMA 20",
                alpha=0.9,
            )
            ax_main.plot(
                window.index,
                window["bb_upper"],
                color="#D29922",
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                label="BB Upper",
            )
            ax_main.plot(
                window.index,
                window["bb_lower"],
                color="#D29922",
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                label="BB Lower",
            )

            # Titles and legends
            ax_main.set_title(
                f"{ticker.upper()} — Institutional Wyckoff & Liquidity Structure",
                fontsize=11,
                fontweight="bold",
                color="#F0F6FC",
                pad=8,
            )
            ax_main.legend(
                loc="upper left",
                framealpha=0.3,
                facecolor="#161B22",
                edgecolor="#30363D",
                labelcolor="#C9D1D9",
                fontsize=8,
            )
            ax_vol.set_ylabel("Vol", color="#8B949E", fontsize=8)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            return buf.getvalue()

    # ------------------------------------------------------------------
    # 2. Algorithmic Smart Money & Wyckoff Analysis Engine
    # ------------------------------------------------------------------
    def analyze_market_structure_algorithmic(
        self, df: pd.DataFrame, ticker: str = "TICKER"
    ) -> WyckoffAnalysisResult:
        """Perform institutional algorithmic Wyckoff & Smart Money Concepts (SMC) analysis.

        Detects:
        - Swing Liquidity Sweeps (Springs / Upthrusts / Turtle Soup).
        - Fair Value Gaps (Bullish & Bearish 3-bar imbalances).
        - Institutional Order Blocks.
        - Wyckoff Phase categorization based on price-action and volume structure.
        """
        if df.empty or len(df) < 15:
            return WyckoffAnalysisResult(
                ticker=ticker,
                wyckoff_phase="RANGE_CONSOLIDATION",
                market_bias="NEUTRAL",
                confidence=0.50,
                liquidity_sweep_detected=False,
                sweep_type="NONE",
                analyst_source="ALGORITHMIC",
                reasoning="Insufficient historical bars for Wyckoff cycle calibration.",
            )

        data = df.copy()
        data.columns = [c.lower() for c in data.columns]
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in data.columns:
                raise ValueError(f"Missing required column '{col}'.")

        n = len(data)
        highs = data["high"].values
        lows = data["low"].values
        closes = data["close"].values
        opens = data["open"].values
        vols = data["volume"].values

        # 1. Identify Key Swing Points (Support / Resistance over last 20 bars)
        recent_window = min(n, 40)
        recent_highs = highs[-recent_window:]
        recent_lows = lows[-recent_window:]
        key_resistance = float(np.max(recent_highs[:-3]))
        key_support = float(np.min(recent_lows[:-3]))

        # 2. Detect Liquidity Sweeps (Wick breaks swing high/low then closes back inside)
        curr_high = highs[-1]
        curr_low = lows[-1]
        curr_close = closes[-1]
        prev_close = closes[-2]
        curr_open = opens[-1]

        # Prior 10-bar swing high/low prior to current bar
        prior_swing_high = float(np.max(highs[-15:-1]))
        prior_swing_low = float(np.min(lows[-15:-1]))

        liquidity_sweep = False
        sweep_type = "NONE"
        sweep_level: Optional[float] = None
        wyckoff_phase = "RANGE_CONSOLIDATION"
        bias = "NEUTRAL"
        confidence = 0.55
        reasoning_parts = []

        # Bullish Liquidity Sweep (Spring / Turtle Soup)
        # Low drops below prior swing low, but close finishes back above it
        if curr_low < prior_swing_low and curr_close > prior_swing_low:
            liquidity_sweep = True
            sweep_type = "SELL_SIDE_LIQUIDITY_RUN"
            sweep_level = prior_swing_low
            wyckoff_phase = "ACCUMULATION_SPRING"
            bias = "BULLISH"
            confidence = 0.82
            reasoning_parts.append(
                f"Bullish Liquidity Sweep: Price pierced below swing low ({prior_swing_low:.2f}) "
                f"capturing sell-stop liquidity and strongly rejected back above to close at {curr_close:.2f}."
            )
        # Bearish Liquidity Sweep (Upthrust After Distribution / UTAD)
        # High pierces above prior swing high, but close finishes back below it
        elif curr_high > prior_swing_high and curr_close < prior_swing_high:
            liquidity_sweep = True
            sweep_type = "BUY_SIDE_LIQUIDITY_RUN"
            sweep_level = prior_swing_high
            wyckoff_phase = "DISTRIBUTION_UTAD"
            bias = "BEARISH"
            confidence = 0.82
            reasoning_parts.append(
                f"Bearish Liquidity Sweep (UTAD): Price probed above swing high ({prior_swing_high:.2f}) "
                f"sweeping buy-stops before closing lower at {curr_close:.2f}."
            )
        else:
            # Trend / Markup / Markdown detection
            sma20 = np.mean(closes[-20:])
            sma50 = np.mean(closes[-min(n, 50) :])
            if curr_close > sma20 > sma50 and curr_close > key_resistance * 0.99:
                wyckoff_phase = "MARKUP"
                bias = "BULLISH"
                confidence = 0.74
                reasoning_parts.append(
                    f"Markup Phase: Expansion above moving averages with constructive higher highs."
                )
            elif curr_close < sma20 < sma50 and curr_close < key_support * 1.01:
                wyckoff_phase = "MARKDOWN"
                bias = "BEARISH"
                confidence = 0.74
                reasoning_parts.append(
                    f"Markdown Phase: Severe weakness with prices tracking below key moving averages."
                )
            else:
                wyckoff_phase = "RANGE_CONSOLIDATION"
                bias = "NEUTRAL"
                confidence = 0.55
                reasoning_parts.append(
                    f"Consolidation Phase: Market oscillating between support ({key_support:.2f}) and resistance ({key_resistance:.2f})."
                )

        # 3. Detect Fair Value Gaps (FVG) across recent bars
        fvg_zones: List[Dict[str, Any]] = []
        for i in range(max(2, n - 15), n):
            # Bullish FVG: Bar i-2 high < Bar i low
            if lows[i] > highs[i - 2]:
                fvg_gap = lows[i] - highs[i - 2]
                if fvg_gap > (closes[i] * 0.001):  # Meaningful gap > 0.1%
                    fvg = FVGZone(
                        top=float(lows[i]),
                        bottom=float(highs[i - 2]),
                        fvg_type="BULLISH",
                        bar_index=i,
                    )
                    fvg_zones.append(asdict(fvg))
            # Bearish FVG: Bar i-2 low > Bar i high
            elif highs[i] < lows[i - 2]:
                fvg_gap = lows[i - 2] - highs[i]
                if fvg_gap > (closes[i] * 0.001):
                    fvg = FVGZone(
                        top=float(lows[i - 2]),
                        bottom=float(highs[i]),
                        fvg_type="BEARISH",
                        bar_index=i,
                    )
                    fvg_zones.append(asdict(fvg))

        if fvg_zones:
            latest_fvg = fvg_zones[-1]
            reasoning_parts.append(
                f"Active {latest_fvg['fvg_type']} FVG zone between {latest_fvg['bottom']:.2f} and {latest_fvg['top']:.2f}."
            )

        # 4. Identify Order Block
        order_block_level: Optional[float] = None
        order_block_type: Optional[str] = None
        # Bullish OB: Most recent down candle before a strong 2-bar rally
        for j in range(n - 2, max(0, n - 10), -1):
            if closes[j] < opens[j] and closes[j + 1] > highs[j]:
                order_block_level = float(lows[j])
                order_block_type = "BULLISH_OB"
                break
        # Bearish OB: Most recent up candle before a strong 2-bar selloff
        if not order_block_level:
            for j in range(n - 2, max(0, n - 10), -1):
                if closes[j] > opens[j] and closes[j + 1] < lows[j]:
                    order_block_level = float(highs[j])
                    order_block_type = "BEARISH_OB"
                    break

        if order_block_level:
            reasoning_parts.append(
                f"Institutional {order_block_type} identified at {order_block_level:.2f}."
            )

        full_reasoning = " | ".join(reasoning_parts)

        return WyckoffAnalysisResult(
            ticker=ticker.upper(),
            wyckoff_phase=wyckoff_phase,
            market_bias=bias,
            confidence=round(confidence, 2),
            liquidity_sweep_detected=liquidity_sweep,
            sweep_type=sweep_type,
            sweep_price_level=round(sweep_level, 2) if sweep_level else None,
            order_block_level=(
                round(order_block_level, 2) if order_block_level else None
            ),
            order_block_type=order_block_type or "NONE",
            fair_value_gaps=fvg_zones[-3:],  # keep last 3 active zones
            key_support=round(key_support, 2),
            key_resistance=round(key_resistance, 2),
            analyst_source="ALGORITHMIC",
            reasoning=full_reasoning,
        )

    # ------------------------------------------------------------------
    # 3. Gemini Multimodal Vision Integration
    # ------------------------------------------------------------------
    def analyze_chart_with_gemini(
        self,
        image_bytes: bytes,
        ticker: str,
        df: Optional[pd.DataFrame] = None,
    ) -> WyckoffAnalysisResult:
        """Submit rendered candlestick chart image to Gemini Vision API for Wyckoff analysis."""
        if not self.api_key:
            logger.info(
                "No GEMINI_API_KEY detected. Using deterministic algorithmic Wyckoff analysis."
            )
            if df is not None:
                return self.analyze_market_structure_algorithmic(df, ticker)
            return WyckoffAnalysisResult(
                ticker=ticker.upper(),
                wyckoff_phase="RANGE_CONSOLIDATION",
                market_bias="NEUTRAL",
                confidence=0.50,
                liquidity_sweep_detected=False,
                sweep_type="NONE",
                analyst_source="ALGORITHMIC",
                reasoning="API key absent and no dataframe provided.",
            )

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt = (
            f"You are an elite quantitative market microstructure analyst specializing in the Wyckoff Method "
            f"and Smart Money Concepts (SMC). Analyze this candlestick chart for {ticker.upper()}.\n\n"
            f"Identify:\n"
            f"1. Wyckoff Phase: 'ACCUMULATION_SPRING', 'MARKUP', 'DISTRIBUTION_UTAD', 'MARKDOWN', or 'RANGE_CONSOLIDATION'.\n"
            f"2. Liquidity Sweep: True/False whether buy-side or sell-side stops were swept.\n"
            f"3. Sweep Type: 'SELL_SIDE_LIQUIDITY_RUN', 'BUY_SIDE_LIQUIDITY_RUN', or 'NONE'.\n"
            f"4. Market Bias: 'BULLISH', 'BEARISH', or 'NEUTRAL'.\n"
            f"5. Confidence score between 0.0 and 1.0.\n"
            f"6. Order Block level and type if visible.\n"
            f"7. Key Support and Resistance levels.\n\n"
            f"Return ONLY a strictly valid JSON object matching this schema:\n"
            f'{{"wyckoff_phase": string, "market_bias": string, "confidence": float, '
            f'"liquidity_sweep_detected": bool, "sweep_type": string, "sweep_price_level": float or null, '
            f'"order_block_level": float or null, "order_block_type": string, "key_support": float, '
            f'"key_resistance": float, "reasoning": string}}'
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": b64_image,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                text_content = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text_content)

                return WyckoffAnalysisResult(
                    ticker=ticker.upper(),
                    wyckoff_phase=parsed.get("wyckoff_phase", "RANGE_CONSOLIDATION"),
                    market_bias=parsed.get("market_bias", "NEUTRAL"),
                    confidence=float(parsed.get("confidence", 0.70)),
                    liquidity_sweep_detected=bool(
                        parsed.get("liquidity_sweep_detected", False)
                    ),
                    sweep_type=parsed.get("sweep_type", "NONE"),
                    sweep_price_level=parsed.get("sweep_price_level"),
                    order_block_level=parsed.get("order_block_level"),
                    order_block_type=parsed.get("order_block_type", "NONE"),
                    key_support=parsed.get("key_support"),
                    key_resistance=parsed.get("key_resistance"),
                    analyst_source="GEMINI_MULTIMODAL",
                    reasoning=parsed.get("reasoning", "Gemini vision analysis"),
                )
            else:
                logger.warning(
                    f"Gemini API returned status {resp.status_code}: {resp.text}. Falling back to algorithmic analysis."
                )
        except Exception as exc:
            logger.warning(
                f"Gemini multimodal analysis encountered exception: {exc}. Falling back to algorithmic analysis."
            )

        if df is not None:
            return self.analyze_market_structure_algorithmic(df, ticker)

        return WyckoffAnalysisResult(
            ticker=ticker.upper(),
            wyckoff_phase="RANGE_CONSOLIDATION",
            market_bias="NEUTRAL",
            confidence=0.50,
            liquidity_sweep_detected=False,
            sweep_type="NONE",
            analyst_source="ALGORITHMIC",
            reasoning="Fallback after API failure.",
        )

    # ------------------------------------------------------------------
    # 4. Master Public Analysis Seam
    # ------------------------------------------------------------------
    def analyze(
        self,
        df: pd.DataFrame,
        ticker: str = "TICKER",
        use_vision_if_available: bool = True,
    ) -> WyckoffAnalysisResult:
        """Perform end-to-end Wyckoff and liquidity sweep analysis on the asset DataFrame.

        If `use_vision_if_available` is True and GEMINI_API_KEY is present, renders
        chart and evaluates with Gemini Multimodal Vision API. Otherwise, uses the
        ultra-fast deterministic algorithmic SMC engine.
        """
        if use_vision_if_available and self.api_key:
            chart_bytes = self.render_chart_image(df, ticker=ticker)
            return self.analyze_chart_with_gemini(chart_bytes, ticker=ticker, df=df)
        return self.analyze_market_structure_algorithmic(df, ticker=ticker)


# Convenience module-level API
def evaluate_chart_structure(
    df: pd.DataFrame, ticker: str = "TICKER", api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Top-level functional entrypoint for market structure and Wyckoff cycle detection."""
    analyst = MultimodalChartAnalyst(api_key=api_key)
    result = analyst.analyze(df, ticker=ticker)
    return result.to_dict()
