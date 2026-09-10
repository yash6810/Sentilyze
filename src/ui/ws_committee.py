"""
Workspace 2: 5-Agent Quantitative War Room Deliberations & Dynamic Pivot S/R Charts.
"""

import os
import json
import streamlit as st
from src.ui.components import render_workspace_header, render_conviction_gauge
from src.agent_committee import convene_trading_committee
from components.agent_war_room import render_multi_agent_war_room
from components.charting import create_candlestick_sr_chart
from components.pipeline_canvas import render_pipeline_topology_canvas
from src.data_ingestion import get_price_history
from src.realtime_tracker import fetch_live_quote


@st.cache_data(ttl=600)
def _load_cached_committee_resolution(ticker: str):
    res_path = os.path.join("results", "committee_resolutions.json")
    if os.path.exists(res_path):
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                res_dict = data.get("resolutions", {})
                if ticker in res_dict:
                    return res_dict[ticker]
        except Exception:
            pass
    return None


@st.cache_data(ttl=600)
def _get_cached_sr_chart(
    ticker: str, spot_price: float, tp1: float, tp2: float, sl: float
):
    try:
        hist_df = get_price_history(ticker, period="1y", use_cache=True)
        if hist_df is not None and not hist_df.empty:
            return create_candlestick_sr_chart(
                ticker=ticker,
                price_df=hist_df,
                spot_price=spot_price,
                tp1=tp1,
                tp2=tp2,
                sl=sl,
            )
    except Exception:
        pass
    return None


def render_committee_workspace(ticker: str):
    """Renders the 5-Agent Trading Committee deliberation panel, dynamic S/R charting, and interactive pipeline canvas."""
    render_workspace_header(
        title=f"🏛️ 5-Agent Quantitative War Room Council ({ticker})",
        subtitle="Specialized AI Agents: Technical Momentum, FinBERT NLP, Forensic DCF, Tape Scout, Adversarial Red-Team, and Chief Risk Officer",
        badge_text="WAR ROOM CONSENSUS",
        badge_color="#3B82F6",
    )

    quote = fetch_live_quote(ticker)
    spot_price = float(quote.get("price", 100.0))

    delib = _load_cached_committee_resolution(ticker)
    recompute = False

    c_status, c_btn = st.columns([3, 1])
    with c_status:
        if delib:
            ts_label = delib.get("timestamp", "Pre-computed")[:16].replace("T", " ")
            st.caption(
                f"⚡ Institutional Resolution Loaded (Timestamp: {ts_label} UTC)"
            )
    with c_btn:
        if st.button(
            "⚡ Re-Convene Council Live",
            key=f"btn_reconvene_{ticker}",
            use_container_width=True,
        ):
            recompute = True

    if delib is None or recompute:
        with st.spinner(f"Convening 5-Agent War Room Council for {ticker}..."):
            try:
                delib = convene_trading_committee(
                    ticker, spot_price=spot_price, save_resolution=True
                )
                _load_cached_committee_resolution.clear()
            except Exception as e:
                st.error(f"Committee deliberation error: {e}")
                return

    tab_war_room, tab_canvas, tab_chart = st.tabs(
        [
            "🏛️ 5-Agent Deliberation Chamber",
            "🌐 Interactive Pipeline & Agent Flow Canvas",
            "📈 Dynamic Pivot S/R & ATR Charting",
        ]
    )

    with tab_war_room:
        # Render Multi-Agent War Room Cards
        render_multi_agent_war_room(
            ticker=ticker, resolution=delib, spot_price=spot_price
        )

    with tab_canvas:
        st.markdown("### 🌐 Live Multi-Agent & Pipeline Architecture Flow Canvas")
        st.caption(
            "Interactive physics-enabled node graph showing live data feeds, agent deliberations, risk gate validation, and broker execution."
        )
        render_pipeline_topology_canvas(
            active_ticker=ticker,
            committee_resolution=delib,
            height=580,
        )

    with tab_chart:
        # Render Dynamic Support/Resistance & ATR Brackets Chart
        st.markdown(
            "### 📈 Automated Dynamic Pivot Support / Resistance & ATR Corridors"
        )
        st.caption(
            "Interactive Plotly candlestick chart with automated pivot support/resistance channels and CRO target brackets."
        )

        tp1 = float(delib.get("tp1_target", spot_price * 1.05))
        tp2 = float(delib.get("tp2_target", spot_price * 1.10))
        sl = float(delib.get("stop_loss_target", spot_price * 0.95))

        fig = _get_cached_sr_chart(
            ticker=ticker, spot_price=spot_price, tp1=tp1, tp2=tp2, sl=sl
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Loading price history for dynamic chart...")
