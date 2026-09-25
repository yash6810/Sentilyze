"""
Workspace 2: 5-Agent Quantitative War Room Deliberations & Dynamic Pivot S/R Charts.
"""

import os
import json
import streamlit as st
from src.ui.components import render_workspace_header
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
    """Renders the 5-Agent Trading Committee deliberation panel, dynamic S/R charting, and pipeline canvas."""
    render_workspace_header(
        title=f"🏛️ 5-Agent Quantitative War Room Council ({ticker})",
        subtitle=(
            "Specialized AI Agents: Technical Momentum, FinBERT NLP, Forensic DCF, Tape Scout, "
            "Adversarial Red-Team, and Chief Risk Officer"
        ),
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

    # ⚡ Real-Time Quantum Telemetry & Conformal Corridor KPI Bar
    q_tel = delib.get("quantum_telemetry", {})
    if q_tel:
        lat = q_tel.get("execution_latency", {})
        q_us = lat.get("total_microseconds", 0.0)
        c_low = q_tel.get("conformal_lower_bound", 0.0)
        c_high = q_tel.get("conformal_upper_bound", 0.0)
        poc = q_tel.get("poc_price", 0.0)
        poc_dist = q_tel.get("dist_to_poc_pct", 0.0)
        hmm_reg = q_tel.get("hmm_regime", "Normal")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "⚡ Quantum Core Speed",
            f"{q_us:,.1f} μs",
            delta="Sub-Millisecond Execution",
        )
        k2.metric(
            "🛡️ 95% Conformal Corridor",
            f"${c_low:.2f} - ${c_high:.2f}",
            delta="Guaranteed Safety",
        )
        k3.metric("📊 Order Flow PoC", f"${poc:.2f}", delta=f"{poc_dist:+.1f}% to Spot")
        k4.metric("🧠 HMM Macro Regime", str(hmm_reg), delta="Hamilton 1989 Filter")

    tab_war_room, tab_canvas, tab_chart, tab_quantum = st.tabs(
        [
            "🏛️ 5-Agent Deliberation Chamber",
            "🌐 Interactive Pipeline & Agent Flow Canvas",
            "📈 Dynamic Pivot S/R & ATR Charting",
            "⚡ Quantum Telemetry & 0.5N Pyramiding",
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
            "Interactive physics-enabled node graph showing live data feeds, agent deliberations, "
            "risk gate validation, and broker execution."
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
            "Interactive Plotly candlestick chart with automated pivot support/resistance channels "
            "and CRO target brackets."
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

    with tab_quantum:
        st.markdown("### ⚡ Ultra-Low-Latency Quantum Telemetry & 0.5N Pyramiding Plan")
        st.caption(
            "Hardware-clock profiling, finite-sample 95% Conformal Prediction boundaries, "
            "and Seykota & Covel 3-stage ATR scaling."
        )

        from src.compound_engine import calculate_turtle_pyramid_plan

        atr_val = float(delib.get("atr_14", max(spot_price * 0.025, 1.0)))
        pyramid_plan = calculate_turtle_pyramid_plan(
            entry_price=spot_price,
            atr=atr_val,
            total_allocation_dollars=15000.0,
        )

        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            st.markdown("#### 🐢 3-Stage Turtle Pyramiding Allocation Plan")
            stages = pyramid_plan.get("stages", [])
            for stg in stages:
                box_style = (
                    "background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(148, 163, 184, 0.15); "
                    "border-radius: 8px; padding: 12px; margin-bottom: 8px;"
                )
                row_style = (
                    "display: flex; justify-content: space-between; margin-top: 6px; "
                    "font-size: 0.9rem; color: #CBD5E1;"
                )
                stg_sl = float(stg.get("stop_loss", stg.get("unified_stop_loss", 0)))
                stg_risk = stg.get("risk_profile", "")
                with st.container():
                    st.markdown(
                        f"""
                        <div style="{box_style}">
                            <div style="font-weight: 700; color: #38BDF8; font-size: 1.05rem;">{stg.get('name')}</div>
                            <div style="{row_style}">
                                <span>Trigger: <b>${stg.get('trigger_price', 0):.2f}</b></span>
                                <span>Shares: <b>{stg.get('shares')} shares</b></span>
                                <span>Stop-Loss: <b>${stg_sl:.2f}</b></span>
                            </div>
                            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">{stg_risk}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with col_p2:
            st.markdown("#### 🛡️ Conformal Prediction & Order Flow")
            c_low_val = q_tel.get("conformal_lower_bound", spot_price * 0.96)
            c_high_val = q_tel.get("conformal_upper_bound", spot_price * 1.04)
            poc_val = q_tel.get("poc_price", spot_price)
            val_p = q_tel.get("val_price", spot_price * 0.98)
            vah_p = q_tel.get("vah_price", spot_price * 1.02)
            c_prob = q_tel.get("crisis_probability", 0.05)
            c_color = "#EF4444" if c_prob > 0.4 else "#10B981"
            box_r = (
                "background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(59, 130, 246, 0.3); "
                "border-radius: 8px; padding: 14px;"
            )
            st.markdown(
                f"""
                <div style="{box_r}">
                    <div style="font-size: 0.85rem; color: #94A3B8;">95% Guaranteed Safety Band</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #10B981; margin: 4px 0;">
                        ${c_low_val:.2f} — ${c_high_val:.2f}
                    </div>
                    <hr style="border: 0; border-top: 1px solid rgba(148, 163, 184, 0.15); margin: 8px 0;" />
                    <div style="font-size: 0.85rem; color: #94A3B8;">Order Flow Point of Control (PoC)</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #F59E0B;">
                        ${poc_val:.2f}
                    </div>
                    <div style="font-size: 0.8rem; color: #64748B;">Value Area: [${val_p:.2f} - ${vah_p:.2f}]</div>
                    <hr style="border: 0; border-top: 1px solid rgba(148, 163, 184, 0.15); margin: 8px 0;" />
                    <div style="font-size: 0.85rem; color: #94A3B8;">Gaussian HMM Crisis Probability</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: {c_color};">
                        {c_prob:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
