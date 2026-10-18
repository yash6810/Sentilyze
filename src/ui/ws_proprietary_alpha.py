"""
Proprietary Alpha Engines & Multi-Resolution Physics Workspace
=============================================================
Interactive UI visualization of Sentilyze's 6 next-generation alpha engines:
1. Wavelet Multi-Resolution Entropy Sifter (MODWT Daubechies D4).
2. FinJEPA Latent Market Physics Engine (Joint-Embedding Predictive Architecture).
3. Jev Sub-Second Fast-Catalyst Engine (Sub-15ms RLCD System-1 Reflex).
4. 24/7 Crypto Funding Rate & Liquidation Arbitrage Engine.
5. Soros Reflexivity Kinematics & Dynamic Sympathy Propagator.
6. FinMem Working Memory Tape (Sliding 50-Tick Microstructure Flow).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.utils import get_logger

logger = get_logger("ws_proprietary_alpha")


def render_proprietary_alpha_workspace(ticker: str):
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
                    border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #F8FAFC; font-size: 1.5rem; font-weight: 700;">
                🧬 Proprietary Alpha & Market Physics Deck: <span style="color: #60A5FA;">{ticker}</span>
            </h2>
            <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 0.9rem;">
                Sub-second event reflexes, multi-resolution wavelets, joint-embedding physics, and 24/7 digital asset arbitrage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alpha_tabs = st.tabs(
        [
            "🌊 Wavelet Sifter",
            "🧠 FinJEPA Physics",
            "⚡ Jev Fast-Catalyst",
            "🪙 24/7 Crypto Arbitrage",
            "🎯 Soros Sympathy",
            "🧵 FinMem Tape",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: WAVELET SIFTER
    # -------------------------------------------------------------
    with alpha_tabs[0]:
        st.subheader("🌊 Wavelet Multi-Resolution Entropy Sifter (MODWT D4)")
        st.caption(
            "Decomposes non-stationary market returns into orthogonal frequency octaves to purge false breakout whipsaws."
        )
        try:
            from src.wavelet_entropy_sifter import evaluate_wavelet_invalidation_gate

            w_res = evaluate_wavelet_invalidation_gate(ticker=ticker)
            veto = w_res.get("veto_active", False)
            entropy = float(w_res.get("entropy", 0.0))
            d1 = float(w_res.get("d1_energy_pct", 0.0))
            d2 = float(w_res.get("d2_energy_pct", 0.0))
            d3 = float(w_res.get("d3_energy_pct", 0.0))
            a3 = float(w_res.get("a3_energy_pct", 0.0))
            regime = w_res.get("regime", "UNKNOWN")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Wavelet Entropy",
                f"{entropy:.3f}",
                delta="Chop > 0.85" if entropy > 0.85 else "Clean Trend",
            )
            c2.metric(
                "D1 Noise Energy",
                f"{d1:.1f}%",
                delta="High Chop" if d1 > 45 else "Low Noise",
            )
            c3.metric(
                "A3 Trend Drift",
                f"{a3:.1f}%",
                delta="Dominant Trend" if a3 > 40 else "Ranging",
            )
            status_text = "🔴 VETO (Chop)" if veto else "🟢 CLEAR (Trending)"
            c4.metric("Council Gate", status_text)

            # Bar chart of octave energies
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=[
                            "D1 (Noise Chop)",
                            "D2 (Swing)",
                            "D3 (Cycle)",
                            "A3 (Macro Trend)",
                        ],
                        y=[d1, d2, d3, a3],
                        marker_color=[
                            "#EF4444" if d1 > 45 else "#3B82F6",
                            "#60A5FA",
                            "#93C5FD",
                            "#10B981" if a3 > 40 else "#6EE7B7",
                        ],
                        text=[f"{v:.1f}%" for v in [d1, d2, d3, a3]],
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(
                title=f"Relative Spectral Energy Distribution: {ticker}",
                yaxis_title="Energy %",
                template="plotly_dark",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Wavelet analysis buffering: {e}")

    # -------------------------------------------------------------
    # TAB 2: FINJEPA LATENT PHYSICS
    # -------------------------------------------------------------
    with alpha_tabs[1]:
        st.subheader("🧠 FinJEPA Latent Market Physics Engine (Joint-Embedding)")
        st.caption(
            "Yann LeCun's non-contrastive predictive architecture predicting latent market representations without hallucination."
        )
        try:
            from src.finjepa_engine import evaluate_finjepa_physics_gate

            jepa_res = evaluate_finjepa_physics_gate(ticker=ticker)
            cro_veto = jepa_res.get("cro_veto", False)
            energy = float(jepa_res.get("latent_energy", 0.0))
            disp = float(jepa_res.get("dispersion_sigma", 0.0))
            p_state = jepa_res.get("physics_state", "NORMAL")

            j1, j2, j3 = st.columns(3)
            j1.metric("Latent Energy Norm", f"{energy:.4f}")
            j2.metric(
                "Dispersion Distance",
                f"{disp:.2f}σ",
                delta="Hazard > 2.5σ" if disp > 2.5 else "Stable",
            )
            j3.metric("CRO Liquidity Veto", "🔴 ACTIVE" if cro_veto else "🟢 PASSED")

            st.info(
                f"**Latent Dynamics State:** `{p_state}` — {'Liquidity exhaustion detected. Risk of sudden adverse fill slippage.' if cro_veto else 'Latent market state shows structural expansion momentum.'}"
            )
        except Exception as e:
            st.warning(f"FinJEPA physics buffering: {e}")

    # -------------------------------------------------------------
    # TAB 3: JEV FAST-CATALYST
    # -------------------------------------------------------------
    with alpha_tabs[2]:
        st.subheader("⚡ Jev Sub-Second Fast-Catalyst Engine")
        st.caption(
            "Sub-15ms RLCD System-1 reflex classifying breaking 8-Ks, FDA approvals, and C-suite departures into typed actions."
        )
        try:
            from src.jev_catalyst import evaluate_jev_catalyst_gate

            jev_v = evaluate_jev_catalyst_gate(ticker=ticker)
            jv1, jv2, jv3, jv4 = st.columns(4)
            jv1.metric("Conviction Tier", jev_v.conviction_tier)
            jv2.metric("Event Category", jev_v.event_type)
            jv3.metric("Kelly Multiplier", f"{jev_v.kelly_multiplier:.2f}x")
            jv4.metric("Reflex Latency", f"{jev_v.latency_ms:.2f} ms")

            st.markdown(f'**Latest Material Headline:**\n> *"{jev_v.raw_headline}"*')
            st.markdown(f"**Reflex Rationale:** `{jev_v.rationale}`")
        except Exception as e:
            st.warning(f"Jev catalyst buffering: {e}")

    # -------------------------------------------------------------
    # TAB 4: 24/7 CRYPTO FUNDING ARBITRAGE
    # -------------------------------------------------------------
    with alpha_tabs[3]:
        st.subheader("🪙 24/7 Crypto Funding & Liquidation Arbitrage")
        st.caption(
            "Public zero-key Binance & Coinbase funding rate scanner routing strictly to isolated crypto sandbox."
        )
        try:
            from src.crypto_arbitrage import get_crypto_arbitrage_engine

            engine = get_crypto_arbitrage_engine()
            majors = ["BTC-USD", "ETH-USD", "SOL-USD"]
            rows = []
            for c_tk in majors:
                opp = engine.scan_funding_opportunity(c_tk)
                rows.append(
                    {
                        "Asset": c_tk,
                        "Binance Symbol": opp.get("symbol"),
                        "8h Funding Rate": f"{opp.get('funding_rate', 0.0):+.4%}",
                        "Annualized Yield": f"{opp.get('annualized_rate_pct', 0.0):+.2f}%",
                        "Opportunity": opp.get("opportunity_type"),
                        "Action": opp.get("action"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            p_path = os.path.join("results", "pilot_crypto_portfolio.json")
            if os.path.exists(p_path):
                import json

                with open(p_path, "r", encoding="utf-8") as f:
                    c_port = json.load(f)
                st.caption(
                    f"🛡️ **Isolated Sandbox State:** Cash: `${c_port.get('cash', 25000.0):,.2f}` | Equity: `${c_port.get('total_equity', 25000.0):,.2f}` | Open Crypto Positions: `{len(c_port.get('open_positions', {}))}`"
                )
        except Exception as e:
            st.warning(f"Crypto arbitrage feed buffering: {e}")

    # -------------------------------------------------------------
    # TAB 5: SOROS SYMPATHY PROPAGATOR
    # -------------------------------------------------------------
    with alpha_tabs[4]:
        st.subheader("🎯 Soros Reflexivity Kinematics & Sympathy Propagator")
        st.caption(
            "Cross-asset lead-lag impulse transmission front-running Tier-2 sympathetic followers."
        )
        try:
            from src.soros_sympathy import get_sympathy_propagator

            prop = get_sympathy_propagator()
            opps = prop.scan_sympathy_network()
            if opps:
                st.write(
                    f"Detected **{len(opps)}** active sympathy transmission corridors:"
                )
                s_df = pd.DataFrame(opps)
                st.dataframe(s_df, use_container_width=True)
            else:
                st.info(
                    "No active cross-asset sympathy impulses (>1.8σ) detected in the last scan window."
                )
        except Exception as e:
            st.warning(f"Sympathy network buffering: {e}")

    # -------------------------------------------------------------
    # TAB 6: FINMEM WORKING MEMORY TAPE
    # -------------------------------------------------------------
    with alpha_tabs[5]:
        st.subheader("🧵 FinMem Working Memory Tape (Sliding 50-Tick Microstructure)")
        st.caption(
            "Ultra-low-latency in-memory sliding tape tracking intra-minute price velocity, flow toxicity, and spread blowout hazard."
        )
        try:
            from src.working_memory_tape import get_working_memory_tape

            tape = get_working_memory_tape()
            m_state = tape.get_microstructure_state(ticker)

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Tape Length", f"{m_state.get('tape_length', 0)} / 50 ticks")
            t2.metric(
                "Flow Toxicity",
                f"{m_state.get('flow_toxicity_score', 0.0):+.2f}",
                delta=(
                    "Buy Flow"
                    if m_state.get("flow_toxicity_score", 0) > 0
                    else "Sell Flow"
                ),
            )
            t3.metric(
                "Current Spread",
                f"{m_state.get('current_spread_bps', 0.0):.1f} bps",
                delta=f"Median: {m_state.get('median_spread_bps', 0.0):.1f} bps",
            )
            t4.metric(
                "Spread Hazard",
                "🔴 DANGER" if m_state.get("spread_expansion_hazard") else "🟢 STABLE",
            )

            st.code(tape.format_deliberation_context(ticker), language="text")
        except Exception as e:
            st.warning(f"FinMem working tape buffering: {e}")
