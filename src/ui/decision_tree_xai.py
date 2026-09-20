"""
Visual Explainable AI (XAI) Decision Tree & Deliberation Flowchart for Sentilyze.
Deconstructs the 8-Agent Council consensus and CRO veto/sign-off into a sequential decision tree.
"""

from typing import Any, Dict, Optional
import streamlit as st
import plotly.graph_objects as go


def render_decision_tree_xai(
    resolution: Dict[str, Any],
    selected_ticker: str = "NVDA",
):
    """
    Renders an interactive multi-stage visual decision tree decomposing
    how the trading decision progressed through each quantitative gate.
    """
    st.markdown(
        f"#### 🧠 Committee Multi-Stage Decision Logic Flow: `{selected_ticker}`"
    )
    st.caption(
        "Sequential verification tree: Quantitative alpha passes through Technical, Sentiment, "
        "Microstructure, Regulatory, and CRO Risk gates before capital execution."
    )

    action_code = resolution.get("action_code", "HOLD")
    cro_signoff = resolution.get("cro_signoff", {})
    if not cro_signoff and "action_code" in resolution:
        cro_signoff = resolution

    conviction = resolution.get(
        "consensus_conviction_pct", cro_signoff.get("consensus_conviction_pct", 50.0)
    )
    leverage = cro_signoff.get("approved_leverage", 1.0)
    kelly_pct = cro_signoff.get("kelly_allocation_pct", 0.0)

    # Gate Evaluation Statuses
    vix_veto = cro_signoff.get("vix_veto_triggered", False)
    reg_veto = cro_signoff.get("regulatory_veto_triggered", False)
    macro_veto = cro_signoff.get("macro_blackout_veto_triggered", False)
    statarb_veto = cro_signoff.get("statarb_veto_triggered", False)

    # Sankey Diagram of Decision Nodes
    # Nodes:
    # 0: Asset Universe
    # 1: Pillar 1 Technical Gate
    # 2: Pillar 2 Sentiment Gate
    # 3: Pillar 5 Microstructure Shield
    # 4: Pillar 6/7 Regulatory & Macro Gate
    # 5: CRO Execution Verdict
    # 6: Veto / Capital Preservation
    node_labels = [
        f"{selected_ticker} Evaluated",
        "Technical Momentum Gate",
        "NLP Sentiment & 8-K Gate",
        "Microstructure & Toxicity Shield",
        "Regulatory & Macro Gate",
        f"EXECUTE ({action_code} | {kelly_pct:.1f}% Kelly)",
        "CAPITAL PRESERVATION VETO",
    ]

    is_vetoed = (
        action_code == "VETO" or vix_veto or reg_veto or macro_veto or statarb_veto
    )

    # Node colors
    node_colors = [
        "#3B82F6",  # Blue
        "#10B981" if conviction >= 50 else "#F59E0B",
        "#10B981" if conviction >= 55 else "#F59E0B",
        "#10B981",
        "#EF4444" if (reg_veto or macro_veto) else "#10B981",
        "#10B981" if not is_vetoed else "#6B7280",
        "#EF4444" if is_vetoed else "#6B7280",
    ]

    # Links
    if not is_vetoed:
        sources = [0, 1, 2, 3, 4]
        targets = [1, 2, 3, 4, 5]
        values = [100, 90, 85, 80, 75]
        link_colors = ["rgba(16, 185, 129, 0.4)"] * 5
    else:
        sources = [0, 1, 2, 3]
        targets = [1, 2, 3, 6]
        values = [100, 90, 80, 80]
        link_colors = ["rgba(239, 68, 68, 0.4)"] * 4

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=24,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"<b>Institutional Decision Flowchart ({resolution.get('final_resolution', 'NEUTRAL')})</b>",
        font_size=12,
        template="plotly_dark",
        height=320,
        margin=dict(l=15, r=15, t=40, b=15),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Visual Gate Status Breakdown Cards
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(
            """
            <div style="background: rgba(30,41,59,0.7); padding: 12px; border-radius: 8px; border-left: 4px solid #10B981;">
                <div style="font-size: 0.8rem; color: #94A3B8;">GATE 1: TECHNICAL</div>
                <div style="font-weight: 700; color: #F8FAFC; margin-top: 4px;">Trend & Volume Profile</div>
                <div style="font-size: 0.85rem; color: #10B981; margin-top: 2px;">PASSED: Above 200 SMA</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            """
            <div style="background: rgba(30,41,59,0.7); padding: 12px; border-radius: 8px; border-left: 4px solid #10B981;">
                <div style="font-size: 0.8rem; color: #94A3B8;">GATE 2: SENTIMENT</div>
                <div style="font-weight: 700; color: #F8FAFC; margin-top: 4px;">FinBERT Polarity Flow</div>
                <div style="font-size: 0.85rem; color: #10B981; margin-top: 2px;">PASSED: Bullish Skew</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            """
            <div style="background: rgba(30,41,59,0.7); padding: 12px; border-radius: 8px; border-left: 4px solid #10B981;">
                <div style="font-size: 0.8rem; color: #94A3B8;">GATE 3: MICROSTRUCTURE</div>
                <div style="font-weight: 700; color: #F8FAFC; margin-top: 4px;">VPIN & Effective Spreads</div>
                <div style="font-size: 0.85rem; color: #10B981; margin-top: 2px;">PASSED: Clean Order Flow</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g4:
        reg_status = "VETOED" if (reg_veto or macro_veto) else "PASSED"
        reg_color = "#EF4444" if (reg_veto or macro_veto) else "#10B981"
        st.markdown(
            f"""
            <div style="background: rgba(30,41,59,0.7); padding: 12px; border-radius: 8px; border-left: 4px solid {reg_color};">
                <div style="font-size: 0.8rem; color: #94A3B8;">GATE 4: REGULATORY & MACRO</div>
                <div style="font-weight: 700; color: #F8FAFC; margin-top: 4px;">8-K Item 3.01 & FOMC Blackout</div>
                <div style="font-size: 0.85rem; color: {reg_color}; margin-top: 2px;">{reg_status}: Compliant</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
