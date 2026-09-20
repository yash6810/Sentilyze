"""
Workspace: SEC EDGAR 8-K / 10-Q Real-Time Radar UI
=================================================
Surfaces material corporate events filed with the SEC:
- Item 1.01: Material Agreements & Partnerships
- Item 2.02: Earnings Announcements & Revenue Surprises
- Item 5.02: Executive Turnover & Board Appointments
"""

import streamlit as st
from src.sec_crawler import get_recent_sec_catalysts, ITEM_TAXONOMY


def render_sec_filings_workspace(selected_ticker: str):
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">
                📑 SEC EDGAR Form 8-K Material Event Radar
            </h3>
            <p style="margin: 2px 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                Zero-latency regulatory event stream mining official SEC EDGAR submissions for material disclosures.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        lookback = st.slider(
            "Lookback Window (Days)", min_value=7, max_value=60, value=30, step=7
        )
    with c2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.button(
            "🔄 Pull Latest Filings", key="btn_sec_pull", use_container_width=True
        )

    with st.spinner(f"Querying SEC EDGAR database for {selected_ticker}..."):
        catalysts = get_recent_sec_catalysts(selected_ticker, days=lookback)

    pos_count = sum(1 for c in catalysts if c.get("is_material_positive"))
    risk_count = sum(1 for c in catalysts if c.get("is_material_risk"))

    m1, m2, m3 = st.columns(3)
    m1.metric("Total 8-K Filings", f"{len(catalysts)} Events")
    m2.metric(
        "Bullish Catalysts",
        f"{pos_count} Filings",
        delta=f"+{pos_count}" if pos_count else None,
    )
    m3.metric(
        "Material Risk Flags",
        f"{risk_count} Flags",
        delta=f"-{risk_count}" if risk_count else None,
        delta_color="inverse",
    )

    st.markdown("---")

    if not catalysts:
        st.info(
            f"No Form 8-K material disclosures filed for {selected_ticker} in the last {lookback} days."
        )
    else:
        for cat in catalysts:
            sentiment = cat.get("sentiment_score", 0.0)
            if sentiment > 0.20:
                color = "#10B981"
                icon = "🟢"
                tag_label = "MATERIAL BULLISH"
            elif sentiment < -0.20:
                color = "#EF4444"
                icon = "🔴"
                tag_label = "MATERIAL RISK"
            else:
                color = "#3B82F6"
                icon = "🔵"
                tag_label = "NEUTRAL / ROUTINE"

            tags_html = " ".join(
                [
                    f"<span style='background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; margin-right: 4px;'>{t}</span>"
                    for t in cat.get("catalyst_types", [])
                ]
            )

            st.markdown(
                f"""
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-left: 4px solid {color}; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #F8FAFC; font-size: 0.92rem;">
                            {icon} {cat.get('ticker')} &bull; Form {cat.get('form_type')}
                        </span>
                        <span style="font-size: 0.75rem; color: #94A3B8; font-family: monospace;">
                            {cat.get('filing_date')[:10]}
                        </span>
                    </div>
                    <div style="margin: 6px 0; color: #E2E8F0; font-size: 0.86rem; line-height: 1.4;">
                        {cat.get('description')}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                        <div>{tags_html}</div>
                        <div style="font-size: 0.78rem; font-weight: 700; color: {color};">
                            {tag_label} ({sentiment:+.2f})
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
