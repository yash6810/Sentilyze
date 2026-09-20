"""
Workspace: BioTech & Pharma Catalyst Radar UI
=============================================
Renders interactive FDA PDUFA decision dates, Phase 3 trial trackers,
and binary event risk telemetry for healthcare equities.
"""

import streamlit as st
from src.biotech_catalyst_radar import (
    compile_biotech_catalyst_radar,
    is_biotech_or_healthcare,
    BIOTECH_TICKERS,
)


def render_biotech_radar_workspace(selected_ticker: str):
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">
                🧬 BioTech & FDA Regulatory Catalyst Radar
            </h3>
            <p style="margin: 2px 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                Tracks OpenFDA drug label approvals, PDUFA action dates, and ClinicalTrials.gov Phase 3 readout schedules.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_bio = is_biotech_or_healthcare(selected_ticker)

    col_t, col_btn = st.columns([3, 1])
    with col_t:
        target_ticker = st.selectbox(
            "🎯 Target Healthcare Asset",
            list(BIOTECH_TICKERS.keys()),
            index=list(BIOTECH_TICKERS.keys()).index(selected_ticker) if is_bio else 0,
            help="Select a pharmaceutical or biotechnology company to inspect its active trial pipeline.",
        )
    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        refresh = st.button(
            "🔄 Scan Catalysts", key="btn_scan_bio", use_container_width=True
        )

    with st.spinner(f"Scanning FDA and ClinicalTrials.gov for {target_ticker}..."):
        intel = compile_biotech_catalyst_radar(target_ticker)

    # Risk Profile Banner
    risk = intel.get("risk_profile", "NORMAL")
    if risk == "HIGH_BINARY_EVENT_RISK":
        st.warning(
            f"⚠️ **HIGH BINARY EVENT RISK DETECTED FOR {target_ticker}**: Active Phase 3 clinical trial readout pending. Implied volatility may expand rapidly into announcement window."
        )
    else:
        st.success(
            f"🟢 **NORMAL VOLATILITY PROFILE**: No immediate high-risk binary FDA PDUFA deadlines identified in the 7-day window."
        )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Active Clinical Trials", f"{intel.get('clinical_trials_count', 0)} Studies"
    )
    m2.metric("Regulatory Status", "Active FDA Pipeline")
    m3.metric("Sector Conviction", "High Dispersion Alpha")

    st.markdown("---")

    t_trials, t_fda = st.tabs(
        ["🧪 Clinical Studies & Readouts", "🏛️ FDA Regulatory Actions"]
    )

    with t_trials:
        trials = intel.get("clinical_trials", [])
        if trials:
            for t in trials:
                with st.expander(
                    f"🔬 {t.get('phase', 'PHASE_UNKNOWN')} • {t.get('title', 'Study')[:75]}..."
                ):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(
                            f"**Target Indication**: `{t.get('conditions', 'Unspecified')}`"
                        )
                        st.markdown(
                            f"**NCT Identifier**: [{t.get('nct_id')}](https://clinicaltrials.gov/study/{t.get('nct_id')})"
                        )
                        st.markdown(
                            f"**Recruitment Status**: `{t.get('status', 'ACTIVE')}`"
                        )
                    with c2:
                        st.metric("Target Readout", t.get("expected_readout", "TBD"))
        else:
            st.info("No active late-stage trials returned from the registry.")

    with t_fda:
        fda_events = intel.get("fda_events", [])
        if fda_events:
            for f in fda_events:
                with st.container():
                    st.markdown(
                        f"""
                        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 700; color: #10B981; font-size: 0.95rem;">💊 {f.get('brand_name', 'Therapeutic')} ({f.get('generic_name', 'Active Molecule')})</span>
                                <span style="font-size: 0.8rem; color: #94A3B8;">Effective: {f.get('effective_date', 'N/A')}</span>
                            </div>
                            <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">
                                <strong>Application:</strong> <code>{f.get('application_number', 'N/A')}</code> &bull; <strong>Status:</strong> {f.get('regulatory_status', 'APPROVED')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No recent openFDA regulatory changes found.")
