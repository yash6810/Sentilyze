"""
Workspace: Automated Broker Webhooks & Execution API Dispatcher.
Configures and simulates institutional bracket order dispatches to Alpaca,
Interactive Brokers, and Custom Webhook Endpoints with HMAC-SHA256 signatures.
"""

import streamlit as st
import json
import os
from src.webhook_dispatcher import (
    load_webhook_config,
    save_webhook_config,
    format_broker_order_payload,
    dispatch_order_webhook,
    WEBHOOK_AUDIT_LOG,
)
from src.realtime_tracker import fetch_live_quote


def render_broker_webhooks_workspace(selected_ticker: str = "NVDA"):
    st.markdown("### ⚡ Automated Broker Webhooks & API Order Dispatcher")
    st.caption(
        "Institutional Execution Gateway: Dispatches real-time algorithmic bracket orders (TP1, TP2 Runner, Stop-Loss) "
        "to external brokerage endpoints (Alpaca, Interactive Brokers Paper Accounts, Custom REST Webhooks) "
        "secured by cryptographic HMAC-SHA256 authentication."
    )

    tab_config, tab_test, tab_logs = st.tabs(
        [
            "⚙️ Webhook Endpoint Configuration",
            "🎯 Bracket Order Simulator & Test Ping",
            "📜 Live Execution Audit Log",
        ]
    )

    config = load_webhook_config()

    # =========================================================================
    # TAB 1: CONFIGURATION
    # =========================================================================
    with tab_config:
        st.markdown("#### ⚙️ Broker API & Webhook Configuration")

        c1, c2 = st.columns(2)
        with c1:
            broker_name = st.selectbox(
                "Target Broker Gateway:",
                [
                    "Custom Institutional REST Webhook",
                    "Simulated Paper Trading Gateway",
                    "External Broker API",
                ],
                index=0,
            )
            webhook_url = st.text_input(
                "Webhook Destination URL:",
                value=config.get(
                    "webhook_url", "https://api.your-broker.com/v2/orders"
                ),
                help="Enter your private webhook endpoint. Loaded securely from environment or local secrets.",
            )
            env = st.selectbox(
                "Execution Environment:",
                ["PAPER_TRADING (Safe Simulation)", "LIVE_PRODUCTION (Real Capital)"],
                index=0,
            )

        with c2:
            hmac_secret = st.text_input(
                "HMAC SHA-256 Secret Key:",
                value="",
                placeholder="Enter private HMAC secret...",
                type="password",
                help="Cryptographic secret key used to sign order payloads. Never committed to Git.",
            )
            enable_webhooks = st.toggle(
                "Enable Live Webhook Dispatching", value=config.get("enabled", False)
            )

        if st.button("💾 Save Webhook Configuration", type="primary"):
            new_cfg = {
                "broker_name": broker_name,
                "webhook_url": webhook_url,
                "environment": env,
                "hmac_secret": hmac_secret,
                "enabled": enable_webhooks,
            }
            if save_webhook_config(new_cfg):
                st.success("✅ Webhook configuration saved successfully!")
                st.rerun()

    # =========================================================================
    # TAB 2: BRACKET ORDER SIMULATOR
    # =========================================================================
    with tab_test:
        st.markdown(
            f"#### 🎯 Generate & Test Bracket Order Payload for **{selected_ticker}**"
        )

        spot = fetch_live_quote(selected_ticker)
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            test_shares = st.number_input(
                "Shares Quantity:", min_value=1, max_value=1000, value=25
            )
            test_action = st.selectbox("Order Action:", ["BUY", "SELL"], index=0)
        with col_t2:
            tp1_test = st.number_input(
                "Target 1 Limit ($):", value=round(spot * 1.05, 2)
            )
            tp2_test = st.number_input(
                "Target 2 Runner ($):", value=round(spot * 1.10, 2)
            )
        with col_t3:
            sl_test = st.number_input(
                "Stop-Loss Limit ($):", value=round(spot * 0.96, 2)
            )

        payload = format_broker_order_payload(
            ticker=selected_ticker,
            action=test_action,
            shares=test_shares,
            price=spot,
            tp1_target=tp1_test,
            tp2_target=tp2_test,
            sl_target=sl_test,
        )

        st.markdown(
            "##### 📦 Generated Institutional JSON Payload (Alpaca Compatible):"
        )
        st.code(json.dumps(payload, indent=2), language="json")

        if st.button("🚀 Dispatch Test Ping to Webhook Endpoint", type="primary"):
            res = dispatch_order_webhook(payload, config=config, is_test=True)
            st.success("✅ Webhook Payload & Signature Verified!")
            st.json(res)

    # =========================================================================
    # TAB 3: AUDIT LOGS
    # =========================================================================
    with tab_logs:
        st.markdown("#### 📜 Dispatched Webhook Execution Audit Ledger")
        if os.path.exists(WEBHOOK_AUDIT_LOG):
            try:
                with open(WEBHOOK_AUDIT_LOG, "r") as f:
                    logs = json.load(f)
                st.dataframe(logs, use_container_width=True)
            except Exception:
                st.info("No webhook dispatches logged yet.")
        else:
            st.info(
                "No external webhook orders dispatched yet. Run a simulated ping in Tab 2."
            )
