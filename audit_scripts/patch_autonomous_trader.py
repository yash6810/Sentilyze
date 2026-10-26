with open("src/autonomous_trader.py", "r", encoding="utf-8") as f:
    at_code = f.read()

target_cash_avail = """        cash_available = self.broker.state.get("cash", 0.0)

        if available_slots > 0 and cash_available > 5000.0:"""

replacement_cash_avail = """        is_live = os.getenv("LIVE_ALPACA_REQUIRED", "false").lower() in ("true", "1", "yes")
        alpaca_live_equity = None
        alpaca_live_cash = None
        if is_live:
            try:
                if not self.alpaca.is_connected():
                    logger.warning("BUY_SKIPPED_ALPACA_UNAVAILABLE: Alpaca client is not connected in live mode")
                    return executed_actions
                acc_summary = self.alpaca.get_account_summary()
                if acc_summary.get("status") != "CONNECTED":
                    logger.warning("BUY_SKIPPED_ALPACA_UNAVAILABLE: Alpaca account summary is not connected")
                    return executed_actions
                alpaca_live_equity = float(acc_summary.get("equity", 0.0))
                alpaca_live_cash = float(acc_summary.get("cash", 0.0))
            except Exception as alp_err:
                logger.warning(f"BUY_SKIPPED_ALPACA_UNAVAILABLE: {alp_err}")
                return executed_actions

        cash_available = alpaca_live_cash if (is_live and alpaca_live_cash is not None) else self.broker.state.get("cash", 0.0)

        if available_slots > 0 and cash_available > 5000.0:"""

assert (
    target_cash_avail in at_code
), "target_cash_avail not found in autonomous_trader.py"
at_code = at_code.replace(target_cash_avail, replacement_cash_avail, 1)

target_tot_eq = """            total_eq = max(
                float(portfolio_summary.get("total_equity") or 100000.0), 1.0
            )"""

replacement_tot_eq = """            total_eq = max(
                float(alpaca_live_equity if (is_live and alpaca_live_equity is not None) else (portfolio_summary.get("total_equity") or 100000.0)),
                1.0,
            )"""

assert target_tot_eq in at_code, "target_tot_eq not found in autonomous_trader.py"
at_code = at_code.replace(target_tot_eq, replacement_tot_eq, 1)

target_sizing = """                target_b = self.get_target_broker(t)
                target_equity = float(target_b.state.get("total_equity", 100000.0))
                if is_crypto_asset(t):
                    shares_est = round(
                        float(target_equity * (approved_kelly / 100.0) / spot_price), 6
                    )
                else:
                    shares_est = max(
                        1.0,
                        float(
                            int(target_equity * (approved_kelly / 100.0) / spot_price)
                        ),
                    )
                gate_eval = self.rule_gate.verify_trade(
                    ticker=t,
                    spot_price=spot_price,
                    shares=shares_est,
                    deliberation=delib,
                    portfolio_state=target_b.state,
                )"""

replacement_sizing = """                target_b = self.get_target_broker(t)
                target_equity = (
                    alpaca_live_equity
                    if (is_live and alpaca_live_equity is not None and not is_crypto_asset(t))
                    else float(target_b.state.get("total_equity", 100000.0))
                )
                if is_crypto_asset(t):
                    shares_est = round(
                        float(target_equity * (approved_kelly / 100.0) / spot_price), 6
                    )
                else:
                    shares_est = max(
                        1.0,
                        float(
                            int(target_equity * (approved_kelly / 100.0) / spot_price)
                        ),
                    )
                live_port_state = (
                    {
                        "cash": alpaca_live_cash,
                        "total_equity": alpaca_live_equity,
                        "open_positions": target_b.state.get("open_positions", {}),
                    }
                    if (is_live and alpaca_live_cash is not None)
                    else target_b.state
                )
                gate_eval = self.rule_gate.verify_trade(
                    ticker=t,
                    spot_price=spot_price,
                    shares=shares_est,
                    deliberation=delib,
                    portfolio_state=live_port_state,
                )"""

assert target_sizing in at_code, "target_sizing not found in autonomous_trader.py"
at_code = at_code.replace(target_sizing, replacement_sizing, 1)

with open("src/autonomous_trader.py", "w", encoding="utf-8") as f:
    f.write(at_code)
print("Updated src/autonomous_trader.py")
