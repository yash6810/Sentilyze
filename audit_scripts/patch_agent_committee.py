with open("src/agent_committee.py", "r", encoding="utf-8") as f:
    code = f.read()

target_cash_floor = '''                    port_path = os.path.join("results", "paper_portfolio.json")
                    if os.path.exists(port_path):
                        with open(port_path, "r", encoding="utf-8") as pp_fp:
                            pp_data = json.load(pp_fp)
                            live_cash = float(pp_data.get("cash", 100000.0))
                            live_equity = float(pp_data.get("total_equity", live_cash))'''

replacement_cash_floor = '''                    live_alpaca_required = os.getenv("LIVE_ALPACA_REQUIRED", "false").lower() in ("true", "1", "yes")
                    if live_alpaca_required:
                        try:
                            from src.alpaca_broker import AlpacaBrokerBridge
                            alp = AlpacaBrokerBridge()
                            if not alp.is_connected():
                                logger.warning("BUY_SKIPPED_ALPACA_UNAVAILABLE: Alpaca disconnected in live mode")
                                return {
                                    "verdict": "VETO",
                                    "confidence": 0.0,
                                    "final_resolution": "VETO",
                                    "veto_active": True,
                                    "veto_reason": "BUY_SKIPPED_ALPACA_UNAVAILABLE: Alpaca disconnected in live mode",
                                    "risk_score": 100.0,
                                    "approved_leverage": 0.0,
                                    "kelly_allocation_pct": 0.0,
                                    "stop_loss_target": None,
                                    "tp1_target": None,
                                    "tp2_target": None,
                                    "action": "VETO",
                                }
                            acc = alp.get_account_summary()
                            if acc.get("status") != "CONNECTED":
                                logger.warning("BUY_SKIPPED_ALPACA_UNAVAILABLE: Alpaca account query failed")
                                return {
                                    "verdict": "VETO",
                                    "confidence": 0.0,
                                    "final_resolution": "VETO",
                                    "veto_active": True,
                                    "veto_reason": "BUY_SKIPPED_ALPACA_UNAVAILABLE: Alpaca account query failed",
                                    "risk_score": 100.0,
                                    "approved_leverage": 0.0,
                                    "kelly_allocation_pct": 0.0,
                                    "stop_loss_target": None,
                                    "tp1_target": None,
                                    "tp2_target": None,
                                    "action": "VETO",
                                }
                            live_cash = float(acc.get("cash", 0.0))
                            live_equity = float(acc.get("equity", live_cash))
                        except Exception as alp_err:
                            logger.warning(f"BUY_SKIPPED_ALPACA_UNAVAILABLE: {alp_err}")
                            return {
                                "verdict": "VETO",
                                "confidence": 0.0,
                                "final_resolution": "VETO",
                                "veto_active": True,
                                "veto_reason": f"BUY_SKIPPED_ALPACA_UNAVAILABLE: {alp_err}",
                                "risk_score": 100.0,
                                "approved_leverage": 0.0,
                                "kelly_allocation_pct": 0.0,
                                "stop_loss_target": None,
                                "tp1_target": None,
                                "tp2_target": None,
                                "action": "VETO",
                            }
                    else:
                        port_path = os.path.join("results", "paper_portfolio.json")
                        if os.path.exists(port_path):
                            with open(port_path, "r", encoding="utf-8") as pp_fp:
                                pp_data = json.load(pp_fp)
                                live_cash = float(pp_data.get("cash", 100000.0))
                                live_equity = float(pp_data.get("total_equity", live_cash))
                        else:
                            live_cash = 100000.0
                            live_equity = 100000.0'''

assert target_cash_floor in code, "target_cash_floor not found in agent_committee.py"
code = code.replace(target_cash_floor, replacement_cash_floor, 1)

target_gz = '''            portfolio_file = os.path.join("results", "paper_portfolio.json")
            if os.path.exists(portfolio_file):
                with open(portfolio_file, "r", encoding="utf-8") as pf:
                    p_state = json.load(pf)
                curr_eq = float(p_state.get("total_equity", 100000.0))
                # Protect baseline initial capital floor ($95k CPPI floor) with 25% max tolerance from peak
                peak_wealth = float(p_state.get("peak_equity", curr_eq))
                if peak_wealth < curr_eq:
                    peak_wealth = curr_eq'''

replacement_gz = '''            live_alpaca_required = os.getenv("LIVE_ALPACA_REQUIRED", "false").lower() in ("true", "1", "yes")
            if live_alpaca_required:
                curr_eq = live_equity
                peak_wealth = max(curr_eq, float(os.getenv("ALPACA_PEAK_EQUITY", curr_eq)))
            else:
                portfolio_file = os.path.join("results", "paper_portfolio.json")
                if os.path.exists(portfolio_file):
                    with open(portfolio_file, "r", encoding="utf-8") as pf:
                        p_state = json.load(pf)
                    curr_eq = float(p_state.get("total_equity", 100000.0))
                    # Protect baseline initial capital floor ($95k CPPI floor) with 25% max tolerance from peak
                    peak_wealth = float(p_state.get("peak_equity", curr_eq))
                    if peak_wealth < curr_eq:
                        peak_wealth = curr_eq
                else:
                    curr_eq = 100000.0
                    peak_wealth = 100000.0'''

assert target_gz in code, "target_gz not found in agent_committee.py"
code = code.replace(target_gz, replacement_gz, 1)

with open("src/agent_committee.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Updated src/agent_committee.py")
