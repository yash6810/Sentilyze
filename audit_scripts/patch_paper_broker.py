with open("src/paper_broker.py", "r", encoding="utf-8") as f:
    pb_code = f.read()

target_query = """        try:
            from src.alpaca_broker import AlpacaBrokerBridge

            alpaca = AlpacaBrokerBridge()
            if alpaca.is_connected():
                alpaca_connected = True
                alpaca_positions = alpaca.get_open_positions()
                alpaca_orders = alpaca.get_open_orders()
            elif live_alpaca_required:
                logger.warning("BUY_SKIPPED_ALPACA_UNAVAILABLE")
                return {
                    "success": False,
                    "error": "BUY_SKIPPED_ALPACA_UNAVAILABLE",
                }
        except Exception as e:
            if live_alpaca_required:
                logger.warning("BUY_SKIPPED_ALPACA_UNAVAILABLE")
                return {
                    "success": False,
                    "error": "BUY_SKIPPED_ALPACA_UNAVAILABLE",
                }
            logger.debug(f"Alpaca pre-buy query notice: {e}")"""

replacement_query = """        alpaca_equity = None
        alpaca_cash = None
        try:
            from src.alpaca_broker import AlpacaBrokerBridge

            alpaca = AlpacaBrokerBridge()
            if alpaca.is_connected():
                alpaca_connected = True
                alpaca_positions = alpaca.get_open_positions()
                alpaca_orders = alpaca.get_open_orders()
                alpaca_acc = alpaca.get_account_summary()
                if alpaca_acc.get("status") == "CONNECTED":
                    alpaca_equity = float(alpaca_acc.get("equity", 0.0))
                    alpaca_cash = float(alpaca_acc.get("cash", 0.0))
                elif live_alpaca_required:
                    logger.warning("BUY_SKIPPED_ALPACA_UNAVAILABLE")
                    return {
                        "success": False,
                        "error": "BUY_SKIPPED_ALPACA_UNAVAILABLE",
                    }
            elif live_alpaca_required:
                logger.warning("BUY_SKIPPED_ALPACA_UNAVAILABLE")
                return {
                    "success": False,
                    "error": "BUY_SKIPPED_ALPACA_UNAVAILABLE",
                }
        except Exception as e:
            if live_alpaca_required:
                logger.warning("BUY_SKIPPED_ALPACA_UNAVAILABLE")
                return {
                    "success": False,
                    "error": "BUY_SKIPPED_ALPACA_UNAVAILABLE",
                }
            logger.debug(f"Alpaca pre-buy query notice: {e}")"""

assert target_query in pb_code, "target_query not found in paper_broker.py"
pb_code = pb_code.replace(target_query, replacement_query, 1)

target_eq = """        total_equity = float(self.state.get("total_equity", 0.0))
        if total_equity <= 0:
            invested_val = sum(
                pos["shares"] * pos.get("current_price", pos.get("entry_price", 0.0))
                for pos in self.state.get("open_positions", {}).values()
            )
            total_equity = float(self.state.get("cash", 0.0) + invested_val)
        if total_equity <= 0:
            total_equity = float(self.initial_cash)"""

replacement_eq = """        if live_alpaca_required or (alpaca_connected and alpaca_equity is not None and alpaca_equity > 0):
            if alpaca_equity is None or alpaca_equity <= 0:
                logger.warning("BUY_SKIPPED_ALPACA_UNAVAILABLE")
                return {
                    "success": False,
                    "error": "BUY_SKIPPED_ALPACA_UNAVAILABLE",
                }
            total_equity = alpaca_equity
        else:
            total_equity = float(self.state.get("total_equity", 0.0))
            if total_equity <= 0:
                invested_val = sum(
                    pos["shares"] * pos.get("current_price", pos.get("entry_price", 0.0))
                    for pos in self.state.get("open_positions", {}).values()
                )
                total_equity = float(self.state.get("cash", 0.0) + invested_val)
            if total_equity <= 0:
                total_equity = float(self.initial_cash)"""

assert target_eq in pb_code, "target_eq not found in paper_broker.py"
pb_code = pb_code.replace(target_eq, replacement_eq, 1)

with open("src/paper_broker.py", "w", encoding="utf-8") as f:
    f.write(pb_code)
print("Updated src/paper_broker.py")
