with open("tests/test_risk_and_position_limits.py", "r", encoding="utf-8") as f:
    content = f.read()

new_tests = '''

def test_learning_frozen_preserves_byte_identical_memory(tmp_path, monkeypatch):
    """
    Asserts that with LEARNING_FROZEN=true, executing post-mortem cycles,
    self-improvement loops, and committee weight recalibrations leaves
    results/agent_learning_memory.json and results/agent_memory/* byte-identical.
    """
    import glob
    import hashlib
    from src.post_mortem_learner import TradePostMortemLearner
    from src.autonomous_trader import AutonomousTradingEngine
    from src.agent_memory import AgentMemoryStore

    monkeypatch.setenv("LEARNING_FROZEN", "true")

    memory_files = sorted(
        ["results/agent_learning_memory.json"]
        + [f.replace("\\\\", "/") for f in glob.glob("results/agent_memory/*")]
    )

    def compute_hashes():
        hashes = {}
        for mf in memory_files:
            if os.path.exists(mf):
                with open(mf, "rb") as fp:
                    hashes[mf] = hashlib.sha256(fp.read()).hexdigest()
        return hashes

    hashes_before = compute_hashes()

    # 1. Run post-mortem learning cycle
    learner = TradePostMortemLearner()
    learner.run_post_mortem_cycle()

    # 2. Run self-improvement loop
    test_broker = PaperBroker(
        portfolio_path=str(tmp_path / "test_portfolio.json"),
        trades_path=str(tmp_path / "test_trades.csv"),
    )
    engine = AutonomousTradingEngine(broker=test_broker)
    engine._run_self_improvement_feedback_loop(date_str="2026-11-02")

    # 3. Run AgentMemoryStore recalibration
    store = AgentMemoryStore()
    store.recalibrate_weights()

    hashes_after = compute_hashes()

    assert hashes_before == hashes_after, (
        f"Memory modified while LEARNING_FROZEN=true! Mismatches: "
        f"{[k for k in hashes_before if hashes_before[k] != hashes_after.get(k)]}"
    )


def test_fresh_container_duplicate_alpaca_position_and_order_blocks_buy(tmp_path):
    """
    A fresh container (NEW PaperBroker with an EMPTY local ledger):
    - When Alpaca returns an open NVDA position: execute_buy returns success=False
      and submit_bracket_order is NOT called.
    - When Alpaca returns an open NVDA order with status 'new': execute_buy returns success=False
      and submit_bracket_order is NOT called.
    """
    # 1. Open position on Alpaca
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_positions",
            return_value=[{"symbol": "NVDA", "qty": "10", "market_value": "1200.0"}],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_orders",
            return_value=[],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_account_summary",
            return_value={"status": "CONNECTED", "equity": 100000.0, "cash": 90000.0},
        ),
        patch("src.alpaca_broker.AlpacaBrokerBridge.submit_bracket_order") as mock_submit1,
    ):
        fresh_broker1 = PaperBroker(
            portfolio_path=str(tmp_path / "fresh_portfolio_1.json"),
            trades_path=str(tmp_path / "fresh_trades_1.csv"),
            initial_cash=100000.0,
        )
        assert len(fresh_broker1.state["open_positions"]) == 0

        res1 = fresh_broker1.execute_buy(ticker="NVDA", shares=10, price=120.0)
        assert res1["success"] is False
        assert "Active position already exists on Alpaca" in res1["error"]
        mock_submit1.assert_not_called()

    # 2. Open pending order with status 'new' on Alpaca
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_positions",
            return_value=[],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_orders",
            return_value=[{"symbol": "NVDA", "status": "new", "qty": "10"}],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_account_summary",
            return_value={"status": "CONNECTED", "equity": 100000.0, "cash": 90000.0},
        ),
        patch("src.alpaca_broker.AlpacaBrokerBridge.submit_bracket_order") as mock_submit2,
    ):
        fresh_broker2 = PaperBroker(
            portfolio_path=str(tmp_path / "fresh_portfolio_2.json"),
            trades_path=str(tmp_path / "fresh_trades_2.csv"),
            initial_cash=100000.0,
        )
        assert len(fresh_broker2.state["open_positions"]) == 0

        res2 = fresh_broker2.execute_buy(ticker="NVDA", shares=10, price=120.0)
        assert res2["success"] is False
        assert "Pending order already exists on Alpaca" in res2["error"]
        mock_submit2.assert_not_called()


def test_live_risk_source_reads_alpaca_account_and_skips_when_unavailable(tmp_path, monkeypatch):
    """
    On live path (LIVE_ALPACA_REQUIRED=true):
    - Reads equity and cash from Alpaca account (/v2/account).
    - If Alpaca is unreachable, skips buy and logs BUY_SKIPPED_ALPACA_UNAVAILABLE at WARNING.
    """
    monkeypatch.setenv("LIVE_ALPACA_REQUIRED", "true")

    broker = PaperBroker(
        portfolio_path=str(tmp_path / "live_risk_pf.json"),
        trades_path=str(tmp_path / "live_risk_tr.csv"),
        initial_cash=50000.0,
    )

    # 1. Alpaca unreachable -> execute_buy skips buy with BUY_SKIPPED_ALPACA_UNAVAILABLE
    with patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=False):
        res = broker.execute_buy("NVDA", shares=10, price=100.0)
        assert res["success"] is False
        assert res["error"] == "BUY_SKIPPED_ALPACA_UNAVAILABLE"

    # 2. Alpaca connected with live equity $200k -> 10% concentration limit scales to $20k
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch("src.alpaca_broker.AlpacaBrokerBridge.get_open_positions", return_value=[]),
        patch("src.alpaca_broker.AlpacaBrokerBridge.get_open_orders", return_value=[]),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_account_summary",
            return_value={"status": "CONNECTED", "equity": 200000.0, "cash": 200000.0},
        ),
    ):
        # Buy cost is $15,000 (which exceeds local $50k * 10% = $5k, but is within Alpaca $200k * 10% = $20k)
        res_live = broker.execute_buy("NVDA", shares=150, price=100.0)
        assert res_live["success"] is True
'''

content += new_tests

with open("tests/test_risk_and_position_limits.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated tests/test_risk_and_position_limits.py")
