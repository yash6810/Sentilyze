from src.telegram_bot import handle_telegram_command, send_telegram_bot_message


def test_telegram_help_command():
    res = handle_telegram_command("/help")
    assert res["status"] == "success"
    assert "Sentilyze Mobile AI Command Desk" in res["markdown_text"]


def test_telegram_signal_command():
    res = handle_telegram_command("/signal NVDA")
    assert res["status"] == "success"
    assert "AI Signal Analysis: NVDA" in res["markdown_text"]
    assert "Take-Profit 1" in res["markdown_text"]


def test_telegram_portfolio_command():
    res = handle_telegram_command("/portfolio")
    assert res["status"] == "success"
    assert "Sentilyze Live Portfolio Status" in res["markdown_text"]


def test_telegram_statarb_command():
    res = handle_telegram_command("/statarb")
    assert res["status"] == "success"
    assert "Statistical Arbitrage" in res["markdown_text"]


def test_telegram_options_command():
    res = handle_telegram_command("/options NVDA")
    assert res["status"] == "success"
    assert "Options Microstructure: NVDA" in res["markdown_text"]
    assert "Max Pain" in res["markdown_text"]


def test_telegram_dcf_command():
    res = handle_telegram_command("/dcf NVDA")
    assert res["status"] == "success"
    assert "Fundamental Health" in res["markdown_text"]
    assert "Piotroski F-Score" in res["markdown_text"]


def test_telegram_killswitch_command():
    res = handle_telegram_command("/killswitch")
    assert res["status"] == "warning"
    assert "EMERGENCY KILL-SWITCH" in res["markdown_text"]


def test_send_telegram_bot_message_fallback():
    # When token is empty, gracefully returns False without throwing exception
    sent = send_telegram_bot_message(bot_token="", chat_id="", text="Test")
    assert sent is False
