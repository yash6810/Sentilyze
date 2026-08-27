import os
import json
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from src.paper_broker import PaperBroker
from src.realtime_tracker import fetch_live_quote, evaluate_intraday_execution
from src.stress_tester import run_monte_carlo_var
from src.utils import get_logger

logger = get_logger(__name__)


def handle_bot_command(command_str: str) -> Dict[str, Any]:
    """
    Parses and processes interactive bot commands:
    - `/signal <ticker>`
    - `/portfolio`
    - `/scan`
    - `/execute`
    - `/var`
    """
    parts = command_str.strip().split()
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].upper() if len(parts) > 1 else "NVDA"

    if cmd in ["/signal", "signal"]:
        ticker = arg
        q = fetch_live_quote(ticker)
        curr_p = float(q.get("price", 0))
        chg = float(q.get("change_pct", 0))

        # Check model
        from src.modeling import load_model, get_prediction_on_latest_data
        from src.preprocessing import preprocess_data
        from src.config import FEATURES

        m_path = f"models/{ticker}_model.json"
        signal_txt = "HOLD"
        conf_val = 0.50
        if os.path.exists(m_path):
            try:
                features_df, _, _ = preprocess_data(ticker, use_cache=True)
                model = load_model(m_path)
                pred_raw, conf_raw = get_prediction_on_latest_data(
                    model, features_df.tail(1), FEATURES
                )
                pred = int(pred_raw[0])
                conf_val = (
                    float(conf_raw[0][1])
                    if len(conf_raw[0]) > 1
                    else float(conf_raw[0][0])
                )
                signal_txt = "BUY" if pred == 1 and conf_val >= 0.50 else "HOLD"
            except Exception as e:
                logger.debug(f"Signal inference error in discord_bot for {ticker}: {e}")

        atr_est = curr_p * 0.03
        tp1 = curr_p + 2.5 * atr_est
        tp2 = curr_p + 4.5 * atr_est
        sl = curr_p - 1.5 * atr_est

        color = 0x00D4AA if signal_txt == "BUY" else 0xF59E0B
        return {
            "title": f"⚡ Sentilyze Signal: {ticker} (${curr_p:,.2f} | {chg:+.2f}%)",
            "description": (
                f"• **AI Verdict:** `{signal_txt}` ({conf_val*100:.1f}% Confidence)\n"
                f"• **Take-Profit 1 (+2.5 ATR):** `${tp1:,.2f}` (50% Scale-Out)\n"
                f"• **Take-Profit 2 (+4.5 ATR):** `${tp2:,.2f}` (Runner Target)\n"
                f"• **Stop-Loss Protection:** `${sl:,.2f}`\n"
            ),
            "color": color,
        }

    elif cmd in ["/portfolio", "portfolio", "/pnl"]:
        broker = PaperBroker()
        summary = broker.get_portfolio_summary()
        open_pos = broker.state.get("open_positions", {})
        pos_txt = "\n".join(
            [
                f"• `{t}`: {p['shares']} shs @ ${p['entry_price']:.2f} (TP1: ${p.get('tp1_target', 0):.2f})"
                for t, p in open_pos.items()
            ]
        )
        if not pos_txt:
            pos_txt = "No active open positions. Cash is 100% liquid."

        return {
            "title": f"💼 Virtual Paper Portfolio ($100k Account)",
            "description": (
                f"• **Total Equity:** **`${summary['total_equity']:,.2f}` ({summary['total_return_pct']:+.2f}%)**\n"
                f"• **Available Cash:** `${summary['cash']:,.2f}`\n"
                f"• **Unrealized PnL:** `${summary['unrealized_pnl']:+,.2f}`\n"
                f"• **Win Rate:** `{summary['win_rate']:.1f}%` ({summary['winning_trades']}/{summary['total_trades']})\n\n"
                f"**Active Holdings:**\n{pos_txt}"
            ),
            "color": 0x7C3AED,
        }

    elif cmd in ["/execute", "execute"]:
        res = evaluate_intraday_execution()
        trades = res.get("executed_trades", [])
        if trades:
            trade_lines = "\n".join(
                [
                    f"• Sold `{t['ticker']}` @ ${t['exit_price']:.2f} | PnL: **${t['pnl']:+,.2f}** ({t['reason']})"
                    for t in trades
                ]
            )
            desc = f"**Executed {len(trades)} live exits:**\n{trade_lines}"
        else:
            desc = "All open positions are within target bands. No exit thresholds triggered."

        return {
            "title": "⚡ 5-Minute Intraday Execution Check",
            "description": desc,
            "color": 0x10B981 if trades else 0x64748B,
        }

    elif cmd in ["/var", "var", "/stress"]:
        var_res = run_monte_carlo_var(initial_equity=100000.0, num_paths=1000, days=45)
        return {
            "title": "🎲 Monte Carlo 1,000-Path VaR Simulation",
            "description": (
                f"• **95% Value-at-Risk (45d):** **`${var_res['var_95_dollar']:,.2f}` ({var_res['var_95_pct']:.2f}%)**\n"
                f"• **95% Expected Shortfall (CVaR):** `${var_res['cvar_95_dollar']:,.2f}`\n"
                f"• **Probability of Profit:** `{var_res['prob_profit_pct']:.1f}%`\n"
                f"• **Worst-Case Drawdown:** `{var_res['worst_case_drawdown_pct']:.1f}%`"
            ),
            "color": 0x00D4AA,
        }

    return {
        "title": "🤖 Sentilyze Interactive Bot Help",
        "description": (
            "Available Commands:\n"
            "• `/signal <ticker>` - Get instant AI prediction & targets (e.g. `/signal AMD`)\n"
            "• `/portfolio` - View live $100k equity, holdings & PnL\n"
            "• `/execute` - Check & execute live intraday exits\n"
            "• `/var` - Run Monte Carlo VaR simulation\n"
        ),
        "color": 0x64748B,
    }


def send_bot_command_reply(command_str: str, webhook_url: Optional[str] = None) -> bool:
    """Executes command and posts formatted embed reply to Discord."""
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return False
    embed = handle_bot_command(command_str)
    embed["footer"] = {"text": "Sentilyze Interactive AI Bot"}
    embed["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        res = requests.post(url, json={"embeds": [embed]}, timeout=8)
        return res.status_code in [200, 204]
    except Exception as e:
        logger.error(f"Discord command reply error: {e}")
        return False
