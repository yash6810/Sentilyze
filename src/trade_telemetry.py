"""
Instant Trade Telemetry & Results Engine.
Provides direct, zero-overhead access to:
  - Today's and Yesterday's executed trades and realized profits
  - Complete portfolio equity, cash, and win-rate statistics
  - Recent closed trade history and active open positions
Designed for immediate reporting via CLI or programmatic query without manual scripting.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta

from src.paper_broker import PaperBroker


def get_trade_telemetry(days_lookback: int = 7) -> Dict[str, Any]:
    """
    Fetches real-time trade execution summary, partitioning by today, yesterday, and recent history.
    """
    broker = PaperBroker()
    summary = broker.get_portfolio_summary()

    # Dates in UTC
    now_utc = datetime.now(timezone.utc)
    today_str = str(now_utc.date())
    yesterday_str = str((now_utc - timedelta(days=1)).date())

    closed_trades = broker.state.get("closed_trades", [])
    open_positions = broker.state.get("open_positions", {})

    today_trades = []
    yesterday_trades = []
    today_realized_pnl = 0.0
    yesterday_realized_pnl = 0.0

    for t in closed_trades:
        exit_date = t.get("exit_date", "")
        pnl = float(t.get("pnl", 0.0))
        if exit_date == today_str:
            today_trades.append(t)
            today_realized_pnl += pnl
        elif exit_date == yesterday_str:
            yesterday_trades.append(t)
            yesterday_realized_pnl += pnl

    # Last N trades
    recent_trades = closed_trades[-days_lookback:] if closed_trades else []

    return {
        "as_of_date": today_str,
        "as_of_timestamp": now_utc.isoformat(),
        "total_equity": round(summary.get("total_equity", 100000.0), 2),
        "cash": round(summary.get("cash", 100000.0), 2),
        "realized_pnl": round(summary.get("realized_pnl", 0.0), 2),
        "unrealized_pnl": round(summary.get("unrealized_pnl", 0.0), 2),
        "win_rate_pct": round(summary.get("win_rate", 0.0), 1),
        "total_trades": summary.get("total_trades", len(closed_trades)),
        "winning_trades": summary.get("winning_trades", 0),
        "losing_trades": summary.get("losing_trades", 0),
        "open_positions_count": len(open_positions),
        "open_positions": open_positions,
        "today": {
            "date": today_str,
            "trades_count": len(today_trades),
            "realized_pnl": round(today_realized_pnl, 2),
            "trades": today_trades,
        },
        "yesterday": {
            "date": yesterday_str,
            "trades_count": len(yesterday_trades),
            "realized_pnl": round(yesterday_realized_pnl, 2),
            "trades": yesterday_trades,
        },
        "recent_closed_trades": recent_trades,
    }


def print_formatted_trade_report(data: Optional[Dict[str, Any]] = None):
    """Prints a clean, concise institutional trade report to the terminal."""
    if data is None:
        data = get_trade_telemetry()

    eq = data["total_equity"]
    cash = data["cash"]
    rpnl = data["realized_pnl"]
    urpnl = data["unrealized_pnl"]
    wr = data["win_rate_pct"]
    tot_t = data["total_trades"]
    w_t = data["winning_trades"]
    l_t = data["losing_trades"]

    print("=" * 68)
    print(
        f"💼 SENTILYZE INSTITUTIONAL PORTFOLIO & TRADE RESULTS ({data['as_of_date']})"
    )
    print("=" * 68)
    print(
        f"💰 Total Equity:      ${eq:,.2f}  (Net Return: +{((eq - 100000.0) / 1000.0):.2f}%)"
    )
    print(f"💵 Cash Balance:      ${cash:,.2f}  ({(cash / eq * 100):.1f}% Liquid)")
    print(
        f"📈 Realized PnL:      ${rpnl:+,.2f}  ({w_t}W / {l_t}L of {tot_t} trades | Win: {wr:.1f}%)"
    )
    print(
        f"⚡ Unrealized PnL:    ${urpnl:+,.2f}  (Open Positions: {data['open_positions_count']})"
    )
    print("-" * 68)

    # Today's trades
    td = data["today"]
    print(
        f"📅 TODAY'S TRADES ({td['date']}): {td['trades_count']} closed | PnL: ${td['realized_pnl']:+,.2f}"
    )
    if td["trades"]:
        for t in td["trades"]:
            print(
                f"   • {t['ticker']:<5} {t['shares']} shs: In ${t['entry_price']:.2f} -> "
                f"Out ${t['exit_price']:.2f} | PnL: ${t['pnl']:+,.2f} ({t['return_pct']:+.2f}%) "
                f"[{t.get('reason', 'EXIT')}]"
            )
    else:
        print("   (No trades closed today)")

    # Yesterday's trades
    yd = data["yesterday"]
    print(
        f"\n📅 YESTERDAY'S TRADES ({yd['date']}): {yd['trades_count']} closed | PnL: ${yd['realized_pnl']:+,.2f}"
    )
    if yd["trades"]:
        for t in yd["trades"]:
            print(
                f"   • {t['ticker']:<5} {t['shares']} shs: In ${t['entry_price']:.2f} -> "
                f"Out ${t['exit_price']:.2f} | PnL: ${t['pnl']:+,.2f} ({t['return_pct']:+.2f}%) "
                f"[{t.get('reason', 'EXIT')}]"
            )
    else:
        print("   (No trades closed yesterday)")

    # Open Positions
    if data["open_positions"]:
        print(f"\n📦 ACTIVE OPEN POSITIONS ({len(data['open_positions'])}):")
        for sym, pos in data["open_positions"].items():
            ep = pos.get("entry_price", 0.0)
            cp = pos.get("current_price", ep)
            sh = pos.get("shares", 0)
            pnl = (cp - ep) * sh
            sl_p = pos.get("sl_target", 0)
            print(
                f"   • {sym:<5} {sh} shs @ ${ep:.2f} (Current: ${cp:.2f} | "
                f"PnL: ${pnl:+,.2f}) [SL: ${sl_p:.2f}]"
            )
    else:
        print("\n📦 ACTIVE OPEN POSITIONS: None (100% Capital Preserved in Cash)")
    print("=" * 68)


if __name__ == "__main__":
    print_formatted_trade_report()
