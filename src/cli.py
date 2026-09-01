"""
Sentilyze Command-Line Interface (CLI).
Interactive terminal tool for 4-Agent Quant Committee deliberations, universe scanning, and portfolio auditing.
"""

import sys
import os
import argparse
import json

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.agent_committee import convene_trading_committee
from src.paper_broker import PaperBroker
from src.realtime_tracker import fetch_live_quote
from src.utils import get_logger

logger = get_logger(__name__)

# ANSI Colors for Terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    banner = f"""{CYAN}{BOLD}
  ███████╗███████╗███╗   ██╗████████╗██╗██╗  ██╗   ██╗███████╗███████╗
  ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
  ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██║   ╚████╔╝   ███╔╝ █████╗  
  ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
  ███████║███████╗██║ ╚████║   ██║   ██║███████╗██║   ███████╗███████╗
  ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
{RESET}{YELLOW}  🤖 24/7 Autonomous Hybrid Quantitative Multi-Agent Intelligence Engine{RESET}
"""
    print(banner)


def cmd_audit(ticker: str):
    """Runs a live 4-Agent Quantitative Committee deliberation for a stock ticker."""
    ticker = ticker.upper().strip()
    print_banner()
    print(f"\n{BOLD}🏛️  CONVENING 4-AGENT QUANTITATIVE COMMITTEE FOR {CYAN}{ticker}{RESET}...\n")

    try:
        quote = fetch_live_quote(ticker)
        spot_price = float(quote.get("price", 100.0))
        delib = convene_trading_committee(ticker, spot_price=spot_price, save_resolution=True)
    except Exception as e:
        print(f"{RED}❌ Error running deliberation for {ticker}: {e}{RESET}")
        return

    verdict = delib.get("final_resolution", "HOLD")
    conviction = delib.get("consensus_conviction_pct", 50.0)
    cro = delib.get("cro_signoff", {})
    kelly = delib.get("kelly_allocation_pct", 0.0)
    tp1 = delib.get("tp1_target", 0.0)
    tp2 = delib.get("tp2_target", 0.0)
    sl = delib.get("stop_loss_target", 0.0)
    testimonies = delib.get("agent_testimonies", [])

    is_buy = "BUY" in str(verdict).upper() or "SCALE_IN" in str(verdict).upper()
    verdict_color = GREEN if is_buy else (YELLOW if "HOLD" in str(verdict).upper() else RED)

    print(f"┌────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ {BOLD}TICKER:{RESET} {CYAN}{ticker:<8}{RESET} │ {BOLD}SPOT PRICE:{RESET} ${spot_price:<10.2f} │ {BOLD}DATE:{RESET} {delib.get('timestamp', '')[:10]:<12}     │")
    print(f"├────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ {BOLD}COUNCIL VERDICT:{RESET} {verdict_color}{BOLD}{verdict:<58}{RESET}│")
    print(f"│ {BOLD}CONSENSUS CONVICTION:{RESET} {conviction:.1f}% │ {BOLD}FRACTIONAL KELLY SIZING:{RESET} {kelly:.1f}% of capital     │")
    print(f"├────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ {BOLD}🎯 TAKE-PROFIT 1 (+2.5 ATR):{RESET} ${tp1:<9.2f} │ {BOLD}🛡️ STOP-LOSS (-1.5 ATR):{RESET} ${sl:<9.2f}    │")
    print(f"│ {BOLD}🏆 TAKE-PROFIT 2 (+4.5 ATR):{RESET} ${tp2:<9.2f} │ {BOLD}⚡ 1st TARGET ACTION:{RESET} Bank 50% & Trail SL   │")
    print(f"└────────────────────────────────────────────────────────────────────────────┘\n")

    print(f"{BOLD}📊 SPECIALIST AGENT ROUND-TABLE TRANSCRIPT:{RESET}\n")
    for t in testimonies:
        agent_name = t.get("agent_name", "Specialist Agent")
        vote = t.get("vote", "NEUTRAL")
        agent_conv = t.get("conviction_score", 50.0)
        thesis = t.get("thesis", "Aligned.")
        vote_c = GREEN if vote == "BUY" else (RED if vote == "SELL" else YELLOW)

        print(f"  ● {BOLD}{agent_name}{RESET}: {vote_c}{BOLD}[{vote} - {agent_conv:.0f}%]{RESET}")
        print(f"    └─ {thesis}\n")


def cmd_portfolio():
    """Displays current portfolio metrics and live holdings."""
    print_banner()
    broker = PaperBroker()
    summary = broker.get_portfolio_summary()
    equity = summary.get("total_equity", 100000.0)
    cash = summary.get("cash", 100000.0)
    unrealized = summary.get("unrealized_pnl", 0.0)
    ret_pct = summary.get("unrealized_pnl_pct", 0.0)
    win_rate = summary.get("win_rate", 0.0)
    open_pos = broker.state.get("open_positions", {})

    pnl_c = GREEN if unrealized >= 0 else RED

    print(f"{BOLD}💼 AUTONOMOUS PAPER PORTFOLIO LEDGER{RESET}")
    print(f"───────────────────────────────────────────────────────────────")
    print(f"  💰 Total Equity:      ${equity:,.2f}")
    print(f"  💵 Cash Balance:      ${cash:,.2f}")
    print(f"  📈 Unrealized PnL:    {pnl_c}${unrealized:+,.2f} ({ret_pct:+.2f}%){RESET}")
    print(f"  🏆 Strategy Win Rate: {win_rate:.1f}%")
    print(f"───────────────────────────────────────────────────────────────\n")

    if open_pos:
        print(f"{BOLD}📦 ACTIVE HOLDINGS ({len(open_pos)} positions):{RESET}")
        for sym, pos in open_pos.items():
            entry_p = float(pos.get("entry_price", 0))
            curr_p = float(pos.get("current_price", entry_p))
            shares = int(pos.get("shares", 0))
            pos_pnl = (curr_p - entry_p) * shares
            color = GREEN if pos_pnl >= 0 else RED
            print(f"  • {BOLD}{sym:<6}{RESET}: {shares} shs @ ${entry_p:,.2f} (Current: ${curr_p:,.2f} | {color}${pos_pnl:+,.2f}{RESET})")
    else:
        print(f"{YELLOW}ℹ️ No open positions. Capital safely preserved in cash.{RESET}")
    print()


def main():
    # Smart argument handling: allow direct ticker inputs like `python sentilyze.py NVDA`
    args = sys.argv[1:]
    if not args:
        print_banner()
        print(f"Usage examples:")
        print(f"  {CYAN}python sentilyze.py NVDA{RESET}          (Audit stock ticker NVDA)")
        print(f"  {CYAN}python sentilyze.py audit AAPL{RESET}    (Audit stock ticker AAPL)")
        print(f"  {CYAN}python sentilyze.py portfolio{RESET}     (View active portfolio and ledger)")
        print()
        return

    first_arg = args[0].strip()

    if first_arg.lower() in ["portfolio", "p", "--portfolio", "-p"]:
        cmd_portfolio()
    elif first_arg.lower() in ["audit", "a", "--audit", "-a"]:
        ticker = args[1] if len(args) > 1 else "NVDA"
        cmd_audit(ticker)
    elif first_arg.startswith("-"):
        print_banner()
        print(f"Usage: python sentilyze.py [TICKER | portfolio]")
    else:
        # Default: treat any plain text argument as a stock ticker!
        cmd_audit(first_arg)


if __name__ == "__main__":
    main()
