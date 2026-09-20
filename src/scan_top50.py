"""
Option B: Top 50 SPY & QQQ Systematic Market Scan Engine.
Deliberates across the 50 largest, most liquid mega-caps from SPY & QQQ using
the high-speed 8-Agent Quantitative Trading Committee (sub-50ms per ticker).
"""

import os
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
import concurrent.futures
from src.utils import get_logger
from src.agent_committee import convene_trading_committee
from src.realtime_tracker import fetch_universe_live_quotes

logger = get_logger(__name__)

# Top 50 Largest Mega-Cap Constituents from S&P 500 (SPY) and Nasdaq-100 (QQQ)
TOP_50_SPY_QQQ = [
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "JPM",
    "LLY",
    "V",
    "UNH",
    "XOM",
    "COST",
    "MA",
    "HD",
    "PG",
    "NFLX",
    "JNJ",
    "BAC",
    "ABBV",
    "CRM",
    "AMD",
    "CVX",
    "WMT",
    "MRK",
    "KO",
    "ORCL",
    "PEP",
    "QCOM",
    "LIN",
    "TMO",
    "CSCO",
    "MCD",
    "IBM",
    "ABT",
    "GE",
    "CAT",
    "TXN",
    "INTU",
    "AMAT",
    "DIS",
    "VZ",
    "CMCSA",
    "PFE",
    "NOW",
    "PM",
    "AMGN",
    "PLTR",
    "ISRG",
]


def run_top50_market_scan(max_workers: int = 4) -> Dict[str, Any]:
    """
    Executes a comprehensive 8-Agent Committee scan across the Top 50 SPY & QQQ tickers.
    Returns structured results and persists to results/top50_scan_summary.json.
    """
    t_start = time.perf_counter()
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.isoformat()

    logger.info(
        f"🌐 [TOP 50 SPY/QQQ SCAN] Starting systematic scan across {len(TOP_50_SPY_QQQ)} mega-caps..."
    )

    # 1. Fetch live or cached quotes in bulk
    quotes_map = fetch_universe_live_quotes(TOP_50_SPY_QQQ)

    # 2. Pre-warm FinBERT once to ensure zero cold-start delay
    try:
        from src.fast_finbert import get_fast_finbert_pipeline

        get_fast_finbert_pipeline()
    except Exception as fe:
        logger.debug(f"FinBERT pre-warm notice: {fe}")

    # 3. Parallel 8-Agent Committee Deliberation
    resolutions_by_ticker = {}

    def _deliberate(sym: str):
        try:
            q = quotes_map.get(sym, {})
            cached_price = float(q.get("price", 0.0))
            res = convene_trading_committee(
                sym, save_resolution=False, spot_price=cached_price
            )
            return sym, res
        except Exception as e:
            logger.warning(f"Deliberation notice for {sym}: {e}")
            return sym, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_deliberate, sym): sym for sym in TOP_50_SPY_QQQ}
        for future in concurrent.futures.as_completed(futures):
            sym, res = future.result()
            if res and isinstance(res, dict):
                resolutions_by_ticker[sym] = res

    # 4. Categorize Setups
    buy_setups = []
    neutral_setups = []
    vetoed_setups = []

    for sym, res in resolutions_by_ticker.items():
        action = res.get("action_code", "HOLD")
        conviction = float(res.get("consensus_conviction_pct", 50.0))
        kelly = float(res.get("kelly_allocation_pct", 0.0))
        spot = float(res.get("spot_price", 0.0))
        tp1 = float(res.get("tp1_target", spot * 1.05))
        tp2 = float(res.get("tp2_target", spot * 1.10))
        sl = float(res.get("stop_loss_target", spot * 0.95))
        cro_thesis = res.get("cro_thesis", "")

        setup_card = {
            "ticker": sym,
            "action": action,
            "conviction_pct": conviction,
            "kelly_pct": kelly,
            "spot_price": spot,
            "tp1": tp1,
            "tp2": tp2,
            "stop_loss": sl,
            "reward_risk_ratio": (
                round((tp1 - spot) / max(spot - sl, 0.01), 2) if spot > sl else 1.5
            ),
            "cro_thesis": cro_thesis,
            "final_resolution": res.get("final_resolution", action),
            "veto_reason": res.get("veto_reason"),
        }

        if action in ["EXECUTE_BUY", "SCALE_IN"]:
            buy_setups.append(setup_card)
        elif action == "HOLD":
            neutral_setups.append(setup_card)
        else:
            vetoed_setups.append(setup_card)

    # Sort buy setups by highest conviction and optimal Kelly sizing
    buy_setups.sort(key=lambda x: (x["conviction_pct"], x["kelly_pct"]), reverse=True)
    neutral_setups.sort(key=lambda x: x["conviction_pct"], reverse=True)
    vetoed_setups.sort(key=lambda x: x["conviction_pct"])

    total_scanned = len(resolutions_by_ticker)
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    scan_summary = {
        "timestamp": now_str,
        "universe_name": "Top 50 Mega-Caps (SPY & QQQ)",
        "total_tickers_scanned": total_scanned,
        "total_buy_signals": len(buy_setups),
        "total_neutral_signals": len(neutral_setups),
        "total_vetoed_signals": len(vetoed_setups),
        "bullish_sentiment_pct": round(
            (len(buy_setups) / max(total_scanned, 1)) * 100.0, 1
        ),
        "scan_latency_ms": round(elapsed_ms, 2),
        "avg_latency_per_ticker_ms": round(elapsed_ms / max(total_scanned, 1), 2),
        "top_3_alpha_picks": buy_setups[:3],
        "all_buy_setups": buy_setups,
        "neutral_setups": neutral_setups,
        "vetoed_setups": vetoed_setups,
    }

    # 5. Persist to results/top50_scan_summary.json
    os.makedirs("results", exist_ok=True)
    summary_path = os.path.join("results", "top50_scan_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(scan_summary, f, indent=2)

    # 6. Safely update results/committee_resolutions.json
    res_path = os.path.join("results", "committee_resolutions.json")
    try:
        current_res = {}
        if os.path.exists(res_path):
            with open(res_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    current_res = loaded.get("resolutions", loaded)

        current_res.update(resolutions_by_ticker)
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": now_str,
                    "total_audited": len(current_res),
                    "resolutions": current_res,
                },
                f,
                indent=2,
            )
    except Exception as e:
        logger.warning(f"Notice updating committee_resolutions.json: {e}")

    logger.info(
        f"✅ [TOP 50 SCAN COMPLETE] Scanned {total_scanned} mega-caps in {elapsed_ms:.1f}ms "
        f"({elapsed_ms/max(total_scanned, 1):.1f}ms/ticker). "
        f"Found {len(buy_setups)} BUY opportunities, {len(neutral_setups)} HOLD, {len(vetoed_setups)} VETO."
    )

    return scan_summary


if __name__ == "__main__":
    summary = run_top50_market_scan()
    print(
        f"Top 50 Scan Complete: {summary['total_buy_signals']} Buys, "
        f"{summary['total_neutral_signals']} Neutral, {summary['total_vetoed_signals']} Vetoes. "
        f"Elapsed: {summary['scan_latency_ms']} ms"
    )
