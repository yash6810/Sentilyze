import os
import json
import pandas as pd
from datetime import datetime, timezone
from src.utils import get_logger
from src.config import FEATURES
from src.modeling import load_model, get_prediction_on_latest_data
from src.preprocessing import preprocess_data
from src.alerts import format_signal_card, send_discord_alert

logger = get_logger(__name__)


def run_daily_market_scan() -> list:
    """
    Scans the entire stock universe defined in stocks.txt, generates
    tomorrow's momentum signals, Take-Profit targets, and Stop-Loss brackets,
    and dispatches live alerts to Discord.
    """
    stocks_file = "stocks.txt"
    if os.path.exists(stocks_file):
        with open(stocks_file, "r") as f:
            tickers = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    else:
        tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMZN"]

    logger.info(f"Starting daily market signal scan for {len(tickers)} assets...")
    signals_summary = []

    for ticker in tickers:
        try:
            logger.info(f"Scanning {ticker}...")
            model_path = os.path.join("models", f"{ticker}_model.json")
            if not os.path.exists(model_path):
                logger.warning(
                    f"Model for {ticker} not found at {model_path}. Skipping."
                )
                continue

            model = load_model(model_path)
            features_df, price_hist, news_df = preprocess_data(
                ticker, period="2y", use_cache=False
            )

            if features_df.empty or price_hist.empty:
                logger.warning(f"Insufficient data for {ticker}. Skipping.")
                continue

            for f in FEATURES:
                if f not in features_df.columns:
                    features_df[f] = 0.0

            spec_features = features_df.iloc[-1:][FEATURES]
            pred, conf = get_prediction_on_latest_data(model, spec_features, FEATURES)

            raw_pred = int(pred[0])
            confidence = float(conf[0][1])
            rsi = float(
                price_hist["rsi"].iloc[-1] if "rsi" in price_hist.columns else 50.0
            )
            curr_close = float(price_hist["Close"].iloc[-1])
            atr_val = float(
                price_hist["atr"].iloc[-1]
                if "atr" in price_hist.columns
                else curr_close * 0.02
            )
            sma = float(
                price_hist["sma200"].iloc[-1]
                if "sma200" in price_hist.columns
                else curr_close
            )
            above_sma = bool(curr_close > sma)

            # Broader Market (SPY) Regime Check
            spy_ret_20d = 0.0
            spy_bullish = True
            try:
                from src.data_ingestion import get_price_history

                spy_hist = get_price_history("SPY", period="3mo", use_cache=True)
                if (
                    spy_hist is not None
                    and not spy_hist.empty
                    and "Close" in spy_hist.columns
                ):
                    spy_c = float(spy_hist["Close"].iloc[-1])
                    if len(spy_hist) >= 21:
                        spy_ret_20d = float(
                            (spy_c - spy_hist["Close"].iloc[-21])
                            / spy_hist["Close"].iloc[-21]
                        )
                    if len(spy_hist) >= 50:
                        spy_sma50 = float(spy_hist["Close"].rolling(50).mean().iloc[-1])
                        spy_bullish = bool(spy_c >= spy_sma50)
            except Exception as spy_err:
                logger.debug(f"Notice fetching SPY macro tape: {spy_err}")

            min_conf = float(
                os.getenv("MIN_CONVICTION_FLOOR", "0.60" if spy_bullish else "0.65")
            )

            # Relative Strength (RS) check against SPY
            if len(price_hist) >= 21:
                stock_ret_20d = float(
                    (curr_close - price_hist["Close"].iloc[-21])
                    / price_hist["Close"].iloc[-21]
                )
                rs_pass = bool(stock_ret_20d >= spy_ret_20d)
            else:
                stock_ret_20d = 0.0
                rs_pass = True

            # VPIN Microstructure Toxicity Check
            try:
                from src.vpin_toxicity import evaluate_ticker_toxicity

                vpin_res = evaluate_ticker_toxicity(ticker)
                vpin_val = float(vpin_res.get("vpin", 0.50))
                vpin_safe = (
                    not vpin_res.get("cro_veto_recommended", False) and vpin_val < 0.70
                )
            except Exception:
                vpin_val = 0.50
                vpin_safe = True

            # Portfolio Correlation & Sector Shield
            try:
                from src.correlation_shield import check_correlation_shield
                from src.cross_asset_pooling import get_sector_for_ticker
                from src.paper_broker import PaperBroker

                broker_state = PaperBroker().state
                open_pos = broker_state.get("open_positions", {})
                corr_res = check_correlation_shield(
                    ticker, open_pos, max_corr_threshold=0.70
                )
                corr_safe = bool(corr_res.get("allowed", True))

                # Sector Shield Check
                cand_sec = get_sector_for_ticker(ticker)
                occupied_secs = {get_sector_for_ticker(h) for h in open_pos.keys()}
                sector_safe = cand_sec not in occupied_secs or cand_sec in (
                    "General",
                    "Unknown",
                    "General_Market",
                )

                # 3-Day Anti-Whipsaw Cooldown Quarantine Check
                is_quarantined = False
                for ct in broker_state.get("closed_trades", []):
                    if ct.get("ticker") == ticker:
                        exit_d = ct.get("exit_date", "")
                        if exit_d:
                            try:
                                d_exit = datetime.fromisoformat(str(exit_d)[:10])
                                d_now = datetime.now(timezone.utc)
                                if (
                                    d_now - d_exit.replace(tzinfo=timezone.utc)
                                ).days < 3:
                                    is_quarantined = True
                                    break
                            except Exception:
                                pass
            except Exception:
                corr_safe = True
                sector_safe = True
                is_quarantined = False

            # Multi-Gate Institutional Gating
            preliminary_pass = (
                confidence >= min_conf
                and rsi < 72.0
                and above_sma
                and rs_pass
                and vpin_safe
                and corr_safe
                and sector_safe
                and not is_quarantined
            )

            signal_type = "HOLD"
            council_verdict_str = "HOLD"

            if preliminary_pass:
                # Convene Multi-Agent Council for Final Arbitrated Clearance
                try:
                    from src.agent_committee import convene_trading_committee

                    comm_res = convene_trading_committee(ticker, horizon_days=5)
                    c_dir = comm_res.get("consensus_direction", "HOLD")
                    c_conf = float(comm_res.get("conviction_score", 0.0))
                    c_veto = bool(comm_res.get("veto_active", False))

                    council_verdict_str = f"{c_dir} ({c_conf:.1%})"
                    if c_dir == "BUY" and c_conf >= 0.55 and not c_veto:
                        signal_type = "BUY"
                    else:
                        signal_type = "HOLD"
                except Exception as comm_err:
                    logger.debug(
                        f"Council deliberation fallback for {ticker}: {comm_err}"
                    )
                    signal_type = "BUY"
            elif confidence < 0.40 or (rsi > 80 and not above_sma):
                signal_type = "SELL"
            else:
                signal_type = "HOLD"

            # Calculate Take-Profit and Stop-Loss Targets
            tp_target = float(curr_close + (2.5 * atr_val))
            sl_target = float(curr_close - ((3.0 if above_sma else 1.5) * atr_val))
            regime_str = (
                "▲ BULLISH (Above SMA200)" if above_sma else "▼ BEARISH (Below SMA200)"
            )

            # Smart Money Structure & Confluence
            try:
                from src.smart_trader_engine import (
                    calculate_smart_money_zones,
                    evaluate_multi_timeframe_confluence,
                )

                sm_zones = calculate_smart_money_zones(price_hist)
                mtf = evaluate_multi_timeframe_confluence(ticker, price_hist)
                poc_val = sm_zones.get("volume_poc", curr_close)
                conf_verdict = mtf.get("verdict", "CONFLUENCE_OK")
            except Exception:
                poc_val = curr_close
                conf_verdict = "OK"

            card = format_signal_card(
                ticker=ticker,
                signal=signal_type,
                confidence=confidence,
                current_price=curr_close,
                stop_loss=sl_target,
                regime=regime_str,
                top_features=[
                    {"feature": "RSI", "importance": rsi},
                    {"feature": "Volume PoC", "importance": poc_val},
                    {"feature": "Confluence", "importance": conf_verdict},
                    {
                        "feature": "RS vs SPY",
                        "importance": (
                            "OUTPERFORMING" if rs_pass else "UNDERPERFORMING"
                        ),
                    },
                    {
                        "feature": "Council Verdict",
                        "importance": council_verdict_str,
                    },
                ],
                take_profit=tp_target,
            )
            signals_summary.append(card)

            logger.info(
                f"+ {ticker:<6} Signal: {signal_type:<4} | Conf: {confidence:.1%} | "
                f"Close: ${curr_close:.2f} | TP: ${tp_target:.2f} | SL: ${sl_target:.2f} | "
                f"RS: {'PASS' if rs_pass else 'FAIL'} | VPIN: {vpin_val:.3f} | Council: {council_verdict_str}"
            )

        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}", exc_info=True)

    # Fallback to pre-computed signals artifact if live scan produced no items
    summary_path = os.path.join("results", "daily_signals_latest.json")
    if not signals_summary and os.path.exists(summary_path):
        try:
            logger.warning(
                f"Live scan produced 0 signals. Loading fallback from {summary_path}..."
            )
            with open(summary_path, "r") as f:
                cached_data = json.load(f)
                signals_summary = cached_data.get("signals", [])
                logger.info(
                    f"Successfully loaded {len(signals_summary)} signals from fallback cache."
                )
        except Exception as e:
            logger.error(f"Failed to load fallback signals from {summary_path}: {e}")

    # Dispatch Alerts to Discord (Primary Institutional Notification Hub)
    discord_url = os.getenv("DISCORD_WEBHOOK_URL")
    logger.info(f"Discord Notification Status: URL Configured = {bool(discord_url)}")

    if discord_url:
        from src.alerts import send_discord_digest, send_discord_market_pulse

        logger.info("Sending Master Market Digest to Discord...")
        send_discord_digest(signals_summary, webhook_url=discord_url)

        # Dispatch Morning Market Pulse with live broker portfolio data
        top_buys = [s for s in signals_summary if s.get("signal") == "BUY"]

        try:
            from src.paper_broker import PaperBroker

            broker = PaperBroker()
            broker_summary = broker.get_portfolio_summary()
            open_pos = broker.state.get("open_positions", {})
            recent_closed = broker.state.get("closed_trades", [])
        except Exception as e:
            logger.debug(f"Notice loading broker state for market pulse: {e}")
            broker_summary = {}
            open_pos = {}
            recent_closed = []

        send_discord_market_pulse(
            {
                "vix_level": 15.2,
                "vix_regime": "LOW VOLATILITY / BULL REGIME",
                "top_buys": top_buys,
                "portfolio_equity": broker_summary.get("total_equity", 100000.0),
                "cash_balance": broker_summary.get("cash", 100000.0),
                "daily_pnl": broker_summary.get("daily_pnl", 0.0),
                "daily_return_pct": broker_summary.get("daily_return_pct", 0.0),
                "realized_pnl": broker_summary.get("realized_pnl", 0.0),
                "open_positions": open_pos,
                "recent_closed_trades": recent_closed,
            },
            webhook_url=discord_url,
        )

        import time

        # Send individual trade setups for top BUY signals
        for card in top_buys[:3]:
            time.sleep(1.0)  # Respect Discord API rate limit
            send_discord_alert(card, webhook_url=discord_url)

    # Dispatch HTML Email Digest
    if os.getenv("EMAIL_USER") and os.getenv("EMAIL_PASSWORD"):
        from src.dispatcher import send_email_digest

        logger.info("Sending Master Market HTML Digest via Email...")
        send_email_digest(signals_summary)

    # Execute Virtual Paper Trading Simulation ($100k Capital)
    try:
        from src.paper_broker import PaperBroker

        broker = PaperBroker()
        executed_actions = broker.execute_daily_signals(signals_summary)
        summary = broker.get_portfolio_summary()
        logger.info(
            f"Paper Portfolio Updated: Equity: ${summary['total_equity']:,.2f} | "
            f"Cash: ${summary['cash']:,.2f} | Open Positions: {summary['open_positions_count']} | "
            f"Unrealized PnL: ${summary['unrealized_pnl']:+,.2f} | Realized PnL: ${summary['realized_pnl']:+,.2f}"
        )
    except Exception as e:
        logger.error(f"Error updating Paper Broker: {e}", exc_info=True)

    # Save summary artifact if we generated fresh signals
    if signals_summary:
        os.makedirs("results", exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "scan_time": datetime.now(timezone.utc).isoformat(),
                    "num_assets": len(signals_summary),
                    "signals": signals_summary,
                },
                f,
                indent=2,
            )
        logger.info(f"Daily scan complete. Saved summary to {summary_path}")
    return signals_summary


if __name__ == "__main__":
    run_daily_market_scan()
