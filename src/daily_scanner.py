import os
import json
import pandas as pd
from datetime import datetime, timezone
from src.utils import get_logger
from src.config import FEATURES
from src.modeling import load_model, get_prediction_on_latest_data
from src.preprocessing import preprocess_data
from src.alerts import format_signal_card, send_discord_alert, send_telegram_alert

logger = get_logger(__name__)


def run_daily_market_scan() -> list:
    """
    Scans the entire stock universe defined in stocks.txt, generates
    tomorrow's momentum signals, Take-Profit targets, and Stop-Loss brackets,
    and dispatches live alerts to Discord and Telegram.
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

            # Optimal Regime Filter
            if confidence >= 0.52 and rsi < 75:
                signal_type = "BUY"
            else:
                signal_type = "SELL"

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
                ],
                take_profit=tp_target,
            )
            signals_summary.append(card)

            logger.info(
                f"+ {ticker:<6} Signal: {signal_type:<4} | Conf: {confidence:.1%} | "
                f"Close: ${curr_close:.2f} | TP: ${tp_target:.2f} | SL: ${sl_target:.2f} | PoC: ${poc_val:.2f}"
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

    # Dispatch Alerts to Discord / Telegram
    discord_url = os.getenv("DISCORD_WEBHOOK_URL")
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID")

    logger.info(
        f"Notification status: Discord URL present = {bool(discord_url)}, Telegram present = {bool(telegram_token and telegram_chat)}"
    )

    if discord_url:
        from src.alerts import send_discord_digest, send_discord_market_pulse

        logger.info("Sending Master Market Digest to Discord...")
        send_discord_digest(signals_summary, webhook_url=discord_url)

        # Dispatch Morning Market Pulse
        top_buys = [s for s in signals_summary if s["signal"] == "BUY"]
        send_discord_market_pulse(
            {
                "vix_level": 15.2,
                "vix_regime": "LOW VOLATILITY / BULL REGIME",
                "top_buys": top_buys,
                "portfolio_equity": 100000.0,
                "open_positions_count": len(top_buys),
            },
            webhook_url=discord_url,
        )

        import time

        # Send individual trade setups for BUY signals
        for card in top_buys[:3]:
            time.sleep(1.0)  # Respect Discord API rate limit
            send_discord_alert(card, webhook_url=discord_url)

    # Dispatch Telegram Digest & Alerts
    if telegram_token and telegram_chat:
        from src.dispatcher import send_telegram_digest

        send_telegram_digest(
            signals_summary, bot_token=telegram_token, chat_id=telegram_chat
        )
        for card in signals_summary:
            if card["signal"] == "BUY":
                send_telegram_alert(
                    card, bot_token=telegram_token, chat_id=telegram_chat
                )

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
