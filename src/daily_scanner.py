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
            tickers = [line.strip() for line in f if line.strip()]
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
            features_df, price_hist, news_df = preprocess_data(ticker, use_cache=False)

            if features_df.empty or price_hist.empty:
                logger.warning(f"Insufficient data for {ticker}. Skipping.")
                continue

            spec_features = features_df.iloc[-1:][FEATURES]
            pred, conf = get_prediction_on_latest_data(model, spec_features, FEATURES)

            raw_pred = pred[0]
            confidence = conf[0][1]
            rsi = price_hist["rsi"].iloc[-1] if "rsi" in price_hist.columns else 50.0
            curr_close = price_hist["Close"].iloc[-1]
            atr_val = (
                price_hist["atr"].iloc[-1]
                if "atr" in price_hist.columns
                else curr_close * 0.02
            )
            sma = (
                price_hist["sma200"].iloc[-1]
                if "sma200" in price_hist.columns
                else curr_close
            )
            above_sma = curr_close > sma

            # Optimal Regime Filter
            if confidence >= 0.52 and rsi < 75:
                signal_type = "BUY"
            else:
                signal_type = "SELL"

            # Calculate Take-Profit and Stop-Loss Targets
            tp_target = curr_close + (2.5 * atr_val)
            sl_target = curr_close - ((3.0 if above_sma else 1.5) * atr_val)
            regime_str = (
                "▲ BULLISH (Above SMA200)" if above_sma else "▼ BEARISH (Below SMA200)"
            )

            card = format_signal_card(
                ticker=ticker,
                signal=signal_type,
                confidence=confidence,
                current_price=curr_close,
                stop_loss=sl_target,
                regime=regime_str,
                top_features=[{"feature": "RSI", "importance": rsi}],
                take_profit=tp_target,
            )
            signals_summary.append(card)

            # Dispatch Alerts if Webhooks / Tokens exist
            if os.getenv("DISCORD_WEBHOOK_URL"):
                send_discord_alert(card)
            if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
                send_telegram_alert(card)

            logger.info(
                f"✓ {ticker:<6} Signal: {signal_type:<4} | Conf: {confidence:.1%} | "
                f"Close: ${curr_close:.2f} | TP: ${tp_target:.2f} | SL: ${sl_target:.2f}"
            )

        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}")

    # Save summary artifact
    os.makedirs("results", exist_ok=True)
    summary_path = os.path.join("results", "daily_signals_latest.json")
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
