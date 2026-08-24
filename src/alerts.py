import os
import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from src.utils import get_logger

logger = get_logger(__name__)


def format_signal_card(
    ticker: str,
    signal: str,
    confidence: float,
    current_price: float,
    stop_loss: float,
    regime: str,
    top_features: List[Dict[str, Any]],
    take_profit: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Construct a standardized trade signal data payload.

    Args:
        ticker (str): Stock ticker symbol (e.g. NVDA).
        signal (str): "BUY" or "SELL".
        confidence (float): Probability confidence (0.0 to 1.0).
        current_price (float): Latest stock price.
        stop_loss (float): Calculated ATR dynamic stop loss price.
        regime (str): Market regime ("BULLISH / ABOVE 200 SMA" or "BEARISH").
        top_features (List[Dict[str, Any]]): Top SHAP feature drivers.
        take_profit (float, optional): Calculated ATR take-profit target.

    Returns:
        Dict[str, Any]: Structured trade alert payload.
    """
    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": confidence,
        "current_price": current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "regime": regime,
        "top_features": top_features,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def send_discord_alert(
    alert_payload: Dict[str, Any], webhook_url: Optional[str] = None
) -> bool:
    """
    Sends a rich formatted trade alert card to a Discord channel via Webhook.

    Args:
        alert_payload (Dict[str, Any]): Trade alert data.
        webhook_url (str, optional): Discord Webhook URL. If None, reads DISCORD_WEBHOOK_URL from env.

    Returns:
        bool: True if message sent successfully, False otherwise.
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        logger.warning("No Discord Webhook URL provided. Alert skipped.")
        return False

    is_buy = alert_payload["signal"].upper() == "BUY"
    color = 0x00FF88 if is_buy else 0xFF3366  # Green or Red
    emoji = "🚀 [STRONG BUY]" if is_buy else "⚠️ [SELL / EXIT]"

    feature_lines = "\n".join(
        [
            f"• **{f.get('feature', 'Feature')}**: {f.get('importance', 0):+.3f} contribution"
            for f in alert_payload.get("top_features", [])[:3]
        ]
    )

    fields = [
        {
            "name": "Confidence",
            "value": f"**{alert_payload['confidence'] * 100:.1f}%**",
            "inline": True,
        },
        {
            "name": "Current Price",
            "value": f"${alert_payload['current_price']:.2f}",
            "inline": True,
        },
        {
            "name": "ATR Stop-Loss",
            "value": f"${alert_payload['stop_loss']:.2f}",
            "inline": True,
        },
    ]

    if alert_payload.get("take_profit"):
        fields.append(
            {
                "name": "🎯 Take-Profit Target",
                "value": f"**${alert_payload['take_profit']:.2f}**",
                "inline": True,
            }
        )

    fields.extend(
        [
            {
                "name": "Macro Regime",
                "value": alert_payload["regime"],
                "inline": False,
            },
            {
                "name": "Key SHAP AI Drivers",
                "value": feature_lines or "N/A",
                "inline": False,
            },
        ]
    )

    embed = {
        "title": f"{emoji} {alert_payload['ticker']} Algorithmic Signal",
        "description": f"**Sentilyze AI Signal Engine** has detected a high-conviction trade setup.",
        "color": color,
        "fields": fields,
        "footer": {"text": f"Sentilyze MLOps • {alert_payload['timestamp']}"},
    }

    try:
        response = requests.post(
            url,
            json={"embeds": [embed]},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if response.status_code in [200, 204]:
            logger.info(
                f"Discord alert successfully delivered for {alert_payload['ticker']}"
            )
            return True
        else:
            logger.error(
                f"Discord webhook failed with status {response.status_code}: {response.text}"
            )
    except Exception as e:
        logger.error(f"Error sending Discord alert: {e}")
        return False


def send_discord_digest(
    signals_list: List[Dict[str, Any]], webhook_url: Optional[str] = None
) -> bool:
    """
    Sends a consolidated master market digest card containing all universe signals.
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        logger.warning("No Discord Webhook URL provided. Digest skipped.")
        return False

    buy_signals = [s for s in signals_list if s["signal"] == "BUY"]
    sell_signals = [s for s in signals_list if s["signal"] == "SELL"]

    lines = []
    lines.append("```")
    lines.append(
        f"{'TICKER':<7} {'SIGNAL':<6} {'CONF':<7} {'PRICE':<9} {'TP ($)':<9} {'SL ($)':<9}"
    )
    lines.append("-" * 52)
    for s in signals_list:
        tp_str = f"${s['take_profit']:.2f}" if s.get("take_profit") else "N/A"
        sl_str = f"${s['stop_loss']:.2f}" if s.get("stop_loss") else "N/A"
        lines.append(
            f"{s['ticker']:<7} {s['signal']:<6} {s['confidence'] * 100:.1f}%  ${s['current_price']:<8.2f} {tp_str:<9} {sl_str:<9}"
        )
    lines.append("```")

    embed = {
        "title": "📊 Sentilyze Master Market Briefing",
        "description": f"**Daily Quantitative Universe Scan Complete**\n\n🟢 **BUY Signals**: `{len(buy_signals)}` | 🔴 **SELL/CASH**: `{len(sell_signals)}`\n\n"
        + "\n".join(lines),
        "color": 0x3B82F6,  # Institutional Blue
        "footer": {
            "text": f"Sentilyze Autonomous MLOps Engine • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        },
    }

    try:
        response = requests.post(
            url,
            json={"embeds": [embed]},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if response.status_code in [200, 204]:
            logger.info("Master market briefing digest sent to Discord successfully.")
            return True
        else:
            logger.error(
                f"Discord digest failed with status {response.status_code}: {response.text}"
            )
            return False
    except Exception as e:
        logger.error(f"Error sending Discord digest: {e}")
        return False


def send_telegram_alert(
    alert_payload: Dict[str, Any],
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """
    Sends a formatted Markdown alert to a Telegram chat or channel.

    Args:
        alert_payload (Dict[str, Any]): Trade alert data.
        bot_token (str, optional): Telegram Bot API Token.
        chat_id (str, optional): Telegram Chat ID.

    Returns:
        bool: True if sent successfully, False otherwise.
    """
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat:
        logger.warning("Telegram Bot Token or Chat ID missing. Alert skipped.")
        return False

    is_buy = alert_payload["signal"].upper() == "BUY"
    emoji = "🟢 *STRONG BUY*" if is_buy else "🔴 *STRONG SELL*"

    message = (
        f"{emoji} *{alert_payload['ticker']} Signal*\n\n"
        f"🎯 *Confidence*: `{alert_payload['confidence'] * 100:.1f}%`\n"
        f"💵 *Price*: `${alert_payload['current_price']:.2f}`\n"
        f"🛡️ *Dynamic Stop-Loss*: `${alert_payload['stop_loss']:.2f}`\n"
        f"📊 *Regime*: `{alert_payload['regime']}`\n\n"
        f"🧠 *Key AI Drivers*:\n"
    )

    for f in alert_payload.get("top_features", [])[:3]:
        message += f"• `{f.get('feature')}`: {f.get('importance', 0):+.3f}\n"

    message += f"\n⏰ _{alert_payload['timestamp']}_"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        res = requests.post(
            url,
            json={
                "chat_id": chat,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        if res.status_code == 200:
            logger.info(f"Telegram alert sent for {alert_payload['ticker']}")
            return True
        else:
            logger.error(f"Telegram API failed with code {res.status_code}: {res.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending Telegram alert: {e}")
        return False
