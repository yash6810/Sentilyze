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
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        logger.warning("No Discord Webhook URL provided. Alert skipped.")
        return False

    is_buy = alert_payload["signal"].upper() == "BUY"
    color = 0x00FF88 if is_buy else 0xFF3366  # Green or Red
    emoji = "🚀 [STRONG BUY]" if is_buy else "⚠️ [SELL / EXIT]"

    feat_list = []
    for f in alert_payload.get("top_features", [])[:3]:
        fname = f.get("feature", "Feature")
        imp = f.get("importance", 0)
        imp_str = f"`{imp:+.3f}`" if isinstance(imp, (int, float)) else f"`{imp}`"
        feat_list.append(f"• **{fname}**: {imp_str}")
    feature_lines = "\n".join(feat_list)

    fields = [
        {
            "name": "🎯 Signal & Confidence",
            "value": f"**{alert_payload['signal']}** ({alert_payload['confidence'] * 100:.1f}%)",
            "inline": True,
        },
        {
            "name": "💵 Market Price",
            "value": f"**${alert_payload['current_price']:.2f}**",
            "inline": True,
        },
        {
            "name": "🛡️ Stop-Loss (Risk Floor)",
            "value": f"`${alert_payload['stop_loss']:.2f}`",
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
                "name": "📊 Macro Regime",
                "value": f"`{alert_payload['regime']}`",
                "inline": False,
            },
            {
                "name": "🧠 Key AI Feature Drivers (SHAP)",
                "value": feature_lines or "Standard Walk-Forward Features",
                "inline": False,
            },
        ]
    )

    embed = {
        "title": f"{emoji} {alert_payload['ticker']} Algorithmic Conviction Signal",
        "description": f"**Sentilyze AI Quantitative Engine** has detected an institutional alpha setup for **{alert_payload['ticker']}**.",
        "color": color,
        "fields": fields,
        "footer": {
            "text": f"Sentilyze Institutional MLOps Wire • {alert_payload.get('timestamp', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))}"
        },
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


def send_discord_execution_alert(
    trade_data: Dict[str, Any], webhook_url: Optional[str] = None
) -> bool:
    """
    Dispatches a high-impact Discord card for live autonomous trade lifecycle events:
    - BUY Entry with Kelly sizing
    - TP1 Hit (+2.5 ATR): 50% Profit Locked & Stop trailed to Breakeven
    - TP2 Hit (+4.5 ATR): Full Profit Realized
    - Stop-Loss / Emergency Exit
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return False

    action = trade_data.get("action", "BUY").upper()
    ticker = trade_data.get("ticker", "ASSET")
    price = trade_data.get("price", 0.0)
    shares = trade_data.get("shares", 0)
    stage = trade_data.get("stage", "ENTRY")

    if action == "BUY":
        color = 0x10B981  # Emerald Green
        title = f"🤖 [AUTONOMOUS BUY EXECUTION] {ticker} Filled @ ${price:.2f}"
        desc = (
            f"**Autonomous Buying Agent** has entered **{shares:,} shares** of **{ticker}** "
            f"via Kelly Criterion Allocation (`{trade_data.get('kelly_pct', 8.0):.1f}%` of portfolio)."
        )
        fields = [
            {
                "name": "💵 Entry Price",
                "value": f"**${price:.2f}**",
                "inline": True,
            },
            {
                "name": "📦 Order Size",
                "value": f"**{shares:,} shares** (${price * shares:,.2f})",
                "inline": True,
            },
            {
                "name": "🎯 Take-Profit 1 (+2.5 ATR)",
                "value": f"`${trade_data.get('tp1', price * 1.06):.2f}` *(50% Profit Lock)*",
                "inline": False,
            },
            {
                "name": "🚀 Take-Profit 2 (+4.5 ATR)",
                "value": f"`${trade_data.get('tp2', price * 1.12):.2f}` *(Remaining Runner)*",
                "inline": True,
            },
            {
                "name": "🛡️ Protective Stop-Loss",
                "value": f"`${trade_data.get('stop_loss', price * 0.965):.2f}`",
                "inline": True,
            },
        ]
    elif "TP1" in stage:
        color = 0xF59E0B  # Amber Gold
        title = f"🎯 [TAKE-PROFIT 1 HIT] {ticker} +50% Profit Banked!"
        desc = (
            f"**Autonomous Selling Agent** has locked in **+50% of the position** at **${price:.2f}**.\n"
            f"🛡️ **Risk-Free Update**: Stop-Loss automatically trailed to **Breakeven (${trade_data.get('entry_price', price):.2f})**! Trade is now completely risk-free."
        )
        fields = [
            {
                "name": "💰 Realized Cash Profit",
                "value": f"**+${trade_data.get('realized_pnl', 0.0):,.2f}**",
                "inline": True,
            },
            {
                "name": "📈 Exit Price",
                "value": f"${price:.2f}",
                "inline": True,
            },
            {
                "name": "🏃 Remaining Runner",
                "value": f"**{shares:,} shares** targeting TP2 (`${trade_data.get('tp2', 0.0):.2f}`)",
                "inline": False,
            },
        ]
    elif "TP2" in stage:
        color = 0x8B5CF6  # Royal Purple
        title = f"🚀 [TAKE-PROFIT 2 MAX RUNNER] {ticker} Full Profit Realized!"
        desc = f"**Autonomous Selling Agent** closed the remaining runner at peak market extension (**${price:.2f}**)."
        fields = [
            {
                "name": "🏆 Total Trade PnL",
                "value": f"**+${trade_data.get('realized_pnl', 0.0):,.2f}**",
                "inline": True,
            },
            {
                "name": "📊 Return on Trade",
                "value": f"**+{trade_data.get('return_pct', 12.0):.2f}%**",
                "inline": True,
            },
        ]
    else:
        color = 0xEF4444  # Crimson Red
        title = f"🛡️ [STOP-LOSS / EXIT EXECUTED] {ticker} Liquidated @ ${price:.2f}"
        desc = f"Position liquidated according to risk-first capital preservation protocols."
        fields = [
            {
                "name": "Exit Price",
                "value": f"${price:.2f}",
                "inline": True,
            },
            {
                "name": "Shares Closed",
                "value": f"{shares:,}",
                "inline": True,
            },
        ]

    embed = {
        "title": title,
        "description": desc,
        "color": color,
        "fields": fields,
        "footer": {
            "text": f"Sentilyze 24/7 Autonomous Trader • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        },
    }

    try:
        res = requests.post(
            url,
            json={"embeds": [embed]},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return res.status_code in [200, 204]
    except Exception as e:
        logger.error(f"Error sending Discord execution alert: {e}")
        return False


def send_discord_committee_alert(
    deliberation: Dict[str, Any], webhook_url: Optional[str] = None
) -> bool:
    """
    Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord.
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return False

    ticker = deliberation.get("ticker", "ASSET")
    verdict = deliberation.get("final_verdict", "HOLD")
    cro = deliberation.get("cro_signoff", {})
    votes = deliberation.get("committee_votes", {})

    is_buy = "BUY" in verdict.upper()
    color = 0x3B82F6 if is_buy else 0x64748B

    tech_v = votes.get("Technical Specialist", {})
    sent_v = votes.get("Sentiment & Alternative Data Specialist", {})
    fund_v = votes.get("Forensic Accounting & Valuation Specialist", {})

    fields = [
        {
            "name": "📈 1. Technical Specialist",
            "value": f"Vote: **{tech_v.get('vote', 'HOLD')}** | RSI: `{tech_v.get('rsi_14', 50.0):.1f}` | 200 SMA: `{tech_v.get('trend_200sma', 'NEUTRAL')}`",
            "inline": False,
        },
        {
            "name": "🧠 2. NLP Sentiment Specialist",
            "value": f"Vote: **{sent_v.get('vote', 'HOLD')}** | Optimism: `{sent_v.get('sentiment_score', 60.0):.1f}/100` | Smart Money: `{sent_v.get('smart_money_score', 50.0):.1f}`",
            "inline": False,
        },
        {
            "name": "🏛️ 3. Forensic DCF Specialist",
            "value": f"Vote: **{fund_v.get('vote', 'HOLD')}** | Fair Value: `${fund_v.get('dcf_fair_value', 0.0):.2f}` | Margin of Safety: `{fund_v.get('margin_of_safety_pct', 0.0):+.1f}%`",
            "inline": False,
        },
        {
            "name": "🛡️ 4. Chief Risk Officer (CRO) Clearance",
            "value": f"Status: **{cro.get('status', 'APPROVED')}** | VIX Gate: `{cro.get('macro_vix_level', 15.0):.1f}` | Kelly Sizing: `+{cro.get('approved_kelly_pct', 8.0):.1f}%`",
            "inline": False,
        },
    ]

    embed = {
        "title": f"🏛️ [4-AGENT COMMITTEE DEBATE] {ticker} Verdict: {verdict}",
        "description": f"The Sentilyze AI Multi-Agent Committee convened to deliberate on **{ticker}**.",
        "color": color,
        "fields": fields,
        "footer": {
            "text": f"Sentilyze Multi-Agent Committee • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        },
    }

    try:
        res = requests.post(
            url,
            json={"embeds": [embed]},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return res.status_code in [200, 204]
    except Exception as e:
        logger.error(f"Error sending Discord committee alert: {e}")
        return False


def send_discord_social_spike_alert(
    social_data: Dict[str, Any], webhook_url: Optional[str] = None
) -> bool:
    """
    Dispatches real-time Reddit r/wallstreetbets & Stocktwits hype spike alerts.
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return False

    ticker = social_data.get("ticker", "ASSET")
    vel = social_data.get("mention_velocity_ratio", 1.0)
    bull_pct = social_data.get("bullish_sentiment_pct", 50.0)
    regime = social_data.get("regime", "SURGE")

    embed = {
        "title": f"🔥 [RETAIL SOCIAL BUZZ SURGE] {ticker} @ {vel:.1f}x Normal Velocity!",
        "description": f"**Pillar 2 Alternative Social Tracker** detected viral retail volume acceleration across Reddit (r/wallstreetbets) and Stocktwits.",
        "color": 0xFF6B00,  # Neon Orange
        "fields": [
            {
                "name": "⚡ 24h Velocity Ratio",
                "value": f"**{vel:.2f}x** vs 7-Day Baseline",
                "inline": True,
            },
            {
                "name": "🐂 Bullish Sentiment",
                "value": f"**{bull_pct:.1f}%** Bullish Posts",
                "inline": True,
            },
            {
                "name": "🏷️ Retail Regime",
                "value": f"`{regime}`",
                "inline": False,
            },
        ],
        "footer": {
            "text": f"Sentilyze Social Alternative Data Wire • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        },
    }

    try:
        res = requests.post(
            url,
            json={"embeds": [embed]},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return res.status_code in [200, 204]
    except Exception as e:
        logger.error(f"Error sending Discord social spike alert: {e}")
        return False


def send_discord_market_pulse(
    pulse_data: Dict[str, Any], webhook_url: Optional[str] = None
) -> bool:
    """
    Sends a consolidated morning macro regime and portfolio health pulse to Discord.
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return False

    vix = pulse_data.get("vix_level", 15.4)
    vix_regime = pulse_data.get("vix_regime", "LOW VOLATILITY / NORMAL")
    top_buys = pulse_data.get("top_buys", [])
    equity = pulse_data.get("portfolio_equity", 100000.0)
    open_pos = pulse_data.get("open_positions_count", 0)

    buys_str = (
        "\n".join(
            [
                f"• **{b['ticker']}**: Conf `{b.get('confidence', 0.6)*100:.1f}%` (${b.get('price', 0):.2f})"
                for b in top_buys[:4]
            ]
        )
        or "No strong BUY setups today."
    )

    embed = {
        "title": "🌅 [SENTILYZE MORNING MARKET RADAR] Institutional Briefing",
        "description": f"**Pre-Market Quantitative Pulse & Macro Risk Status**",
        "color": 0x38BDF8,  # Sky Blue
        "fields": [
            {
                "name": "🌪️ Macro VIX Regime",
                "value": f"VIX: **{vix:.2f}** (`{vix_regime}`)",
                "inline": True,
            },
            {
                "name": "💼 Autonomous Portfolio",
                "value": f"Equity: **${equity:,.2f}** | Positions: `{open_pos}`",
                "inline": True,
            },
            {
                "name": "🚀 Top Institutional AI Setups",
                "value": buys_str,
                "inline": False,
            },
        ],
        "footer": {
            "text": f"Sentilyze Pre-Market Intelligence • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        },
    }

    try:
        res = requests.post(
            url,
            json={"embeds": [embed]},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return res.status_code in [200, 204]
    except Exception as e:
        logger.error(f"Error sending Discord market pulse: {e}")
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
        return response.status_code in [200, 204]
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
    """
    token = (bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    chat = (chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()

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
        fname = f.get("feature", "Feature")
        imp = f.get("importance", 0)
        imp_str = f"`{imp:+.3f}`" if isinstance(imp, (int, float)) else f"`{imp}`"
        message += f"• `{fname}`: {imp_str}\n"

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


def send_discord_holdings_heartbeat(
    portfolio_state: Dict[str, Any],
    webhook_url: Optional[str] = None,
) -> bool:
    """
    Sends a sleek, institutional Discord embed with live prices, PnL, and distance to targets for all active holdings.
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return False

    open_pos = portfolio_state.get("open_positions", {})
    if not open_pos:
        return False

    total_eq = float(portfolio_state.get("total_equity", 100000.0))
    unrealized_pnl = float(portfolio_state.get("unrealized_pnl", 0.0))
    cash = float(portfolio_state.get("cash", 0.0))
    pnl_pct = (unrealized_pnl / total_eq) * 100.0 if total_eq > 0 else 0.0

    color = 0x10B981 if unrealized_pnl >= 0 else 0xEF4444

    fields = []
    for ticker, pos in open_pos.items():
        curr_p = float(pos.get("current_price", 0.0))
        entry_p = float(pos.get("entry_price", curr_p))
        shares = int(pos.get("shares", 0))
        tp1 = float(pos.get("tp1_target", curr_p * 1.06))
        sl = float(pos.get("sl_target", curr_p * 0.95))
        pos_pnl = (curr_p - entry_p) * shares
        pos_ret = ((curr_p - entry_p) / entry_p) * 100.0 if entry_p > 0 else 0.0

        dist_tp1 = ((tp1 - curr_p) / curr_p) * 100.0 if curr_p > 0 else 0.0
        dist_sl = ((curr_p - sl) / curr_p) * 100.0 if curr_p > 0 else 0.0

        status_tag = (
            "🛡️ 50% Banked (Risk-Free)" if pos.get("scaled_out") else "⚡ 100% Active"
        )
        emoji = "🟢" if pos_pnl >= 0 else "🔴"

        fields.append(
            {
                "name": f"{emoji} {ticker} • ${curr_p:.2f} ({pos_ret:+.2f}%)",
                "value": (
                    f"• **Shares:** `{shares:,}` | **Entry Basis:** `${entry_p:.2f}`\n"
                    f"• **Unrealized PnL:** **`${pos_pnl:+,.2f}`**\n"
                    f"• **Target 1 (+2.5 ATR):** `${tp1:.2f}` (`{dist_tp1:+.1f}%` away)\n"
                    f"• **Stop Loss Floor:** `${sl:.2f}` (`{dist_sl:.1f}%` buffer)\n"
                    f"• **State:** `{status_tag}`"
                ),
                "inline": False,
            }
        )

    embed = {
        "title": "📈 Sentilyze Intraday Live Holdings Price Update",
        "description": (
            f"**Portfolio Equity:** `${total_eq:,.2f}`\n"
            f"**Unrealized PnL:** **`${unrealized_pnl:+,.2f}` (`{pnl_pct:+.2f}%`)**\n"
            f"**Cash Balance:** `${cash:,.2f}`\n"
            f"**Active Positions:** `{len(open_pos)} Assets`"
        ),
        "color": color,
        "fields": fields,
        "footer": {
            "text": f"Sentilyze Sub-Second Price Guardian • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        res = requests.post(url, json={"embeds": [embed]}, timeout=10)
        return res.status_code in [200, 204]
    except Exception as e:
        logger.error(f"Error sending Discord holdings heartbeat: {e}")
        return False
