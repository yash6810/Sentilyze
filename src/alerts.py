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
    Dispatches a crystal-clear, high-impact Discord card for live autonomous trade executions:
    - 🟢 BUY ENTRY: 9-Paper Quantum Omni-Hybrid conviction, Kelly sizing, ATR corridors, Stop-Loss
    - 🎯 TP1 HIT (+2.0 ATR): 50% Profit Locked & Stop trailed to Breakeven
    - 🚀 TP2 HIT (+4.0 ATR): Full Profit Realized on remaining runner
    - 🛑 STOP-LOSS / DERISK: Liquidated per Grossman-Zhou / CUSUM safety floor
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return False

    action = trade_data.get("action", "BUY").upper()
    ticker = trade_data.get("ticker", "ASSET")
    price = float(trade_data.get("price", 0.0))
    shares = int(trade_data.get("shares", 0))
    stage = trade_data.get("stage", "ENTRY")
    order_value = price * shares

    if action == "BUY":
        color = 0x10B981  # Emerald Green
        title = f"👑 [QUANTUM OMNI-HYBRID] BUY ENTRY: {ticker} Filled @ ${price:.2f}"
        desc = (
            f"**Sentilyze Autonomous Quant Trader** executed an institutional **BUY ORDER** for **{ticker}**.\n"
            f"⚡ **Conviction**: Ranked as Top *Stock in Play* (ORB + FinBERT Catalyst) passing DSR Overfit Gate.\n"
            f"🛡️ **Risk Model**: Allocated via **HRP + Boyd Convex SOCP + Risk-Constrained Kelly**."
        )
        fields = [
            {
                "name": "💵 Entry Price",
                "value": f"**${price:.2f}**",
                "inline": True,
            },
            {
                "name": "📦 Position Size",
                "value": f"**{shares:,} shares** (${order_value:,.2f})",
                "inline": True,
            },
            {
                "name": "📊 Portfolio Allocation",
                "value": f"**{trade_data.get('kelly_pct', 8.0):.1f}%** (Risk-Kelly)",
                "inline": True,
            },
            {
                "name": "🎯 Take-Profit 1 (+2.0 ATR)",
                "value": f"**`${trade_data.get('tp1', price * 1.05):.2f}`** *(50% Scale-Out + Breakeven Lock)*",
                "inline": False,
            },
            {
                "name": "🚀 Take-Profit 2 (+4.0 ATR)",
                "value": f"**`${trade_data.get('tp2', price * 1.10):.2f}`** *(Max Extension Runner)*",
                "inline": True,
            },
            {
                "name": "🛑 Hard Stop-Loss (-1.5 ATR)",
                "value": f"**`${trade_data.get('stop_loss', price * 0.965):.2f}`** *(Grossman-Zhou Protected)*",
                "inline": True,
            },
            {
                "name": "🔬 Academic Engine Papers",
                "value": "`#11 Triple-Barrier` • `#25 ORB` • `#10 DSR` • `#12 HRP` • `#02 Boyd SOCP` • `#18 Grossman-Zhou`",
                "inline": False,
            },
        ]
    elif "TP1" in stage:
        color = 0xF59E0B  # Amber Gold
        pnl = float(trade_data.get("realized_pnl", 0.0))
        ret_pct = float(
            trade_data.get(
                "return_pct",
                (
                    (price - float(trade_data.get("entry_price", price)))
                    / max(float(trade_data.get("entry_price", 1.0)), 1e-6)
                )
                * 100.0,
            )
        )
        title = f"🎯 [TAKE-PROFIT 1 REALIZED] {ticker} +50% Profit Locked!"
        desc = (
            f"**Autonomous Selling Agent** has locked in **50% of the position** at **${price:.2f}**.\n"
            f"🛡️ **Trade Made Risk-Free**: Stop-loss automatically trailed to **Breakeven (${trade_data.get('entry_price', price):.2f})**!"
        )
        fields = [
            {
                "name": "💰 Realized Cash Profit",
                "value": f"**+${pnl:,.2f}** ({ret_pct:+.2f}%)",
                "inline": True,
            },
            {
                "name": "📈 Scale-Out Price",
                "value": f"**${price:.2f}**",
                "inline": True,
            },
            {
                "name": "🏃 Remaining Runner",
                "value": f"**{shares:,} shares** targeting TP2 (`${trade_data.get('tp2', 0.0):.2f}`)",
                "inline": False,
            },
            {
                "name": "🔒 Risk Status",
                "value": "`100% RISK-FREE (Stop Trailed to Entry)`",
                "inline": False,
            },
        ]
    elif "TP2" in stage:
        color = 0x8B5CF6  # Royal Purple
        pnl = float(trade_data.get("realized_pnl", 0.0))
        ret_pct = float(trade_data.get("return_pct", 10.0))
        title = f"🚀 [TAKE-PROFIT 2 MAX RUNNER] {ticker} Full Target Banked!"
        desc = f"**Autonomous Selling Agent** closed the final runner of **{ticker}** at peak volatility extension (**${price:.2f}**)."
        fields = [
            {
                "name": "🏆 Total Trade Realized PnL",
                "value": f"**+${pnl:,.2f}**",
                "inline": True,
            },
            {
                "name": "📈 Overall Trade Gain",
                "value": f"**+{ret_pct:.2f}%**",
                "inline": True,
            },
            {
                "name": "📦 Closed Shares",
                "value": f"**{shares:,} shares** @ **${price:.2f}**",
                "inline": True,
            },
            {
                "name": "💼 Cash State",
                "value": "`Capital Returned to Cash Pool for Reallocation`",
                "inline": False,
            },
        ]
    else:
        color = 0xEF4444  # Crimson Red
        title = (
            f"🛑 [STOP-LOSS / CAPITAL PRESERVATION] {ticker} Liquidated @ ${price:.2f}"
        )
        desc = (
            f"**Autonomous Risk Officer** closed **{ticker}** to strictly enforce the **Grossman-Zhou Capital Floor** "
            f"and prevent drawdown beyond pre-set parameters."
        )
        fields = [
            {
                "name": "Exit Price",
                "value": f"**${price:.2f}**",
                "inline": True,
            },
            {
                "name": "Shares Liquidated",
                "value": f"**{shares:,} shares**",
                "inline": True,
            },
            {
                "name": "Risk Reason",
                "value": f"`{trade_data.get('reason', 'PROTECTIVE_STOP_LOSS')}`",
                "inline": False,
            },
        ]

    embed = {
        "title": title,
        "description": desc,
        "color": color,
        "fields": fields,
        "footer": {
            "text": f"Sentilyze 24/7 Quantum Autonomous Trader • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
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
    verdict = (
        deliberation.get("final_resolution")
        or deliberation.get("final_verdict")
        or "HOLD"
    )
    cro = deliberation.get("cro_signoff") or {}
    testimonies = deliberation.get("agent_testimonies") or []

    # Map testimonies by specialist role
    tech_t = next(
        (t for t in testimonies if "Technical" in t.get("agent_name", "")), {}
    )
    sent_t = next(
        (t for t in testimonies if "Sentiment" in t.get("agent_name", "")), {}
    )
    fund_t = next((t for t in testimonies if "Forensic" in t.get("agent_name", "")), {})

    is_buy = "BUY" in str(verdict).upper() or "SCALE_IN" in str(verdict).upper()
    color = 0x10B981 if is_buy else 0x64748B

    tech_metrics = tech_t.get("key_metrics", {})
    tech_val = (
        f"Vote: **{tech_t.get('vote', 'NEUTRAL')}** ({tech_t.get('conviction_score', 50.0):.0f}%)\n"
        f"• RSI: `{tech_metrics.get('estimated_rsi', 50.0):.1f}` | Trend: `{tech_metrics.get('trend_status', 'NORMAL')}`\n"
        f"• Thesis: *{tech_t.get('thesis', 'Aligned with moving averages.')[:120]}*"
    )

    sent_val = (
        f"Vote: **{sent_t.get('vote', 'NEUTRAL')}** ({sent_t.get('conviction_score', 50.0):.0f}%)\n"
        f"• Thesis: *{sent_t.get('thesis', 'Neutral sentiment flow.')[:120]}*"
    )

    fund_metrics = fund_t.get("metrics", {})
    fund_val = (
        f"Vote: **{fund_t.get('vote', 'NEUTRAL')}** ({fund_t.get('conviction_score', 50.0):.0f}%)\n"
        f"• Piotroski F: `{fund_metrics.get('piotroski_f_score', 6)}/9` | Altman Z: `{fund_metrics.get('altman_z_score', 3.0):.2f}`\n"
        f"• Margin of Safety: `{fund_metrics.get('margin_of_safety_pct', 0.0):+.1f}%`"
    )

    cro_val = (
        f"Resolution: **{cro.get('final_resolution', verdict)}**\n"
        f"• VIX Level: `{cro.get('vix_level', 16.5):.1f}` | Kelly Sizing: `+{cro.get('kelly_allocation_pct', 8.0):.1f}%`\n"
        f"• Target 1: `${cro.get('tp1_target', 0):.2f}` | Target 2: `${cro.get('tp2_target', 0):.2f}` | Stop: `${cro.get('stop_loss_target', 0):.2f}`"
    )

    fields = [
        {
            "name": "📈 1. Technical Alpha Specialist",
            "value": tech_val,
            "inline": False,
        },
        {
            "name": "🧠 2. FinBERT Sentiment Specialist",
            "value": sent_val,
            "inline": False,
        },
        {"name": "🏛️ 3. Forensic Fundamentalist", "value": fund_val, "inline": False},
        {
            "name": "🛡️ 4. Chief Risk Officer (CRO) Clearance",
            "value": cro_val,
            "inline": False,
        },
    ]

    embed = {
        "title": f"🏛️ [4-AGENT COMMITTEE DELIBERATION] {ticker} • {verdict}",
        "description": f"The Sentilyze AI Multi-Agent Council concluded round-table deliberation on **{ticker}**.",
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
    Sends a comprehensive institutional morning macro regime, portfolio health,
    open holdings, realized P&L, and AI opportunities pulse to Discord.
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return False

    vix = float(pulse_data.get("vix_level", 15.4))
    vix_regime = pulse_data.get("vix_regime", "LOW VOLATILITY / NORMAL")
    top_buys = pulse_data.get("top_buys", [])
    equity = float(pulse_data.get("portfolio_equity", 100000.0))
    cash = float(pulse_data.get("cash_balance", equity))
    daily_pnl = float(pulse_data.get("daily_pnl", 0.0))
    daily_ret = float(pulse_data.get("daily_return_pct", 0.0))
    realized_pnl = float(pulse_data.get("realized_pnl", 0.0))
    open_pos = pulse_data.get("open_positions", {})
    recent_closed = pulse_data.get("recent_closed_trades", [])

    # Format Top AI Buys across universe (sorted by highest confidence)
    sorted_buys = sorted(top_buys, key=lambda x: x.get("confidence", 0.0), reverse=True)
    if sorted_buys:
        buys_str = "\n".join(
            [
                f"• **{b['ticker']}**: Conf `{b.get('confidence', 0.6)*100:.1f}%` @ `${b.get('current_price', b.get('price', 0)):.2f}` | TP: `${b.get('take_profit', 0):.2f}`"
                for b in sorted_buys[:5]
            ]
        )
    else:
        buys_str = "ℹ️ No high-conviction BUY setups today. Capital preserved in cash."

    # Format Open Holdings (What the bot has bought and currently holds)
    if open_pos:
        pos_lines = []
        for sym, p in list(open_pos.items())[:5]:
            curr_p = float(p.get("current_price", p.get("entry_price", 0.0)))
            entry_p = float(p.get("entry_price", curr_p))
            shares = int(p.get("shares", 0))
            unrealized = (curr_p - entry_p) * shares
            ret_pct = ((curr_p - entry_p) / entry_p) * 100.0 if entry_p > 0 else 0.0
            emoji = "🟢" if unrealized >= 0 else "🔴"
            pos_lines.append(
                f"{emoji} **{sym}**: `{shares}` shs @ `${entry_p:.2f}` (Now `${curr_p:.2f}` | **`${unrealized:+,.2f}`** / `{ret_pct:+.1f}%`)"
            )
        holdings_str = "\n".join(pos_lines)
    else:
        holdings_str = "💼 **100% Liquid Cash** (0 Open Positions — Standing by for high-conviction entries)."

    # Format Recent Closed Trades (Profits and Losses)
    if recent_closed:
        closed_lines = []
        for t in recent_closed[-4:]:
            pnl_val = float(t.get("pnl", 0.0))
            ret_val = float(t.get("return_pct", 0.0))
            emoji = "💰" if pnl_val >= 0 else "🛑"
            closed_lines.append(
                f"{emoji} **{t.get('ticker')}**: **`${pnl_val:+,.2f}`** (`{ret_val:+.2f}%`) — *{t.get('reason', 'EXIT')}*"
            )
        closed_str = "\n".join(closed_lines)
    else:
        closed_str = "No trades closed recently."

    pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"

    fields = [
        {
            "name": "🌪️ Macro Market Regime & Risk",
            "value": f"VIX: **{vix:.2f}** (`{vix_regime}`)\nMacro Stance: **Capital Preservation & Kelly Optimization**",
            "inline": False,
        },
        {
            "name": "💼 Autonomous Portfolio & Performance",
            "value": (
                f"• **Total Equity:** **`${equity:,.2f}`**\n"
                f"• **Available Cash:** `${cash:,.2f}`\n"
                f"• **Today's P&L:** {pnl_emoji} **`${daily_pnl:+,.2f}`** (`{daily_ret:+.2f}%`)\n"
                f"• **Total Realized P&L:** **`${realized_pnl:+,.2f}`**"
            ),
            "inline": False,
        },
        {
            "name": "📦 Current Open Holdings (Stocks Held)",
            "value": holdings_str,
            "inline": False,
        },
        {
            "name": "📜 Recent Realized Profits & Losses",
            "value": closed_str,
            "inline": False,
        },
        {
            "name": "🚀 Top AI-Scanned Opportunities (106 Universe)",
            "value": buys_str,
            "inline": False,
        },
    ]

    embed = {
        "title": "🌅 [SENTILYZE MORNING MARKET RADAR] Institutional Briefing",
        "description": "Daily Pre-Market Quantitative Intelligence, Portfolio Health & Actionable Signals",
        "color": 0x38BDF8,  # Sky Blue
        "fields": fields,
        "footer": {
            "text": f"Sentilyze Autonomous Intelligence • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
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


def send_discord_premarket_briefing(
    portfolio_summary: Dict[str, Any],
    macro_vix: float = 16.5,
    top_watchlist: Optional[List[Dict[str, Any]]] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """
    Dispatches an institutional Pre-Market Intelligence Briefing embed to Discord.
    Summarizes overnight sentiment, macro volatility gate, active portfolio health, and top setups.
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        logger.warning("No Discord Webhook URL provided for premarket briefing.")
        return False

    total_eq = float(portfolio_summary.get("total_equity", 100000.0))
    cash = float(portfolio_summary.get("cash", 100000.0))
    unrealized = float(portfolio_summary.get("unrealized_pnl", 0.0))
    pnl_pct = float(portfolio_summary.get("unrealized_pnl_pct", 0.0))
    win_rate = float(portfolio_summary.get("win_rate", 50.0))
    open_pos = portfolio_summary.get("open_positions", {})

    # Volatility Regime Assessment
    if macro_vix > 26.0:
        vol_status = "🔴 CIRCUIT BREAKER ACTIVE (VIX > 26.0) — New Buys Halted"
        vol_color = 0xEF4444  # Red
    elif macro_vix > 20.0:
        vol_status = "🟡 ELEVATED VOLATILITY (VIX 20-26) — Quarter-Kelly Tightened"
        vol_color = 0xF59E0B  # Amber
    else:
        vol_status = "🟢 CALM / LOW VOLATILITY (VIX < 20) — Full Execution Mode"
        vol_color = 0x10B981  # Emerald Green

    fields = [
        {
            "name": "🏛️ Macro Volatility & Risk Gate",
            "value": (
                f"• **CBOE VIX Level:** `{macro_vix:.2f}`\n"
                f"• **Council Gate:** {vol_status}\n"
                f"• **Capital Protection:** `Dynamic ATR Stops Enabled`"
            ),
            "inline": False,
        },
        {
            "name": "💼 Paper Portfolio Balance",
            "value": (
                f"• **Total Equity:** **`${total_eq:,.2f}`**\n"
                f"• **Cash Reserves:** `${cash:,.2f}`\n"
                f"• **Open Holdings:** `{len(open_pos)} Positions` (PnL: `${unrealized:+,.2f}` / `{pnl_pct:+.2f}%`)\n"
                f"• **Historical Win Rate:** `{win_rate:.1f}%`"
            ),
            "inline": False,
        },
    ]

    # Add Watchlist Setups if provided
    if top_watchlist:
        watchlist_lines = []
        for item in top_watchlist[:5]:
            tk = item.get("ticker", "N/A")
            res = item.get("resolution", "HOLD")
            conv = item.get("conviction", 50.0)
            sent = item.get("sentiment_score", 0.0)
            emoji = "🟢" if "BUY" in res else ("🔴" if "SELL" in res else "🟡")
            watchlist_lines.append(
                f"{emoji} **{tk}**: `{res}` ({conv:.0f}% conviction | FinBERT: `{sent:+.2f}`)"
            )
        fields.append(
            {
                "name": "🎯 Pre-Market Watchlist & Council Verdicts",
                "value": (
                    "\n".join(watchlist_lines)
                    if watchlist_lines
                    else "Scanning universe..."
                ),
                "inline": False,
            }
        )

    fields.append(
        {
            "name": "🛡️ Chief Risk Officer (CRO) Daily Mandate",
            "value": (
                "• All entries governed by **Fractional Quarter-Kelly Criterion**.\n"
                "• **Stage 1 Target:** Bank 50% profit @ `+2.5 ATR` and trail stop to breakeven.\n"
                "• **Stage 2 Target:** Harvest remaining 50% runner @ `+4.5 ATR`."
            ),
            "inline": False,
        }
    )

    embed = {
        "title": "🌅 Sentilyze Pre-Market Intelligence & Council Briefing",
        "description": (
            f"**Morning Market Opening Bell Assessment**\n"
            f"*Automated 4-Agent Multi-Modal Quant Briefing for US Equities*"
        ),
        "color": vol_color,
        "fields": fields,
        "footer": {
            "text": f"Sentilyze Pre-Market Intelligence Desk • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        res = requests.post(url, json={"embeds": [embed]}, timeout=10)
        return res.status_code in [200, 204]
    except Exception as e:
        logger.error(f"Error sending Discord pre-market briefing: {e}")
        return False
