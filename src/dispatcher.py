import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from src.utils import get_logger, get_market_timestamp

logger = get_logger(__name__)


def send_email_digest(
    signals_summary: List[Dict[str, Any]],
    recipient_email: Optional[str] = None,
) -> bool:
    """
    Dispatches formatted HTML morning market digest email via Gmail SMTP.
    """
    host = os.getenv("EMAIL_HOST", "smtp.gmail.com")
    port = int(os.getenv("EMAIL_PORT", "587"))
    user = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    recipient = recipient_email or os.getenv("EMAIL_RECIPIENT", user)

    if not user or not password or not recipient:
        logger.warning("Email credentials missing. Skipping email digest.")
        return False

    try:
        timestamp_str = get_market_timestamp()
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"📊 Sentilyze Quantitative Alpha Digest • {timestamp_str}"
        msg["From"] = f"Sentilyze AI <{user}>"
        msg["To"] = recipient

        buy_count = sum(1 for s in signals_summary if s.get("signal") == "BUY")
        sell_count = sum(1 for s in signals_summary if s.get("signal") == "SELL")

        # HTML Table Rows
        rows_html = ""
        for s in signals_summary:
            is_buy = s.get("signal") == "BUY"
            color = "#10B981" if is_buy else "#EF4444"
            badge = f"<span style='background: {color}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;'>{s.get('signal')}</span>"
            conf = f"{float(s.get('confidence', 0.5)) * 100:.1f}%"
            price = f"${float(s.get('current_price', 0)):.2f}"
            tp = f"${float(s.get('take_profit', 0)):.2f}"
            sl = f"${float(s.get('stop_loss', 0)):.2f}"
            rows_html += f"""
            <tr style='border-bottom: 1px solid #334155;'>
                <td style='padding: 10px; font-weight: bold; color: #FFFFFF;'>{s.get('ticker')}</td>
                <td style='padding: 10px;'>{badge}</td>
                <td style='padding: 10px; color: #94A3B8;'>{conf}</td>
                <td style='padding: 10px; color: #FFFFFF;'>{price}</td>
                <td style='padding: 10px; color: #10B981;'>{tp}</td>
                <td style='padding: 10px; color: #EF4444;'>{sl}</td>
            </tr>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #0F172A; color: #F8FAFC; padding: 20px;">
            <div style="max-width: 650px; margin: 0 auto; background: #1E293B; border-radius: 12px; padding: 25px; border: 1px solid #334155;">
                <h1 style="color: #00D4AA; margin-top: 0; font-size: 22px;">📊 Sentilyze Quantitative Alpha Digest</h1>
                <p style="color: #94A3B8; font-size: 14px;">Scan across universe complete • {timestamp_str}</p>
                <div style="background: #0F172A; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; font-weight: bold;">
                    🟢 <span style="color: #10B981;">BUY Signals: {buy_count}</span> &nbsp;|&nbsp; 
                    🔴 <span style="color: #EF4444;">SELL / CASH: {sell_count}</span>
                </div>
                <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 13px;">
                    <thead>
                        <tr style="background: #334155; color: #94A3B8;">
                            <th style="padding: 8px;">TICKER</th>
                            <th style="padding: 8px;">SIGNAL</th>
                            <th style="padding: 8px;">CONF</th>
                            <th style="padding: 8px;">PRICE</th>
                            <th style="padding: 8px;">TP ($)</th>
                            <th style="padding: 8px;">SL ($)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
                <p style="margin-top: 25px; font-size: 11px; color: #64748B; text-align: center;">
                    Sentilyze Autonomous MLOps Engine • {timestamp_str}
                </p>
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, recipient, msg.as_string())

        logger.info(f"Successfully sent morning HTML digest to {recipient}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email digest: {e}")
        return False
