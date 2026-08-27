import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from src.utils import get_logger

logger = get_logger(__name__)


def generate_audio_script(
    signals_list: List[Dict[str, Any]], total_equity: float = 100000.0
) -> str:
    """Generates an institutional Wall Street morning audio briefing script."""
    now_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")
    buys = [s for s in signals_list if s.get("signal") == "BUY"]

    script = (
        f"Good morning. This is your Sentilyze Autonomous AI Market Intelligence Briefing for {now_str}. "
        f"Total portfolio equity stands at {total_equity:,.2f} dollars. "
    )

    if buys:
        top_buys = sorted(buys, key=lambda x: x.get("confidence", 0), reverse=True)[:3]
        buy_names = ", ".join(
            [
                f"{s['ticker']} with {float(s.get('confidence', 0.5))*100:.0f} percent confidence"
                for s in top_buys
            ]
        )
        script += f"Today's top quantitative momentum BUY setups are: {buy_names}. "
        script += "Positions will be monitored automatically by the five minute intraday guardian with dynamic take profit targets. "
    else:
        script += "Markets are currently in neutral or defensive mode. Preserving capital in high yield cash. "

    script += "Have a disciplined and profitable trading session."
    return script


def synthesize_morning_audio(
    signals_list: Optional[List[Dict[str, Any]]] = None,
    total_equity: float = 100000.0,
    output_path: str = "results/morning_briefing.mp3",
) -> Optional[str]:
    """
    Synthesizes the morning briefing audio MP3 file.
    Uses gTTS if available, or returns script.
    """
    signals_list = signals_list or []
    if not signals_list:
        # Load from latest scan
        sig_file = "results/daily_signals_latest.json"
        if os.path.exists(sig_file):
            try:
                with open(sig_file, "r") as f:
                    data = json.load(f)
                    signals_list = data.get("signals", [])
            except Exception as e:
                logger.debug(f"Could not load {sig_file}: {e}")

    script = generate_audio_script(signals_list, total_equity=total_equity)

    try:
        from gtts import gTTS

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tts = gTTS(text=script, lang="en", tld="com", slow=False)
        tts.save(output_path)
        logger.info(f"🎙️ Morning AI audio briefing synthesized to {output_path}")
        return output_path
    except ImportError:
        logger.info("gTTS not installed. Audio script generated in text format.")
    except Exception as e:
        logger.warning(f"Audio synthesis notice: {e}")

    return None
