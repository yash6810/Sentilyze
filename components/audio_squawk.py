"""
Real-Time Audio Trade Squawk Component for Streamlit.

Functions:
- Uses browser-native HTML5 Web Speech API (speechSynthesis) to announce
  high-conviction trade approvals, execution fills, and risk vetoes.
- 100% zero-dependency, works directly inside Streamlit without external audio files.
"""

import streamlit as st
import streamlit.components.v1 as components


def render_audio_squawk_button(
    announcement_text: str,
    button_label: str = "🔊 Play Audio Squawk",
    auto_play: bool = False,
):
    """
    Renders an HTML5 Web Speech API audio squawk generator inside Streamlit.
    """
    safe_text = announcement_text.replace("'", "\\'").replace('"', '\\"')
    auto_trigger_js = f"speakAlert('{safe_text}');" if auto_play else ""

    html_code = f"""
    <div style="padding: 6px 0px;">
        <button id="squawkBtn" onclick="speakAlert('{safe_text}')" style="
            background: linear-gradient(135deg, #00D4AA 0%, #009977 100%);
            color: #0d1117;
            border: none;
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,212,170,0.25);
            transition: all 0.2s ease;
        ">
            {button_label}
        </button>
        <span id="squawkStatus" style="color: #8b949e; font-size: 12px; margin-left: 10px;"></span>
    </div>

    <script>
    function speakAlert(text) {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel(); // Stop prior speech
            var utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.05;
            utterance.pitch = 1.0;
            
            // Prefer an English voice if available
            var voices = window.speechSynthesis.getVoices();
            var usVoice = voices.find(v => v.lang.includes('en-US') || v.lang.includes('en-GB'));
            if (usVoice) {{
                utterance.voice = usVoice;
            }}

            document.getElementById('squawkStatus').innerText = '🎙️ Broadcasting squawk...';
            utterance.onend = function() {{
                document.getElementById('squawkStatus').innerText = '';
            }};
            
            window.speechSynthesis.speak(utterance);
        }} else {{
            document.getElementById('squawkStatus').innerText = '⚠️ Web Speech API not supported in browser.';
        }}
    }}
    {auto_trigger_js}
    </script>
    """
    components.html(html_code, height=50)
