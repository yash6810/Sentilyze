import pytest
from src.ui.theme import THEMES, inject_custom_theme
from src.ui.components import render_workspace_header, render_conviction_gauge


def test_theme_configurations():
    assert len(THEMES) == 3
    assert "🌌 Obsidian Terminal" in THEMES
    assert "💎 Cyberpunk Quant" in THEMES
    assert "🏛️ Goldman Slate" in THEMES

    for name, palette in THEMES.items():
        assert "bg_color" in palette
        assert "accent_color" in palette
        assert "card_bg" in palette
