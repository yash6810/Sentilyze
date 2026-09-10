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


def test_ws_alpha_dag_module_import():
    from src.ui.ws_alpha_dag import (
        load_cached_tournament_results,
        render_alpha_dag_workspace,
    )

    results = load_cached_tournament_results()
    assert results is not None
    assert "gen1" in results
    assert "gen2" in results
    assert "gen3" in results
    assert callable(render_alpha_dag_workspace)
