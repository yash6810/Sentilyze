import pytest
from unittest.mock import patch, MagicMock
from src.ui.ws_biotech_radar import render_biotech_radar_workspace
from src.ui.ws_sec_filings import render_sec_filings_workspace
from src.ui.ws_arxiv_research import render_arxiv_research_workspace


def mock_columns_side_effect(spec):
    count = len(spec) if isinstance(spec, list) else int(spec)
    return [MagicMock() for _ in range(count)]


@patch("streamlit.selectbox", return_value="LLY")
@patch("streamlit.button", return_value=False)
@patch("streamlit.columns", side_effect=mock_columns_side_effect)
@patch("streamlit.tabs", return_value=[MagicMock(), MagicMock()])
@patch("streamlit.spinner")
def test_render_biotech_radar_workspace(
    mock_spin, mock_tabs, mock_cols, mock_btn, mock_select
):
    render_biotech_radar_workspace("LLY")


@patch("streamlit.slider", return_value=30)
@patch("streamlit.button", return_value=False)
@patch("streamlit.columns", side_effect=mock_columns_side_effect)
@patch("streamlit.spinner")
def test_render_sec_filings_workspace(mock_spin, mock_cols, mock_btn, mock_slider):
    render_sec_filings_workspace("NVDA")


@patch("streamlit.selectbox", return_value="Market Microstructure & Order Flow")
@patch("streamlit.button", return_value=False)
@patch("streamlit.columns", side_effect=mock_columns_side_effect)
@patch("streamlit.spinner")
def test_render_arxiv_research_workspace(mock_spin, mock_cols, mock_btn, mock_select):
    render_arxiv_research_workspace()
