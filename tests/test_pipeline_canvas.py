"""
Tests for Interactive Multi-Agent & Pipeline Architecture Canvas (components/pipeline_canvas.py).
Verifies node hierarchy, edge connectivity, and dynamic status color mapping.
"""

import pytest
from components.pipeline_canvas import generate_pipeline_graph_data


def test_generate_pipeline_graph_data_defaults():
    data = generate_pipeline_graph_data("NVDA")
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) >= 10
    assert len(data["edges"]) >= 10

    node_ids = [n["id"] for n in data["nodes"]]
    assert "ohlcv_feed" in node_ids
    assert "news_nlp_feed" in node_ids
    assert "agent_tech" in node_ids
    assert "agent_red_team" in node_ids
    assert "cro_supervisor" in node_ids
    assert "corr_shield" in node_ids
    assert "auto_broker" in node_ids


def test_generate_pipeline_graph_data_with_veto():
    mock_delib = {
        "cro_signoff": {"action_code": "VETO"},
        "agent_testimonies": [
            {"agent_name": "Adversarial Red-Team Specialist", "vote": "VETO"}
        ],
    }
    data = generate_pipeline_graph_data("TSLA", committee_resolution=mock_delib)

    red_team_node = next(n for n in data["nodes"] if n["id"] == "agent_red_team")
    assert red_team_node["color"] == "#FF4B4B"

    cro_node = next(n for n in data["nodes"] if n["id"] == "cro_supervisor")
    assert cro_node["color"] == "#FF4B4B"
