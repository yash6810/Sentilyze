import numpy as np
from src.gnn_supply_chain import SupplyChainGraphNetwork, analyze_supply_chain_spillover


def test_supply_chain_network_initialization():
    gnn = SupplyChainGraphNetwork()
    assert gnn.n_nodes >= 15
    assert gnn.adj_matrix.shape == (gnn.n_nodes, gnn.n_nodes)
    assert np.all(np.diag(gnn.adj_matrix) == 1.0)
    assert gnn.normalized_adj.shape == (gnn.n_nodes, gnn.n_nodes)


def test_graph_convolution_layer():
    gnn = SupplyChainGraphNetwork()
    # Node features: 17 nodes x 4 features
    features = np.random.randn(gnn.n_nodes, 4)
    h_out = gnn.propagate_graph_convolution(features)

    assert h_out.shape == (gnn.n_nodes, 4)
    assert np.all(h_out >= 0.0)  # ReLU non-negativity


def test_simulate_upstream_shock_tsm():
    gnn = SupplyChainGraphNetwork()
    # Shock to TSMC (-5%)
    impacts = gnn.simulate_upstream_shock("TSM", shock_magnitude_pct=-5.0)

    assert len(impacts) > 0
    # NVDA, AAPL, and AMD should be primary direct downstream customers
    impacted_tickers = [x["target"] for x in impacts]
    assert "NVDA" in impacted_tickers
    assert "AAPL" in impacted_tickers
    assert "AMD" in impacted_tickers

    for imp in impacts:
        assert imp["origin"] == "TSM"
        assert imp["predicted_spillover_pct"] < 0.0


def test_analyze_supply_chain_spillover():
    res = analyze_supply_chain_spillover(origin_ticker="NVDA", shock_pct=-6.0)
    assert res["origin_ticker"] == "NVDA"
    assert res["total_impacted_nodes"] > 0
    assert len(res["downstream_impacts"]) > 0
