"""
Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for Sentilyze.
Pillar 1 Advanced AI Module:
- Models interconnected tech supply chain topology across universe stocks.
- Implements Spectral Graph Convolution (GCN) message passing over revenue dependency matrices.
- Simulates upstream supply shocks (e.g., TSMC fab disruptions) and propagates impact downstream to Big Tech hyperscalers.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)

# Standard Supply Chain Revenue Dependency Graph (Directed edges: Supplier -> Customer)
SUPPLY_CHAIN_EDGES = [
    (
        "TSM",
        "NVDA",
        0.85,
        "Foundry: Advanced 3nm/4nm CoWoS packaging for Blackwell/Hopper GPUs",
    ),
    (
        "TSM",
        "AAPL",
        0.90,
        "Foundry: Sole manufacturer of A-series and M-series silicon",
    ),
    (
        "TSM",
        "AMD",
        0.80,
        "Foundry: Sole manufacturer of EPYC and Instinct MI300 accelerators",
    ),
    (
        "TSM",
        "AVGO",
        0.75,
        "Foundry: Custom ASIC networking and TPU manufacturing partner",
    ),
    (
        "NVDA",
        "MSFT",
        0.75,
        "Hardware: Primary AI accelerator supplier for Azure Cloud data centers",
    ),
    (
        "NVDA",
        "META",
        0.80,
        "Hardware: Largest GPU customer for Llama 3 training clusters",
    ),
    (
        "NVDA",
        "AMZN",
        0.70,
        "Hardware: Major GPU provider for AWS Bedrock & EC2 instances",
    ),
    ("NVDA", "GOOGL", 0.65, "Hardware: GPU supplier alongside Google internal TPUs"),
    ("AVGO", "GOOGL", 0.70, "Silicon Design: Co-developer of Google custom TPU chips"),
    (
        "AVGO",
        "AAPL",
        0.60,
        "RF Components: 5G front-end modules and custom wireless chips",
    ),
    (
        "MSFT",
        "PLTR",
        0.55,
        "Cloud Infrastructure: Strategic Azure government cloud integration",
    ),
    (
        "AMZN",
        "NFLX",
        0.65,
        "Cloud Provider: AWS hosts Netflix global streaming infrastructure",
    ),
    (
        "AAPL",
        "COST",
        0.40,
        "Retail Distribution: Mega-volume consumer hardware retail channel",
    ),
]


class SupplyChainGraphNetwork:
    """
    Graph Neural Network implementing 2-Hop Graph Convolutional Message Passing
    over the technology ecosystem graph.
    """

    def __init__(self, tickers: Optional[List[str]] = None):
        self.tickers = tickers or [
            "TSM",
            "NVDA",
            "AMD",
            "AVGO",
            "AAPL",
            "MSFT",
            "GOOGL",
            "META",
            "AMZN",
            "TSLA",
            "PLTR",
            "NFLX",
            "COST",
            "LLY",
            "JPM",
            "QQQ",
            "SPY",
        ]
        self.n_nodes = len(self.tickers)
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}
        self.adj_matrix, self.edge_descriptions = self._build_adjacency_matrix()
        self.normalized_adj = self._compute_laplacian_normalization(self.adj_matrix)

    def _build_adjacency_matrix(self) -> Tuple[np.ndarray, Dict[Tuple[str, str], str]]:
        A = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        descriptions = {}

        for src, dst, weight, desc in SUPPLY_CHAIN_EDGES:
            if src in self.ticker_to_idx and dst in self.ticker_to_idx:
                i, j = self.ticker_to_idx[src], self.ticker_to_idx[dst]
                A[i, j] = weight
                descriptions[(src, dst)] = desc

        # Add self-loops (identity connections for GCN stability)
        np.fill_diagonal(A, 1.0)
        return A, descriptions

    def _compute_laplacian_normalization(self, A: np.ndarray) -> np.ndarray:
        """Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)."""
        degrees = np.sum(A, axis=1)
        degrees_inv_sqrt = np.zeros_like(degrees, dtype=float)
        nonzero = degrees > 0
        degrees_inv_sqrt[nonzero] = degrees[nonzero] ** -0.5
        D_inv = np.diag(degrees_inv_sqrt)
        return D_inv @ A @ D_inv

    def propagate_graph_convolution(
        self, node_features: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Executes a Graph Convolutional Network (GCN) layer:
        H_new = ReLU(A_hat * H * W)

        Args:
            node_features: Node feature matrix of shape (N, F)
            weights: Learnable projection weights (F, F)

        Returns:
            Updated node representations (N, F)
        """
        if weights is None:
            weights = np.eye(node_features.shape[1])

        # Message passing step
        aggregated = self.normalized_adj @ node_features
        # Linear transform + ReLU activation
        transformed = aggregated @ weights
        h_new = np.maximum(0.0, transformed)
        return h_new

    def simulate_upstream_shock(
        self, source_ticker: str, shock_magnitude_pct: float = -5.0
    ) -> List[Dict[str, Any]]:
        """
        Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab bottleneck)
        and propagates predicted revenue and price shocks through downstream customers.

        Args:
            source_ticker: Symbol where the shock originates (e.g. TSM or NVDA)
            shock_magnitude_pct: Shock percentage (e.g. -5.0%)

        Returns:
            List of downstream companies ranked by shock transmission impact.
        """
        if source_ticker not in self.ticker_to_idx:
            return []

        src_idx = self.ticker_to_idx[source_ticker]
        # Initial shock vector
        shock_v = np.zeros(self.n_nodes)
        shock_v[src_idx] = shock_magnitude_pct

        # 1-Hop propagation
        hop1_spillover = self.adj_matrix[src_idx, :] * (shock_magnitude_pct * 0.75)
        # 2-Hop propagation (e.g. TSM -> NVDA -> MSFT)
        hop2_matrix = self.adj_matrix @ self.adj_matrix
        hop2_spillover = hop2_matrix[src_idx, :] * (shock_magnitude_pct * 0.40)

        results = []
        for i, target_ticker in enumerate(self.tickers):
            if target_ticker == source_ticker:
                continue

            impact_pct = float(hop1_spillover[i] + hop2_spillover[i])
            if abs(impact_pct) < 0.05:
                continue

            # Determine dependency context
            edge_desc = self.edge_descriptions.get(
                (source_ticker, target_ticker),
                "2-Hop indirect supply spillover via tech ecosystem",
            )

            results.append(
                {
                    "origin": source_ticker,
                    "target": target_ticker,
                    "predicted_spillover_pct": round(impact_pct, 2),
                    "relationship": edge_desc,
                    "sensitivity": (
                        "🔴 HIGH EXPOSURE (> 3.0%)"
                        if abs(impact_pct) >= 3.0
                        else (
                            "🟡 MODERATE (1.0%–3.0%)"
                            if abs(impact_pct) >= 1.0
                            else "🟢 LOW (< 1.0%)"
                        )
                    ),
                }
            )

        results.sort(key=lambda x: abs(x["predicted_spillover_pct"]), reverse=True)
        return results


def analyze_supply_chain_spillover(
    origin_ticker: str = "TSM", shock_pct: float = -5.0
) -> Dict[str, Any]:
    """
    High-level entry point to run GNN supply chain shock propagation.
    """
    gnn = SupplyChainGraphNetwork()
    spillovers = gnn.simulate_upstream_shock(
        origin_ticker, shock_magnitude_pct=shock_pct
    )

    return {
        "origin_ticker": origin_ticker,
        "input_shock_pct": shock_pct,
        "total_impacted_nodes": len(spillovers),
        "downstream_impacts": spillovers,
    }
