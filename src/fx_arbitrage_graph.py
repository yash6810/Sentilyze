"""
Paper 5: Negative Cycle Detection on Exchange Log-Rate Digraphs (Bellman-Ford).

Detects multi-currency / cross-pair statistical and triangular arbitrage in O(V * E) polynomial time.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)


def detect_negative_cycle_arbitrage(
    currency_pairs: List[Tuple[str, str, float]],
) -> Dict[str, Any]:
    """
    Finds triangular arbitrage using Bellman-Ford on log exchange rates:
    w = -ln(Rate). A negative cycle indicates an arbitrage loop where product(Rates) > 1.
    """
    nodes = set()
    for src, dst, _ in currency_pairs:
        nodes.add(src)
        nodes.add(dst)

    nodes = list(nodes)
    V = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Graph edges with negative log weights
    edges = []
    for src, dst, rate in currency_pairs:
        if rate > 0:
            edges.append((node_to_idx[src], node_to_idx[dst], -np.log(rate), rate))

    dist = [0.0] * V
    pred = [-1] * V

    # Relax edges V - 1 times
    for _ in range(V - 1):
        for u, v, w, _ in edges:
            if dist[u] + w < dist[v] - 1e-8:
                dist[v] = dist[u] + w
                pred[v] = u

    # Check for negative cycle
    arbitrage_cycle = []
    cycle_profit_pct = 0.0
    has_arbitrage = False

    for u, v, w, rate in edges:
        if dist[u] + w < dist[v] - 1e-8:
            has_arbitrage = True
            # Reconstruct cycle
            curr = v
            for _ in range(V):
                curr = pred[curr]
            cycle = [curr]
            p = pred[curr]
            while p != curr and p != -1 and len(cycle) < V + 2:
                cycle.append(p)
                p = pred[p]
            cycle.append(curr)
            cycle.reverse()

            named_cycle = [nodes[idx] for idx in cycle]
            cycle_profit_pct = float(np.exp(-dist[v]) - 1.0) * 100.0
            arbitrage_cycle = named_cycle
            break

    return {
        "has_arbitrage_opportunity": has_arbitrage,
        "arbitrage_path": arbitrage_cycle,
        "implied_risk_free_profit_pct": round(max(0.0, cycle_profit_pct), 3),
        "complexity_class": f"Polynomial Time O(V*E) = O({V}*{len(edges)})",
    }
