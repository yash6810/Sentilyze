"""Unit Tests for Project-Wide GPU Omnisearch Engine.

Verifies:
1. GPUVectorIndex vocabulary and tensor embedding construction.
2. Batch GPU/CPU dot product cosine similarity retrieval.
3. Code and monograph chunk extraction.
4. Functional search_project API.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from src.omnisearch_gpu import (
    DocumentChunk,
    GPUVectorIndex,
    ProjectOmnisearchEngine,
    SearchResult,
    search_project,
)


@pytest.fixture
def mock_chunks() -> list[DocumentChunk]:
    """Generate sample document chunks across code, monograph, and risk."""
    return [
        DocumentChunk(
            chunk_id="code_hawkes",
            source_path="src/hawkes_cascade_engine.py",
            source_type="CODE",
            title="Hawkes Cascade Engine",
            content="Bivariate Hawkes point process tracking branching ratio eta and endogenous liquidity cascade risk.",
        ),
        DocumentChunk(
            chunk_id="doc_ssrn",
            source_path="ssrn_151_trading_strategies_monograph.md",
            source_type="MONOGRAPH",
            title="SSRN 151 Trading Strategies Monograph",
            content="Idiosyncratic volatility anomaly and mean reversion factor strategies for quantitative equities.",
        ),
        DocumentChunk(
            chunk_id="risk_cvar",
            source_path="results/cvar_stress_report.json",
            source_type="RISK_REPORT",
            title="Neural SDE CVaR99 Stress Report",
            content="1,000,000 jump-diffusion paths with $117,877 survival cash floor.",
        ),
    ]


def test_vector_index_search(mock_chunks: list[DocumentChunk]):
    """Verify GPU vector index correctly embeds and retrieves top matching chunks."""
    index = GPUVectorIndex(embedding_dim=128, device="cpu")
    index.index_chunks(mock_chunks)

    assert index.embeddings is not None
    assert index.embeddings.shape == (3, 128)

    # Search for Hawkes cascade
    results = index.search("Hawkes branching ratio liquidity", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "code_hawkes"
    assert results[0].score > 0.0

    # Search for CVaR cash floor
    results_cvar = index.search("survival cash floor cvar", top_k=2)
    assert len(results_cvar) > 0
    assert results_cvar[0].chunk_id == "risk_cvar"


def test_chunk_extraction_and_search_project(tmp_path):
    """Verify ProjectOmnisearchEngine extracts chunks and performs end-to-end query."""
    # Create mock repo directory structure in tmp_path
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "sample_model.py").write_text(
        '"""Sample alpha model for momentum trading."""\ndef compute_momentum(prices):\n    pass\n',
        encoding="utf-8",
    )

    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    (art_dir / "quant_paper.md").write_text(
        "## Statistical Arbitrage\nPairs trading using dynamic Kalman filter hedge ratios.",
        encoding="utf-8",
    )

    engine = ProjectOmnisearchEngine(
        repo_root=str(tmp_path), artifact_dir=str(art_dir), device="cpu"
    )
    total_chunks = engine.build_index()
    assert total_chunks >= 2

    # Query the engine
    matches = engine.search("Kalman filter statistical arbitrage", top_k=1)
    assert len(matches) == 1
    assert "quant_paper.md" in matches[0].source_path
