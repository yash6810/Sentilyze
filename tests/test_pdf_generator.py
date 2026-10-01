import pytest
import os
from src.pdf_generator import generate_wall_street_factsheet_pdf


def test_generate_wall_street_factsheet_pdf_bytes():
    pdf_bytes = generate_wall_street_factsheet_pdf(
        strategy_name="Unit Test Alpha",
        ticker="SPY",
        custom_metrics={"sharpe_ratio": 2.85, "cagr_pct": 42.1},
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF Magic bytes header: %PDF
    assert pdf_bytes.startswith(b"%PDF")


def test_generate_wall_street_factsheet_pdf_file(tmp_path):
    out_file = str(tmp_path / "factsheet_test.pdf")
    pdf_bytes = generate_wall_street_factsheet_pdf(output_path=out_file)

    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000
