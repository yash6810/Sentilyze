import pytest
import os
from src.tearsheet_generator import generate_institutional_pdf_tearsheet


def test_generate_institutional_pdf_tearsheet_bytes():
    """Verify that PDF generation creates valid, non-empty binary content."""
    pdf_bytes = generate_institutional_pdf_tearsheet(ticker="NVDA")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 10000
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_institutional_pdf_tearsheet_file(tmp_path):
    """Verify that PDF file is saved correctly to a specified path."""
    out_file = str(tmp_path / "test_factsheet.pdf")
    pdf_bytes = generate_institutional_pdf_tearsheet(
        output_path=out_file, ticker="AAPL"
    )
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 10000
    assert pdf_bytes.startswith(b"%PDF-")
