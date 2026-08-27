import os
from src.rebalancer import calculate_share_allocation
from src.tearsheet import generate_executive_pdf_tearsheet
from src.stress_tester import run_monte_carlo_stress_test


def test_calculate_share_allocation():
    signals = [
        {
            "ticker": "AMD",
            "confidence": 0.76,
            "current_price": 400.0,
            "take_profit": 440.0,
            "stop_loss": 370.0,
        },
        {
            "ticker": "TSLA",
            "confidence": 0.65,
            "current_price": 200.0,
            "take_profit": 220.0,
            "stop_loss": 185.0,
        },
        {
            "ticker": "META",
            "confidence": 0.58,
            "current_price": 500.0,
            "take_profit": 540.0,
            "stop_loss": 470.0,
        },
    ]

    result = calculate_share_allocation(
        capital=25000.0, selected_signals=signals, method="risk_parity"
    )

    assert result["total_capital"] == 25000.0
    assert result["total_invested"] <= 25000.0
    assert result["cash_reserve"] >= 0.0
    assert not result["allocation_table"].empty
    assert len(result["allocation_table"]) == 3
    assert result["positions_count"] > 0


def test_generate_pdf_tearsheet(tmp_path):
    pdf_path = str(tmp_path / "test_factsheet.pdf")
    pdf_bytes = generate_executive_pdf_tearsheet(output_path=pdf_path)

    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
    assert os.path.exists(pdf_path)


def test_monte_carlo_stress_test():
    result = run_monte_carlo_stress_test(
        initial_capital=50000.0,
        num_simulations=500,
        time_horizon_days=20,
    )

    assert result["initial_capital"] == 50000.0
    assert "var_95_dollar" in result
    assert "prob_profit" in result
    assert 0 <= result["prob_profit"] <= 100
    assert not result["percentile_paths_df"].empty
    assert len(result["percentile_paths_df"]) == 21
