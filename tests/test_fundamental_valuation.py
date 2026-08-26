from src.fundamental_valuation import (
    fetch_financial_statements,
    calculate_piotroski_f_score,
    calculate_altman_z_score,
    calculate_dcf_fair_value,
    generate_spider_radar_profile,
)


def test_fetch_financial_statements():
    res = fetch_financial_statements("NVDA")
    assert "ticker" in res
    assert "spot_price" in res
    assert "market_cap" in res
    assert "info" in res


def test_calculate_piotroski_f_score():
    fin_data = fetch_financial_statements("NVDA")
    res = calculate_piotroski_f_score("NVDA", fin_data)

    assert "f_score" in res
    assert 0 <= res["f_score"] <= 9
    assert "category" in res
    assert "breakdown" in res


def test_calculate_altman_z_score():
    fin_data = fetch_financial_statements("NVDA")
    res = calculate_altman_z_score("NVDA", fin_data)

    assert "z_score" in res
    assert res["z_score"] > 0
    assert "zone" in res
    assert "components" in res


def test_calculate_dcf_fair_value():
    fin_data = fetch_financial_statements("NVDA")
    res = calculate_dcf_fair_value("NVDA", fin_data, growth_rate=0.15)

    assert "fair_value_price" in res
    assert "margin_of_safety_pct" in res
    assert "verdict" in res
    assert res["fair_value_price"] > 0


def test_generate_spider_radar_profile():
    fin_data = fetch_financial_statements("NVDA")
    f_res = calculate_piotroski_f_score("NVDA", fin_data)
    z_res = calculate_altman_z_score("NVDA", fin_data)
    dcf_res = calculate_dcf_fair_value("NVDA", fin_data)

    radar = generate_spider_radar_profile("NVDA", 0.75, f_res, z_res, dcf_res)
    assert "AI Technical Momentum" in radar
    assert "Solvency (Altman Z)" in radar
    assert "Quality (Piotroski F)" in radar
    assert "Valuation Discount (DCF)" in radar
    for v in radar.values():
        assert 0 <= v <= 100
