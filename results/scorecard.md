# Sentilyze Forward Execution Scorecard

**Audit Inception Date**: `2026-11-02`  
**As Of Date**: `2026-10-26` (`0 days (not started)`)  
**Last Alpaca Data Point Used**: `2026-10-24`  
**Audit Status**: **`IN PROGRESS`**  
**Status Detail**: Evaluation active: 0 days (not started) (0.0/8.0 weeks elapsed). Minimum 8 weeks required.

## Performance Comparison

| Metric | Alpaca Paper Portfolio | SPY Benchmark (Adjusted) | Spread / Delta |
| :--- | :---: | :---: | :---: |
| **Starting Equity** | **$100,000.00** | **$100,000.00** | $0.00 |
| **Current Equity** | **$100,000.00** | **$100,000.00** | $0.00 |
| **Net Total Return** | **+0.00%** | **+0.00%** | **+0.00%** |
| **Maximum Drawdown** | **0.00%** | **0.00%** | +0.00% |
| **Closed Forward Trades** | **0** (min 30 required) | Buy & Hold | — |
| **Average Invested Exposure** | **0.00%** (min 25% required) | 100.0% | — |

## Strict Evaluation Gate Rules

1. **Evaluation Horizon**: Minimum 8 full weeks (56 calendar days) forward from `2026-11-02`.
2. **Minimum Trade Activity**: At least 30 closed round-trip orders executed on Alpaca paper markets.
3. **Alpha Outperformance**: Alpaca net return must strictly exceed SPY adjusted total return.
4. **Capital Utilization**: Average daily invested exposure (`position_cost / equity`) must be at least 25.0% to reject cash-hiding survivorship.

## 📦 Open Alpaca Positions Telemetry

- **Total Open Positions**: **2**

| Symbol | Open Quantity | Entry Date | Position Age | Avg Entry Price |
| :--- | :---: | :---: | :---: | :---: |
| `AMZN` | 6.0 | `2026-10-02` | 24 days | $250.55 |
| `QCOM` | 35.0 | `2026-10-02` | 24 days | $186.08 |

## 🎯 Bracket Order Leg Exits (Alpaca Brokerage)

Breakdown of all filled sell orders executed by Alpaca exchange-side bracket orders:

| Exit Mechanism | Leg Order Type | Orders Closed | Distribution |
| :--- | :---: | :---: | :---: |
| **Take-Profit Leg** | `limit` sell | **17** | 89.5% |
| **Stop-Loss Leg** | `stop` / `stop_limit` sell | **1** | 5.3% |
| **Discretionary / Other** | `market` sell | **1** | 5.3% |
| **Total Realized Exits** | — | **19** | 100.0% |

*Generated automatically by `scripts/weekly_scorecard.py` on 2026-10-26.*