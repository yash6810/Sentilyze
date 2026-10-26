# Invalid Local Simulation Ledger (2026 Archive)

> **Notice**: This local simulation ledger is invalid; entry prices don't match real market prices (38 of 95 entry prices fall outside that day's high-low range), and it was manually edited. Do not use it for performance claims.

## Summary of Audit Findings
1. **Decoupled Prices**: 38 out of 95 trade entry prices and 9 exit prices fell strictly outside the authentic daily market [Low, High] range recorded by Yahoo Finance (`auto_adjust=False`). Examples: PLTR entered on 2026-09-02 at $28.50 vs actual low of $165.71; AMD entered on 2026-09-02 at $98.20 vs actual low of $452.36.
2. **Manual Edits**: Git history shows repeated manual additions, deletions, and restorations of equity balance and trade entries by human commits rather than automated market execution fills.
3. **Decoupling from Broker Truth**: The real Alpaca paper account (`PA3NFG0UWP3R`) recorded an authentic return of +2.44% ($102,443.49 equity) over this period, completely distinct from this simulation ledger's claimed $136,489.18 balance.

This archive preserves the raw files (`paper_portfolio.json`, `executed_trades.csv`) for forensic and audit reproducibility without presenting them as legitimate performance.
