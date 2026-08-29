# 🧪 Experimental & Simulated Research Prototypes

> **⚠️ STATUS: ISOLATED FROM PRODUCTION TRADING DECISIONS**
> 
> The modules in this directory are research prototypes and architectural scaffolding designed for prospective API integration (e.g. SEC EDGAR direct scraping, ATS Dark Pool feeds, and USPTO patent databases).
>
> **None of the modules in this directory feed into live trading decisions, position sizing, or backtest execution in `src/`.**

---

## 📁 Prototype Inventory

| Module | Intended Domain | Current State | Prospective Live Feed |
| :--- | :--- | :--- | :--- |
| `dark_pool_radar.py` | ATS Dark Pool & Block Flow | Simulated structural block templates | FINRA TRACE / Cboe ATS |
| `insider_tracker.py` | SEC Form 4 Insider Trades | Simulated insider cluster transactions | SEC EDGAR API / OpenInsider |
| `patent_contract_radar.py` | USPTO Patents & Defense Contracts | Simulated innovation indices | USPTO API / USASpending.gov |
| `sec_filing_diff.py` | 10-K / 10-Q Text Diffing | Simulated Item 1A Risk Factor diffs | SEC EDGAR Full-Text Search |

---

## 🔒 Production Isolation Guarantee

The core quantitative engine in `src/` (`agent_committee.py`, `autonomous_trader.py`, `backtesting.py`) does **NOT** import or depend on any file in this `experimental/` folder.
