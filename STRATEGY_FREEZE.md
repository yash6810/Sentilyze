# Sentilyze Quantitative Strategy Freeze Protocol

**Effective Freeze Date**: `2026-10-26`  
**Freeze Expiration**: When `results/scorecard.md` outputs `PASS` or `FAIL` (minimum 8 full weeks and 30 closed trades).  
**Enforcement Mechanism**: Automated CI test [`tests/test_strategy_freeze.py`](tests/test_strategy_freeze.py).  

## 1. Freeze Mandate

From **October 26, 2026 forward**, no changes, adjustments, re-calibrations, or hotfixes are permitted to:
1. **Entry Rules & Multi-Agent Deliberation**: [`src/agent_committee.py`](src/agent_committee.py)
2. **Position Sizing & Execution Schedulers**: [`src/autonomous_trader.py`](src/autonomous_trader.py)
3. **Production Model Weights**: All XGBoost JSON models in [`models/*.json`](models/)

Any modification to strategy parameters, thresholds, risk gates, or model weights will cause the continuous integration test `tests/test_strategy_freeze.py` to immediately fail.

## 2. Core Execution Engine Cryptographic Hashes (SHA-256)

| File Path | SHA-256 Digest |
| :--- | :--- |
| `src/agent_committee.py` | `b81fce666b50e8c51c1ddae397c35e180c46225af995f0959216bdfced7e5a95` |
| `src/autonomous_trader.py` | `a5e08b13cfe21f09bee196073fbc25fe95bff39566edaa7e4b8da2f4f438f6bf` |

## 3. Production Model Weights (1974 Models)

All 1974 XGBoost production models in `models/*.json` are cryptographically frozen. The complete list of SHA-256 signatures is tracked in [`results/strategy_freeze_manifest.json`](results/strategy_freeze_manifest.json). Below is a sample of key active universe models:

| Model File | SHA-256 Digest |
| :--- | :--- |
| `models/NVDA_model.json` | `cb6708b9e4614729063bd4f38d103f9de43b40127c3b67fdbbc03674c032dcb2` |
| `models/AAPL_model.json` | `b8c1e20dda370c0525100c0e1ca6cd94370f89a60a13c850aae235d1bb9573cf` |
| `models/MSFT_model.json` | `e752e0e9b1a0264391e740af18e18af4fd9341a907019d8f1ce4d37497e2caf3` |
| `models/GOOGL_model.json` | `b1a5521057ba0296a04f7588c577661a3b3f8b5271655121dd02fb5de9361271` |
| `models/AMZN_model.json` | `d8333357a895074917209d001bb49093bb087c03aa53e2505188cd5168dc45c2` |
| `models/META_model.json` | `2500941fda415e9e4cb5f9127984a814bb97d965e4997fe43f6c9277921cde78` |
| `models/TSLA_model.json` | `21af4b8905c11173dbf55b7c5e5e2c104be26840836c9df1f99470423361a27b` |
| `models/SPY_model.json` | `4789d6461e45b3243b5ce3549cdb324a5387ab31b5416687ebeacfb06b5cb331` |
| `models/QQQ_model.json` | `c38284fe59614dbd133b1f3439ca354b89b673a08f03cdf6c29cdbe900c29268` |
| `models/WMT_model.json` | `ab09fa25e44c169b3df1a9847218d103ea87c26d1a35b6fe2bfccc8b53b25d80` |

---
*Enforced by Sentilyze Audit Protocol and `tests/test_strategy_freeze.py`.*