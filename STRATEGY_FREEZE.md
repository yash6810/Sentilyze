# Sentilyze Quantitative Strategy Freeze Protocol

**Effective Freeze Date**: `2026-11-02`  
**Baseline Fix Commit Hash**: `GIT_COMMIT_HASH_PLACEHOLDER`  
**Freeze Expiration**: When `results/scorecard.md` outputs `PASS` or `FAIL` (minimum 8 full weeks and 30 closed trades).  
**Enforcement Mechanism**: Automated CI test [`tests/test_strategy_freeze.py`](tests/test_strategy_freeze.py).  

## 1. Freeze Mandate

From **November 02, 2026 forward**, no changes, adjustments, re-calibrations, or hotfixes are permitted to:
1. **Core Strategy & Risk Source Modules (195 files)**: All modules in [`src/*.py`](src/)
2. **Production Model Weights (1974 models)**: All XGBoost JSON models in [`models/*.json`](models/)
3. **Persistent Learned State & Memory**: [`results/agent_learning_memory.json`](results/agent_learning_memory.json) and all files in [`results/agent_memory/`](results/agent_memory/)
4. **Live Execution Workflow & Scripts**: [`.github/workflows/autonomous_live_trader.yml`](.github/workflows/autonomous_live_trader.yml), [`scripts/run_247_daemon.py`](scripts/run_247_daemon.py), [`scripts/weekly_scorecard.py`](scripts/weekly_scorecard.py)
5. **Pinned Dependency Requirements**: [`requirements.txt`](requirements.txt) & [`requirements-dev.txt`](requirements-dev.txt)

> [!IMPORTANT]
> **Any change to a frozen file restarts the 8-week clock and must be logged in `CHANGELOG.md` with date and reason.**

Any modification to strategy parameters, thresholds, risk gates, learning state, dependencies, or model weights will cause the continuous integration test `tests/test_strategy_freeze.py` to immediately fail.

## 2. Learned State & Pinned Configuration Hashes (SHA-256)

| File Path | SHA-256 Digest |
| :--- | :--- |
| `.github/workflows/autonomous_live_trader.yml` | `54e423ac2bea2e3c882b0859414f8d46680ae72778fad0e716f9d8da9331d52a` |
| `requirements.txt` | `ee23bdc75df9a6fbf1b0951988c74378793a76217a03ff0be5048f2cc5fb5796` |
| `requirements-dev.txt` | `39a00628a6af2a70af77909db146ecb9fcbadd0ee4375ede98ce47468a65ba5a` |
| `scripts/run_247_daemon.py` | `05fc3fc025ff2f71ecd749144fce2687ac498b3df6347bd2269811ff483f84ce` |
| `scripts/weekly_scorecard.py` | `135f2c072f10184465d2624afdc31c2ddc701939fdc904716388b86fd7ce40ea` |
| `results/agent_learning_memory.json` | `95d03a2ca3deddaaffbec754840bc4df934b631c91503688c4119a7b4a04c605` |
| `results/agent_memory/committee_weights.json` | `c527d4b9dba60da44522500ea16e5b66390de36bb5b3146263823105ded808ee` |
| `results/agent_memory/semantic_rules.json` | `e835d22aa834acb8cdf6d90d78e46bcbf29d9b61c368338e62e28555edc514e1` |
| `results/agent_memory/session_episodic_memory.json` | `f43e9a9cd8dec4269a0f47b90fbb71f2f34858fd50b44f9a71d7429ebc4fa0ea` |
| `results/agent_memory/trade_postmortems.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## 3. Core Strategy Modules Sample (SHA-256)

| Core Module | SHA-256 Digest |
| :--- | :--- |
| `src/agent_committee.py` | `20aaaa19a158eeb2085f7342d9870afd9debe588cbb88aa9fcccca228bd27a54` |
| `src/autonomous_trader.py` | `b66c49a42062c81a6612fcfef5fbd0845a5e44c7edb7687bd9f9cc63e289c459` |
| `src/paper_broker.py` | `ab2c84ffea7913b8bedb32ce53ab9df56a401440db92f8cd3f333f05e4c1e8ac` |
| `src/alpaca_broker.py` | `09b44590ad7fc032f62a127f63201a4a2be44459366786a97f437075503add24` |
| `src/post_mortem_learner.py` | `efcbf27b9a5721690e68642b6dc3004b75d81a4b11545bba5844781fa72a1902` |
| `src/agent_memory.py` | `d6c4ef9b2a91a16b685006bce8a1a8d633108cae61ec14c3aee9cfd29ecc7ca2` |

## 4. Production Model Weights (1974 Models)

All 1974 XGBoost production models in `models/*.json` are cryptographically frozen. The complete list of SHA-256 signatures is tracked in [`results/strategy_freeze_manifest.json`](results/strategy_freeze_manifest.json). Below is a sample of key active universe models:

| Model File | SHA-256 Digest |
| :--- | :--- |
| `models/NVDA_model.json` | `f0405af43ff1d3ea8a5d9b52960a2bb59f05504373ef78c8c599a2ee7c786737` |
| `models/AAPL_model.json` | `f8a0eb6d516210fa59a96d0a543d51655d2f4bfdd1a62656d65fa2f7b68bbd4a` |
| `models/MSFT_model.json` | `0ca03a46c8b561d51029295ef19b6273a2b77f8ed65921d4df63eed37a224bc3` |
| `models/GOOGL_model.json` | `ba98ab5daab3b58776817f9b1a36d90003709de38bed25f4eb8a18df4606980e` |
| `models/AMZN_model.json` | `dbd283df3ff33f70a8645b5141880cd3774ca9174cc9323eace8e3f31f5bb391` |
| `models/META_model.json` | `1fa7488cd180b606f156a0ff0a7b1882c8c272ea87287d97916853cf76e1fc5c` |
| `models/TSLA_model.json` | `37ef678b73fc0cbea1587858d445d727dbc9789eb6a503d3f8d20e0ad3478334` |
| `models/SPY_model.json` | `37f7a77b738265d7928f0c515988716047f28da1286b2a3736e21178706efae3` |
| `models/QQQ_model.json` | `c6a3f4154c0ab40a2289c7164f84bf20c698ff25d488785b3aa1bfaa7a151c13` |
| `models/WMT_model.json` | `248b1bef6bb8120f4b548b352e71c95c84b0faffcb7756e741c4e851c8433371` |

## 5. Runtime Inputs & Neural Tensorial Weights (SHA-256)

Every non-.py file read on the live execution path is cryptographically signed:

| File Path | SHA-256 Digest |
| :--- | :--- |
| `stocks.txt` | `74a6c0880b46ba0bcda6f080b5f2e41848be588cf2457c52a7ee061d82d6f3a6` |
| `models/drl_policy/ADP_drl_weights.pt` | `52b74dc2b6db9ff286d0846da42ae8885903d4006f8c76971f06b5e635e23d44` |
| `models/drl_policy/NVDA_drl_weights.pt` | `de5e314f2a2da6e91fa404509fa89247abe74c11e46485831b2c6b8b5a9d6e81` |
| `models/drl_policy/universal_drl_master_policy.pt` | `a638fc083a4f6ed58b453f26330cb739c1d7e702e2e2fe13837086f5ad296410` |
| `models/drl_policy/universal_drl_policy.pt` | `65fff01e1b6af260b7f6c6aa759bcfc961d3d0f8388b6d5765b5ae66fd1946b0` |
| `models/finbert-fine-tuned/config.json` | `6772f6b50f6b04d7fa86549cbd326995f2d84d85e3416b7f7ab4473c127ee09f` |
| `models/finbert-fine-tuned/model.safetensors` | `efac5d3b168d32c8c5bd11205822717122018bc05f70404ff0b4e1094a6ce6f5` |
| `models/finbert-fine-tuned/special_tokens_map.json` | `3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6` |
| `models/finbert-fine-tuned/tokenizer.json` | `0a1e1d9fa0bafddab4e45e3f06255aa9b808b9091b4f3da5beaf65a76d101d3a` |
| `models/finbert-fine-tuned/tokenizer_config.json` | `00bd5b2d9a46e60d32549452fac5d9b15d8f9ac1945681dfcbc03e3e068090ab` |
| `models/finbert-fine-tuned/vocab.txt` | `85f2481b4e3c25a7ece5e0e4930631a75c1a5fe2d31b0ecd9b5d51136ab98de8` |
| `models/fingpt_materiality_adapter.pt` | `8e544950cb2f5fcae621c098b0332a3cbc5519effe65cd4c9f7b6c2e2c898912` |
| `models/finjepa_latent_physics.pt` | `feb22f2f4dfd350fde0f6c85f44342151b9722256989e963534c26f3969b476a` |
| `models/finmamba_jepa_master.pt` | `ec372dbd1e8ced55220b49e6edce19b4a39867b60c46bd88a114f9bc9df61e0b` |
| `models/ft_transformer.pt` | `4f482b9deeabdad0d28c9d72adee1da030e858997cde7c2fbfc710be0b60bf60` |
| `models/neural_sde_jump.pt` | `023b112a53e2640b769a48b54f46ff41328e0ccec13fa23d33fcdcee67c03826` |

## 6. Exit Behavior & Brokerage Execution Model

- **Alpaca Live Account**: In the live autonomous execution path (`autonomous_live_trader.yml`), all equity exits on Alpaca (`PA3NFG0UWP3R`) occur **exclusively through native entry-time exchange-side OCO bracket legs** (`take_profit.limit_price` and `stop_loss.stop_price`). Alpaca's matching engine executes either the limit sell (take-profit) or stop sell (stop-loss) automatically; when one leg fills, Alpaca server-side cancels the other leg.
- **Simulation Ledger**: Intraday discretionary exits, breakeven trails, and committee veto exits evaluated in `AutonomousTradingEngine._evaluate_portfolio_exits()` and `PaperBroker.execute_manual_sell()` apply **strictly to the local simulation ledger** (`results/paper_portfolio.json`). They do not submit market or limit sell orders to Alpaca.
- **Historical Orders Source of Truth**: Across all 19 realized sell executions recorded in `results/alpaca_orders.csv`, exactly 17 closed via limit bracket legs (take-profit fills), 1 closed via a stop bracket leg (AKAM stop-loss fill), and 1 closed via an immediate test market order on `BTC/USD` (order `78c12d32-43d8-4d9f-9ba1-1eb5053baea5` placed on 2026-10-16 at 15:43:21Z; Alpaca does not support OCO brackets on crypto assets). The function `execute_daily_signals()` (which contains the `MODEL_SELL` hook and calls `alpaca.close_position()` at `src/paper_broker.py:697`) is never called by the live trader (`autonomous_live_trader.yml` / `src/autonomous_trader.py`) nor by any scheduled GitHub Actions workflow (0 occurrences across `.github/workflows/`).

---
*Enforced by Sentilyze Audit Protocol and `tests/test_strategy_freeze.py`.*