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
| `.github/workflows/autonomous_live_trader.yml` | `60d935bd019d4f1d411d2f19203b6d14c9db57a2702ecbea5c2fe89a203ab21c` |
| `requirements.txt` | `ecb8365cf752da713623a1c701ac6c158dfc5b7efd1c91b412e33c7f0869475c` |
| `requirements-dev.txt` | `4c9e67f52217c5c10dfd725804913f3d36294f8981f8b3f4d9d0b2a30330f780` |
| `scripts/run_247_daemon.py` | `05fc3fc025ff2f71ecd749144fce2687ac498b3df6347bd2269811ff483f84ce` |
| `scripts/weekly_scorecard.py` | `cb737f143ced2c6d0a9cc2d0b7be4297dc0bdcf973b29cbcef7486621192656f` |
| `results/agent_learning_memory.json` | `9c5b84f27812d179d845d4722e6b90d781941199b728916baea7893f622a644b` |
| `results/agent_memory/committee_weights.json` | `523f964dcc6bccf3cd383aa0ecfb32ab8278468c1befdb0f6f89ebaa513bbd58` |
| `results/agent_memory/semantic_rules.json` | `244897ab6250ee84963d1ab99c15a3bd74251cf735d739d11bf40c5c5b26a206` |
| `results/agent_memory/session_episodic_memory.json` | `2f5b75a57fcdfb06c51c9868c25978c9afc87d7b7b6e9b02d0e5a7f53525e5e9` |
| `results/agent_memory/trade_postmortems.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## 3. Core Strategy Modules Sample (SHA-256)

| Core Module | SHA-256 Digest |
| :--- | :--- |
| `src/agent_committee.py` | `2e9ede3b5ad446febfe2a56d80ea004fa655133abce20a906b40c4aba837c404` |
| `src/autonomous_trader.py` | `e45736bd80db6e5bb2e777906a89b44d44eb6ac8b391b86a0f0ce585689500a5` |
| `src/paper_broker.py` | `5b365b19f8691f178e269675e600c917131eed175b2a74a6bfc406cdf979009d` |
| `src/alpaca_broker.py` | `36f2acee8b77082226ed084fc1985313576d9773ace08ae745826504d23d140a` |
| `src/post_mortem_learner.py` | `e56670a47ae6074d2a2e528d019c9eb8fbfe8a024c1c57c7fcbb3fa192f67eed` |
| `src/agent_memory.py` | `d6c4ef9b2a91a16b685006bce8a1a8d633108cae61ec14c3aee9cfd29ecc7ca2` |

## 4. Production Model Weights (1974 Models)

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