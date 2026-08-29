# Graph Report - Sentilyze  (2026-08-30)

## Corpus Check
- 467 files · ~725,262 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1132 nodes · 2393 edges · 55 communities (51 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e7ebdf2e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- quant_engine.py
- get_price_history
- test_omnichannel_mobile.py
- run_daily_market_scan
- run_backtest
- app.py
- ws_live_prediction.py
- test_statistical_arbitrage.py
- agent_committee.py
- TradingEnvironment
- get_sentiment
- CloudDataLake
- ws_alternative_data.py
- SupplyChainGraphNetwork
- ws_portfolio.py
- run_temporal_fusion_forecast
- test_feature_engineering.py
- train_meta_ensemble
- ._save
- chart_pattern_learning.py
- get_logger
- compute_dark_pool_sentiment
- test_rebalancer_and_tearsheet.py
- load_model
- safe_path_join
- compute_lead_lag_matrix
- black_swan_simulator.py
- test_pillar2_alternative_data.py
- AlpacaBrokerBridge
- AutonomousTradingEngine
- PaperBroker
- opening_range_engine.py
- options_surface.py
- .get_closed_trades_df
- compute_cross_asset_correlation
- preprocess_data
- SuperEnsembleClassifier
- TickerSentinelSwarm
- Sentilyze — Systematic Sentiment & Momentum Trading Research Platform
- autonomous_trader.py
- render_workspace_header
- test_api.py
- strategy_optimizer.py
- reddit_premarket_station.py
- ui/__init__.py
- Contributor Covenant Code of Conduct
- temp_data_dir
- calculate_doubling_progress
- How Can I Contribute?
- audio_briefing.py
- 🧪 Experimental & Simulated Research Prototypes
- rules/graphify.md
- workflows/graphify.md

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 69 edges
2. `PaperBroker` - 41 edges
3. `get_price_history()` - 35 edges
4. `run_unified_institutional_pipeline()` - 32 edges
5. `fetch_live_quote()` - 29 edges
6. `preprocess_data()` - 26 edges
7. `get_news()` - 24 edges
8. `handle_telegram_command()` - 23 edges
9. `render_workspace_header()` - 22 edges
10. `fetch_financial_statements()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `test_autonomous_cycle_execution()` --uses--> `AutonomousTradingEngine`  [INFERRED]
  tests/test_autonomous_trader.py → src/autonomous_trader.py
- `test_autonomous_cycle_execution()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_idempotency_lock_prevents_overlap()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_preprocess_data_orchestrates_correctly()` --calls--> `preprocess_data()`  [EXTRACTED]
  tests/test_preprocessing.py → src/preprocessing.py
- `test_fetch_live_quote()` --calls--> `fetch_live_quote()`  [EXTRACTED]
  tests/test_realtime_tracker.py → src/realtime_tracker.py

## Import Cycles
- None detected.

## Communities (55 total, 4 thin omitted)

### Community 0 - "quant_engine.py"
Cohesion: 0.05
Nodes (76): MasterQuantPipelineResult, Any, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), analyze_debt_maturity_wall(), calculate_beneish_m_score() (+68 more)

### Community 1 - "get_price_history"
Cohesion: 0.07
Nodes (50): _fetch_alpaca_news(), _fetch_alpaca_price_history(), _fetch_direct_yahoo_chart(), _fetch_eodhd_price_history(), _fetch_finnhub_news(), _fetch_fmp_price_history(), _fetch_google_news_rss(), _fetch_marketaux_news() (+42 more)

### Community 2 - "test_omnichannel_mobile.py"
Cohesion: 0.13
Nodes (17): answer_financial_query(), Any, Natural Language Financial Q&A Agent for Sentilyze. Pillar 7 Mobile &…, Parses natural language questions and routes them to quantitative engines.…, generate_smartwatch_glance_payload(), Any, Apple Watch & Wear OS Glance Complications API for Sentilyze. Pillar 7 Mobile &…, Generates structured complication JSON for Apple Watch (watchOS) and Wear OS. (+9 more)

### Community 3 - "run_daily_market_scan"
Cohesion: 0.08
Nodes (43): format_signal_card(), Any, Dispatches a high-impact Discord card for live autonomous trade lifecycle…, Construct a standardized trade signal data payload. Args: ticker (str): Stock…, Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord., Dispatches real-time Reddit r/wallstreetbets & Stocktwits hype spike alerts., Sends a consolidated morning macro regime and portfolio health pulse to Discord., Sends a consolidated master market digest card containing all universe signals. (+35 more)

### Community 4 - "run_backtest"
Cohesion: 0.10
Nodes (35): Figure, _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition(), calculate_performance_metrics(), _calculate_trade_outcomes() (+27 more)

### Community 5 - "app.py"
Cohesion: 0.15
Nodes (16): load_universe_tickers(), main(), Sentilyze - Institutional Algorithmic Trading & MLOps Platform. Modular Master…, Loads active S&P 100 universe tickers., inject_custom_theme(), Dynamic Bespoke Theme Engine for Sentilyze. Supports 3 Institutional Presets:…, Injects high-performance, bespoke CSS styling into the Streamlit app., Renders the 24/7 Autonomous Live Trading & News Agent interface. (+8 more)

### Community 6 - "ws_live_prediction.py"
Cohesion: 0.17
Nodes (19): calculate_smart_money_zones(), calculate_structural_trailing_stop(), evaluate_multi_timeframe_confluence(), find_swing_pivots(), Any, DataFrame, Institutional Smart Money Market Structure & Price-Action Engine for Sentilyze.…, Ratchets the Stop-Loss up structurally behind higher swing lows. Rules: 1.… (+11 more)

### Community 7 - "test_statistical_arbitrage.py"
Cohesion: 0.19
Nodes (25): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any, Series (+17 more)

### Community 8 - "agent_committee.py"
Cohesion: 0.05
Nodes (55): _persist_ablation_results(), Any, 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), audit_full_universe_committee(), ChiefRiskOfficerAgent, compute_fractional_kelly_sizing() (+47 more)

### Community 9 - "TradingEnvironment"
Cohesion: 0.14
Nodes (16): optimize_rl_position_allocation(), PPOPolicyAgent, Any, ndarray, ⚠️ EXPERIMENTAL / RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM PRODUCTION…, Computes mean action (leverage) between 0.0 and 2.0., Estimates state value., Trains Actor-Critic parameters across historical episodes. (+8 more)

### Community 10 - "get_sentiment"
Cohesion: 0.08
Nodes (35): clean_headline_data(), _load_sentiment_analyzer(), Any, Loads the FinBERT sentiment analysis model and tokenizer from the local…, Cleans a headline CSV file by removing rows with invalid stock tickers. Caches…, analyze_sentiment(), clean_financial_text(), get_sentiment() (+27 more)

### Community 11 - "CloudDataLake"
Cohesion: 0.14
Nodes (14): CloudDataLake, Any, Supabase / PostgreSQL Cloud Data Lake Connector., Validates or generates cloud database schema., Syncs local trade executions to the cloud database., Publishes real-time portfolio snapshot to cloud WebSockets channel., generate_twap_order_schedule(), generate_vwap_order_schedule() (+6 more)

### Community 12 - "ws_alternative_data.py"
Cohesion: 0.14
Nodes (20): auto_register_ipo_ticker(), fetch_pre_ipo_radar_summary(), fetch_sec_edgar_ipo_filings(), get_pre_ipo_pipeline_df(), Any, DataFrame, IPO & Pre-IPO Intelligence Radar for Sentilyze. Pillar 9 Alternative Asset…, Fetches real-time SEC Form S-1 / S-1/A IPO registration statements from SEC… (+12 more)

### Community 13 - "SupplyChainGraphNetwork"
Cohesion: 0.15
Nodes (14): analyze_supply_chain_spillover(), Any, ndarray, Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for…, Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)., Executes a Graph Convolutional Network (GCN) layer: H_new = ReLU(A_hat * H * W)…, Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab…, High-level entry point to run GNN supply chain shock propagation. (+6 more)

### Community 14 - "ws_portfolio.py"
Cohesion: 0.17
Nodes (18): build_unified_portfolio(), calculate_risk_parity_weights(), load_all_ticker_portfolios(), Any, DataFrame, Series, Combines individual stock strategies into a single managed multi-asset fund.…, Load individual backtest portfolio CSVs for all available tickers. Args:… (+10 more)

### Community 15 - "run_temporal_fusion_forecast"
Cohesion: 0.14
Nodes (15): Any, DataFrame, ndarray, Temporal Fusion Transformer (TFT) & Multi-Horizon Self-Attention Engine for…, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with… (+7 more)

### Community 16 - "test_feature_engineering.py"
Cohesion: 0.17
Nodes (19): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, DataFrame (+11 more)

### Community 17 - "train_meta_ensemble"
Cohesion: 0.16
Nodes (14): MetaEnsembleClassifier, DataFrame, ndarray, Series, Institutional Multi-Model Meta-Ensemble Engine for Sentilyze. Pillar 1 Core…, Generates binary class prediction (0 = Hold/Sell, 1 = Buy) using soft-voting…, Instantiates and fits the Meta-Ensemble classifier., Multi-Model Meta-Ensemble stacking XGBoost, Random Forest, and Calibrated… (+6 more)

### Community 18 - "._save"
Cohesion: 0.16
Nodes (9): Any, Loads existing portfolio state from JSON or initializes a fresh $100k account., Updates total equity, unrealized PnL, and win rates., Returns high-level KPI metrics for the portfolio dashboard., Executes an immediate manual live/simulated BUY order from UI., Executes an immediate manual live/simulated exit of an open position., Executes a 50% scale-out on an open position and moves stop to break-even., Persists portfolio ledger to disk. (+1 more)

### Community 19 - "chart_pattern_learning.py"
Cohesion: 0.19
Nodes (17): detect_classical_chart_patterns(), generate_ai_chart_explanation(), match_historical_chart_twins(), normalize_waveform(), Any, DataFrame, ndarray, AI Chart Pattern Recognition, Geometric Wave Learning & Visual Understanding… (+9 more)

### Community 20 - "get_logger"
Cohesion: 0.11
Nodes (23): datetime, Logger, Cloud PostgreSQL Data Lake (Supabase / Neon) Connector for Sentilyze. Pillar 6…, check_live_news_sentiment_shock(), evaluate_intraday_execution(), get_us_market_session_info(), Any, Checks if breaking news in the last few hours has a severe negative sentiment… (+15 more)

### Community 21 - "compute_dark_pool_sentiment"
Cohesion: 0.21
Nodes (14): compute_dark_pool_sentiment(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent institutional off-exchange block trades and dark pool prints., Scans option chain contracts where daily volume significantly exceeds open…, Synthesizes dark pool prints and unusual options flow into a unified…, scan_abnormal_options_vol_oi(), scan_dark_pool_blocks() (+6 more)

### Community 22 - "test_rebalancer_and_tearsheet.py"
Cohesion: 0.13
Nodes (20): fetch_universe_live_quotes(), Fetches real-time quotes across the entire universe concurrently in parallel., calculate_custom_rebalance(), calculate_share_allocation(), Any, Helper to calculate share allocation from latest daily signals file or universe…, Computes exact whole-share buy allocations for a given capital budget across…, Any (+12 more)

### Community 23 - "load_model"
Cohesion: 0.18
Nodes (18): get_prediction_on_latest_data(), load_model(), Any, DataFrame, Series, Load a trained model from a file using XGBoost's native format. Args: filepath…, Gets a prediction from the model for the latest available data point. Args:…, Train the XGBoost model using Walk-Forward Optimization (WFO) alongside a… (+10 more)

### Community 24 - "safe_path_join"
Cohesion: 0.17
Nodes (17): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+9 more)

### Community 25 - "compute_lead_lag_matrix"
Cohesion: 0.20
Nodes (14): compute_lead_lag_matrix(), _granger_f_test(), Any, DataFrame, ndarray, Series, rank_market_price_leaders(), Lead-Lag Granger Causality & Supply Chain Price Discovery Engine for Sentilyze.… (+6 more)

### Community 26 - "black_swan_simulator.py"
Cohesion: 0.23
Nodes (11): calculate_kelly_sizing(), estimate_market_impact_slippage(), Any, Historical Black Swan Crisis Simulator & Kelly Position Sizing for Sentilyze.…, Calculates optimal position sizing using the Kelly Criterion: Kelly % = W - (1…, Estimates market execution slippage using the Almgren-Chriss square-root impact…, Stress-tests the current portfolio against major historical market crashes.…, simulate_portfolio_crises() (+3 more)

### Community 27 - "test_pillar2_alternative_data.py"
Cohesion: 0.07
Nodes (43): compute_smart_money_insider_score(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent SEC Form 4 insider transactions for a given stock., Retrieves recent Congressional STOCK Act disclosure reports for a ticker., Synthesizes SEC Form 4 and Congressional activity into an overall Smart Money…, track_congressional_stock_disclosures(), track_corporate_insider_filings() (+35 more)

### Community 28 - "AlpacaBrokerBridge"
Cohesion: 0.24
Nodes (7): AlpacaBrokerBridge, Any, Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.…, Fetches active positions from Alpaca brokerage., Verifies active connection to Alpaca Brokerage API., Fetches live Alpaca account equity, buying power, and cash., Submits an institutional Bracket Order: - Entry: Market order - Exit 1: Limit…

### Community 29 - "AutonomousTradingEngine"
Cohesion: 0.15
Nodes (13): AutonomousTradingEngine, check_daily_loss_circuit_breaker(), Any, Dispatches an institutional execution alert to Discord Webhook if configured., Executes one full autonomous decision and execution cycle with: - Task 6:…, Core cycle execution body., Executes the Self-Improving Feedback Loop: 1. Analyzes trade autopsies on…, Task 8: Independent Max-Daily-Loss Circuit Breaker. Returns True if current… (+5 more)

### Community 30 - "PaperBroker"
Cohesion: 0.19
Nodes (12): PaperBroker, Institutional Multi-Stage Quantitative Execution Broker ($100k Account).…, Alias for _save to ensure 100% backward compatibility., fixture, temp_portfolio_file(), test_paper_broker_dataframes(), test_paper_broker_execute_buy_signals(), test_paper_broker_initialization() (+4 more)

### Community 31 - "opening_range_engine.py"
Cohesion: 0.12
Nodes (23): check_market_hours_preflight(), get_current_ny_time(), get_us_market_session(), Any, Unified US Stock Market (NYSE / NASDAQ) Session & Calendar Engine for…, Pre-flight sanity check for automated workflows. Returns True if execution…, Returns the current precise timestamp in America/New_York (Eastern Time)., Computes the exact real-time US equity market session (NYSE / NASDAQ). Session… (+15 more)

### Community 32 - "options_surface.py"
Cohesion: 0.27
Nodes (10): calculate_multileg_payoff(), generate_volatility_surface_mesh(), Any, 3D Implied Volatility Surface & Multi-Leg Options Strategy Desk for Sentilyze.…, Constructs a 3D Implied Volatility Surface across strike prices and expiration…, Calculates profit and loss (P&L) curves at expiration for institutional multi-…, test_calculate_multileg_payoff_bull_call_spread(), test_calculate_multileg_payoff_iron_condor() (+2 more)

### Community 33 - ".get_closed_trades_df"
Cohesion: 0.29
Nodes (4): DataFrame, Returns a DataFrame of current open holdings with Scale-Out status., Returns a DataFrame of trade history with full company names., Returns equity history as a DatetimeIndex DataFrame.

### Community 34 - "compute_cross_asset_correlation"
Cohesion: 0.40
Nodes (6): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…

### Community 35 - "preprocess_data"
Cohesion: 0.13
Nodes (20): main(), Batch Universe Trainer for Remaining S&P 100 Tickers., run_single(), enrich_features_with_alpha_interactions(), execute_continuous_retrain_cycle(), Any, DataFrame, Continuous Model Self-Training & Accuracy Boosting Engine for Sentilyze. Self-… (+12 more)

### Community 36 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (16): Any, DataFrame, ndarray, Series, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.…, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used. (+8 more)

### Community 37 - "TickerSentinelSwarm"
Cohesion: 0.14
Nodes (14): detect_peak_crest_exhaustion(), Any, Dedicated Micro-Agent assigned to monitor a single stock position 24/7., Audits live price tick and determines peak crest execution., Manages the full swarm of Dedicated Ticker Sentinels across all open positions., Synchronizes active sentinels with current portfolio open positions., Audits all active sentinels concurrently., Detects if a stock has reached the crest/peak of its 15-minute momentum wave… (+6 more)

### Community 38 - "Sentilyze — Systematic Sentiment & Momentum Trading Research Platform"
Cohesion: 0.10
Nodes (19): 1. Installation, **1. Multi-Ticker Empirical Attribution Decomposition (50 Monte Carlo Trials per Asset)**, **2. 4-Agent Committee Ablation Matrix (400-Day Out-of-Sample Horizon)**, 2. Running the Streamlit Dashboard, 3. Running the FastAPI REST Microservice, 4. Running the Test Suite & Attribution Engine, 🧪 4-Year Sizing & Management Benchmark ($100,000 Capital), ⚡ Asymmetric Trade Execution Mechanics (+11 more)

### Community 39 - "autonomous_trader.py"
Cohesion: 0.15
Nodes (14): is_kill_switch_active(), load_universe_tickers(), Autonomous Live Trading & News Intelligence Engine for Sentilyze. Institutional…, Task 7: Master Kill Switch Check. Returns True if SENTILYZE_KILL_SWITCH…, Loads universe of tickers from stocks.txt., Runs the Autonomous Trading Engine continuously on an interval., run_autonomous_daemon(), patch (+6 more)

### Community 40 - "render_workspace_header"
Cohesion: 0.13
Nodes (14): get_market_status(), Any, Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Wraps HTML content inside an institutional frosted glass container., Calculates live US Market (NYSE/NASDAQ) status based on Eastern Time., Renders an executive header banner with live status badge and market clock., render_glass_card(), render_workspace_header() (+6 more)

### Community 42 - "strategy_optimizer.py"
Cohesion: 0.50
Nodes (3): Any, Fast vectorized backtesting simulation sandbox for custom leverage, confidence,…, simulate_strategy_sandbox()

### Community 43 - "reddit_premarket_station.py"
Cohesion: 0.25
Nodes (12): fetch_4station_premarket_intelligence(), _fetch_subreddit_rss_entries(), Any, Systematic 4-Station 1-Day-Prior Reddit Market Intelligence Engine. Pillar 2…, Calculates ticker mentions and sentiment within a specific Reddit station., Orchestrates real-time 1-day-prior intelligence across all 4 key Reddit…, Fetches real-time Atom RSS feed for a subreddit using safe defusedxml., scrape_station_ticker_sentiment() (+4 more)

### Community 47 - "Contributor Covenant Code of Conduct"
Cohesion: 0.15
Nodes (12): 1. Correction, 2. Warning, 3. Temporary Ban, 4. Permanent Ban, Attribution, Contributor Covenant Code of Conduct, Enforcement, Enforcement Guidelines (+4 more)

### Community 48 - "temp_data_dir"
Cohesion: 0.67
Nodes (3): fixture, Fixture to set a temporary data directory for tests., temp_data_dir()

### Community 49 - "calculate_doubling_progress"
Cohesion: 0.27
Nodes (9): calculate_doubling_progress(), compute_compound_position_size(), Any, Max Compound Acceleration & +100% Target Doubling Engine for Sentilyze.…, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Unit tests for Max Compound Acceleration Engine., test_calculate_doubling_progress() (+1 more)

### Community 53 - "How Can I Contribute?"
Cohesion: 0.33
Nodes (5): Contributing to Sentilyze, How Can I Contribute?, Pull Requests, Reporting Bugs, Suggesting Enhancements

### Community 54 - "audio_briefing.py"
Cohesion: 0.47
Nodes (5): generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses gTTS if available, or…, synthesize_morning_audio()

### Community 55 - "🧪 Experimental & Simulated Research Prototypes"
Cohesion: 0.50
Nodes (3): 🧪 Experimental & Simulated Research Prototypes, 🔒 Production Isolation Guarantee, 📁 Prototype Inventory

## Knowledge Gaps
- **31 isolated node(s):** `graphify`, `Workflow: graphify`, `Our Pledge`, `Our Standards`, `Enforcement Responsibilities` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `get_logger` to `quant_engine.py`, `get_price_history`, `test_omnichannel_mobile.py`, `run_daily_market_scan`, `run_backtest`, `ws_live_prediction.py`, `test_statistical_arbitrage.py`, `agent_committee.py`, `TradingEnvironment`, `get_sentiment`, `CloudDataLake`, `ws_alternative_data.py`, `SupplyChainGraphNetwork`, `ws_portfolio.py`, `run_temporal_fusion_forecast`, `train_meta_ensemble`, `chart_pattern_learning.py`, `compute_dark_pool_sentiment`, `test_rebalancer_and_tearsheet.py`, `safe_path_join`, `compute_lead_lag_matrix`, `black_swan_simulator.py`, `test_pillar2_alternative_data.py`, `opening_range_engine.py`, `options_surface.py`, `preprocess_data`, `SuperEnsembleClassifier`, `autonomous_trader.py`, `strategy_optimizer.py`, `reddit_premarket_station.py`, `calculate_doubling_progress`, `audio_briefing.py`?**
  _High betweenness centrality (0.235) - this node is a cross-community bridge._
- **Why does `get_price_history()` connect `get_price_history` to `compute_cross_asset_correlation`, `preprocess_data`, `run_backtest`, `run_daily_market_scan`, `app.py`, `test_statistical_arbitrage.py`, `agent_committee.py`, `render_workspace_header`, `strategy_optimizer.py`, `get_logger`, `safe_path_join`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Workflow: graphify`, `Our Pledge` to the rest of the system?**
  _31 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `quant_engine.py` be split into smaller, more focused modules?**
  _Cohesion score 0.054858934169279 - nodes in this community are weakly interconnected._
- **Should `get_price_history` be split into smaller, more focused modules?**
  _Cohesion score 0.06988120195667366 - nodes in this community are weakly interconnected._
- **Should `test_omnichannel_mobile.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._