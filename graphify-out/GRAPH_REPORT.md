# Graph Report - Sentilyze  (2026-08-30)

## Corpus Check
- 463 files · ~725,348 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1130 nodes · 2401 edges · 57 communities (56 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `397a9ae3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- fetch_financial_statements
- get_price_history
- options_flow.py
- daily_scanner.py
- run_backtest
- app.py
- smart_trader_engine.py
- test_statistical_arbitrage.py
- agent_committee.py
- TradingEnvironment
- preprocess_data
- CloudDataLake
- ws_alternative_data.py
- SupplyChainGraphNetwork
- ws_portfolio.py
- run_temporal_fusion_forecast
- quant_engine.py
- train_meta_ensemble
- ._save
- ws_live_prediction.py
- realtime_tracker.py
- compute_dark_pool_sentiment
- test_omnichannel_mobile.py
- utils.py
- handle_bot_command
- compute_lead_lag_matrix
- black_swan_simulator.py
- test_pillar2_alternative_data.py
- AlpacaBrokerBridge
- telegram_bot.py
- PaperBroker
- datetime
- options_surface.py
- AICopilotEngine
- liquidity_heatmap.py
- run_committee_ablation_backtest
- SuperEnsembleClassifier
- TickerSentinelSwarm
- Sentilyze — Systematic Sentiment & Momentum Trading Research Platform
- test_autonomous_trader.py
- components.py
- test_rebalancer_and_tearsheet.py
- AutonomousTradingEngine
- reddit_premarket_station.py
- ui/__init__.py
- Contributor Covenant Code of Conduct
- calculate_15min_opening_range
- calculate_doubling_progress
- fetch_universe_live_quotes
- get_logger
- autonomous_trader.py
- How Can I Contribute?
- audio_briefing.py
- 🧪 Experimental & Simulated Research Prototypes
- generate_smartwatch_glance_payload

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 69 edges
2. `PaperBroker` - 41 edges
3. `get_price_history()` - 35 edges
4. `run_unified_institutional_pipeline()` - 32 edges
5. `fetch_live_quote()` - 29 edges
6. `preprocess_data()` - 26 edges
7. `get_news()` - 24 edges
8. `fetch_financial_statements()` - 23 edges
9. `handle_telegram_command()` - 23 edges
10. `render_workspace_header()` - 22 edges

## Surprising Connections (you probably didn't know these)
- `test_autonomous_cycle_execution()` --uses--> `AutonomousTradingEngine`  [INFERRED]
  tests/test_autonomous_trader.py → src/autonomous_trader.py
- `test_autonomous_cycle_execution()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_idempotency_lock_prevents_overlap()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `predict()` --calls--> `preprocess_data()`  [EXTRACTED]
  api.py → src/preprocessing.py
- `main()` --calls--> `render_alternative_data_workspace()`  [EXTRACTED]
  app.py → src/ui/ws_alternative_data.py

## Import Cycles
- None detected.

## Communities (57 total, 1 thin omitted)

### Community 0 - "fetch_financial_statements"
Cohesion: 0.18
Nodes (22): calculate_altman_z_score(), calculate_dcf_fair_value(), calculate_piotroski_f_score(), fetch_financial_statements(), _generate_calibrated_financials(), generate_spider_radar_profile(), Any, Institutional Fundamental Valuation & Forensic Accounting Engine for Sentilyze.… (+14 more)

### Community 1 - "get_price_history"
Cohesion: 0.05
Nodes (62): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…, _fetch_alpaca_news(), _fetch_alpaca_price_history() (+54 more)

### Community 2 - "options_flow.py"
Cohesion: 0.17
Nodes (21): calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain(), Any, DataFrame, Live Options Microstructure, Gamma Exposure (GEX) & Max Pain Terminal for… (+13 more)

### Community 3 - "daily_scanner.py"
Cohesion: 0.11
Nodes (34): format_signal_card(), Any, Dispatches a high-impact Discord card for live autonomous trade lifecycle…, Construct a standardized trade signal data payload. Args: ticker (str): Stock…, Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord., Dispatches real-time Reddit r/wallstreetbets & Stocktwits hype spike alerts., Sends a consolidated morning macro regime and portfolio health pulse to Discord., Sends a consolidated master market digest card containing all universe signals. (+26 more)

### Community 4 - "run_backtest"
Cohesion: 0.09
Nodes (37): Figure, _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition(), calculate_performance_metrics(), _calculate_trade_outcomes() (+29 more)

### Community 5 - "app.py"
Cohesion: 0.11
Nodes (26): load_universe_tickers(), main(), Sentilyze - Institutional Algorithmic Trading & MLOps Platform. Modular Master…, Loads active S&P 100 universe tickers., get_market_status(), Any, Calculates live US Market (NYSE/NASDAQ) status based on Eastern Time., Renders an executive header banner with live status badge and market clock. (+18 more)

### Community 6 - "smart_trader_engine.py"
Cohesion: 0.15
Nodes (21): apply_high_watermark_profit_lock(), calculate_smart_money_zones(), calculate_structural_trailing_stop(), evaluate_multi_timeframe_confluence(), find_swing_pivots(), Any, DataFrame, Institutional Smart Money Market Structure & Price-Action Engine for Sentilyze.… (+13 more)

### Community 7 - "test_statistical_arbitrage.py"
Cohesion: 0.19
Nodes (25): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any, Series (+17 more)

### Community 8 - "agent_committee.py"
Cohesion: 0.13
Nodes (26): 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, audit_full_universe_committee(), ChiefRiskOfficerAgent, compute_fractional_kelly_sizing(), convene_trading_committee(), ForensicFundamentalAgent, _persist_committee_resolution(), Any (+18 more)

### Community 9 - "TradingEnvironment"
Cohesion: 0.14
Nodes (16): optimize_rl_position_allocation(), PPOPolicyAgent, Any, ndarray, ⚠️ EXPERIMENTAL / RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM PRODUCTION…, Computes mean action (leverage) between 0.0 and 2.0., Estimates state value., Trains Actor-Critic parameters across historical episodes. (+8 more)

### Community 10 - "preprocess_data"
Cohesion: 0.05
Nodes (59): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, clean_headline_data() (+51 more)

### Community 11 - "CloudDataLake"
Cohesion: 0.13
Nodes (15): CloudDataLake, Any, Cloud PostgreSQL Data Lake (Supabase / Neon) Connector for Sentilyze. Pillar 6…, Supabase / PostgreSQL Cloud Data Lake Connector., Validates or generates cloud database schema., Syncs local trade executions to the cloud database., Publishes real-time portfolio snapshot to cloud WebSockets channel., generate_twap_order_schedule() (+7 more)

### Community 12 - "ws_alternative_data.py"
Cohesion: 0.15
Nodes (18): auto_register_ipo_ticker(), fetch_pre_ipo_radar_summary(), fetch_sec_edgar_ipo_filings(), get_pre_ipo_pipeline_df(), Any, DataFrame, IPO & Pre-IPO Intelligence Radar for Sentilyze. Pillar 9 Alternative Asset…, Fetches real-time SEC Form S-1 / S-1/A IPO registration statements from SEC… (+10 more)

### Community 13 - "SupplyChainGraphNetwork"
Cohesion: 0.15
Nodes (14): analyze_supply_chain_spillover(), Any, ndarray, Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for…, Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)., Executes a Graph Convolutional Network (GCN) layer: H_new = ReLU(A_hat * H * W)…, Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab…, High-level entry point to run GNN supply chain shock propagation. (+6 more)

### Community 14 - "ws_portfolio.py"
Cohesion: 0.17
Nodes (18): build_unified_portfolio(), calculate_risk_parity_weights(), load_all_ticker_portfolios(), Any, DataFrame, Series, Combines individual stock strategies into a single managed multi-asset fund.…, Load individual backtest portfolio CSVs for all available tickers. Args:… (+10 more)

### Community 15 - "run_temporal_fusion_forecast"
Cohesion: 0.14
Nodes (15): Any, DataFrame, ndarray, Temporal Fusion Transformer (TFT) & Multi-Horizon Self-Attention Engine for…, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with… (+7 more)

### Community 16 - "quant_engine.py"
Cohesion: 0.15
Nodes (17): MasterQuantPipelineResult, Any, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), analyze_debt_maturity_wall(), calculate_beneish_m_score() (+9 more)

### Community 17 - "train_meta_ensemble"
Cohesion: 0.16
Nodes (14): MetaEnsembleClassifier, DataFrame, ndarray, Series, Institutional Multi-Model Meta-Ensemble Engine for Sentilyze. Pillar 1 Core…, Generates binary class prediction (0 = Hold/Sell, 1 = Buy) using soft-voting…, Instantiates and fits the Meta-Ensemble classifier., Multi-Model Meta-Ensemble stacking XGBoost, Random Forest, and Calibrated… (+6 more)

### Community 18 - "._save"
Cohesion: 0.14
Nodes (10): Any, Loads existing portfolio state from JSON or initializes a fresh $100k account., Updates total equity, unrealized PnL, and win rates., Returns high-level KPI metrics for the portfolio dashboard., Executes an immediate manual live/simulated BUY order from UI., Alias for _save to ensure 100% backward compatibility., Executes an immediate manual live/simulated exit of an open position., Executes a 50% scale-out on an open position and moves stop to break-even. (+2 more)

### Community 19 - "ws_live_prediction.py"
Cohesion: 0.17
Nodes (20): detect_classical_chart_patterns(), generate_ai_chart_explanation(), match_historical_chart_twins(), normalize_waveform(), Any, DataFrame, ndarray, AI Chart Pattern Recognition, Geometric Wave Learning & Visual Understanding… (+12 more)

### Community 20 - "realtime_tracker.py"
Cohesion: 0.12
Nodes (26): AI Trade Copilot & Conversational Analyst for Sentilyze. Provides natural…, check_live_news_sentiment_shock(), evaluate_intraday_execution(), fetch_live_quote(), _get_browser_session(), get_us_market_session_info(), Any, Session (+18 more)

### Community 21 - "compute_dark_pool_sentiment"
Cohesion: 0.21
Nodes (14): compute_dark_pool_sentiment(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent institutional off-exchange block trades and dark pool prints., Scans option chain contracts where daily volume significantly exceeds open…, Synthesizes dark pool prints and unusual options flow into a unified…, scan_abnormal_options_vol_oi(), scan_dark_pool_blocks() (+6 more)

### Community 22 - "test_omnichannel_mobile.py"
Cohesion: 0.19
Nodes (12): answer_financial_query(), Any, Natural Language Financial Q&A Agent for Sentilyze. Pillar 7 Mobile &…, Parses natural language questions and routes them to quantitative engines.…, format_whatsapp_trade_alert(), Any, WhatsApp Push Notifications & Execution Receipts for Sentilyze. Pillar 7 Mobile…, Constructs a formatted WhatsApp messaging receipt. (+4 more)

### Community 23 - "utils.py"
Cohesion: 0.07
Nodes (45): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+37 more)

### Community 24 - "handle_bot_command"
Cohesion: 0.40
Nodes (5): handle_bot_command(), Any, Executes command and posts formatted embed reply to Discord., Parses and processes interactive bot commands: - `/signal <ticker>` -…, send_bot_command_reply()

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
Cohesion: 0.22
Nodes (7): AlpacaBrokerBridge, Any, Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.…, Fetches active positions from Alpaca brokerage., Verifies active connection to Alpaca Brokerage API., Fetches live Alpaca account equity, buying power, and cash., Submits an institutional Bracket Order: - Entry: Market order - Exit 1: Limit…

### Community 29 - "telegram_bot.py"
Cohesion: 0.20
Nodes (16): build_interactive_inline_keyboard(), handle_telegram_command(), Any, 2-Way Interactive Telegram Bot Controller & Remote Execution Bridge for…, Sends a formatted markdown message with inline buttons to a Telegram chat., Builds interactive inline quick-action buttons for mobile Telegram., Parses and executes Telegram slash commands and callback buttons., send_telegram_bot_message() (+8 more)

### Community 30 - "PaperBroker"
Cohesion: 0.16
Nodes (14): PaperBroker, DataFrame, Institutional Multi-Stage Quantitative Execution Broker ($100k Account).…, Returns a DataFrame of current open holdings with Scale-Out status., Returns a DataFrame of trade history with full company names., Returns equity history as a DatetimeIndex DataFrame., fixture, temp_portfolio_file() (+6 more)

### Community 31 - "datetime"
Cohesion: 0.20
Nodes (13): datetime, check_market_hours_preflight(), get_current_ny_time(), get_us_market_session(), Any, Unified US Stock Market (NYSE / NASDAQ) Session & Calendar Engine for…, Pre-flight sanity check for automated workflows. Returns True if execution…, Returns the current precise timestamp in America/New_York (Eastern Time). (+5 more)

### Community 32 - "options_surface.py"
Cohesion: 0.27
Nodes (10): calculate_multileg_payoff(), generate_volatility_surface_mesh(), Any, 3D Implied Volatility Surface & Multi-Leg Options Strategy Desk for Sentilyze.…, Constructs a 3D Implied Volatility Surface across strike prices and expiration…, Calculates profit and loss (P&L) curves at expiration for institutional multi-…, test_calculate_multileg_payoff_bull_call_spread(), test_calculate_multileg_payoff_iron_condor() (+2 more)

### Community 33 - "AICopilotEngine"
Cohesion: 0.24
Nodes (8): AICopilotEngine, Any, Conversational intelligence engine that parses queries and generates analytical…, Interprets user prompt and routes to appropriate financial analytical…, test_copilot_committee_query(), test_copilot_portfolio_query(), test_copilot_stress_query(), test_copilot_ticker_analysis_query()

### Community 34 - "liquidity_heatmap.py"
Cohesion: 0.31
Nodes (8): compute_order_book_depth_and_clusters(), compute_volume_profile_and_poc(), Any, Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap for…, Simulates Level 2 market depth and identifies institutional buy/sell liquidity…, Computes Point of Control (POC), Value Area High (VAH), and Value Area Low…, test_compute_order_book_depth_and_clusters(), test_compute_volume_profile_and_poc()

### Community 35 - "run_committee_ablation_backtest"
Cohesion: 0.36
Nodes (7): _persist_ablation_results(), Any, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), test_committee_ablation_study_execution()

### Community 36 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (16): Any, DataFrame, ndarray, Series, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.…, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used. (+8 more)

### Community 37 - "TickerSentinelSwarm"
Cohesion: 0.14
Nodes (15): detect_peak_crest_exhaustion(), Any, Dedicated Ticker Sentinel & Peak-Crest Volume Harvester Swarm for Sentilyze.…, Dedicated Micro-Agent assigned to monitor a single stock position 24/7., Audits live price tick and determines peak crest execution., Manages the full swarm of Dedicated Ticker Sentinels across all open positions., Synchronizes active sentinels with current portfolio open positions., Audits all active sentinels concurrently. (+7 more)

### Community 38 - "Sentilyze — Systematic Sentiment & Momentum Trading Research Platform"
Cohesion: 0.10
Nodes (19): 1. Installation, **1. Multi-Ticker Empirical Attribution Decomposition (50 Monte Carlo Trials per Asset)**, **2. 4-Agent Committee Ablation Matrix (500-Day Out-of-Sample Horizon)**, 2. Running the Streamlit Dashboard, 3. Running the FastAPI REST Microservice, 4. Running the Test Suite & Attribution Engine, 🧪 4-Year Sizing & Management Benchmark ($100,000 Capital), ⚡ Asymmetric Trade Execution Mechanics (+11 more)

### Community 39 - "test_autonomous_trader.py"
Cohesion: 0.13
Nodes (15): is_kill_switch_active(), load_universe_tickers(), Task 7: Master Kill Switch Check. Returns True if SENTILYZE_KILL_SWITCH…, Loads universe of tickers from stocks.txt., patch, Task 8: Verify circuit breaker triggers when true daily drawdown exceeds…, Task 9: Verify unhandled exception in cycle is caught and handled safely., Task 6: Verify active lock file prevents overlapping concurrent cycles. (+7 more)

### Community 40 - "components.py"
Cohesion: 0.24
Nodes (6): Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Wraps HTML content inside an institutional frosted glass container., Renders a progress meter with dynamic color coding., render_conviction_gauge(), render_glass_card(), Workspace 2: 4-Agent Trading Committee Round-Table Deliberations.

### Community 41 - "test_rebalancer_and_tearsheet.py"
Cohesion: 0.19
Nodes (12): Any, DataFrame, Helper wrapper for Monte Carlo VaR simulation., Runs an institutional Monte Carlo forward stress test and Value-at-Risk (VaR)…, run_monte_carlo_stress_test(), run_monte_carlo_var(), generate_executive_pdf_tearsheet(), Any (+4 more)

### Community 42 - "AutonomousTradingEngine"
Cohesion: 0.23
Nodes (9): AutonomousTradingEngine, check_daily_loss_circuit_breaker(), Any, Autonomous Execution Engine that integrates Live News Ingestion, 4-Agent…, Dispatches an institutional execution alert to Discord Webhook if configured., Executes one full autonomous decision and execution cycle with: - Task 6:…, Core cycle execution body., Task 8: Independent Max-Daily-Loss Circuit Breaker. Compares current total… (+1 more)

### Community 43 - "reddit_premarket_station.py"
Cohesion: 0.25
Nodes (12): fetch_4station_premarket_intelligence(), _fetch_subreddit_rss_entries(), Any, Systematic 4-Station 1-Day-Prior Reddit Market Intelligence Engine. Pillar 2…, Calculates ticker mentions and sentiment within a specific Reddit station., Orchestrates real-time 1-day-prior intelligence across all 4 key Reddit…, Fetches real-time Atom RSS feed for a subreddit using safe defusedxml., scrape_station_ticker_sentiment() (+4 more)

### Community 47 - "Contributor Covenant Code of Conduct"
Cohesion: 0.15
Nodes (12): 1. Correction, 2. Warning, 3. Temporary Ban, 4. Permanent Ban, Attribution, Contributor Covenant Code of Conduct, Enforcement, Enforcement Guidelines (+4 more)

### Community 48 - "calculate_15min_opening_range"
Cohesion: 0.26
Nodes (11): calculate_15min_opening_range(), find_low_of_day_pullback_entry(), is_opening_15min_whipsaw_period(), Any, DataFrame, Checks if current Eastern Time is within the hectic 09:30 - 09:45 EDT opening…, Calculates the 15-minute Opening Range (High, Low, Midpoint) established…, Evaluates whether a stock is in the optimal 'Low-of-Day Pullback & Volume… (+3 more)

### Community 49 - "calculate_doubling_progress"
Cohesion: 0.27
Nodes (9): calculate_doubling_progress(), compute_compound_position_size(), Any, Max Compound Acceleration & +100% Target Doubling Engine for Sentilyze.…, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Unit tests for Max Compound Acceleration Engine., test_calculate_doubling_progress() (+1 more)

### Community 50 - "fetch_universe_live_quotes"
Cohesion: 0.31
Nodes (8): fetch_universe_live_quotes(), Fetches real-time quotes across the entire universe concurrently in parallel., calculate_custom_rebalance(), calculate_share_allocation(), Any, Helper to calculate share allocation from latest daily signals file or universe…, Computes exact whole-share buy allocations for a given capital budget across…, test_calculate_share_allocation()

### Community 51 - "get_logger"
Cohesion: 0.40
Nodes (5): Logger, get_logger(), Configures and returns a logger with a standard format and utf-8 safe…, Tests that the get_logger function returns a configured logger., test_get_logger()

### Community 52 - "autonomous_trader.py"
Cohesion: 0.33
Nodes (5): execute_committee_order(), Executes a committee-approved buy order into the virtual paper broker ledger., Autonomous Live Trading & News Intelligence Engine for Sentilyze. Institutional…, Runs the Autonomous Trading Engine continuously on an interval., run_autonomous_daemon()

### Community 53 - "How Can I Contribute?"
Cohesion: 0.33
Nodes (5): Contributing to Sentilyze, How Can I Contribute?, Pull Requests, Reporting Bugs, Suggesting Enhancements

### Community 54 - "audio_briefing.py"
Cohesion: 0.47
Nodes (5): generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses gTTS if available, or…, synthesize_morning_audio()

### Community 55 - "🧪 Experimental & Simulated Research Prototypes"
Cohesion: 0.50
Nodes (3): 🧪 Experimental & Simulated Research Prototypes, 🔒 Production Isolation Guarantee, 📁 Prototype Inventory

### Community 56 - "generate_smartwatch_glance_payload"
Cohesion: 0.33
Nodes (5): generate_smartwatch_glance_payload(), Any, Apple Watch & Wear OS Glance Complications API for Sentilyze. Pillar 7 Mobile &…, Generates structured complication JSON for Apple Watch (watchOS) and Wear OS., test_smartwatch_api()

## Knowledge Gaps
- **29 isolated node(s):** `Our Pledge`, `Our Standards`, `Enforcement Responsibilities`, `Scope`, `Enforcement` (+24 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `get_logger` to `fetch_financial_statements`, `get_price_history`, `options_flow.py`, `daily_scanner.py`, `run_backtest`, `smart_trader_engine.py`, `test_statistical_arbitrage.py`, `agent_committee.py`, `TradingEnvironment`, `preprocess_data`, `CloudDataLake`, `ws_alternative_data.py`, `SupplyChainGraphNetwork`, `ws_portfolio.py`, `run_temporal_fusion_forecast`, `quant_engine.py`, `train_meta_ensemble`, `ws_live_prediction.py`, `realtime_tracker.py`, `compute_dark_pool_sentiment`, `test_omnichannel_mobile.py`, `utils.py`, `compute_lead_lag_matrix`, `black_swan_simulator.py`, `test_pillar2_alternative_data.py`, `AlpacaBrokerBridge`, `telegram_bot.py`, `datetime`, `options_surface.py`, `liquidity_heatmap.py`, `SuperEnsembleClassifier`, `TickerSentinelSwarm`, `test_rebalancer_and_tearsheet.py`, `reddit_premarket_station.py`, `calculate_doubling_progress`, `fetch_universe_live_quotes`, `autonomous_trader.py`, `audio_briefing.py`, `generate_smartwatch_glance_payload`?**
  _High betweenness centrality (0.270) - this node is a cross-community bridge._
- **Why does `PaperBroker` connect `PaperBroker` to `AICopilotEngine`, `daily_scanner.py`, `test_autonomous_trader.py`, `AutonomousTradingEngine`, `._save`, `realtime_tracker.py`, `autonomous_trader.py`, `handle_bot_command`, `telegram_bot.py`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Our Pledge`, `Our Standards`, `Enforcement Responsibilities` to the rest of the system?**
  _29 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `get_price_history` be split into smaller, more focused modules?**
  _Cohesion score 0.05267778753292362 - nodes in this community are weakly interconnected._
- **Should `daily_scanner.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11201079622132254 - nodes in this community are weakly interconnected._
- **Should `run_backtest` be split into smaller, more focused modules?**
  _Cohesion score 0.09146341463414634 - nodes in this community are weakly interconnected._