# Graph Report - Sentilyze  (2026-09-03)

## Corpus Check
- 543 files · ~748,869 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1696 nodes · 3570 edges · 100 communities (96 shown, 4 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6c347466`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- deep_learning_model.py
- data_ingestion.py
- test_options_flow.py
- daily_scanner.py
- run_backtest
- drl_policy_agent.py
- test_all_14_papers.py
- test_statistical_arbitrage.py
- fetch_financial_statements
- TradingEnvironment
- benchmark_papers_15_24.py
- CloudDataLake
- ws_alternative_data.py
- analyze_supply_chain_spillover
- calculate_hrp_weights
- run_temporal_fusion_forecast
- PaperBroker
- meta_ensemble.py
- OnlineNewtonStepOptimizer
- ws_live_prediction.py
- social_sentiment.py
- compute_dark_pool_sentiment
- test_omnichannel_mobile.py
- test_rebalancer_and_tearsheet.py
- utils.py
- compute_lead_lag_matrix
- black_swan_simulator.py
- ipo_radar.py
- AlpacaBrokerBridge
- realtime_tracker.py
- ._save
- datetime
- get_prediction_on_latest_data
- AICopilotEngine
- PolyTimeConvexOptimizer
- preprocessing.py
- SuperEnsembleClassifier
- triple_convex_engine.py
- Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform
- autonomous_trader.py
- app.py
- TickerSentinelSwarm
- discord_bot.py
- analyze_sec_filing_diff
- ui/__init__.py
- Contributor Covenant Code of Conduct
- get_sentiment
- calculate_doubling_progress
- test_pillar2_alternative_data.py
- get_logger
- strategy_incubator.py
- How Can I Contribute?
- run_daily_market_scan
- 🧪 Experimental & Simulated Research Prototypes
- Sentilyze — Standing Audit Protocol
- webhook_dispatcher.py
- get_vix_data
- _load_sentiment_analyzer
- preprocess_data
- quant_engine.py
- rules/graphify.md
- workflows/graphify.md
- GaussianHMMRegimeDetector
- EWMACorrelationMonitor
- OpeningRangeBreakout
- CUSUMDetector
- PageHinkleyDetector
- macro_liquidity.py
- insider_signals.py
- get_news
- test_api.py
- mega_tournament_simulation.py
- ADWINDetector
- test_papers_15_24.py
- morning_briefing.py
- agent_committee.py
- run_cppi_backtest
- DCCCorrelation
- model_ensemble.py
- generate_comprehensive_factsheet
- risk_constrained_kelly_allocation
- render_workspace_header
- calculate_portfolio_diversity_grade
- ws_quantum_tournament.py
- audio_briefing.py
- main
- correlation_matrix.py
- .get_closed_trades_df
- cli.py
- block_external_alerts
- get_price_history
- fetch_live_quote
- options_surface.py
- calculate_beneish_m_score
- execute_continuous_retrain_cycle
- run_opening_range_session
- _get_browser_session

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 92 edges
2. `get_price_history()` - 58 edges
3. `PaperBroker` - 53 edges
4. `fetch_live_quote()` - 35 edges
5. `run_unified_institutional_pipeline()` - 32 edges
6. `get_news()` - 28 edges
7. `preprocess_data()` - 27 edges
8. `render_workspace_header()` - 24 edges
9. `main()` - 23 edges
10. `convene_trading_committee()` - 22 edges

## Surprising Connections (you probably didn't know these)
- `test_paper7_quant_agents_trader()` --calls--> `AutonomousTradingEngine`  [EXTRACTED]
  tests/test_all_14_papers.py → src/autonomous_trader.py
- `test_paper9_when_agents_trade_scanner()` --calls--> `run_daily_market_scan()`  [EXTRACTED]
  tests/test_all_14_papers.py → src/daily_scanner.py
- `test_autonomous_cycle_execution()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_idempotency_lock_prevents_overlap()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_paper12_hierarchical_risk_parity()` --calls--> `calculate_hrp_weights()`  [EXTRACTED]
  tests/test_all_14_papers.py → src/portfolio.py

## Import Cycles
- None detected.

## Communities (100 total, 4 thin omitted)

### Community 0 - "deep_learning_model.py"
Cohesion: 0.10
Nodes (28): create_sliding_window_tensors(), DLinearTCNModel, load_dlinear_model(), predict_momentum_probability(), Any, DataFrame, Tensor, High-Efficiency Deep Learning Engine: DLinear + Temporal Convolutional Network… (+20 more)

### Community 1 - "data_ingestion.py"
Cohesion: 0.12
Nodes (23): _fetch_alpaca_news(), _fetch_alpaca_price_history(), _fetch_eodhd_price_history(), _fetch_finnhub_news(), _fetch_fmp_price_history(), _fetch_google_news_rss(), _fetch_marketaux_news(), _fetch_polygon_news_feed() (+15 more)

### Community 2 - "test_options_flow.py"
Cohesion: 0.17
Nodes (21): calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain(), Any, DataFrame, Live Options Microstructure, Gamma Exposure (GEX) & Max Pain Terminal for… (+13 more)

### Community 3 - "daily_scanner.py"
Cohesion: 0.15
Nodes (28): format_signal_card(), Any, Dispatches a crystal-clear, high-impact Discord card for live autonomous trade…, Construct a standardized trade signal data payload. Args: ticker (str): Stock…, Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord., Dispatches real-time Reddit r/wallstreetbets & Stocktwits hype spike alerts., Sends a comprehensive institutional morning macro regime, portfolio health,…, Sends a rich formatted trade alert card to a Discord channel via Webhook. (+20 more)

### Community 4 - "run_backtest"
Cohesion: 0.10
Nodes (35): Figure, _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition(), calculate_performance_metrics(), _calculate_trade_outcomes() (+27 more)

### Community 5 - "drl_policy_agent.py"
Cohesion: 0.14
Nodes (18): ActorCriticPolicy, DRLTradingEnvironment, evaluate_drl_policy_action(), Any, ndarray, Tensor, Deep Reinforcement Learning (DRL) Autonomous Policy Agent for Sentilyze.…, Trains an Actor-Critic DRL policy agent on historical market returns and news… (+10 more)

### Community 6 - "test_all_14_papers.py"
Cohesion: 0.09
Nodes (28): Any, Master Academic Research Papers Empirical Benchmark Suite (All 14 Papers).…, Executes empirical backtests comparing all 14 academic paper methodologies., run_all_14_papers_benchmark(), calculate_almgren_chriss_trajectory(), Any, Paper 3: Almgren & Chriss (2000) - Optimal Execution of Portfolio Transactions.…, Computes Almgren-Chriss optimal trading trajectory. x_j = 2 * sinh(0.5 * kappa… (+20 more)

### Community 7 - "test_statistical_arbitrage.py"
Cohesion: 0.19
Nodes (27): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any, Series (+19 more)

### Community 8 - "fetch_financial_statements"
Cohesion: 0.18
Nodes (22): calculate_altman_z_score(), calculate_dcf_fair_value(), calculate_piotroski_f_score(), fetch_financial_statements(), _generate_calibrated_financials(), generate_spider_radar_profile(), Any, Institutional Fundamental Valuation & Forensic Accounting Engine for Sentilyze.… (+14 more)

### Community 9 - "TradingEnvironment"
Cohesion: 0.14
Nodes (16): optimize_rl_position_allocation(), PPOPolicyAgent, Any, ndarray, ⚠️ EXPERIMENTAL / RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM PRODUCTION…, Computes mean action (leverage) between 0.0 and 2.0., Estimates state value., Trains Actor-Critic parameters across historical episodes. (+8 more)

### Community 10 - "benchmark_papers_15_24.py"
Cohesion: 0.11
Nodes (25): benchmark_adwin(), benchmark_cdar(), benchmark_cppi(), benchmark_cusum(), benchmark_dcc(), benchmark_ewma(), benchmark_grossman_zhou(), benchmark_hmm() (+17 more)

### Community 11 - "CloudDataLake"
Cohesion: 0.13
Nodes (15): CloudDataLake, Any, Cloud PostgreSQL Data Lake (Supabase / Neon) Connector for Sentilyze. Pillar 6…, Supabase / PostgreSQL Cloud Data Lake Connector., Validates or generates cloud database schema., Syncs local trade executions to the cloud database., Publishes real-time portfolio snapshot to cloud WebSockets channel., generate_twap_order_schedule() (+7 more)

### Community 12 - "ws_alternative_data.py"
Cohesion: 0.16
Nodes (18): get_pre_ipo_pipeline_df(), DataFrame, Returns a formatted pandas DataFrame of all pre-IPO target assets., fetch_4station_premarket_intelligence(), _fetch_subreddit_rss_entries(), Any, Systematic 4-Station 1-Day-Prior Reddit Market Intelligence Engine. Pillar 2…, Calculates ticker mentions and sentiment within a specific Reddit station. (+10 more)

### Community 13 - "analyze_supply_chain_spillover"
Cohesion: 0.14
Nodes (15): analyze_supply_chain_spillover(), Any, ndarray, Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for…, Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)., Executes a Graph Convolutional Network (GCN) layer: H_new = ReLU(A_hat * H * W)…, Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab…, High-level entry point to run GNN supply chain shock propagation. (+7 more)

### Community 14 - "calculate_hrp_weights"
Cohesion: 0.08
Nodes (39): compute_performance_metrics(), Any, DataFrame, Series, Empirical Quant Experimentation & Multi-Asset Ablation Benchmark Suite.…, Executes empirical ablation benchmark across the full asset universe., Simulates walk-forward strategy execution with or without advanced quant…, Computes key quant performance metrics. (+31 more)

### Community 15 - "run_temporal_fusion_forecast"
Cohesion: 0.14
Nodes (15): Any, DataFrame, ndarray, Temporal Fusion Transformer (TFT) & Multi-Horizon Self-Attention Engine for…, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with… (+7 more)

### Community 16 - "PaperBroker"
Cohesion: 0.19
Nodes (12): PaperBroker, Institutional Multi-Stage Quantitative Execution Broker ($100k Account).…, Alias for _save to ensure 100% backward compatibility., fixture, temp_portfolio_file(), test_paper_broker_dataframes(), test_paper_broker_execute_buy_signals(), test_paper_broker_initialization() (+4 more)

### Community 17 - "meta_ensemble.py"
Cohesion: 0.10
Nodes (19): DynamicSharpeMetaEnsemble, MetaEnsembleClassifier, DataFrame, ndarray, Series, Institutional Multi-Model Meta-Ensemble Engine for Sentilyze. Pillar 1 Core…, Generates binary class prediction (0 = Hold/Sell, 1 = Buy) using soft-voting…, Instantiates and fits the Meta-Ensemble classifier. (+11 more)

### Community 18 - "OnlineNewtonStepOptimizer"
Cohesion: 0.16
Nodes (11): OnlineNewtonStepOptimizer, DataFrame, ndarray, Online Newton Step (ONS) Portfolio Engine (Agarwal, Hazan, Kale). A polynomial-…, Polynomial-Time ONS Portfolio Engine (Hazan et al.)., Processes price relatives (r_t = Close_t / Close_{t-1}) and updates weights in…, Fast O(d log d) Euclidean projection onto probability simplex., Runs ONS sequence through time and outputs daily allocations and portfolio… (+3 more)

### Community 19 - "ws_live_prediction.py"
Cohesion: 0.09
Nodes (41): detect_classical_chart_patterns(), generate_ai_chart_explanation(), match_historical_chart_twins(), normalize_waveform(), Any, DataFrame, ndarray, AI Chart Pattern Recognition, Geometric Wave Learning & Visual Understanding… (+33 more)

### Community 20 - "social_sentiment.py"
Cohesion: 0.23
Nodes (13): calculate_social_buzz_metrics(), fetch_social_sentiment_tracker(), Any, Social Sentiment Velocity & Retail Multi-Platform Scraper for Sentilyze. Pillar…, Scrapes real-time streaming retail sentiment from Stocktwits public symbol…, Scrapes tech community discussions on AI catalysts (OpenAI, Anthropic, Nvidia)…, Computes retail sentiment velocity and flow conviction metrics., High-level entry point to retrieve calibrated real-time social buzz metrics for… (+5 more)

### Community 21 - "compute_dark_pool_sentiment"
Cohesion: 0.21
Nodes (14): compute_dark_pool_sentiment(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent institutional off-exchange block trades and dark pool prints., Scans option chain contracts where daily volume significantly exceeds open…, Synthesizes dark pool prints and unusual options flow into a unified…, scan_abnormal_options_vol_oi(), scan_dark_pool_blocks() (+6 more)

### Community 22 - "test_omnichannel_mobile.py"
Cohesion: 0.13
Nodes (17): answer_financial_query(), Any, Natural Language Financial Q&A Agent for Sentilyze. Pillar 7 Mobile &…, Parses natural language questions and routes them to quantitative engines.…, generate_smartwatch_glance_payload(), Any, Apple Watch & Wear OS Glance Complications API for Sentilyze. Pillar 7 Mobile &…, Generates structured complication JSON for Apple Watch (watchOS) and Wear OS. (+9 more)

### Community 23 - "test_rebalancer_and_tearsheet.py"
Cohesion: 0.18
Nodes (13): fetch_universe_live_quotes(), Fetches real-time quotes across universe with fast batching (sub-2s)., calculate_custom_rebalance(), calculate_share_allocation(), Any, Helper to calculate share allocation from latest daily signals file or universe…, Computes exact whole-share buy allocations for a given capital budget across…, generate_executive_pdf_tearsheet() (+5 more)

### Community 24 - "utils.py"
Cohesion: 0.29
Nodes (10): Continuous Model Self-Training & Accuracy Boosting Engine for Sentilyze. Self-…, load_model(), Save the trained model to a file using XGBoost's native format. This is safer…, Load a trained model from a file using XGBoost's native format. Args: filepath…, save_model(), Sanitizes user/external inputs to prevent path injection / path traversal…, Safely joins paths, ensuring the resolved target path remains strictly within…, safe_path_join() (+2 more)

### Community 25 - "compute_lead_lag_matrix"
Cohesion: 0.20
Nodes (14): compute_lead_lag_matrix(), _granger_f_test(), Any, DataFrame, ndarray, Series, rank_market_price_leaders(), Lead-Lag Granger Causality & Supply Chain Price Discovery Engine for Sentilyze.… (+6 more)

### Community 26 - "black_swan_simulator.py"
Cohesion: 0.23
Nodes (11): calculate_kelly_sizing(), estimate_market_impact_slippage(), Any, Historical Black Swan Crisis Simulator & Kelly Position Sizing for Sentilyze.…, Calculates optimal position sizing using the Kelly Criterion: Kelly % = W - (1…, Estimates market execution slippage using the Almgren-Chriss square-root impact…, Stress-tests the current portfolio against major historical market crashes.…, simulate_portfolio_crises() (+3 more)

### Community 27 - "ipo_radar.py"
Cohesion: 0.21
Nodes (12): auto_register_ipo_ticker(), fetch_pre_ipo_radar_summary(), fetch_sec_edgar_ipo_filings(), Any, IPO & Pre-IPO Intelligence Radar for Sentilyze. Pillar 9 Alternative Asset…, Fetches real-time SEC Form S-1 / S-1/A IPO registration statements from SEC…, Appends a newly public IPO ticker to stocks.txt to initiate model ingestion., High-level entry point returning the complete Pre-IPO and SEC S-1 pipeline. (+4 more)

### Community 28 - "AlpacaBrokerBridge"
Cohesion: 0.16
Nodes (14): AlpacaBrokerBridge, Any, Fetches active positions from Alpaca brokerage., Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.…, Verifies active connection to Alpaca Brokerage API., Fetches live Alpaca account equity, buying power, and cash., Submits an institutional Bracket Order: - Entry: Market order - Exit 1: Limit…, patch (+6 more)

### Community 29 - "realtime_tracker.py"
Cohesion: 0.18
Nodes (17): check_live_news_sentiment_shock(), evaluate_intraday_execution(), get_us_market_session_info(), Any, Checks if breaking news in the last few hours has a severe negative sentiment…, Autonomous 5-Minute Intraday Market Execution Engine: 1. Evaluates 50/50 Scale-…, Computes current US stock market session status (Pre-Market, Regular Hours,…, Sends immediate Discord alert when a new position is opened. (+9 more)

### Community 30 - "._save"
Cohesion: 0.15
Nodes (10): Any, Executes daily quantitative scan results using the Concentrated Top-2 + Scale-…, Loads existing portfolio state from JSON or initializes a fresh $100k account., Updates total equity, unrealized PnL, and win rates., Returns high-level KPI metrics for the portfolio dashboard., Executes an institutional BUY order into the virtual paper broker ledger.…, Executes an immediate manual live/simulated BUY order from UI., Executes an immediate manual live/simulated exit of an open position. (+2 more)

### Community 31 - "datetime"
Cohesion: 0.11
Nodes (25): datetime, check_market_hours_preflight(), get_current_ny_time(), get_us_market_session(), Any, Unified US Stock Market (NYSE / NASDAQ) Session & Calendar Engine for…, Pre-flight sanity check for automated workflows. Returns True if execution…, Returns the current precise timestamp in America/New_York (Eastern Time). (+17 more)

### Community 32 - "get_prediction_on_latest_data"
Cohesion: 0.20
Nodes (15): get_prediction_on_latest_data(), Any, DataFrame, Series, Gets a prediction from the model for the latest available data point. Args:…, Train the XGBoost model using Walk-Forward Optimization (WFO) alongside a…, train_model(), Generates sample data for testing. Needs at least 520 rows for WFO… (+7 more)

### Community 33 - "AICopilotEngine"
Cohesion: 0.21
Nodes (9): AICopilotEngine, Any, AI Trade Copilot & Conversational Analyst for Sentilyze. Provides natural…, Conversational intelligence engine that parses queries and generates analytical…, Interprets user prompt and routes to appropriate financial analytical…, test_copilot_committee_query(), test_copilot_portfolio_query(), test_copilot_stress_query() (+1 more)

### Community 34 - "PolyTimeConvexOptimizer"
Cohesion: 0.16
Nodes (9): PolyTimeConvexOptimizer, Any, DataFrame, Series, Stanford Multi-Period Convex Portfolio Optimization Engine (Boyd et al.).…, Polynomial-Time Convex Portfolio Optimizer with Market Frictions (Boyd et al.)., Solves the friction-aware convex optimization problem in polynomial time. Args:…, test_paper2_boyd_convex_optimizer() (+1 more)

### Community 35 - "preprocessing.py"
Cohesion: 0.15
Nodes (21): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, _get_api_key() (+13 more)

### Community 36 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (16): Any, DataFrame, ndarray, Series, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.…, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used. (+8 more)

### Community 37 - "triple_convex_engine.py"
Cohesion: 0.10
Nodes (24): Any, Multi-Trial Empirical Benchmark Suite for Triple-Convex Quantum Engine.…, Runs multi-trial empirical testing and saves verified metrics to JSON., run_triple_convex_multi_trial_benchmark(), apply_triple_barrier_labeling(), calculate_deflated_sharpe_ratio(), DataFrame, Series (+16 more)

### Community 38 - "Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform"
Cohesion: 0.13
Nodes (14): 1. Run in 1-Click (No Installation Required), 2. Local Setup & Installation, 3. Run the 4-Agent Quantitative CLI, 4. Launch the Streamlit Mission Control, 5. Run Full Test Suite (240+ Unit Tests), 📊 Empirical Alpha Attribution & Benchmarks, 🏛️ Grounded 4-Agent Deliberation Council, 🖥️ Interactive Streamlit App & 23 Mission Control Workspaces (+6 more)

### Community 39 - "autonomous_trader.py"
Cohesion: 0.08
Nodes (32): AutonomousTradingEngine, check_daily_loss_circuit_breaker(), is_kill_switch_active(), load_universe_tickers(), Any, Autonomous Live Trading & News Intelligence Engine for Sentilyze. Institutional…, Autonomous Execution Engine that integrates Live News Ingestion, 4-Agent…, Dispatches an institutional execution alert to Discord Webhook if configured. (+24 more)

### Community 40 - "app.py"
Cohesion: 0.13
Nodes (17): load_universe_tickers(), main(), Sentilyze - Institutional Algorithmic Trading & MLOps Platform. Modular Master…, Loads active S&P 100 universe tickers., inject_custom_theme(), Dynamic Bespoke Theme Engine for Sentilyze. Supports 3 Institutional Presets:…, Injects high-performance, bespoke CSS styling into the Streamlit app., Renders the Walk-Forward Backtesting and Strategy Tearsheet workspace. (+9 more)

### Community 41 - "TickerSentinelSwarm"
Cohesion: 0.14
Nodes (15): detect_peak_crest_exhaustion(), Any, Dedicated Ticker Sentinel & Peak-Crest Volume Harvester Swarm for Sentilyze.…, Dedicated Micro-Agent assigned to monitor a single stock position 24/7., Audits live price tick and determines peak crest execution., Manages the full swarm of Dedicated Ticker Sentinels across all open positions., Synchronizes active sentinels with current portfolio open positions., Audits all active sentinels concurrently. (+7 more)

### Community 42 - "discord_bot.py"
Cohesion: 0.20
Nodes (12): handle_bot_command(), Any, Executes command and posts formatted embed reply to Discord., Parses and processes interactive bot commands: - `/signal <ticker>` -…, send_bot_command_reply(), Any, DataFrame, Helper wrapper for Monte Carlo VaR simulation. (+4 more)

### Community 43 - "analyze_sec_filing_diff"
Cohesion: 0.32
Nodes (7): analyze_sec_filing_diff(), compute_text_similarity_and_diff(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Computes lexical and semantic diff metrics between consecutive filings., Retrieves and compares the most recent and prior SEC filings for a company.…, test_sec_filing_diff_analysis()

### Community 47 - "Contributor Covenant Code of Conduct"
Cohesion: 0.15
Nodes (12): 1. Correction, 2. Warning, 3. Temporary Ban, 4. Permanent Ban, Attribution, Contributor Covenant Code of Conduct, Enforcement, Enforcement Guidelines (+4 more)

### Community 48 - "get_sentiment"
Cohesion: 0.12
Nodes (23): analyze_sentiment(), clean_financial_text(), get_sentiment(), _parse_analyzer_output(), Any, DataFrame, Analyzes the sentiment of news articles using high-precision FinBERT pipeline,…, Cleans raw financial headlines/descriptions by stripping boilerplate publisher… (+15 more)

### Community 49 - "calculate_doubling_progress"
Cohesion: 0.27
Nodes (9): calculate_doubling_progress(), compute_compound_position_size(), Any, Max Compound Acceleration & +100% Target Doubling Engine for Sentilyze.…, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Unit tests for Max Compound Acceleration Engine., test_calculate_doubling_progress() (+1 more)

### Community 50 - "test_pillar2_alternative_data.py"
Cohesion: 0.19
Nodes (18): compute_smart_money_insider_score(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent SEC Form 4 insider transactions for a given stock., Retrieves recent Congressional STOCK Act disclosure reports for a ticker., Synthesizes SEC Form 4 and Congressional activity into an overall Smart Money…, track_congressional_stock_disclosures(), track_corporate_insider_filings() (+10 more)

### Community 51 - "get_logger"
Cohesion: 0.40
Nodes (5): Logger, get_logger(), Configures and returns a logger with a standard format and utf-8 safe…, Tests that the get_logger function returns a configured logger., test_get_logger()

### Community 52 - "strategy_incubator.py"
Cohesion: 0.20
Nodes (15): breed_strategy_generation(), evaluate_3zone_robustness(), load_strategy_vault(), Any, Evolutionary Strategy Incubator & Robustness Lab for Sentilyze. Institutional…, Evaluates a Strategy Genome across 3 distinct zones: 1. In-Sample Train (70%)…, Runs evolutionary genetic algorithm across generations, breeding top survivors., Represents an algorithmic strategy rule DNA. (+7 more)

### Community 53 - "How Can I Contribute?"
Cohesion: 0.33
Nodes (5): Contributing to Sentilyze, How Can I Contribute?, Pull Requests, Reporting Bugs, Suggesting Enhancements

### Community 54 - "run_daily_market_scan"
Cohesion: 0.25
Nodes (7): Scans the entire stock universe defined in stocks.txt, generates tomorrow's…, run_daily_market_scan(), Any, Dispatches formatted HTML morning market digest email via Gmail SMTP., send_email_digest(), Test that run_daily_market_scan executes successfully across mock tickers., test_run_daily_market_scan()

### Community 55 - "🧪 Experimental & Simulated Research Prototypes"
Cohesion: 0.50
Nodes (3): 🧪 Experimental & Simulated Research Prototypes, 🔒 Production Isolation Guarantee, 📁 Prototype Inventory

### Community 56 - "Sentilyze — Standing Audit Protocol"
Cohesion: 0.15
Nodes (12): 10. Strict Portfolio Preservation & Realized Gain Integrity, 1. Fabricated / fake data check, 2. Ticker/input-invariant bugs, 3. Mislabeled methodology, 4. Results-file / README consistency, 5. Duplicate-output smell test, 6. Safety-critical logic sanity check, 7. Silent failure check (+4 more)

### Community 57 - "webhook_dispatcher.py"
Cohesion: 0.19
Nodes (19): Workspace: Automated Broker Webhooks & Execution API Dispatcher. Configures and…, render_broker_webhooks_workspace(), _append_audit_log(), dispatch_order_webhook(), format_broker_order_payload(), generate_hmac_signature(), load_webhook_config(), Any (+11 more)

### Community 58 - "get_vix_data"
Cohesion: 0.21
Nodes (11): _fetch_direct_yahoo_chart(), _get_browser_session(), get_vix_data(), Session, Creates a requests Session with modern desktop browser headers to prevent 429…, Fetches full historical price data directly from Yahoo Finance Chart API up to…, Fetches historical data for the CBOE Volatility Index (VIX). Args: period…, Nightly sync script that pre-fetches and caches 10-year OHLCV prices, VIX macro… (+3 more)

### Community 59 - "_load_sentiment_analyzer"
Cohesion: 0.21
Nodes (12): clean_headline_data(), _load_sentiment_analyzer(), Any, Thread-safely loads the FinBERT sentiment analysis model and tokenizer once…, Cleans a headline CSV file by removing rows with invalid stock tickers. Caches…, fixture, temp_data_dirs(), test_clean_headline_data_invalid_tickers_no_cache() (+4 more)

### Community 60 - "preprocess_data"
Cohesion: 0.26
Nodes (11): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+3 more)

### Community 61 - "quant_engine.py"
Cohesion: 0.17
Nodes (13): MasterQuantPipelineResult, Any, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), analyze_earnings_call_transcript(), Any (+5 more)

### Community 64 - "GaussianHMMRegimeDetector"
Cohesion: 0.15
Nodes (10): GaussianHMMRegimeDetector, Any, DataFrame, ndarray, 3-state Gaussian HMM regime classifier: Bull / Normal / Crisis. Uses hand-…, Gaussian emission probability for each state., Forward algorithm step: update filtered state probabilities with one new…, Classify an entire return series into regimes. (+2 more)

### Community 65 - "EWMACorrelationMonitor"
Cohesion: 0.14
Nodes (10): EWMACorrelationMonitor, Any, DataFrame, ndarray, Paper 17: RiskMetrics EWMA Volatility & Correlation Monitor. Source: J.P.…, Real-time EWMA-based correlation and volatility monitor. Tracks time-varying…, Initialize EWMA state from a seed window of returns., Update EWMA state with one day's returns across all assets. Returns current… (+2 more)

### Community 67 - "OpeningRangeBreakout"
Cohesion: 0.16
Nodes (12): OpeningRangeBreakout, Any, DataFrame, Paper 25: Opening Range Breakout (ORB) with Stocks-in-Play Filter. Source:…, Simulate multi-year daily ORB strategy returns., Opening Range Breakout (ORB) trading engine with Stocks-in-Play filter. Paper…, Rank universe and select top 'Stocks in Play' based on volume, ATR, and…, Evaluate ORB signal using daily OHLC + volatility approximation. (+4 more)

### Community 68 - "CUSUMDetector"
Cohesion: 0.15
Nodes (9): CUSUMDetector, Any, ndarray, Paper 16: CUSUM Sequential Change-Point Detection. Source: E.S. Page (1954) —…, Cumulative Sum detector for online mean-shift detection. Monitors a numeric…, Process one observation. Returns alarm status., Process a batch of observations., Reset detector state. (+1 more)

### Community 69 - "PageHinkleyDetector"
Cohesion: 0.15
Nodes (9): PageHinkleyDetector, Any, ndarray, Paper 22: Page-Hinkley Sequential Test for Concept Drift Detection. Source:…, Page-Hinkley test for detecting changes in the mean of a stream. Monitors…, Process one observation. Returns drift status., Process a batch of observations., Reset detector state. (+1 more)

### Community 70 - "macro_liquidity.py"
Cohesion: 0.38
Nodes (5): calculate_macro_liquidity_metrics(), Any, Real-Time Macro Liquidity & Treasury Yield Curve Radar for Sentilyze. Analyzes…, Computes real-time macroeconomic liquidity indicators and yield curve dynamics.…, test_calculate_macro_liquidity_metrics()

### Community 71 - "insider_signals.py"
Cohesion: 0.25
Nodes (13): calculate_insider_conviction_score(), fetch_insider_transactions(), Any, Smart-Money Executive & Institutional Insider Radar for Sentilyze.…, Computes the Quantitative Insider Conviction Index (0 to 100 Score) and detects…, Screens a universe of tickers and returns the highest-ranking insider buying…, Fetches recent SEC Form 4 insider transactions for a specific ticker. Includes…, scan_universe_insider_catalysts() (+5 more)

### Community 72 - "get_news"
Cohesion: 0.21
Nodes (11): get_news(), Enterprise Multi-Source News Router: Cascades through Google News RSS -> Yahoo…, fixture, Fixture to set a temporary data directory for tests., Test that get_news fetches data from NewsAPI and saves it to a cache file., Test that get_news loads data from the cache if it's not stale., Test that get_news re-fetches data if the cache is stale., temp_data_dir() (+3 more)

### Community 74 - "mega_tournament_simulation.py"
Cohesion: 0.18
Nodes (9): grossman_zhou_allocation(), Any, Paper 18: Grossman-Zhou Optimal Drawdown-Constrained Strategy. Source: Grossman…, Compute the Grossman-Zhou optimal risky allocation under a drawdown constraint…, Paper 15: Gaussian Hidden Markov Model for Market Regime Detection. Source:…, load_cached_universe(), Mega-Tournament Simulation: Mixing 25 Quant Papers across Trading Teams.…, run_team_tournament() (+1 more)

### Community 75 - "ADWINDetector"
Cohesion: 0.19
Nodes (7): ADWINDetector, Any, ndarray, Paper 21: ADWIN (Adaptive Windowing) Drift Detector. Source: Bifet & Gavaldà…, ADWIN drift detector with Hoeffding bound. Maintains a variable-length window…, Add one observation. Returns whether drift was detected. If drift is detected,…, TestADWIN

### Community 76 - "test_papers_15_24.py"
Cohesion: 0.20
Nodes (10): calculate_cdar(), optimize_cdar_portfolio(), Any, DataFrame, ndarray, Paper 19: Conditional Drawdown-at-Risk (CDaR) Portfolio Optimization. Source:…, Calculate Conditional Drawdown-at-Risk: the expected drawdown in the worst…, Optimize portfolio weights to minimize CDaR. Simplified approach: compute CDaR… (+2 more)

### Community 77 - "morning_briefing.py"
Cohesion: 0.17
Nodes (20): generate_morning_briefing_text(), get_portfolio_intelligence(), load_universe_candidates(), Any, AI Pre-Market Audio & Executive Morning Briefing Generator for Sentilyze.…, Reads live paper portfolio state for broadcast reporting., Assembles a comprehensive, institutional Wall Street Morning Podcast and…, Synthesizes broadcast audio podcast (.mp3) using Google Text-to-Speech (gTTS).… (+12 more)

### Community 78 - "agent_committee.py"
Cohesion: 0.09
Nodes (37): _persist_ablation_results(), Any, 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), audit_full_universe_committee() (+29 more)

### Community 79 - "run_cppi_backtest"
Cohesion: 0.23
Nodes (8): calculate_cppi_allocation(), Any, ndarray, Paper 20: Constant Proportion Portfolio Insurance (CPPI). Source: Black & Jones…, CPPI allocation: Exposure = M * (Portfolio - Floor). Args: portfolio_value:…, Run a full CPPI backtest over a return series. Args: returns: Array of daily…, run_cppi_backtest(), TestCPPI

### Community 80 - "DCCCorrelation"
Cohesion: 0.20
Nodes (7): DCCCorrelation, Any, DataFrame, Paper 24: Dynamic Conditional Correlation (DCC-GARCH). Source: Engle (2002) —…, Simplified DCC model using EWMA-GARCH(1,1) for individual volatilities and DCC…, Fit the DCC model to a returns DataFrame. Returns time-varying correlation…, TestDCC

### Community 81 - "model_ensemble.py"
Cohesion: 0.20
Nodes (12): blend_model_predictions(), calculate_triple_barrier_corridors(), Any, Dual-Model Consensus Alpha Engine: XGBoost + DLinear-TCN Deep Learning Fusion.…, Blends XGBoost and Deep Learning probabilities with consensus-gated execution.…, Calculates dynamic institutional take-profit and stop-loss levels based on ATR…, Verify that conflict between models blocks the trade and returns NEUTRAL., Verify ATR-based multi-stage profit and stop corridors. (+4 more)

### Community 82 - "generate_comprehensive_factsheet"
Cohesion: 0.24
Nodes (9): generate_comprehensive_factsheet(), Any, Series, Institutional Risk & Alpha Performance Factsheet Engine for Sentilyze.…, Computes over 30 institutional hedge-fund risk, performance, and drawdown…, Workspace: Institutional Risk & Alpha Performance Factsheet. Quantitative…, render_performance_factsheet_workspace(), test_generate_comprehensive_factsheet_custom_series() (+1 more)

### Community 83 - "risk_constrained_kelly_allocation"
Cohesion: 0.24
Nodes (6): Any, ndarray, Paper 23: Risk-Constrained Kelly Gambling. Source: Busseti, Ryu, Boyd —…, Compute optimal Kelly allocation with drawdown probability constraint.…, risk_constrained_kelly_allocation(), TestRiskKelly

### Community 84 - "render_workspace_header"
Cohesion: 0.13
Nodes (17): get_market_status(), Any, Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Wraps HTML content inside an institutional frosted glass container., Renders a progress meter with dynamic color coding., Calculates live US Market (NYSE/NASDAQ) status based on Eastern Time., Renders an executive header banner with live status badge and market clock., render_conviction_gauge() (+9 more)

### Community 85 - "calculate_portfolio_diversity_grade"
Cohesion: 0.27
Nodes (9): calculate_portfolio_diversity_grade(), Any, DataFrame, Workspace: Portfolio Diversity & Correlation Health Grader. Institutional…, render_portfolio_diversity_workspace(), test_custom_returns_correlated(), test_custom_returns_diverse(), test_empty_portfolio() (+1 more)

### Community 86 - "ws_quantum_tournament.py"
Cohesion: 0.22
Nodes (12): generate_institutional_pdf_tearsheet(), Generates a publication-grade institutional quantitative tearsheet PDF. Returns…, get_market_timestamp(), load_safety_benchmarks(), load_tournament_results(), Any, Workspace 14: 25-Paper Quantum Tournament, Live Omni-Hybrid Pipeline & Risk…, render_quantum_tournament_workspace() (+4 more)

### Community 87 - "audio_briefing.py"
Cohesion: 0.47
Nodes (5): generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses gTTS if available, or…, synthesize_morning_audio()

### Community 88 - "main"
Cohesion: 0.40
Nodes (5): main(), Batch Universe Trainer for Remaining S&P 100 Tickers., run_single(), main(), Main function to run the training pipeline for a given stock ticker. Args:…

### Community 89 - "correlation_matrix.py"
Cohesion: 0.38
Nodes (6): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…

### Community 90 - ".get_closed_trades_df"
Cohesion: 0.29
Nodes (4): DataFrame, Returns a DataFrame of current open holdings with Scale-Out status., Returns a DataFrame of trade history with full company names., Returns equity history as a DatetimeIndex DataFrame.

### Community 91 - "cli.py"
Cohesion: 0.29
Nodes (10): Root entry point for Sentilyze CLI. Run directly with: python sentilyze.py NVDA…, cmd_audit(), cmd_briefing(), cmd_portfolio(), main(), print_banner(), Sentilyze Command-Line Interface (CLI). Interactive terminal tool for 4-Agent…, Displays current portfolio metrics and live holdings. (+2 more)

### Community 92 - "block_external_alerts"
Cohesion: 0.50
Nodes (3): block_external_alerts(), fixture, Autouse fixture that prevents tests from sending real outbound network calls to…

### Community 93 - "get_price_history"
Cohesion: 0.18
Nodes (11): get_price_history(), Enterprise Data Router: Fetches historical price data up to today using the…, Any, Fast vectorized backtesting simulation sandbox for custom leverage, confidence,…, simulate_strategy_sandbox(), Test that get_price_history fetches data from yfinance and saves it to a cache…, Test that get_price_history loads data from the cache if it's not stale., Test that get_price_history re-fetches data if the cache is stale. (+3 more)

### Community 94 - "fetch_live_quote"
Cohesion: 0.27
Nodes (10): compute_order_book_depth_and_clusters(), compute_volume_profile_and_poc(), Any, Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap for…, Simulates Level 2 market depth and identifies institutional buy/sell liquidity…, Computes Point of Control (POC), Value Area High (VAH), and Value Area Low…, fetch_live_quote(), Fetches sub-second real-time market quote using Yahoo Finance Direct Chart API… (+2 more)

### Community 95 - "options_surface.py"
Cohesion: 0.27
Nodes (10): calculate_multileg_payoff(), generate_volatility_surface_mesh(), Any, 3D Implied Volatility Surface & Multi-Leg Options Strategy Desk for Sentilyze.…, Constructs a 3D Implied Volatility Surface across strike prices and expiration…, Calculates profit and loss (P&L) curves at expiration for institutional multi-…, test_calculate_multileg_payoff_bull_call_spread(), test_calculate_multileg_payoff_iron_condor() (+2 more)

### Community 96 - "calculate_beneish_m_score"
Cohesion: 0.27
Nodes (9): analyze_debt_maturity_wall(), calculate_beneish_m_score(), Any, DataFrame, Beneish M-Score Forensic Analyzer & Debt Maturity Wall Radar for Sentilyze.…, Evaluates corporate interest coverage and debt maturity wall runway., Computes the 8-Ratio Beneish M-Score from 2-year comparative SEC financial…, test_beneish_m_score() (+1 more)

### Community 97 - "execute_continuous_retrain_cycle"
Cohesion: 0.32
Nodes (7): enrich_features_with_alpha_interactions(), execute_continuous_retrain_cycle(), Any, DataFrame, Enriches standard feature matrix with non-linear interaction terms., Executes an end-to-end continuous learning and model boosting cycle: 1.…, test_enrich_features_with_alpha_interactions()

### Community 98 - "run_opening_range_session"
Cohesion: 0.40
Nodes (5): Any, Executes a live 5-Minute Opening Range Breakout scan across top liquid assets:…, run_opening_range_session(), Verify that ORB live session executes, filters stocks in play, and saves latest…, test_run_opening_range_session()

### Community 99 - "_get_browser_session"
Cohesion: 0.67
Nodes (3): _get_browser_session(), Session, Creates a requests Session with modern desktop browser headers.

## Knowledge Gaps
- **40 isolated node(s):** `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs`, `3. Mislabeled methodology`, `4. Results-file / README consistency` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `get_logger` to `deep_learning_model.py`, `data_ingestion.py`, `test_options_flow.py`, `daily_scanner.py`, `run_backtest`, `drl_policy_agent.py`, `test_all_14_papers.py`, `test_statistical_arbitrage.py`, `fetch_financial_statements`, `TradingEnvironment`, `CloudDataLake`, `ws_alternative_data.py`, `analyze_supply_chain_spillover`, `calculate_hrp_weights`, `run_temporal_fusion_forecast`, `PaperBroker`, `meta_ensemble.py`, `OnlineNewtonStepOptimizer`, `ws_live_prediction.py`, `social_sentiment.py`, `compute_dark_pool_sentiment`, `test_omnichannel_mobile.py`, `test_rebalancer_and_tearsheet.py`, `utils.py`, `compute_lead_lag_matrix`, `black_swan_simulator.py`, `ipo_radar.py`, `AlpacaBrokerBridge`, `realtime_tracker.py`, `datetime`, `AICopilotEngine`, `PolyTimeConvexOptimizer`, `preprocessing.py`, `SuperEnsembleClassifier`, `triple_convex_engine.py`, `autonomous_trader.py`, `TickerSentinelSwarm`, `discord_bot.py`, `analyze_sec_filing_diff`, `get_sentiment`, `calculate_doubling_progress`, `test_pillar2_alternative_data.py`, `strategy_incubator.py`, `run_daily_market_scan`, `webhook_dispatcher.py`, `get_vix_data`, `preprocess_data`, `quant_engine.py`, `macro_liquidity.py`, `insider_signals.py`, `morning_briefing.py`, `agent_committee.py`, `model_ensemble.py`, `generate_comprehensive_factsheet`, `calculate_portfolio_diversity_grade`, `audio_briefing.py`, `main`, `correlation_matrix.py`, `cli.py`, `get_price_history`, `fetch_live_quote`, `options_surface.py`, `calculate_beneish_m_score`?**
  _High betweenness centrality (0.257) - this node is a cross-community bridge._
- **Why does `get_price_history()` connect `get_price_history` to `data_ingestion.py`, `daily_scanner.py`, `run_backtest`, `drl_policy_agent.py`, `test_all_14_papers.py`, `test_statistical_arbitrage.py`, `calculate_hrp_weights`, `utils.py`, `realtime_tracker.py`, `preprocessing.py`, `triple_convex_engine.py`, `app.py`, `strategy_incubator.py`, `get_vix_data`, `preprocess_data`, `macro_liquidity.py`, `get_news`, `morning_briefing.py`, `agent_committee.py`, `render_workspace_header`, `calculate_portfolio_diversity_grade`, `correlation_matrix.py`, `run_opening_range_session`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `PaperBroker` connect `PaperBroker` to `AICopilotEngine`, `run_opening_range_session`, `daily_scanner.py`, `drl_policy_agent.py`, `autonomous_trader.py`, `insider_signals.py`, `discord_bot.py`, `morning_briefing.py`, `generate_comprehensive_factsheet`, `strategy_incubator.py`, `calculate_portfolio_diversity_grade`, `run_daily_market_scan`, `.get_closed_trades_df`, `cli.py`, `realtime_tracker.py`, `._save`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs` to the rest of the system?**
  _40 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `deep_learning_model.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10160427807486631 - nodes in this community are weakly interconnected._
- **Should `data_ingestion.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11956521739130435 - nodes in this community are weakly interconnected._