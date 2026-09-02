# Graph Report - Sentilyze  (2026-09-03)

## Corpus Check
- 542 files · ~731,196 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1684 nodes · 3541 edges · 94 communities (90 shown, 4 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `53047014`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DLinearTCNModel
- data_ingestion.py
- quant_engine.py
- get_market_timestamp
- run_backtest
- drl_policy_agent.py
- test_all_14_papers.py
- test_statistical_arbitrage.py
- agent_committee.py
- TradingEnvironment
- benchmark_papers_15_24.py
- CloudDataLake
- ws_alternative_data.py
- analyze_supply_chain_spillover
- calculate_hrp_weights
- run_temporal_fusion_forecast
- ws_deep_quant.py
- meta_ensemble.py
- OnlineNewtonStepOptimizer
- ws_live_prediction.py
- social_sentiment.py
- compute_dark_pool_sentiment
- test_omnichannel_mobile.py
- .optimize_allocation
- daily_scanner.py
- compute_lead_lag_matrix
- black_swan_simulator.py
- compute_smart_money_insider_score
- AlpacaBrokerBridge
- realtime_tracker.py
- PaperBroker
- datetime
- options_surface.py
- AICopilotEngine
- liquidity_heatmap.py
- preprocessing.py
- SuperEnsembleClassifier
- calculate_deflated_sharpe_ratio
- Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform
- autonomous_trader.py
- app.py
- TickerSentinelSwarm
- ._execute_cycle_body
- analyze_sec_filing_diff
- ui/__init__.py
- Contributor Covenant Code of Conduct
- get_sentiment
- calculate_doubling_progress
- test_pillar2_alternative_data.py
- utils.py
- strategy_incubator.py
- How Can I Contribute?
- run_daily_market_scan
- 🧪 Experimental & Simulated Research Prototypes
- Sentilyze — Standing Audit Protocol
- webhook_dispatcher.py
- test_rebalancer_and_tearsheet.py
- preprocess_data
- get_news
- run_unified_institutional_pipeline
- rules/graphify.md
- workflows/graphify.md
- GaussianHMMRegimeDetector
- EWMACorrelationMonitor
- OpeningRangeBreakout
- CUSUMDetector
- PageHinkleyDetector
- get_price_history
- insider_signals.py
- test_data_ingestion.py
- api.py
- mega_tournament_simulation.py
- ADWINDetector
- test_papers_15_24.py
- morning_briefing.py
- cli.py
- run_cppi_backtest
- DCCCorrelation
- blend_model_predictions
- generate_comprehensive_factsheet
- risk_constrained_kelly_allocation
- run_full_quant_experiment
- calculate_portfolio_diversity_grade
- generate_institutional_pdf_tearsheet
- execute_continuous_retrain_cycle
- opening_range_runner.py
- correlation_matrix.py
- .get_closed_trades_df
- ws_quantum_tournament.py
- block_external_alerts
- render_glass_card

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 92 edges
2. `get_price_history()` - 57 edges
3. `PaperBroker` - 53 edges
4. `fetch_live_quote()` - 33 edges
5. `run_unified_institutional_pipeline()` - 32 edges
6. `get_news()` - 27 edges
7. `preprocess_data()` - 27 edges
8. `render_workspace_header()` - 24 edges
9. `main()` - 23 edges
10. `convene_trading_committee()` - 22 edges

## Surprising Connections (you probably didn't know these)
- `test_autonomous_cycle_execution()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_idempotency_lock_prevents_overlap()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_evaluate_intraday_scale_out_and_tp2()` --calls--> `PaperBroker`  [EXTRACTED]
  tests/test_realtime_tracker.py → src/paper_broker.py
- `predict()` --calls--> `get_prediction_on_latest_data()`  [EXTRACTED]
  api.py → src/modeling.py
- `predict()` --calls--> `load_model()`  [EXTRACTED]
  api.py → src/modeling.py

## Import Cycles
- None detected.

## Communities (94 total, 4 thin omitted)

### Community 0 - "DLinearTCNModel"
Cohesion: 0.11
Nodes (23): create_sliding_window_tensors(), DLinearTCNModel, load_dlinear_model(), predict_momentum_probability(), Any, DataFrame, Tensor, Converts a pandas DataFrame into sliding window sequence tensors for Deep… (+15 more)

### Community 1 - "data_ingestion.py"
Cohesion: 0.14
Nodes (19): _fetch_alpaca_news(), _fetch_alpaca_price_history(), _fetch_eodhd_price_history(), _fetch_finnhub_news(), _fetch_fmp_price_history(), _fetch_marketaux_news(), _fetch_polygon_news_feed(), _fetch_polygon_price_history() (+11 more)

### Community 2 - "quant_engine.py"
Cohesion: 0.15
Nodes (24): MasterQuantPipelineResult, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain() (+16 more)

### Community 3 - "get_market_timestamp"
Cohesion: 0.16
Nodes (27): format_signal_card(), Any, Dispatches a crystal-clear, high-impact Discord card for live autonomous trade…, Construct a standardized trade signal data payload. Args: ticker (str): Stock…, Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord., Dispatches real-time Reddit r/wallstreetbets & Stocktwits hype spike alerts., Sends a comprehensive institutional morning macro regime, portfolio health,…, Sends a rich formatted trade alert card to a Discord channel via Webhook. (+19 more)

### Community 4 - "run_backtest"
Cohesion: 0.08
Nodes (40): Figure, main(), Batch Universe Trainer for Remaining S&P 100 Tickers., run_single(), _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample… (+32 more)

### Community 5 - "drl_policy_agent.py"
Cohesion: 0.14
Nodes (18): ActorCriticPolicy, DRLTradingEnvironment, evaluate_drl_policy_action(), Any, ndarray, Tensor, Deep Reinforcement Learning (DRL) Autonomous Policy Agent for Sentilyze.…, Trains an Actor-Critic DRL policy agent on historical market returns and news… (+10 more)

### Community 6 - "test_all_14_papers.py"
Cohesion: 0.08
Nodes (30): Any, Master Academic Research Papers Empirical Benchmark Suite (All 14 Papers).…, Executes empirical backtests comparing all 14 academic paper methodologies., run_all_14_papers_benchmark(), calculate_almgren_chriss_trajectory(), Any, Paper 3: Almgren & Chriss (2000) - Optimal Execution of Portfolio Transactions.…, Computes Almgren-Chriss optimal trading trajectory. x_j = 2 * sinh(0.5 * kappa… (+22 more)

### Community 7 - "test_statistical_arbitrage.py"
Cohesion: 0.19
Nodes (27): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any, Series (+19 more)

### Community 8 - "agent_committee.py"
Cohesion: 0.06
Nodes (59): _persist_ablation_results(), Any, 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), audit_full_universe_committee() (+51 more)

### Community 9 - "TradingEnvironment"
Cohesion: 0.14
Nodes (16): optimize_rl_position_allocation(), PPOPolicyAgent, Any, ndarray, ⚠️ EXPERIMENTAL / RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM PRODUCTION…, Computes mean action (leverage) between 0.0 and 2.0., Estimates state value., Trains Actor-Critic parameters across historical episodes. (+8 more)

### Community 10 - "benchmark_papers_15_24.py"
Cohesion: 0.11
Nodes (25): benchmark_adwin(), benchmark_cdar(), benchmark_cppi(), benchmark_cusum(), benchmark_dcc(), benchmark_ewma(), benchmark_grossman_zhou(), benchmark_hmm() (+17 more)

### Community 11 - "CloudDataLake"
Cohesion: 0.14
Nodes (14): CloudDataLake, Any, Supabase / PostgreSQL Cloud Data Lake Connector., Validates or generates cloud database schema., Syncs local trade executions to the cloud database., Publishes real-time portfolio snapshot to cloud WebSockets channel., generate_twap_order_schedule(), generate_vwap_order_schedule() (+6 more)

### Community 12 - "ws_alternative_data.py"
Cohesion: 0.14
Nodes (20): auto_register_ipo_ticker(), fetch_pre_ipo_radar_summary(), fetch_sec_edgar_ipo_filings(), get_pre_ipo_pipeline_df(), Any, DataFrame, IPO & Pre-IPO Intelligence Radar for Sentilyze. Pillar 9 Alternative Asset…, Fetches real-time SEC Form S-1 / S-1/A IPO registration statements from SEC… (+12 more)

### Community 13 - "analyze_supply_chain_spillover"
Cohesion: 0.14
Nodes (15): analyze_supply_chain_spillover(), Any, ndarray, Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for…, Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)., Executes a Graph Convolutional Network (GCN) layer: H_new = ReLU(A_hat * H * W)…, Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab…, High-level entry point to run GNN supply chain shock propagation. (+7 more)

### Community 14 - "calculate_hrp_weights"
Cohesion: 0.11
Nodes (31): Empirical Quant Experimentation & Multi-Asset Ablation Benchmark Suite.…, build_unified_portfolio(), calculate_hrp_weights(), calculate_risk_parity_weights(), get_cluster_var(), get_quasi_diag(), get_rec_bisection(), load_all_ticker_portfolios() (+23 more)

### Community 15 - "run_temporal_fusion_forecast"
Cohesion: 0.14
Nodes (15): Any, DataFrame, ndarray, Temporal Fusion Transformer (TFT) & Multi-Horizon Self-Attention Engine for…, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with… (+7 more)

### Community 16 - "ws_deep_quant.py"
Cohesion: 0.18
Nodes (14): analyze_debt_maturity_wall(), calculate_beneish_m_score(), Any, DataFrame, Beneish M-Score Forensic Analyzer & Debt Maturity Wall Radar for Sentilyze.…, Evaluates corporate interest coverage and debt maturity wall runway., Computes the 8-Ratio Beneish M-Score from 2-year comparative SEC financial…, Helper wrapper for Monte Carlo VaR simulation. (+6 more)

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

### Community 23 - ".optimize_allocation"
Cohesion: 0.40
Nodes (4): Any, DataFrame, Series, Solves the friction-aware convex optimization problem in polynomial time. Args:…

### Community 24 - "daily_scanner.py"
Cohesion: 0.12
Nodes (30): Continuous Model Self-Training & Accuracy Boosting Engine for Sentilyze. Self-…, handle_bot_command(), Any, Executes command and posts formatted embed reply to Discord., Parses and processes interactive bot commands: - `/signal <ticker>` -…, send_bot_command_reply(), get_prediction_on_latest_data(), load_model() (+22 more)

### Community 25 - "compute_lead_lag_matrix"
Cohesion: 0.20
Nodes (14): compute_lead_lag_matrix(), _granger_f_test(), Any, DataFrame, ndarray, Series, rank_market_price_leaders(), Lead-Lag Granger Causality & Supply Chain Price Discovery Engine for Sentilyze.… (+6 more)

### Community 26 - "black_swan_simulator.py"
Cohesion: 0.23
Nodes (11): calculate_kelly_sizing(), estimate_market_impact_slippage(), Any, Historical Black Swan Crisis Simulator & Kelly Position Sizing for Sentilyze.…, Calculates optimal position sizing using the Kelly Criterion: Kelly % = W - (1…, Estimates market execution slippage using the Almgren-Chriss square-root impact…, Stress-tests the current portfolio against major historical market crashes.…, simulate_portfolio_crises() (+3 more)

### Community 27 - "compute_smart_money_insider_score"
Cohesion: 0.33
Nodes (9): compute_smart_money_insider_score(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent SEC Form 4 insider transactions for a given stock., Retrieves recent Congressional STOCK Act disclosure reports for a ticker., Synthesizes SEC Form 4 and Congressional activity into an overall Smart Money…, track_congressional_stock_disclosures(), track_corporate_insider_filings() (+1 more)

### Community 28 - "AlpacaBrokerBridge"
Cohesion: 0.16
Nodes (14): AlpacaBrokerBridge, Any, Fetches active positions from Alpaca brokerage., Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.…, Verifies active connection to Alpaca Brokerage API., Fetches live Alpaca account equity, buying power, and cash., Submits an institutional Bracket Order: - Entry: Market order - Exit 1: Limit…, patch (+6 more)

### Community 29 - "realtime_tracker.py"
Cohesion: 0.13
Nodes (24): check_live_news_sentiment_shock(), evaluate_intraday_execution(), fetch_live_quote(), _get_browser_session(), get_us_market_session_info(), Any, Session, Fetches sub-second real-time market quote using Yahoo Finance Direct Chart API… (+16 more)

### Community 30 - "PaperBroker"
Cohesion: 0.11
Nodes (20): PaperBroker, Any, Executes daily quantitative scan results using the Concentrated Top-2 + Scale-…, Institutional Multi-Stage Quantitative Execution Broker ($100k Account).…, Loads existing portfolio state from JSON or initializes a fresh $100k account., Updates total equity, unrealized PnL, and win rates., Returns high-level KPI metrics for the portfolio dashboard., Executes an institutional BUY order into the virtual paper broker ledger.… (+12 more)

### Community 31 - "datetime"
Cohesion: 0.07
Nodes (41): datetime, generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses gTTS if available, or…, synthesize_morning_audio(), check_market_hours_preflight(), get_current_ny_time() (+33 more)

### Community 32 - "options_surface.py"
Cohesion: 0.27
Nodes (10): calculate_multileg_payoff(), generate_volatility_surface_mesh(), Any, 3D Implied Volatility Surface & Multi-Leg Options Strategy Desk for Sentilyze.…, Constructs a 3D Implied Volatility Surface across strike prices and expiration…, Calculates profit and loss (P&L) curves at expiration for institutional multi-…, test_calculate_multileg_payoff_bull_call_spread(), test_calculate_multileg_payoff_iron_condor() (+2 more)

### Community 33 - "AICopilotEngine"
Cohesion: 0.21
Nodes (9): AICopilotEngine, Any, AI Trade Copilot & Conversational Analyst for Sentilyze. Provides natural…, Conversational intelligence engine that parses queries and generates analytical…, Interprets user prompt and routes to appropriate financial analytical…, test_copilot_committee_query(), test_copilot_portfolio_query(), test_copilot_stress_query() (+1 more)

### Community 34 - "liquidity_heatmap.py"
Cohesion: 0.31
Nodes (8): compute_order_book_depth_and_clusters(), compute_volume_profile_and_poc(), Any, Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap for…, Simulates Level 2 market depth and identifies institutional buy/sell liquidity…, Computes Point of Control (POC), Value Area High (VAH), and Value Area Low…, test_compute_order_book_depth_and_clusters(), test_compute_volume_profile_and_poc()

### Community 35 - "preprocessing.py"
Cohesion: 0.17
Nodes (19): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, DataFrame (+11 more)

### Community 36 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (16): Any, DataFrame, ndarray, Series, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.…, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used. (+8 more)

### Community 37 - "calculate_deflated_sharpe_ratio"
Cohesion: 0.12
Nodes (17): Any, Runs multi-trial empirical testing and saves verified metrics to JSON., run_triple_convex_multi_trial_benchmark(), apply_triple_barrier_labeling(), calculate_deflated_sharpe_ratio(), DataFrame, Series, Computes Bailey & López de Prado's Deflated Sharpe Ratio (DSR). Adjusts for: -… (+9 more)

### Community 38 - "Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform"
Cohesion: 0.13
Nodes (14): 1. Run in 1-Click (No Installation Required), 2. Local Setup & Installation, 3. Run the 4-Agent Quantitative CLI, 4. Launch the Streamlit Mission Control, 5. Run Full Test Suite (240+ Unit Tests), 📊 Empirical Alpha Attribution & Benchmarks, 🏛️ Grounded 4-Agent Deliberation Council, 🖥️ Interactive Streamlit App & 23 Mission Control Workspaces (+6 more)

### Community 39 - "autonomous_trader.py"
Cohesion: 0.10
Nodes (27): AutonomousTradingEngine, check_daily_loss_circuit_breaker(), is_kill_switch_active(), load_universe_tickers(), Autonomous Live Trading & News Intelligence Engine for Sentilyze. Institutional…, Autonomous Execution Engine that integrates Live News Ingestion, 4-Agent…, Task 7: Master Kill Switch Check. Returns True if SENTILYZE_KILL_SWITCH…, Task 8: Independent Max-Daily-Loss Circuit Breaker. Compares current total… (+19 more)

### Community 40 - "app.py"
Cohesion: 0.10
Nodes (27): load_universe_tickers(), main(), Sentilyze - Institutional Algorithmic Trading & MLOps Platform. Modular Master…, Loads active S&P 100 universe tickers., get_market_status(), Any, Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Calculates live US Market (NYSE/NASDAQ) status based on Eastern Time. (+19 more)

### Community 41 - "TickerSentinelSwarm"
Cohesion: 0.14
Nodes (14): detect_peak_crest_exhaustion(), Any, Dedicated Micro-Agent assigned to monitor a single stock position 24/7., Audits live price tick and determines peak crest execution., Manages the full swarm of Dedicated Ticker Sentinels across all open positions., Synchronizes active sentinels with current portfolio open positions., Audits all active sentinels concurrently., Detects if a stock has reached the crest/peak of its 15-minute momentum wave… (+6 more)

### Community 42 - "._execute_cycle_body"
Cohesion: 0.22
Nodes (6): Any, Dispatches an institutional execution alert to Discord Webhook if configured., Executes one full autonomous decision and execution cycle with: - Task 6:…, Core cycle execution body., Executes the Self-Improving Feedback Loop: 1. Analyzes trade autopsies on…, Gathers overnight macro VIX volatility regime, paper portfolio balance, and top…

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
Cohesion: 0.31
Nodes (8): calculate_doubling_progress(), compute_compound_position_size(), Any, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Unit tests for Max Compound Acceleration Engine., test_calculate_doubling_progress(), test_compute_compound_position_size()

### Community 50 - "test_pillar2_alternative_data.py"
Cohesion: 0.21
Nodes (14): compute_government_and_patent_index(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Synthesizes federal contracting dollars and patent velocity into a single…, Retrieves recent prime federal government contract awards for a company., Tracks recent USPTO patent grants in AI/ML, Semiconductor Design, and Cloud…, track_federal_contract_awards(), track_uspto_patent_momentum() (+6 more)

### Community 51 - "utils.py"
Cohesion: 0.11
Nodes (19): Logger, Multi-Trial Empirical Benchmark Suite for Triple-Convex Quantum Engine.…, Cloud PostgreSQL Data Lake (Supabase / Neon) Connector for Sentilyze. Pillar 6…, Max Compound Acceleration & +100% Target Doubling Engine for Sentilyze.…, High-Efficiency Deep Learning Engine: DLinear + Temporal Convolutional Network…, Dual-Model Consensus Alpha Engine: XGBoost + DLinear-TCN Deep Learning Fusion.…, Dedicated Ticker Sentinel & Peak-Crest Volume Harvester Swarm for Sentilyze.…, Marcos López de Prado's Triple-Barrier Method & Deflated Sharpe Ratio (DSR).… (+11 more)

### Community 52 - "strategy_incubator.py"
Cohesion: 0.20
Nodes (15): breed_strategy_generation(), evaluate_3zone_robustness(), load_strategy_vault(), Any, Evolutionary Strategy Incubator & Robustness Lab for Sentilyze. Institutional…, Evaluates a Strategy Genome across 3 distinct zones: 1. In-Sample Train (70%)…, Runs evolutionary genetic algorithm across generations, breeding top survivors., Represents an algorithmic strategy rule DNA. (+7 more)

### Community 53 - "How Can I Contribute?"
Cohesion: 0.33
Nodes (5): Contributing to Sentilyze, How Can I Contribute?, Pull Requests, Reporting Bugs, Suggesting Enhancements

### Community 54 - "run_daily_market_scan"
Cohesion: 0.25
Nodes (8): Scans the entire stock universe defined in stocks.txt, generates tomorrow's…, run_daily_market_scan(), Any, Dispatches formatted HTML morning market digest email via Gmail SMTP., send_email_digest(), test_paper9_when_agents_trade_scanner(), Test that run_daily_market_scan executes successfully across mock tickers., test_run_daily_market_scan()

### Community 55 - "🧪 Experimental & Simulated Research Prototypes"
Cohesion: 0.50
Nodes (3): 🧪 Experimental & Simulated Research Prototypes, 🔒 Production Isolation Guarantee, 📁 Prototype Inventory

### Community 56 - "Sentilyze — Standing Audit Protocol"
Cohesion: 0.17
Nodes (11): 1. Fabricated / fake data check, 2. Ticker/input-invariant bugs, 3. Mislabeled methodology, 4. Results-file / README consistency, 5. Duplicate-output smell test, 6. Safety-critical logic sanity check, 7. Silent failure check, 8. Scope creep check (+3 more)

### Community 57 - "webhook_dispatcher.py"
Cohesion: 0.19
Nodes (19): Workspace: Automated Broker Webhooks & Execution API Dispatcher. Configures and…, render_broker_webhooks_workspace(), _append_audit_log(), dispatch_order_webhook(), format_broker_order_payload(), generate_hmac_signature(), load_webhook_config(), Any (+11 more)

### Community 58 - "test_rebalancer_and_tearsheet.py"
Cohesion: 0.13
Nodes (18): fetch_universe_live_quotes(), Fetches real-time quotes across universe with fast batching (sub-2s)., calculate_custom_rebalance(), calculate_share_allocation(), Any, Helper to calculate share allocation from latest daily signals file or universe…, Computes exact whole-share buy allocations for a given capital budget across…, Any (+10 more)

### Community 59 - "preprocess_data"
Cohesion: 0.13
Nodes (19): clean_headline_data(), _get_api_key(), _load_sentiment_analyzer(), preprocess_data(), Any, DataFrame, Safely attempts to retrieve the API key from Streamlit secrets, falling back to…, Orchestrates the data acquisition, sentiment analysis, and feature engineering… (+11 more)

### Community 60 - "get_news"
Cohesion: 0.15
Nodes (17): _fetch_direct_yahoo_chart(), _fetch_google_news_rss(), _fetch_yfinance_news(), _get_browser_session(), get_news(), get_vix_data(), Session, Fetches real-time live market news headlines directly from Yahoo Finance. (+9 more)

### Community 61 - "run_unified_institutional_pipeline"
Cohesion: 0.47
Nodes (5): Any, Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), test_all_8_pillars_present_in_output(), test_run_unified_institutional_pipeline()

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

### Community 70 - "get_price_history"
Cohesion: 0.19
Nodes (12): get_price_history(), Enterprise Data Router: Fetches historical price data up to today using the…, calculate_macro_liquidity_metrics(), Any, Real-Time Macro Liquidity & Treasury Yield Curve Radar for Sentilyze. Analyzes…, Computes real-time macroeconomic liquidity indicators and yield curve dynamics.…, Any, Fast vectorized backtesting simulation sandbox for custom leverage, confidence,… (+4 more)

### Community 71 - "insider_signals.py"
Cohesion: 0.25
Nodes (13): calculate_insider_conviction_score(), fetch_insider_transactions(), Any, Smart-Money Executive & Institutional Insider Radar for Sentilyze.…, Computes the Quantitative Insider Conviction Index (0 to 100 Score) and detects…, Screens a universe of tickers and returns the highest-ranking insider buying…, Fetches recent SEC Form 4 insider transactions for a specific ticker. Includes…, scan_universe_insider_catalysts() (+5 more)

### Community 72 - "test_data_ingestion.py"
Cohesion: 0.12
Nodes (15): fixture, Fixture to set a temporary data directory for tests., Test that get_news fetches data from NewsAPI and saves it to a cache file., Test that get_news loads data from the cache if it's not stale., Test that get_news re-fetches data if the cache is stale., Test that get_price_history fetches data from yfinance and saves it to a cache…, Test that get_price_history loads data from the cache if it's not stale., Test that get_price_history re-fetches data if the cache is stale. (+7 more)

### Community 73 - "api.py"
Cohesion: 0.18
Nodes (10): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+2 more)

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
Cohesion: 0.26
Nodes (10): generate_morning_briefing_text(), Any, AI Pre-Market Audio & Executive Morning Briefing Generator for Sentilyze.…, Synthesizes executive audio file (.mp3) using Google Text-to-Speech (gTTS).…, Assembles a comprehensive, institutional Wall Street Pre-Market Morning…, synthesize_briefing_audio(), Workspace: AI Pre-Market Audio & Executive Morning Intelligence Briefing.…, render_morning_briefing_workspace() (+2 more)

### Community 78 - "cli.py"
Cohesion: 0.29
Nodes (10): Root entry point for Sentilyze CLI. Run directly with: python sentilyze.py NVDA…, cmd_audit(), cmd_briefing(), cmd_portfolio(), main(), print_banner(), Sentilyze Command-Line Interface (CLI). Interactive terminal tool for 4-Agent…, Displays current portfolio metrics and live holdings. (+2 more)

### Community 79 - "run_cppi_backtest"
Cohesion: 0.23
Nodes (8): calculate_cppi_allocation(), Any, ndarray, Paper 20: Constant Proportion Portfolio Insurance (CPPI). Source: Black & Jones…, CPPI allocation: Exposure = M * (Portfolio - Floor). Args: portfolio_value:…, Run a full CPPI backtest over a return series. Args: returns: Array of daily…, run_cppi_backtest(), TestCPPI

### Community 80 - "DCCCorrelation"
Cohesion: 0.20
Nodes (7): DCCCorrelation, Any, DataFrame, Paper 24: Dynamic Conditional Correlation (DCC-GARCH). Source: Engle (2002) —…, Simplified DCC model using EWMA-GARCH(1,1) for individual volatilities and DCC…, Fit the DCC model to a returns DataFrame. Returns time-varying correlation…, TestDCC

### Community 81 - "blend_model_predictions"
Cohesion: 0.21
Nodes (11): blend_model_predictions(), calculate_triple_barrier_corridors(), Any, Blends XGBoost and Deep Learning probabilities with consensus-gated execution.…, Calculates dynamic institutional take-profit and stop-loss levels based on ATR…, Verify that conflict between models blocks the trade and returns NEUTRAL., Verify ATR-based multi-stage profit and stop corridors., Verify that high agreement between XGBoost and Deep Learning boosts conviction. (+3 more)

### Community 82 - "generate_comprehensive_factsheet"
Cohesion: 0.24
Nodes (9): generate_comprehensive_factsheet(), Any, Series, Institutional Risk & Alpha Performance Factsheet Engine for Sentilyze.…, Computes over 30 institutional hedge-fund risk, performance, and drawdown…, Workspace: Institutional Risk & Alpha Performance Factsheet. Quantitative…, render_performance_factsheet_workspace(), test_generate_comprehensive_factsheet_custom_series() (+1 more)

### Community 83 - "risk_constrained_kelly_allocation"
Cohesion: 0.24
Nodes (6): Any, ndarray, Paper 23: Risk-Constrained Kelly Gambling. Source: Busseti, Ryu, Boyd —…, Compute optimal Kelly allocation with drawdown probability constraint.…, risk_constrained_kelly_allocation(), TestRiskKelly

### Community 84 - "run_full_quant_experiment"
Cohesion: 0.22
Nodes (9): compute_performance_metrics(), Any, DataFrame, Series, Executes empirical ablation benchmark across the full asset universe., Simulates walk-forward strategy execution with or without advanced quant…, Computes key quant performance metrics., run_full_quant_experiment() (+1 more)

### Community 85 - "calculate_portfolio_diversity_grade"
Cohesion: 0.36
Nodes (7): calculate_portfolio_diversity_grade(), Any, DataFrame, test_custom_returns_correlated(), test_custom_returns_diverse(), test_empty_portfolio(), test_single_asset_portfolio()

### Community 86 - "generate_institutional_pdf_tearsheet"
Cohesion: 0.31
Nodes (7): generate_institutional_pdf_tearsheet(), Institutional Quantitative Tearsheet & Factsheet Generator for Sentilyze.…, Generates a publication-grade institutional quantitative tearsheet PDF. Returns…, Verify that PDF file is saved correctly to a specified path., Verify that PDF generation creates valid, non-empty binary content., test_generate_institutional_pdf_tearsheet_bytes(), test_generate_institutional_pdf_tearsheet_file()

### Community 87 - "execute_continuous_retrain_cycle"
Cohesion: 0.32
Nodes (7): enrich_features_with_alpha_interactions(), execute_continuous_retrain_cycle(), Any, DataFrame, Enriches standard feature matrix with non-linear interaction terms., Executes an end-to-end continuous learning and model boosting cycle: 1.…, test_enrich_features_with_alpha_interactions()

### Community 88 - "opening_range_runner.py"
Cohesion: 0.32
Nodes (6): Any, Paper 25 Live Runner: Opening Range Breakout (ORB) on Top Stocks in Play.…, Executes a live 5-Minute Opening Range Breakout scan across top liquid assets:…, run_opening_range_session(), Verify that ORB live session executes, filters stocks in play, and saves latest…, test_run_opening_range_session()

### Community 89 - "correlation_matrix.py"
Cohesion: 0.38
Nodes (6): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…

### Community 90 - ".get_closed_trades_df"
Cohesion: 0.29
Nodes (4): DataFrame, Returns a DataFrame of current open holdings with Scale-Out status., Returns a DataFrame of trade history with full company names., Returns equity history as a DatetimeIndex DataFrame.

### Community 91 - "ws_quantum_tournament.py"
Cohesion: 0.53
Nodes (5): load_safety_benchmarks(), load_tournament_results(), Any, Workspace 14: 25-Paper Quantum Tournament, Live Omni-Hybrid Pipeline & Risk…, render_quantum_tournament_workspace()

### Community 92 - "block_external_alerts"
Cohesion: 0.50
Nodes (3): block_external_alerts(), fixture, Autouse fixture that prevents tests from sending real outbound network calls to…

## Knowledge Gaps
- **39 isolated node(s):** `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs`, `3. Mislabeled methodology`, `4. Results-file / README consistency` (+34 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `utils.py` to `data_ingestion.py`, `quant_engine.py`, `get_market_timestamp`, `run_backtest`, `drl_policy_agent.py`, `test_all_14_papers.py`, `test_statistical_arbitrage.py`, `agent_committee.py`, `TradingEnvironment`, `CloudDataLake`, `ws_alternative_data.py`, `analyze_supply_chain_spillover`, `calculate_hrp_weights`, `run_temporal_fusion_forecast`, `ws_deep_quant.py`, `meta_ensemble.py`, `OnlineNewtonStepOptimizer`, `ws_live_prediction.py`, `social_sentiment.py`, `compute_dark_pool_sentiment`, `test_omnichannel_mobile.py`, `daily_scanner.py`, `compute_lead_lag_matrix`, `black_swan_simulator.py`, `compute_smart_money_insider_score`, `AlpacaBrokerBridge`, `realtime_tracker.py`, `PaperBroker`, `datetime`, `options_surface.py`, `AICopilotEngine`, `liquidity_heatmap.py`, `preprocessing.py`, `SuperEnsembleClassifier`, `autonomous_trader.py`, `analyze_sec_filing_diff`, `get_sentiment`, `test_pillar2_alternative_data.py`, `strategy_incubator.py`, `webhook_dispatcher.py`, `test_rebalancer_and_tearsheet.py`, `get_news`, `get_price_history`, `insider_signals.py`, `api.py`, `morning_briefing.py`, `cli.py`, `generate_comprehensive_factsheet`, `calculate_portfolio_diversity_grade`, `generate_institutional_pdf_tearsheet`, `opening_range_runner.py`, `correlation_matrix.py`?**
  _High betweenness centrality (0.264) - this node is a cross-community bridge._
- **Why does `get_price_history()` connect `get_price_history` to `data_ingestion.py`, `run_backtest`, `drl_policy_agent.py`, `test_all_14_papers.py`, `test_statistical_arbitrage.py`, `agent_committee.py`, `calculate_hrp_weights`, `daily_scanner.py`, `realtime_tracker.py`, `preprocessing.py`, `calculate_deflated_sharpe_ratio`, `app.py`, `utils.py`, `strategy_incubator.py`, `preprocess_data`, `get_news`, `test_data_ingestion.py`, `morning_briefing.py`, `run_full_quant_experiment`, `calculate_portfolio_diversity_grade`, `opening_range_runner.py`, `correlation_matrix.py`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `GaussianHMMRegimeDetector` connect `GaussianHMMRegimeDetector` to `benchmark_papers_15_24.py`, `test_papers_15_24.py`, `mega_tournament_simulation.py`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs` to the rest of the system?**
  _39 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `DLinearTCNModel` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._
- **Should `data_ingestion.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14210526315789473 - nodes in this community are weakly interconnected._