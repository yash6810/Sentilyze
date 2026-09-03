# Graph Report - Sentilyze  (2026-09-03)

## Corpus Check
- 542 files · ~746,034 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1685 nodes · 3543 edges · 92 communities (88 shown, 4 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c332b156`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- deep_learning_model.py
- get_price_history
- quant_engine.py
- daily_scanner.py
- run_backtest
- train_drl_policy
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
- smart_trader_engine.py
- meta_ensemble.py
- OnlineNewtonStepOptimizer
- ws_live_prediction.py
- social_sentiment.py
- compute_dark_pool_sentiment
- test_omnichannel_mobile.py
- .optimize_allocation
- preprocess_data
- compute_lead_lag_matrix
- black_swan_simulator.py
- compute_smart_money_insider_score
- AlpacaBrokerBridge
- realtime_tracker.py
- PaperBroker
- calculate_15min_opening_range
- get_prediction_on_latest_data
- AICopilotEngine
- triple_convex_engine.py
- test_feature_engineering.py
- SuperEnsembleClassifier
- calculate_deflated_sharpe_ratio
- Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform
- autonomous_trader.py
- app.py
- TickerSentinelSwarm
- run_all_14_papers_benchmark
- analyze_sec_filing_diff
- ui/__init__.py
- Contributor Covenant Code of Conduct
- get_sentiment
- ws_autonomous_trader.py
- test_pillar2_alternative_data.py
- utils.py
- strategy_incubator.py
- How Can I Contribute?
- run_daily_market_scan
- 🧪 Experimental & Simulated Research Prototypes
- Sentilyze — Standing Audit Protocol
- fetch_live_quote
- get_us_market_session
- _load_sentiment_analyzer
- update_live_holdings_prices_and_alert_discord
- run_unified_institutional_pipeline
- rules/graphify.md
- workflows/graphify.md
- GaussianHMMRegimeDetector
- EWMACorrelationMonitor
- OpeningRangeBreakout
- CUSUMDetector
- PageHinkleyDetector
- calculate_macro_liquidity_metrics
- ws_insider_radar.py
- temp_data_dir
- test_api.py
- mega_tournament_simulation.py
- ADWINDetector
- test_papers_15_24.py
- generate_morning_briefing_text
- convene_trading_committee
- run_cppi_backtest
- DCCCorrelation
- blend_model_predictions
- generate_comprehensive_factsheet
- risk_constrained_kelly_allocation
- test_ui_modules.py
- calculate_portfolio_diversity_grade
- ws_quantum_tournament.py
- generate_audio_script
- run_batch_training.py
- correlation_matrix.py
- .get_closed_trades_df
- block_external_alerts

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
- `test_paper9_when_agents_trade_scanner()` --calls--> `run_daily_market_scan()`  [EXTRACTED]
  tests/test_all_14_papers.py → src/daily_scanner.py
- `test_autonomous_cycle_execution()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_idempotency_lock_prevents_overlap()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_evaluate_intraday_scale_out_and_tp2()` --calls--> `PaperBroker`  [EXTRACTED]
  tests/test_realtime_tracker.py → src/paper_broker.py
- `test_preprocess_data_orchestrates_correctly()` --calls--> `preprocess_data()`  [EXTRACTED]
  tests/test_preprocessing.py → src/preprocessing.py

## Import Cycles
- None detected.

## Communities (92 total, 4 thin omitted)

### Community 0 - "deep_learning_model.py"
Cohesion: 0.11
Nodes (26): create_sliding_window_tensors(), DLinearTCNModel, load_dlinear_model(), predict_momentum_probability(), Any, DataFrame, Tensor, High-Efficiency Deep Learning Engine: DLinear + Temporal Convolutional Network… (+18 more)

### Community 1 - "get_price_history"
Cohesion: 0.06
Nodes (54): Multi-Trial Empirical Benchmark Suite for Triple-Convex Quantum Engine.…, _fetch_alpaca_news(), _fetch_alpaca_price_history(), _fetch_direct_yahoo_chart(), _fetch_eodhd_price_history(), _fetch_finnhub_news(), _fetch_fmp_price_history(), _fetch_google_news_rss() (+46 more)

### Community 2 - "quant_engine.py"
Cohesion: 0.15
Nodes (24): MasterQuantPipelineResult, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain() (+16 more)

### Community 3 - "daily_scanner.py"
Cohesion: 0.16
Nodes (27): format_signal_card(), Any, Dispatches a crystal-clear, high-impact Discord card for live autonomous trade…, Construct a standardized trade signal data payload. Args: ticker (str): Stock…, Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord., Dispatches real-time Reddit r/wallstreetbets & Stocktwits hype spike alerts., Sends a comprehensive institutional morning macro regime, portfolio health,…, Sends a rich formatted trade alert card to a Discord channel via Webhook. (+19 more)

### Community 4 - "run_backtest"
Cohesion: 0.10
Nodes (35): Figure, _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition(), calculate_performance_metrics(), _calculate_trade_outcomes() (+27 more)

### Community 5 - "train_drl_policy"
Cohesion: 0.14
Nodes (17): ActorCriticPolicy, DRLTradingEnvironment, evaluate_drl_policy_action(), Any, ndarray, Tensor, Trains an Actor-Critic DRL policy agent on historical market returns and news…, Performs fast sub-millisecond inference with the trained Actor-Critic policy. (+9 more)

### Community 6 - "test_all_14_papers.py"
Cohesion: 0.11
Nodes (23): Master Academic Research Papers Empirical Benchmark Suite (All 14 Papers).…, calculate_almgren_chriss_trajectory(), Any, Paper 3: Almgren & Chriss (2000) - Optimal Execution of Portfolio Transactions.…, Computes Almgren-Chriss optimal trading trajectory. x_j = 2 * sinh(0.5 * kappa…, detect_negative_cycle_arbitrage(), Any, Paper 5: Negative Cycle Detection on Exchange Log-Rate Digraphs (Bellman-Ford).… (+15 more)

### Community 7 - "test_statistical_arbitrage.py"
Cohesion: 0.19
Nodes (27): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any, Series (+19 more)

### Community 8 - "fetch_financial_statements"
Cohesion: 0.06
Nodes (59): _persist_ablation_results(), Any, 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), analyze_debt_maturity_wall() (+51 more)

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
Cohesion: 0.10
Nodes (30): auto_register_ipo_ticker(), fetch_pre_ipo_radar_summary(), fetch_sec_edgar_ipo_filings(), get_pre_ipo_pipeline_df(), Any, DataFrame, IPO & Pre-IPO Intelligence Radar for Sentilyze. Pillar 9 Alternative Asset…, Fetches real-time SEC Form S-1 / S-1/A IPO registration statements from SEC… (+22 more)

### Community 13 - "analyze_supply_chain_spillover"
Cohesion: 0.14
Nodes (15): analyze_supply_chain_spillover(), Any, ndarray, Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for…, Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)., Executes a Graph Convolutional Network (GCN) layer: H_new = ReLU(A_hat * H * W)…, Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab…, High-level entry point to run GNN supply chain shock propagation. (+7 more)

### Community 14 - "calculate_hrp_weights"
Cohesion: 0.08
Nodes (40): compute_performance_metrics(), Any, DataFrame, Series, Empirical Quant Experimentation & Multi-Asset Ablation Benchmark Suite.…, Executes empirical ablation benchmark across the full asset universe., Simulates walk-forward strategy execution with or without advanced quant…, Computes key quant performance metrics. (+32 more)

### Community 15 - "run_temporal_fusion_forecast"
Cohesion: 0.14
Nodes (15): Any, DataFrame, ndarray, Temporal Fusion Transformer (TFT) & Multi-Horizon Self-Attention Engine for…, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with… (+7 more)

### Community 16 - "smart_trader_engine.py"
Cohesion: 0.21
Nodes (16): calculate_smart_money_zones(), calculate_structural_trailing_stop(), evaluate_multi_timeframe_confluence(), find_swing_pivots(), Any, DataFrame, Institutional Smart Money Market Structure & Price-Action Engine for Sentilyze.…, Ratchets the Stop-Loss up structurally behind higher swing lows. Rules: 1.… (+8 more)

### Community 17 - "meta_ensemble.py"
Cohesion: 0.10
Nodes (19): DynamicSharpeMetaEnsemble, MetaEnsembleClassifier, DataFrame, ndarray, Series, Institutional Multi-Model Meta-Ensemble Engine for Sentilyze. Pillar 1 Core…, Generates binary class prediction (0 = Hold/Sell, 1 = Buy) using soft-voting…, Instantiates and fits the Meta-Ensemble classifier. (+11 more)

### Community 18 - "OnlineNewtonStepOptimizer"
Cohesion: 0.16
Nodes (11): OnlineNewtonStepOptimizer, DataFrame, ndarray, Online Newton Step (ONS) Portfolio Engine (Agarwal, Hazan, Kale). A polynomial-…, Polynomial-Time ONS Portfolio Engine (Hazan et al.)., Processes price relatives (r_t = Close_t / Close_{t-1}) and updates weights in…, Fast O(d log d) Euclidean projection onto probability simplex., Runs ONS sequence through time and outputs daily allocations and portfolio… (+3 more)

### Community 19 - "ws_live_prediction.py"
Cohesion: 0.15
Nodes (21): detect_classical_chart_patterns(), generate_ai_chart_explanation(), match_historical_chart_twins(), normalize_waveform(), Any, DataFrame, ndarray, Normalizes a price series to [0, 1] range and interpolates to fixed length. (+13 more)

### Community 20 - "social_sentiment.py"
Cohesion: 0.23
Nodes (13): calculate_social_buzz_metrics(), fetch_social_sentiment_tracker(), Any, Social Sentiment Velocity & Retail Multi-Platform Scraper for Sentilyze. Pillar…, Scrapes real-time streaming retail sentiment from Stocktwits public symbol…, Scrapes tech community discussions on AI catalysts (OpenAI, Anthropic, Nvidia)…, Computes retail sentiment velocity and flow conviction metrics., High-level entry point to retrieve calibrated real-time social buzz metrics for… (+5 more)

### Community 21 - "compute_dark_pool_sentiment"
Cohesion: 0.24
Nodes (13): compute_dark_pool_sentiment(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent institutional off-exchange block trades and dark pool prints., Scans option chain contracts where daily volume significantly exceeds open…, Synthesizes dark pool prints and unusual options flow into a unified…, scan_abnormal_options_vol_oi(), scan_dark_pool_blocks() (+5 more)

### Community 22 - "test_omnichannel_mobile.py"
Cohesion: 0.15
Nodes (15): answer_financial_query(), Any, Natural Language Financial Q&A Agent for Sentilyze. Pillar 7 Mobile &…, Parses natural language questions and routes them to quantitative engines.…, generate_smartwatch_glance_payload(), Any, Generates structured complication JSON for Apple Watch (watchOS) and Wear OS., format_whatsapp_trade_alert() (+7 more)

### Community 23 - ".optimize_allocation"
Cohesion: 0.40
Nodes (4): Any, DataFrame, Series, Solves the friction-aware convex optimization problem in polynomial time. Args:…

### Community 24 - "preprocess_data"
Cohesion: 0.10
Nodes (36): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+28 more)

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
Cohesion: 0.11
Nodes (23): Autonomous Multi-Agent Trading Committee & Deliberation Engine for Sentilyze.…, AI Trade Copilot & Conversational Analyst for Sentilyze. Provides natural…, Any, Paper 25 Live Runner: Opening Range Breakout (ORB) on Top Stocks in Play.…, Executes a live 5-Minute Opening Range Breakout scan across top liquid assets:…, run_opening_range_session(), check_live_news_sentiment_shock(), evaluate_intraday_execution() (+15 more)

### Community 30 - "PaperBroker"
Cohesion: 0.11
Nodes (20): PaperBroker, Any, Executes daily quantitative scan results using the Concentrated Top-2 + Scale-…, Institutional Multi-Stage Quantitative Execution Broker ($100k Account).…, Loads existing portfolio state from JSON or initializes a fresh $100k account., Updates total equity, unrealized PnL, and win rates., Returns high-level KPI metrics for the portfolio dashboard., Executes an institutional BUY order into the virtual paper broker ledger.… (+12 more)

### Community 31 - "calculate_15min_opening_range"
Cohesion: 0.26
Nodes (11): calculate_15min_opening_range(), find_low_of_day_pullback_entry(), is_opening_15min_whipsaw_period(), Any, DataFrame, Checks if current Eastern Time is within the hectic 09:30 - 09:45 EDT opening…, Calculates the 15-minute Opening Range (High, Low, Midpoint) established…, Evaluates whether a stock is in the optimal 'Low-of-Day Pullback & Volume… (+3 more)

### Community 32 - "get_prediction_on_latest_data"
Cohesion: 0.19
Nodes (16): get_prediction_on_latest_data(), Any, DataFrame, Series, Gets a prediction from the model for the latest available data point. Args:…, Train the XGBoost model using Walk-Forward Optimization (WFO) alongside a…, train_model(), Generates sample data for testing. Needs at least 520 rows for WFO… (+8 more)

### Community 33 - "AICopilotEngine"
Cohesion: 0.24
Nodes (8): AICopilotEngine, Any, Conversational intelligence engine that parses queries and generates analytical…, Interprets user prompt and routes to appropriate financial analytical…, test_copilot_committee_query(), test_copilot_portfolio_query(), test_copilot_stress_query(), test_copilot_ticker_analysis_query()

### Community 34 - "triple_convex_engine.py"
Cohesion: 0.19
Nodes (10): PolyTimeConvexOptimizer, Stanford Multi-Period Convex Portfolio Optimization Engine (Boyd et al.).…, Polynomial-Time Convex Portfolio Optimizer with Market Frictions (Boyd et al.)., Triple-Convex Quantum Execution Engine. Fuses the top quantitative research…, Unified High-Expectancy, Minimum-Drawdown, Sub-15ms Execution Engine., TripleConvexEngine, test_paper2_boyd_convex_optimizer(), test_polytime_convex_optimizer() (+2 more)

### Community 35 - "test_feature_engineering.py"
Cohesion: 0.17
Nodes (19): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, DataFrame (+11 more)

### Community 36 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (15): Any, DataFrame, ndarray, Series, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used., Loads all 3 models natively. (+7 more)

### Community 37 - "calculate_deflated_sharpe_ratio"
Cohesion: 0.12
Nodes (18): Any, Runs multi-trial empirical testing and saves verified metrics to JSON., run_triple_convex_multi_trial_benchmark(), apply_triple_barrier_labeling(), calculate_deflated_sharpe_ratio(), DataFrame, Series, Marcos López de Prado's Triple-Barrier Method & Deflated Sharpe Ratio (DSR).… (+10 more)

### Community 38 - "Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform"
Cohesion: 0.13
Nodes (14): 1. Run in 1-Click (No Installation Required), 2. Local Setup & Installation, 3. Run the 4-Agent Quantitative CLI, 4. Launch the Streamlit Mission Control, 5. Run Full Test Suite (240+ Unit Tests), 📊 Empirical Alpha Attribution & Benchmarks, 🏛️ Grounded 4-Agent Deliberation Council, 🖥️ Interactive Streamlit App & 23 Mission Control Workspaces (+6 more)

### Community 39 - "autonomous_trader.py"
Cohesion: 0.07
Nodes (35): execute_committee_order(), Executes a committee-approved buy order into the virtual paper broker ledger., AutonomousTradingEngine, check_daily_loss_circuit_breaker(), is_kill_switch_active(), load_universe_tickers(), Any, Autonomous Live Trading & News Intelligence Engine for Sentilyze. Institutional… (+27 more)

### Community 40 - "app.py"
Cohesion: 0.11
Nodes (26): load_universe_tickers(), main(), Sentilyze - Institutional Algorithmic Trading & MLOps Platform. Modular Master…, Loads active S&P 100 universe tickers., get_market_status(), Any, Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Wraps HTML content inside an institutional frosted glass container. (+18 more)

### Community 41 - "TickerSentinelSwarm"
Cohesion: 0.14
Nodes (14): detect_peak_crest_exhaustion(), Any, Dedicated Micro-Agent assigned to monitor a single stock position 24/7., Audits live price tick and determines peak crest execution., Manages the full swarm of Dedicated Ticker Sentinels across all open positions., Synchronizes active sentinels with current portfolio open positions., Audits all active sentinels concurrently., Detects if a stock has reached the crest/peak of its 15-minute momentum wave… (+6 more)

### Community 42 - "run_all_14_papers_benchmark"
Cohesion: 0.18
Nodes (11): Any, Executes empirical backtests comparing all 14 academic paper methodologies., run_all_14_papers_benchmark(), ChiefRiskOfficerAgent, compute_fractional_kelly_sizing(), Agent 4: Chief Risk Officer (CRO) — Synthesizes Votes, Computes Kelly Sizing,…, Computes true mathematical fractional Kelly Criterion position sizing: f* = (p…, test_cro_agent_approval_and_veto() (+3 more)

### Community 43 - "analyze_sec_filing_diff"
Cohesion: 0.32
Nodes (7): analyze_sec_filing_diff(), compute_text_similarity_and_diff(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Computes lexical and semantic diff metrics between consecutive filings., Retrieves and compares the most recent and prior SEC filings for a company.…, test_sec_filing_diff_analysis()

### Community 47 - "Contributor Covenant Code of Conduct"
Cohesion: 0.15
Nodes (12): 1. Correction, 2. Warning, 3. Temporary Ban, 4. Permanent Ban, Attribution, Contributor Covenant Code of Conduct, Enforcement, Enforcement Guidelines (+4 more)

### Community 48 - "get_sentiment"
Cohesion: 0.12
Nodes (21): clean_financial_text(), get_sentiment(), _parse_analyzer_output(), Any, DataFrame, Analyzes the sentiment of news articles using high-precision FinBERT pipeline,…, Cleans raw financial headlines/descriptions by stripping boilerplate publisher…, Normalizes pipeline output whether given full multi-class probabilities (list… (+13 more)

### Community 49 - "ws_autonomous_trader.py"
Cohesion: 0.21
Nodes (11): calculate_doubling_progress(), compute_compound_position_size(), Any, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Workspace 3: 24/7 Autonomous Broker, Kelly Sizing & Staged Profit Scaler.…, Renders the 24/7 Autonomous Live Trading & News Agent interface., render_autonomous_trader_workspace() (+3 more)

### Community 50 - "test_pillar2_alternative_data.py"
Cohesion: 0.24
Nodes (13): compute_government_and_patent_index(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Synthesizes federal contracting dollars and patent velocity into a single…, Retrieves recent prime federal government contract awards for a company., Tracks recent USPTO patent grants in AI/ML, Semiconductor Design, and Cloud…, track_federal_contract_awards(), track_uspto_patent_momentum() (+5 more)

### Community 51 - "utils.py"
Cohesion: 0.07
Nodes (27): datetime, Logger, AI Chart Pattern Recognition, Geometric Wave Learning & Visual Understanding…, Cloud PostgreSQL Data Lake (Supabase / Neon) Connector for Sentilyze. Pillar 6…, Max Compound Acceleration & +100% Target Doubling Engine for Sentilyze.…, Deep Reinforcement Learning (DRL) Autonomous Policy Agent for Sentilyze.…, Real-Time Earnings Call Transcript & Management Tone Analyzer for Sentilyze.…, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.… (+19 more)

### Community 52 - "strategy_incubator.py"
Cohesion: 0.20
Nodes (15): breed_strategy_generation(), evaluate_3zone_robustness(), load_strategy_vault(), Any, Evolutionary Strategy Incubator & Robustness Lab for Sentilyze. Institutional…, Evaluates a Strategy Genome across 3 distinct zones: 1. In-Sample Train (70%)…, Runs evolutionary genetic algorithm across generations, breeding top survivors., Represents an algorithmic strategy rule DNA. (+7 more)

### Community 53 - "How Can I Contribute?"
Cohesion: 0.33
Nodes (5): Contributing to Sentilyze, How Can I Contribute?, Pull Requests, Reporting Bugs, Suggesting Enhancements

### Community 54 - "run_daily_market_scan"
Cohesion: 0.29
Nodes (7): Scans the entire stock universe defined in stocks.txt, generates tomorrow's…, run_daily_market_scan(), Any, Dispatches formatted HTML morning market digest email via Gmail SMTP., send_email_digest(), Test that run_daily_market_scan executes successfully across mock tickers., test_run_daily_market_scan()

### Community 55 - "🧪 Experimental & Simulated Research Prototypes"
Cohesion: 0.50
Nodes (3): 🧪 Experimental & Simulated Research Prototypes, 🔒 Production Isolation Guarantee, 📁 Prototype Inventory

### Community 56 - "Sentilyze — Standing Audit Protocol"
Cohesion: 0.17
Nodes (11): 1. Fabricated / fake data check, 2. Ticker/input-invariant bugs, 3. Mislabeled methodology, 4. Results-file / README consistency, 5. Duplicate-output smell test, 6. Safety-critical logic sanity check, 7. Silent failure check, 8. Scope creep check (+3 more)

### Community 57 - "fetch_live_quote"
Cohesion: 0.07
Nodes (44): compute_order_book_depth_and_clusters(), compute_volume_profile_and_poc(), Any, Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap for…, Simulates Level 2 market depth and identifies institutional buy/sell liquidity…, Computes Point of Control (POC), Value Area High (VAH), and Value Area Low…, calculate_multileg_payoff(), generate_volatility_surface_mesh() (+36 more)

### Community 58 - "get_us_market_session"
Cohesion: 0.27
Nodes (8): check_market_hours_preflight(), get_us_market_session(), Any, Pre-flight sanity check for automated workflows. Returns True if execution…, Computes the exact real-time US equity market session (NYSE / NASDAQ). Session…, Unit tests for US Market Session & Calendar Engine., test_get_us_market_session_keys(), test_preflight_check()

### Community 59 - "_load_sentiment_analyzer"
Cohesion: 0.21
Nodes (12): clean_headline_data(), _load_sentiment_analyzer(), Any, Thread-safely loads the FinBERT sentiment analysis model and tokenizer once…, Cleans a headline CSV file by removing rows with invalid stock tickers. Caches…, fixture, temp_data_dirs(), test_clean_headline_data_invalid_tickers_no_cache() (+4 more)

### Community 60 - "update_live_holdings_prices_and_alert_discord"
Cohesion: 0.24
Nodes (9): Sub-second live spot price poller for active holdings. Updates current price,…, Continuous 5-Minute Intraday Guardian Loop during active market hours., run_5min_guardian_loop(), update_live_holdings_prices_and_alert_discord(), apply_high_watermark_profit_lock(), Guarantees that once a trade reaches peak profit, the bot NEVER gives back >…, Unit tests for High-Watermark Peak Profit Ratchet (75% Lock Floor)., test_high_watermark_does_not_lower_sl_on_pullback() (+1 more)

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

### Community 70 - "calculate_macro_liquidity_metrics"
Cohesion: 0.50
Nodes (4): calculate_macro_liquidity_metrics(), Any, Computes real-time macroeconomic liquidity indicators and yield curve dynamics.…, test_calculate_macro_liquidity_metrics()

### Community 71 - "ws_insider_radar.py"
Cohesion: 0.26
Nodes (12): calculate_insider_conviction_score(), fetch_insider_transactions(), Any, Computes the Quantitative Insider Conviction Index (0 to 100 Score) and detects…, Screens a universe of tickers and returns the highest-ranking insider buying…, Fetches recent SEC Form 4 insider transactions for a specific ticker. Includes…, scan_universe_insider_catalysts(), Workspace: Smart-Money Executive & Institutional Insider Radar. SEC Form 4… (+4 more)

### Community 72 - "temp_data_dir"
Cohesion: 0.67
Nodes (3): fixture, Fixture to set a temporary data directory for tests., temp_data_dir()

### Community 74 - "mega_tournament_simulation.py"
Cohesion: 0.18
Nodes (9): grossman_zhou_allocation(), Any, Paper 18: Grossman-Zhou Optimal Drawdown-Constrained Strategy. Source: Grossman…, Compute the Grossman-Zhou optimal risky allocation under a drawdown constraint…, Paper 15: Gaussian Hidden Markov Model for Market Regime Detection. Source:…, load_cached_universe(), Mega-Tournament Simulation: Mixing 25 Quant Papers across Trading Teams.…, run_team_tournament() (+1 more)

### Community 75 - "ADWINDetector"
Cohesion: 0.19
Nodes (7): ADWINDetector, Any, ndarray, Paper 21: ADWIN (Adaptive Windowing) Drift Detector. Source: Bifet & Gavaldà…, ADWIN drift detector with Hoeffding bound. Maintains a variable-length window…, Add one observation. Returns whether drift was detected. If drift is detected,…, TestADWIN

### Community 76 - "test_papers_15_24.py"
Cohesion: 0.20
Nodes (10): calculate_cdar(), optimize_cdar_portfolio(), Any, DataFrame, ndarray, Paper 19: Conditional Drawdown-at-Risk (CDaR) Portfolio Optimization. Source:…, Calculate Conditional Drawdown-at-Risk: the expected drawdown in the worst…, Optimize portfolio weights to minimize CDaR. Simplified approach: compute CDaR… (+2 more)

### Community 77 - "generate_morning_briefing_text"
Cohesion: 0.27
Nodes (9): generate_morning_briefing_text(), Any, Synthesizes executive audio file (.mp3) using Google Text-to-Speech (gTTS).…, Assembles a comprehensive, institutional Wall Street Pre-Market Morning…, synthesize_briefing_audio(), Workspace: AI Pre-Market Audio & Executive Morning Intelligence Briefing.…, render_morning_briefing_workspace(), test_generate_morning_briefing_text() (+1 more)

### Community 78 - "convene_trading_committee"
Cohesion: 0.10
Nodes (28): Root entry point for Sentilyze CLI. Run directly with: python sentilyze.py NVDA…, audit_full_universe_committee(), convene_trading_committee(), ForensicFundamentalAgent, _persist_committee_resolution(), Any, Agent 2: Evaluates FinBERT Deep NLP Sentiment across Live News Streams., Agent 3: Evaluates Real Financial Statements, Piotroski F-Score, and DCF… (+20 more)

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
Cohesion: 0.27
Nodes (8): generate_comprehensive_factsheet(), Any, Series, Computes over 30 institutional hedge-fund risk, performance, and drawdown…, Workspace: Institutional Risk & Alpha Performance Factsheet. Quantitative…, render_performance_factsheet_workspace(), test_generate_comprehensive_factsheet_custom_series(), test_generate_comprehensive_factsheet_default()

### Community 83 - "risk_constrained_kelly_allocation"
Cohesion: 0.24
Nodes (6): Any, ndarray, Paper 23: Risk-Constrained Kelly Gambling. Source: Busseti, Ryu, Boyd —…, Compute optimal Kelly allocation with drawdown probability constraint.…, risk_constrained_kelly_allocation(), TestRiskKelly

### Community 84 - "test_ui_modules.py"
Cohesion: 0.40
Nodes (3): inject_custom_theme(), Dynamic Bespoke Theme Engine for Sentilyze. Supports 3 Institutional Presets:…, Injects high-performance, bespoke CSS styling into the Streamlit app.

### Community 85 - "calculate_portfolio_diversity_grade"
Cohesion: 0.27
Nodes (9): calculate_portfolio_diversity_grade(), Any, DataFrame, Workspace: Portfolio Diversity & Correlation Health Grader. Institutional…, render_portfolio_diversity_workspace(), test_custom_returns_correlated(), test_custom_returns_diverse(), test_empty_portfolio() (+1 more)

### Community 86 - "ws_quantum_tournament.py"
Cohesion: 0.19
Nodes (13): generate_institutional_pdf_tearsheet(), Institutional Quantitative Tearsheet & Factsheet Generator for Sentilyze.…, Generates a publication-grade institutional quantitative tearsheet PDF. Returns…, get_market_timestamp(), load_safety_benchmarks(), load_tournament_results(), Any, Workspace 14: 25-Paper Quantum Tournament, Live Omni-Hybrid Pipeline & Risk… (+5 more)

### Community 87 - "generate_audio_script"
Cohesion: 0.50
Nodes (5): generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses gTTS if available, or…, synthesize_morning_audio()

### Community 88 - "run_batch_training.py"
Cohesion: 0.67
Nodes (3): main(), Batch Universe Trainer for Remaining S&P 100 Tickers., run_single()

### Community 89 - "correlation_matrix.py"
Cohesion: 0.38
Nodes (6): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…

### Community 90 - ".get_closed_trades_df"
Cohesion: 0.29
Nodes (4): DataFrame, Returns a DataFrame of current open holdings with Scale-Out status., Returns a DataFrame of trade history with full company names., Returns equity history as a DatetimeIndex DataFrame.

### Community 92 - "block_external_alerts"
Cohesion: 0.50
Nodes (3): block_external_alerts(), fixture, Autouse fixture that prevents tests from sending real outbound network calls to…

## Knowledge Gaps
- **39 isolated node(s):** `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs`, `3. Mislabeled methodology`, `4. Results-file / README consistency` (+34 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `utils.py` to `deep_learning_model.py`, `get_price_history`, `quant_engine.py`, `daily_scanner.py`, `run_backtest`, `test_all_14_papers.py`, `test_statistical_arbitrage.py`, `fetch_financial_statements`, `TradingEnvironment`, `CloudDataLake`, `ws_alternative_data.py`, `analyze_supply_chain_spillover`, `calculate_hrp_weights`, `run_temporal_fusion_forecast`, `smart_trader_engine.py`, `meta_ensemble.py`, `OnlineNewtonStepOptimizer`, `social_sentiment.py`, `compute_dark_pool_sentiment`, `test_omnichannel_mobile.py`, `preprocess_data`, `compute_lead_lag_matrix`, `black_swan_simulator.py`, `compute_smart_money_insider_score`, `AlpacaBrokerBridge`, `realtime_tracker.py`, `triple_convex_engine.py`, `calculate_deflated_sharpe_ratio`, `autonomous_trader.py`, `analyze_sec_filing_diff`, `test_pillar2_alternative_data.py`, `strategy_incubator.py`, `fetch_live_quote`, `convene_trading_committee`, `ws_quantum_tournament.py`, `run_batch_training.py`, `correlation_matrix.py`?**
  _High betweenness centrality (0.265) - this node is a cross-community bridge._
- **Why does `get_price_history()` connect `get_price_history` to `run_backtest`, `train_drl_policy`, `test_all_14_papers.py`, `test_statistical_arbitrage.py`, `fetch_financial_statements`, `calculate_hrp_weights`, `preprocess_data`, `realtime_tracker.py`, `calculate_deflated_sharpe_ratio`, `app.py`, `run_all_14_papers_benchmark`, `ws_autonomous_trader.py`, `utils.py`, `strategy_incubator.py`, `update_live_holdings_prices_and_alert_discord`, `calculate_macro_liquidity_metrics`, `generate_morning_briefing_text`, `convene_trading_committee`, `calculate_portfolio_diversity_grade`, `correlation_matrix.py`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Why does `GaussianHMMRegimeDetector` connect `GaussianHMMRegimeDetector` to `benchmark_papers_15_24.py`, `test_papers_15_24.py`, `mega_tournament_simulation.py`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs` to the rest of the system?**
  _39 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `deep_learning_model.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10887096774193548 - nodes in this community are weakly interconnected._
- **Should `get_price_history` be split into smaller, more focused modules?**
  _Cohesion score 0.060109289617486336 - nodes in this community are weakly interconnected._