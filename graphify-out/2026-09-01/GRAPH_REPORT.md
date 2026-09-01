# Graph Report - Sentilyze  (2026-09-01)

## Corpus Check
- 472 files · ~667,752 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1204 nodes · 2544 edges · 67 communities (63 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `be35f499`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- fetch_financial_statements
- get_price_history
- quant_engine.py
- daily_scanner.py
- run_backtest
- app.py
- preprocessing.py
- test_statistical_arbitrage.py
- convene_trading_committee
- TradingEnvironment
- test_feature_engineering.py
- CloudDataLake
- ws_alternative_data.py
- SupplyChainGraphNetwork
- advanced_quant_experiments.py
- run_temporal_fusion_forecast
- ws_deep_quant.py
- meta_ensemble.py
- preprocess_data
- smart_trader_engine.py
- update_live_holdings_prices_and_alert_discord
- compute_dark_pool_sentiment
- test_omnichannel_mobile.py
- get_prediction_on_latest_data
- load_model
- compute_lead_lag_matrix
- black_swan_simulator.py
- test_pillar2_alternative_data.py
- AlpacaBrokerBridge
- telegram_bot.py
- PaperBroker
- opening_range_engine.py
- options_surface.py
- AICopilotEngine
- fetch_live_quote
- run_committee_ablation_backtest
- SuperEnsembleClassifier
- TickerSentinelSwarm
- Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform
- test_autonomous_trader.py
- render_workspace_header
- test_rebalancer_and_tearsheet.py
- AutonomousTradingEngine
- reddit_premarket_station.py
- ui/__init__.py
- Contributor Covenant Code of Conduct
- get_sentiment
- calculate_doubling_progress
- temp_data_dir
- get_logger
- cli.py
- How Can I Contribute?
- audio_briefing.py
- 🧪 Experimental & Simulated Research Prototypes
- Sentilyze — Standing Audit Protocol
- execute_continuous_retrain_cycle
- render_conviction_gauge
- correlation_matrix.py
- .get_closed_trades_df
- run_unified_institutional_pipeline
- rules/graphify.md
- workflows/graphify.md
- main
- test_api.py

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 71 edges
2. `PaperBroker` - 44 edges
3. `get_price_history()` - 37 edges
4. `run_unified_institutional_pipeline()` - 32 edges
5. `fetch_live_quote()` - 32 edges
6. `preprocess_data()` - 28 edges
7. `handle_telegram_command()` - 28 edges
8. `get_news()` - 24 edges
9. `fetch_financial_statements()` - 23 edges
10. `convene_trading_committee()` - 22 edges

## Surprising Connections (you probably didn't know these)
- `test_autonomous_cycle_execution()` --uses--> `AutonomousTradingEngine`  [INFERRED]
  tests/test_autonomous_trader.py → src/autonomous_trader.py
- `test_autonomous_cycle_execution()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_idempotency_lock_prevents_overlap()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_evaluate_intraday_scale_out_and_tp2()` --calls--> `PaperBroker`  [EXTRACTED]
  tests/test_realtime_tracker.py → src/paper_broker.py
- `predict()` --calls--> `get_prediction_on_latest_data()`  [EXTRACTED]
  api.py → src/modeling.py

## Import Cycles
- None detected.

## Communities (67 total, 4 thin omitted)

### Community 0 - "fetch_financial_statements"
Cohesion: 0.20
Nodes (19): calculate_altman_z_score(), calculate_dcf_fair_value(), calculate_piotroski_f_score(), fetch_financial_statements(), _generate_calibrated_financials(), generate_spider_radar_profile(), Any, Institutional Fundamental Valuation & Forensic Accounting Engine for Sentilyze.… (+11 more)

### Community 1 - "get_price_history"
Cohesion: 0.06
Nodes (53): _fetch_alpaca_news(), _fetch_alpaca_price_history(), _fetch_direct_yahoo_chart(), _fetch_eodhd_price_history(), _fetch_finnhub_news(), _fetch_fmp_price_history(), _fetch_google_news_rss(), _fetch_marketaux_news() (+45 more)

### Community 2 - "quant_engine.py"
Cohesion: 0.15
Nodes (24): MasterQuantPipelineResult, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain() (+16 more)

### Community 3 - "daily_scanner.py"
Cohesion: 0.10
Nodes (37): format_signal_card(), Any, Dispatches a high-impact Discord card for live autonomous trade lifecycle…, Construct a standardized trade signal data payload. Args: ticker (str): Stock…, Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord., Dispatches real-time Reddit r/wallstreetbets & Stocktwits hype spike alerts., Sends a comprehensive institutional morning macro regime, portfolio health,…, Sends a rich formatted trade alert card to a Discord channel via Webhook. (+29 more)

### Community 4 - "run_backtest"
Cohesion: 0.10
Nodes (35): Figure, _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition(), calculate_performance_metrics(), _calculate_trade_outcomes() (+27 more)

### Community 5 - "app.py"
Cohesion: 0.16
Nodes (16): load_universe_tickers(), main(), Sentilyze - Institutional Algorithmic Trading & MLOps Platform. Modular Master…, Loads active S&P 100 universe tickers., get_market_status(), Any, Calculates live US Market (NYSE/NASDAQ) status based on Eastern Time., inject_custom_theme() (+8 more)

### Community 6 - "preprocessing.py"
Cohesion: 0.31
Nodes (4): Continuous Model Self-Training & Accuracy Boosting Engine for Sentilyze. Self-…, _get_api_key(), Safely attempts to retrieve the API key from Streamlit secrets, falling back to…, Workspace 1: Live Directional Predictions & Fast Real-Time Inference. Features:…

### Community 7 - "test_statistical_arbitrage.py"
Cohesion: 0.19
Nodes (25): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any, Series (+17 more)

### Community 8 - "convene_trading_committee"
Cohesion: 0.11
Nodes (27): 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, audit_full_universe_committee(), ChiefRiskOfficerAgent, compute_fractional_kelly_sizing(), convene_trading_committee(), execute_committee_order(), ForensicFundamentalAgent, _persist_committee_resolution() (+19 more)

### Community 9 - "TradingEnvironment"
Cohesion: 0.14
Nodes (16): optimize_rl_position_allocation(), PPOPolicyAgent, Any, ndarray, ⚠️ EXPERIMENTAL / RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM PRODUCTION…, Computes mean action (leverage) between 0.0 and 2.0., Estimates state value., Trains Actor-Critic parameters across historical episodes. (+8 more)

### Community 10 - "test_feature_engineering.py"
Cohesion: 0.17
Nodes (19): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, DataFrame (+11 more)

### Community 11 - "CloudDataLake"
Cohesion: 0.14
Nodes (14): CloudDataLake, Any, Cloud PostgreSQL Data Lake (Supabase / Neon) Connector for Sentilyze. Pillar 6…, Supabase / PostgreSQL Cloud Data Lake Connector., Validates or generates cloud database schema., Syncs local trade executions to the cloud database., Publishes real-time portfolio snapshot to cloud WebSockets channel., generate_twap_order_schedule() (+6 more)

### Community 12 - "ws_alternative_data.py"
Cohesion: 0.15
Nodes (18): auto_register_ipo_ticker(), fetch_pre_ipo_radar_summary(), fetch_sec_edgar_ipo_filings(), get_pre_ipo_pipeline_df(), Any, DataFrame, IPO & Pre-IPO Intelligence Radar for Sentilyze. Pillar 9 Alternative Asset…, Fetches real-time SEC Form S-1 / S-1/A IPO registration statements from SEC… (+10 more)

### Community 13 - "SupplyChainGraphNetwork"
Cohesion: 0.15
Nodes (14): analyze_supply_chain_spillover(), Any, ndarray, Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for…, Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)., Executes a Graph Convolutional Network (GCN) layer: H_new = ReLU(A_hat * H * W)…, Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab…, High-level entry point to run GNN supply chain shock propagation. (+6 more)

### Community 14 - "advanced_quant_experiments.py"
Cohesion: 0.08
Nodes (39): compute_performance_metrics(), Any, DataFrame, Series, Empirical Quant Experimentation & Multi-Asset Ablation Benchmark Suite.…, Executes empirical ablation benchmark across the full asset universe., Simulates walk-forward strategy execution with or without advanced quant…, Computes key quant performance metrics. (+31 more)

### Community 15 - "run_temporal_fusion_forecast"
Cohesion: 0.14
Nodes (15): Any, DataFrame, ndarray, Temporal Fusion Transformer (TFT) & Multi-Horizon Self-Attention Engine for…, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with… (+7 more)

### Community 16 - "ws_deep_quant.py"
Cohesion: 0.22
Nodes (12): analyze_debt_maturity_wall(), calculate_beneish_m_score(), Any, DataFrame, Beneish M-Score Forensic Analyzer & Debt Maturity Wall Radar for Sentilyze.…, Evaluates corporate interest coverage and debt maturity wall runway., Computes the 8-Ratio Beneish M-Score from 2-year comparative SEC financial…, Workspaces 9-13: Deep Quantitative Modeling, GNN Supply Chain, Stress Tests &… (+4 more)

### Community 17 - "meta_ensemble.py"
Cohesion: 0.10
Nodes (19): DynamicSharpeMetaEnsemble, MetaEnsembleClassifier, DataFrame, ndarray, Series, Institutional Multi-Model Meta-Ensemble Engine for Sentilyze. Pillar 1 Core…, Generates binary class prediction (0 = Hold/Sell, 1 = Buy) using soft-voting…, Instantiates and fits the Meta-Ensemble classifier. (+11 more)

### Community 18 - "preprocess_data"
Cohesion: 0.15
Nodes (17): clean_headline_data(), _load_sentiment_analyzer(), preprocess_data(), Any, DataFrame, Orchestrates the data acquisition, sentiment analysis, and feature engineering…, Thread-safely loads the FinBERT sentiment analysis model and tokenizer once…, Cleans a headline CSV file by removing rows with invalid stock tickers. Caches… (+9 more)

### Community 19 - "smart_trader_engine.py"
Cohesion: 0.09
Nodes (38): detect_classical_chart_patterns(), generate_ai_chart_explanation(), match_historical_chart_twins(), normalize_waveform(), Any, DataFrame, ndarray, AI Chart Pattern Recognition, Geometric Wave Learning & Visual Understanding… (+30 more)

### Community 20 - "update_live_holdings_prices_and_alert_discord"
Cohesion: 0.50
Nodes (4): Sub-second live spot price poller for active holdings. Updates current price,…, Continuous 5-Minute Intraday Guardian Loop during active market hours., run_5min_guardian_loop(), update_live_holdings_prices_and_alert_discord()

### Community 21 - "compute_dark_pool_sentiment"
Cohesion: 0.21
Nodes (14): compute_dark_pool_sentiment(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent institutional off-exchange block trades and dark pool prints., Scans option chain contracts where daily volume significantly exceeds open…, Synthesizes dark pool prints and unusual options flow into a unified…, scan_abnormal_options_vol_oi(), scan_dark_pool_blocks() (+6 more)

### Community 22 - "test_omnichannel_mobile.py"
Cohesion: 0.15
Nodes (15): answer_financial_query(), Any, Parses natural language questions and routes them to quantitative engines.…, generate_smartwatch_glance_payload(), Any, Apple Watch & Wear OS Glance Complications API for Sentilyze. Pillar 7 Mobile &…, Generates structured complication JSON for Apple Watch (watchOS) and Wear OS., format_whatsapp_trade_alert() (+7 more)

### Community 23 - "get_prediction_on_latest_data"
Cohesion: 0.18
Nodes (18): get_prediction_on_latest_data(), Any, DataFrame, Series, Save the trained model to a file using XGBoost's native format. This is safer…, Gets a prediction from the model for the latest available data point. Args:…, Train the XGBoost model using Walk-Forward Optimization (WFO) alongside a…, save_model() (+10 more)

### Community 24 - "load_model"
Cohesion: 0.16
Nodes (21): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+13 more)

### Community 25 - "compute_lead_lag_matrix"
Cohesion: 0.21
Nodes (13): compute_lead_lag_matrix(), _granger_f_test(), Any, DataFrame, ndarray, Series, rank_market_price_leaders(), Ranks stocks by their predictive influence (number of peers they statistically… (+5 more)

### Community 26 - "black_swan_simulator.py"
Cohesion: 0.23
Nodes (11): calculate_kelly_sizing(), estimate_market_impact_slippage(), Any, Historical Black Swan Crisis Simulator & Kelly Position Sizing for Sentilyze.…, Calculates optimal position sizing using the Kelly Criterion: Kelly % = W - (1…, Estimates market execution slippage using the Almgren-Chriss square-root impact…, Stress-tests the current portfolio against major historical market crashes.…, simulate_portfolio_crises() (+3 more)

### Community 27 - "test_pillar2_alternative_data.py"
Cohesion: 0.08
Nodes (42): compute_smart_money_insider_score(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent SEC Form 4 insider transactions for a given stock., Retrieves recent Congressional STOCK Act disclosure reports for a ticker., Synthesizes SEC Form 4 and Congressional activity into an overall Smart Money…, track_congressional_stock_disclosures(), track_corporate_insider_filings() (+34 more)

### Community 28 - "AlpacaBrokerBridge"
Cohesion: 0.16
Nodes (14): AlpacaBrokerBridge, Any, Fetches active positions from Alpaca brokerage., Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.…, Verifies active connection to Alpaca Brokerage API., Fetches live Alpaca account equity, buying power, and cash., Submits an institutional Bracket Order: - Entry: Market order - Exit 1: Limit…, patch (+6 more)

### Community 29 - "telegram_bot.py"
Cohesion: 0.20
Nodes (16): build_interactive_inline_keyboard(), handle_telegram_command(), Any, 2-Way Interactive Telegram Bot Controller & Remote Execution Bridge for…, Sends a formatted markdown message with inline buttons to a Telegram chat., Builds interactive inline quick-action buttons for mobile Telegram., Parses and executes Telegram slash commands and callback buttons., send_telegram_bot_message() (+8 more)

### Community 30 - "PaperBroker"
Cohesion: 0.11
Nodes (20): PaperBroker, Any, Executes daily quantitative scan results using the Concentrated Top-2 + Scale-…, Institutional Multi-Stage Quantitative Execution Broker ($100k Account).…, Loads existing portfolio state from JSON or initializes a fresh $100k account., Updates total equity, unrealized PnL, and win rates., Returns high-level KPI metrics for the portfolio dashboard., Executes an institutional BUY order into the virtual paper broker ledger.… (+12 more)

### Community 31 - "opening_range_engine.py"
Cohesion: 0.12
Nodes (23): check_market_hours_preflight(), get_current_ny_time(), get_us_market_session(), Any, Unified US Stock Market (NYSE / NASDAQ) Session & Calendar Engine for…, Pre-flight sanity check for automated workflows. Returns True if execution…, Returns the current precise timestamp in America/New_York (Eastern Time)., Computes the exact real-time US equity market session (NYSE / NASDAQ). Session… (+15 more)

### Community 32 - "options_surface.py"
Cohesion: 0.27
Nodes (10): calculate_multileg_payoff(), generate_volatility_surface_mesh(), Any, 3D Implied Volatility Surface & Multi-Leg Options Strategy Desk for Sentilyze.…, Constructs a 3D Implied Volatility Surface across strike prices and expiration…, Calculates profit and loss (P&L) curves at expiration for institutional multi-…, test_calculate_multileg_payoff_bull_call_spread(), test_calculate_multileg_payoff_iron_condor() (+2 more)

### Community 33 - "AICopilotEngine"
Cohesion: 0.24
Nodes (8): AICopilotEngine, Any, Conversational intelligence engine that parses queries and generates analytical…, Interprets user prompt and routes to appropriate financial analytical…, test_copilot_committee_query(), test_copilot_portfolio_query(), test_copilot_stress_query(), test_copilot_ticker_analysis_query()

### Community 34 - "fetch_live_quote"
Cohesion: 0.16
Nodes (15): compute_order_book_depth_and_clusters(), compute_volume_profile_and_poc(), Any, Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap for…, Simulates Level 2 market depth and identifies institutional buy/sell liquidity…, Computes Point of Control (POC), Value Area High (VAH), and Value Area Low…, fetch_live_quote(), _get_browser_session() (+7 more)

### Community 35 - "run_committee_ablation_backtest"
Cohesion: 0.36
Nodes (7): _persist_ablation_results(), Any, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), test_committee_ablation_study_execution()

### Community 36 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (16): Any, DataFrame, ndarray, Series, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.…, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used. (+8 more)

### Community 37 - "TickerSentinelSwarm"
Cohesion: 0.14
Nodes (15): detect_peak_crest_exhaustion(), Any, Dedicated Ticker Sentinel & Peak-Crest Volume Harvester Swarm for Sentilyze.…, Dedicated Micro-Agent assigned to monitor a single stock position 24/7., Audits live price tick and determines peak crest execution., Manages the full swarm of Dedicated Ticker Sentinels across all open positions., Synchronizes active sentinels with current portfolio open positions., Audits all active sentinels concurrently. (+7 more)

### Community 38 - "Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform"
Cohesion: 0.12
Nodes (16): 1. Run in 1-Click (No Installation Required), 2. Local Setup & Installation, 3. Run the 4-Agent Quantitative CLI, 4. Launch the Streamlit Mission Control, 5. Run Automated Multi-Agent Backtests & Full Test Suite, 🎯 Asymmetric Risk Management & Staged Profit Scaling, 📊 Empirical Alpha Attribution & Benchmarks, 🏛️ Grounded 4-Agent Deliberation Council (+8 more)

### Community 39 - "test_autonomous_trader.py"
Cohesion: 0.10
Nodes (21): check_daily_loss_circuit_breaker(), is_kill_switch_active(), load_universe_tickers(), Task 7: Master Kill Switch Check. Returns True if SENTILYZE_KILL_SWITCH…, Task 8: Independent Max-Daily-Loss Circuit Breaker. Compares current total…, Loads universe of tickers from stocks.txt., clean_lock_file(), fixture (+13 more)

### Community 40 - "render_workspace_header"
Cohesion: 0.17
Nodes (11): Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Wraps HTML content inside an institutional frosted glass container., Renders an executive header banner with live status badge and market clock., render_glass_card(), render_workspace_header(), Workspace 3: 24/7 Autonomous Broker, Kelly Sizing & Staged Profit Scaler.…, Workspace 6: Walk-Forward Backtesting & Performance Tearsheet., Workspace 8: 3D Options Volatility Surface & Dark Pool Liquidity Heatmap. (+3 more)

### Community 41 - "test_rebalancer_and_tearsheet.py"
Cohesion: 0.08
Nodes (29): fetch_universe_live_quotes(), get_us_market_session_info(), Any, Fetches real-time quotes across the entire universe concurrently in parallel., Computes current US stock market session status (Pre-Market, Regular Hours,…, Sends immediate Discord alert when a new position is opened., Sends instant flash notifications to Discord., Sends immediate Discord flash notification. (+21 more)

### Community 42 - "AutonomousTradingEngine"
Cohesion: 0.17
Nodes (10): AutonomousTradingEngine, Any, Autonomous Execution Engine that integrates Live News Ingestion, 4-Agent…, Dispatches an institutional execution alert to Discord Webhook if configured., Executes one full autonomous decision and execution cycle with: - Task 6:…, Core cycle execution body., Executes the Self-Improving Feedback Loop: 1. Analyzes trade autopsies on…, Gathers overnight macro VIX volatility regime, paper portfolio balance, and top… (+2 more)

### Community 43 - "reddit_premarket_station.py"
Cohesion: 0.25
Nodes (12): fetch_4station_premarket_intelligence(), _fetch_subreddit_rss_entries(), Any, Systematic 4-Station 1-Day-Prior Reddit Market Intelligence Engine. Pillar 2…, Calculates ticker mentions and sentiment within a specific Reddit station., Orchestrates real-time 1-day-prior intelligence across all 4 key Reddit…, Fetches real-time Atom RSS feed for a subreddit using safe defusedxml., scrape_station_ticker_sentiment() (+4 more)

### Community 47 - "Contributor Covenant Code of Conduct"
Cohesion: 0.15
Nodes (12): 1. Correction, 2. Warning, 3. Temporary Ban, 4. Permanent Ban, Attribution, Contributor Covenant Code of Conduct, Enforcement, Enforcement Guidelines (+4 more)

### Community 48 - "get_sentiment"
Cohesion: 0.12
Nodes (21): clean_financial_text(), get_sentiment(), _parse_analyzer_output(), Any, DataFrame, Analyzes the sentiment of news articles using high-precision FinBERT pipeline,…, Cleans raw financial headlines/descriptions by stripping boilerplate publisher…, Normalizes pipeline output whether given full multi-class probabilities (list… (+13 more)

### Community 49 - "calculate_doubling_progress"
Cohesion: 0.27
Nodes (9): calculate_doubling_progress(), compute_compound_position_size(), Any, Max Compound Acceleration & +100% Target Doubling Engine for Sentilyze.…, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Unit tests for Max Compound Acceleration Engine., test_calculate_doubling_progress() (+1 more)

### Community 50 - "temp_data_dir"
Cohesion: 0.67
Nodes (3): fixture, Fixture to set a temporary data directory for tests., temp_data_dir()

### Community 51 - "get_logger"
Cohesion: 0.12
Nodes (18): datetime, Logger, Autonomous Multi-Agent Trading Committee & Deliberation Engine for Sentilyze.…, AI Trade Copilot & Conversational Analyst for Sentilyze. Provides natural…, Autonomous Live Trading & News Intelligence Engine for Sentilyze. Institutional…, Real-Time Earnings Call Transcript & Management Tone Analyzer for Sentilyze.…, Natural Language Financial Q&A Agent for Sentilyze. Pillar 7 Mobile &…, Lead-Lag Granger Causality & Supply Chain Price Discovery Engine for Sentilyze.… (+10 more)

### Community 52 - "cli.py"
Cohesion: 0.29
Nodes (10): Root entry point for Sentilyze CLI. Run directly with: python sentilyze.py NVDA…, cmd_audit(), cmd_briefing(), cmd_portfolio(), main(), print_banner(), Sentilyze Command-Line Interface (CLI). Interactive terminal tool for 4-Agent…, Displays current portfolio metrics and live holdings. (+2 more)

### Community 53 - "How Can I Contribute?"
Cohesion: 0.33
Nodes (5): Contributing to Sentilyze, How Can I Contribute?, Pull Requests, Reporting Bugs, Suggesting Enhancements

### Community 54 - "audio_briefing.py"
Cohesion: 0.47
Nodes (5): generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses gTTS if available, or…, synthesize_morning_audio()

### Community 55 - "🧪 Experimental & Simulated Research Prototypes"
Cohesion: 0.50
Nodes (3): 🧪 Experimental & Simulated Research Prototypes, 🔒 Production Isolation Guarantee, 📁 Prototype Inventory

### Community 56 - "Sentilyze — Standing Audit Protocol"
Cohesion: 0.17
Nodes (11): 1. Fabricated / fake data check, 2. Ticker/input-invariant bugs, 3. Mislabeled methodology, 4. Results-file / README consistency, 5. Duplicate-output smell test, 6. Safety-critical logic sanity check, 7. Silent failure check, 8. Scope creep check (+3 more)

### Community 57 - "execute_continuous_retrain_cycle"
Cohesion: 0.32
Nodes (7): enrich_features_with_alpha_interactions(), execute_continuous_retrain_cycle(), Any, DataFrame, Enriches standard feature matrix with non-linear interaction terms., Executes an end-to-end continuous learning and model boosting cycle: 1.…, test_enrich_features_with_alpha_interactions()

### Community 58 - "render_conviction_gauge"
Cohesion: 0.29
Nodes (5): Renders a progress meter with dynamic color coding., render_conviction_gauge(), Workspace 2: 4-Agent Trading Committee Round-Table Deliberations., Renders the 4-Agent Trading Committee deliberation panel., render_committee_workspace()

### Community 59 - "correlation_matrix.py"
Cohesion: 0.38
Nodes (6): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…

### Community 60 - ".get_closed_trades_df"
Cohesion: 0.29
Nodes (4): DataFrame, Returns a DataFrame of current open holdings with Scale-Out status., Returns a DataFrame of trade history with full company names., Returns equity history as a DatetimeIndex DataFrame.

### Community 61 - "run_unified_institutional_pipeline"
Cohesion: 0.47
Nodes (5): Any, Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), test_all_8_pillars_present_in_output(), test_run_unified_institutional_pipeline()

### Community 64 - "main"
Cohesion: 0.40
Nodes (5): main(), Batch Universe Trainer for Remaining S&P 100 Tickers., run_single(), main(), Main function to run the training pipeline for a given stock ticker. Args:…

## Knowledge Gaps
- **40 isolated node(s):** `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs`, `3. Mislabeled methodology`, `4. Results-file / README consistency` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `get_logger` to `fetch_financial_statements`, `get_price_history`, `quant_engine.py`, `daily_scanner.py`, `run_backtest`, `preprocessing.py`, `test_statistical_arbitrage.py`, `convene_trading_committee`, `TradingEnvironment`, `CloudDataLake`, `ws_alternative_data.py`, `SupplyChainGraphNetwork`, `advanced_quant_experiments.py`, `run_temporal_fusion_forecast`, `ws_deep_quant.py`, `meta_ensemble.py`, `smart_trader_engine.py`, `compute_dark_pool_sentiment`, `test_omnichannel_mobile.py`, `load_model`, `black_swan_simulator.py`, `test_pillar2_alternative_data.py`, `AlpacaBrokerBridge`, `telegram_bot.py`, `opening_range_engine.py`, `options_surface.py`, `fetch_live_quote`, `SuperEnsembleClassifier`, `TickerSentinelSwarm`, `test_rebalancer_and_tearsheet.py`, `reddit_premarket_station.py`, `calculate_doubling_progress`, `cli.py`, `audio_briefing.py`, `correlation_matrix.py`, `main`?**
  _High betweenness centrality (0.232) - this node is a cross-community bridge._
- **Why does `PaperBroker` connect `PaperBroker` to `AICopilotEngine`, `fetch_live_quote`, `daily_scanner.py`, `test_autonomous_trader.py`, `AutonomousTradingEngine`, `get_logger`, `cli.py`, `update_live_holdings_prices_and_alert_discord`, `load_model`, `.get_closed_trades_df`, `telegram_bot.py`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `get_price_history()` connect `get_price_history` to `run_committee_ablation_backtest`, `run_backtest`, `app.py`, `preprocessing.py`, `test_statistical_arbitrage.py`, `convene_trading_committee`, `render_workspace_header`, `advanced_quant_experiments.py`, `preprocess_data`, `get_logger`, `update_live_holdings_prices_and_alert_discord`, `load_model`, `correlation_matrix.py`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs` to the rest of the system?**
  _40 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `get_price_history` be split into smaller, more focused modules?**
  _Cohesion score 0.06412583182093164 - nodes in this community are weakly interconnected._
- **Should `daily_scanner.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1024390243902439 - nodes in this community are weakly interconnected._