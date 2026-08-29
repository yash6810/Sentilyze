# Graph Report - Sentilyze  (2026-08-30)

## Corpus Check
- 466 files · ~726,645 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1146 nodes · 2424 edges · 64 communities (60 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0735b7b0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- fetch_financial_statements
- get_price_history
- telegram_bot.py
- daily_scanner.py
- run_backtest
- app.py
- ws_live_prediction.py
- test_statistical_arbitrage.py
- agent_committee.py
- TradingEnvironment
- preprocess_data
- CloudDataLake
- ws_alternative_data.py
- SupplyChainGraphNetwork
- ws_portfolio.py
- run_temporal_fusion_forecast
- run_unified_institutional_pipeline
- train_meta_ensemble
- ._save
- render_live_prediction_workspace
- realtime_tracker.py
- compute_dark_pool_sentiment
- test_omnichannel_mobile.py
- load_model
- handle_bot_command
- compute_lead_lag_matrix
- black_swan_simulator.py
- test_pillar2_alternative_data.py
- AlpacaBrokerBridge
- handle_telegram_command
- PaperBroker
- opening_range_engine.py
- options_surface.py
- AICopilotEngine
- liquidity_heatmap.py
- ablation_study.py
- SuperEnsembleClassifier
- TickerSentinelSwarm
- Sentilyze — Systematic Sentiment & Momentum Trading Research Platform
- test_autonomous_trader.py
- render_glass_card
- test_rebalancer_and_tearsheet.py
- AutonomousTradingEngine
- reddit_premarket_station.py
- ui/__init__.py
- Contributor Covenant Code of Conduct
- safe_path_join
- calculate_doubling_progress
- get_news
- get_logger
- social_sentiment.py
- How Can I Contribute?
- audio_briefing.py
- 🧪 Experimental & Simulated Research Prototypes
- Sentilyze — Standing Audit Protocol
- quant_engine.py
- analyze_sec_filing_diff
- correlation_matrix.py
- .get_closed_trades_df
- apply_high_watermark_profit_lock
- rules/graphify.md
- workflows/graphify.md

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 69 edges
2. `PaperBroker` - 41 edges
3. `get_price_history()` - 35 edges
4. `run_unified_institutional_pipeline()` - 32 edges
5. `fetch_live_quote()` - 29 edges
6. `preprocess_data()` - 28 edges
7. `handle_telegram_command()` - 28 edges
8. `get_news()` - 24 edges
9. `fetch_financial_statements()` - 23 edges
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
- `predict()` --calls--> `safe_path_join()`  [EXTRACTED]
  api.py → src/utils.py

## Import Cycles
- None detected.

## Communities (64 total, 4 thin omitted)

### Community 0 - "fetch_financial_statements"
Cohesion: 0.18
Nodes (22): calculate_altman_z_score(), calculate_dcf_fair_value(), calculate_piotroski_f_score(), fetch_financial_statements(), _generate_calibrated_financials(), generate_spider_radar_profile(), Any, Institutional Fundamental Valuation & Forensic Accounting Engine for Sentilyze.… (+14 more)

### Community 1 - "get_price_history"
Cohesion: 0.09
Nodes (35): _fetch_alpaca_news(), _fetch_alpaca_price_history(), _fetch_direct_yahoo_chart(), _fetch_eodhd_price_history(), _fetch_finnhub_news(), _fetch_fmp_price_history(), _fetch_google_news_rss(), _fetch_polygon_news_feed() (+27 more)

### Community 2 - "telegram_bot.py"
Cohesion: 0.17
Nodes (22): calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain(), Any, DataFrame, Live Options Microstructure, Gamma Exposure (GEX) & Max Pain Terminal for… (+14 more)

### Community 3 - "daily_scanner.py"
Cohesion: 0.12
Nodes (34): format_signal_card(), Any, Dispatches a high-impact Discord card for live autonomous trade lifecycle…, Construct a standardized trade signal data payload. Args: ticker (str): Stock…, Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord., Dispatches real-time Reddit r/wallstreetbets & Stocktwits hype spike alerts., Sends a consolidated morning macro regime and portfolio health pulse to Discord., Sends a consolidated master market digest card containing all universe signals. (+26 more)

### Community 4 - "run_backtest"
Cohesion: 0.10
Nodes (35): Figure, _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition(), calculate_performance_metrics(), _calculate_trade_outcomes() (+27 more)

### Community 5 - "app.py"
Cohesion: 0.10
Nodes (27): load_universe_tickers(), main(), Sentilyze - Institutional Algorithmic Trading & MLOps Platform. Modular Master…, Loads active S&P 100 universe tickers., get_market_status(), Any, Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Calculates live US Market (NYSE/NASDAQ) status based on Eastern Time. (+19 more)

### Community 6 - "ws_live_prediction.py"
Cohesion: 0.19
Nodes (17): calculate_smart_money_zones(), calculate_structural_trailing_stop(), evaluate_multi_timeframe_confluence(), find_swing_pivots(), Any, DataFrame, Institutional Smart Money Market Structure & Price-Action Engine for Sentilyze.…, Ratchets the Stop-Loss up structurally behind higher swing lows. Rules: 1.… (+9 more)

### Community 7 - "test_statistical_arbitrage.py"
Cohesion: 0.19
Nodes (25): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any, Series (+17 more)

### Community 8 - "agent_committee.py"
Cohesion: 0.12
Nodes (27): audit_full_universe_committee(), ChiefRiskOfficerAgent, compute_fractional_kelly_sizing(), convene_trading_committee(), execute_committee_order(), ForensicFundamentalAgent, _persist_committee_resolution(), Any (+19 more)

### Community 9 - "TradingEnvironment"
Cohesion: 0.14
Nodes (15): optimize_rl_position_allocation(), PPOPolicyAgent, Any, ndarray, Computes mean action (leverage) between 0.0 and 2.0., Estimates state value., Trains Actor-Critic parameters across historical episodes., Runs live RL Actor-Critic inference to determine optimal trade leverage and… (+7 more)

### Community 10 - "preprocess_data"
Cohesion: 0.09
Nodes (36): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, clean_headline_data() (+28 more)

### Community 11 - "CloudDataLake"
Cohesion: 0.15
Nodes (13): CloudDataLake, Any, Supabase / PostgreSQL Cloud Data Lake Connector., Validates or generates cloud database schema., Syncs local trade executions to the cloud database., Publishes real-time portfolio snapshot to cloud WebSockets channel., generate_twap_order_schedule(), generate_vwap_order_schedule() (+5 more)

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
Cohesion: 0.15
Nodes (14): Any, DataFrame, ndarray, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with…, Executes forward pass through Variable Selection, Multi-Head Attention, and… (+6 more)

### Community 16 - "run_unified_institutional_pipeline"
Cohesion: 0.18
Nodes (14): Any, Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), analyze_debt_maturity_wall(), calculate_beneish_m_score(), Any, DataFrame, Beneish M-Score Forensic Analyzer & Debt Maturity Wall Radar for Sentilyze.… (+6 more)

### Community 17 - "train_meta_ensemble"
Cohesion: 0.16
Nodes (14): MetaEnsembleClassifier, DataFrame, ndarray, Series, Institutional Multi-Model Meta-Ensemble Engine for Sentilyze. Pillar 1 Core…, Generates binary class prediction (0 = Hold/Sell, 1 = Buy) using soft-voting…, Instantiates and fits the Meta-Ensemble classifier., Multi-Model Meta-Ensemble stacking XGBoost, Random Forest, and Calibrated… (+6 more)

### Community 18 - "._save"
Cohesion: 0.17
Nodes (9): Any, Updates total equity, unrealized PnL, and win rates., Returns high-level KPI metrics for the portfolio dashboard., Executes an immediate manual live/simulated BUY order from UI., Alias for _save to ensure 100% backward compatibility., Executes an immediate manual live/simulated exit of an open position., Executes a 50% scale-out on an open position and moves stop to break-even., Persists portfolio ledger to disk. (+1 more)

### Community 19 - "render_live_prediction_workspace"
Cohesion: 0.18
Nodes (19): detect_classical_chart_patterns(), generate_ai_chart_explanation(), match_historical_chart_twins(), normalize_waveform(), Any, DataFrame, ndarray, AI Chart Pattern Recognition, Geometric Wave Learning & Visual Understanding… (+11 more)

### Community 20 - "realtime_tracker.py"
Cohesion: 0.11
Nodes (29): datetime, Autonomous Live Trading & News Intelligence Engine for Sentilyze. Institutional…, check_live_news_sentiment_shock(), evaluate_intraday_execution(), fetch_live_quote(), fetch_universe_live_quotes(), _get_browser_session(), get_us_market_session_info() (+21 more)

### Community 21 - "compute_dark_pool_sentiment"
Cohesion: 0.21
Nodes (14): compute_dark_pool_sentiment(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent institutional off-exchange block trades and dark pool prints., Scans option chain contracts where daily volume significantly exceeds open…, Synthesizes dark pool prints and unusual options flow into a unified…, scan_abnormal_options_vol_oi(), scan_dark_pool_blocks() (+6 more)

### Community 22 - "test_omnichannel_mobile.py"
Cohesion: 0.13
Nodes (17): answer_financial_query(), Any, Natural Language Financial Q&A Agent for Sentilyze. Pillar 7 Mobile &…, Parses natural language questions and routes them to quantitative engines.…, generate_smartwatch_glance_payload(), Any, Apple Watch & Wear OS Glance Complications API for Sentilyze. Pillar 7 Mobile &…, Generates structured complication JSON for Apple Watch (watchOS) and Wear OS. (+9 more)

### Community 23 - "load_model"
Cohesion: 0.07
Nodes (43): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+35 more)

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
Cohesion: 0.19
Nodes (18): compute_smart_money_insider_score(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent SEC Form 4 insider transactions for a given stock., Retrieves recent Congressional STOCK Act disclosure reports for a ticker., Synthesizes SEC Form 4 and Congressional activity into an overall Smart Money…, track_congressional_stock_disclosures(), track_corporate_insider_filings() (+10 more)

### Community 28 - "AlpacaBrokerBridge"
Cohesion: 0.24
Nodes (7): AlpacaBrokerBridge, Any, Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.…, Fetches active positions from Alpaca brokerage., Verifies active connection to Alpaca Brokerage API., Fetches live Alpaca account equity, buying power, and cash., Submits an institutional Bracket Order: - Entry: Market order - Exit 1: Limit…

### Community 29 - "handle_telegram_command"
Cohesion: 0.21
Nodes (15): build_interactive_inline_keyboard(), handle_telegram_command(), Any, Sends a formatted markdown message with inline buttons to a Telegram chat., Builds interactive inline quick-action buttons for mobile Telegram., Parses and executes Telegram slash commands and callback buttons., send_telegram_bot_message(), test_send_telegram_bot_message_fallback() (+7 more)

### Community 30 - "PaperBroker"
Cohesion: 0.21
Nodes (11): PaperBroker, Institutional Multi-Stage Quantitative Execution Broker ($100k Account).…, Loads existing portfolio state from JSON or initializes a fresh $100k account., fixture, temp_portfolio_file(), test_paper_broker_dataframes(), test_paper_broker_execute_buy_signals(), test_paper_broker_initialization() (+3 more)

### Community 31 - "opening_range_engine.py"
Cohesion: 0.12
Nodes (23): check_market_hours_preflight(), get_current_ny_time(), get_us_market_session(), Any, Unified US Stock Market (NYSE / NASDAQ) Session & Calendar Engine for…, Pre-flight sanity check for automated workflows. Returns True if execution…, Returns the current precise timestamp in America/New_York (Eastern Time)., Computes the exact real-time US equity market session (NYSE / NASDAQ). Session… (+15 more)

### Community 32 - "options_surface.py"
Cohesion: 0.27
Nodes (10): calculate_multileg_payoff(), generate_volatility_surface_mesh(), Any, 3D Implied Volatility Surface & Multi-Leg Options Strategy Desk for Sentilyze.…, Constructs a 3D Implied Volatility Surface across strike prices and expiration…, Calculates profit and loss (P&L) curves at expiration for institutional multi-…, test_calculate_multileg_payoff_bull_call_spread(), test_calculate_multileg_payoff_iron_condor() (+2 more)

### Community 33 - "AICopilotEngine"
Cohesion: 0.21
Nodes (9): AICopilotEngine, Any, AI Trade Copilot & Conversational Analyst for Sentilyze. Provides natural…, Conversational intelligence engine that parses queries and generates analytical…, Interprets user prompt and routes to appropriate financial analytical…, test_copilot_committee_query(), test_copilot_portfolio_query(), test_copilot_stress_query() (+1 more)

### Community 34 - "liquidity_heatmap.py"
Cohesion: 0.31
Nodes (8): compute_order_book_depth_and_clusters(), compute_volume_profile_and_poc(), Any, Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap for…, Simulates Level 2 market depth and identifies institutional buy/sell liquidity…, Computes Point of Control (POC), Value Area High (VAH), and Value Area Low…, test_compute_order_book_depth_and_clusters(), test_compute_volume_profile_and_poc()

### Community 35 - "ablation_study.py"
Cohesion: 0.33
Nodes (8): _persist_ablation_results(), Any, 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), test_committee_ablation_study_execution()

### Community 36 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (15): Any, DataFrame, ndarray, Series, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used., Loads all 3 models natively. (+7 more)

### Community 37 - "TickerSentinelSwarm"
Cohesion: 0.14
Nodes (14): detect_peak_crest_exhaustion(), Any, Dedicated Micro-Agent assigned to monitor a single stock position 24/7., Audits live price tick and determines peak crest execution., Manages the full swarm of Dedicated Ticker Sentinels across all open positions., Synchronizes active sentinels with current portfolio open positions., Audits all active sentinels concurrently., Detects if a stock has reached the crest/peak of its 15-minute momentum wave… (+6 more)

### Community 38 - "Sentilyze — Systematic Sentiment & Momentum Trading Research Platform"
Cohesion: 0.10
Nodes (19): 1. Installation, **1. Multi-Ticker Empirical Attribution Decomposition (50 Monte Carlo Trials per Asset)**, **2. 4-Agent Committee Ablation Matrix (500-Day Out-of-Sample Horizon)**, 2. Running the Streamlit Dashboard, 3. Running the FastAPI REST Microservice, 4. Running the Test Suite & Attribution Engine, 🧪 4-Year Sizing & Management Benchmark ($100,000 Capital), ⚡ Asymmetric Trade Execution Mechanics (+11 more)

### Community 39 - "test_autonomous_trader.py"
Cohesion: 0.13
Nodes (15): is_kill_switch_active(), load_universe_tickers(), Task 7: Master Kill Switch Check. Returns True if SENTILYZE_KILL_SWITCH…, Loads universe of tickers from stocks.txt., patch, Task 8: Verify circuit breaker triggers when true daily drawdown exceeds…, Task 9: Verify unhandled exception in cycle is caught and handled safely., Task 6: Verify active lock file prevents overlapping concurrent cycles. (+7 more)

### Community 41 - "test_rebalancer_and_tearsheet.py"
Cohesion: 0.13
Nodes (18): calculate_custom_rebalance(), calculate_share_allocation(), Any, Helper to calculate share allocation from latest daily signals file or universe…, Computes exact whole-share buy allocations for a given capital budget across…, Any, DataFrame, Helper wrapper for Monte Carlo VaR simulation. (+10 more)

### Community 42 - "AutonomousTradingEngine"
Cohesion: 0.19
Nodes (11): AutonomousTradingEngine, check_daily_loss_circuit_breaker(), Any, Autonomous Execution Engine that integrates Live News Ingestion, 4-Agent…, Dispatches an institutional execution alert to Discord Webhook if configured., Executes one full autonomous decision and execution cycle with: - Task 6:…, Core cycle execution body., Task 8: Independent Max-Daily-Loss Circuit Breaker. Compares current total… (+3 more)

### Community 43 - "reddit_premarket_station.py"
Cohesion: 0.25
Nodes (12): fetch_4station_premarket_intelligence(), _fetch_subreddit_rss_entries(), Any, Systematic 4-Station 1-Day-Prior Reddit Market Intelligence Engine. Pillar 2…, Calculates ticker mentions and sentiment within a specific Reddit station., Orchestrates real-time 1-day-prior intelligence across all 4 key Reddit…, Fetches real-time Atom RSS feed for a subreddit using safe defusedxml., scrape_station_ticker_sentiment() (+4 more)

### Community 47 - "Contributor Covenant Code of Conduct"
Cohesion: 0.15
Nodes (12): 1. Correction, 2. Warning, 3. Temporary Ban, 4. Permanent Ban, Attribution, Contributor Covenant Code of Conduct, Enforcement, Enforcement Guidelines (+4 more)

### Community 48 - "safe_path_join"
Cohesion: 0.10
Nodes (27): analyze_sentiment(), clean_financial_text(), get_sentiment(), _parse_analyzer_output(), Any, DataFrame, Analyzes the sentiment of news articles using high-precision FinBERT pipeline,…, Cleans raw financial headlines/descriptions by stripping boilerplate publisher… (+19 more)

### Community 49 - "calculate_doubling_progress"
Cohesion: 0.27
Nodes (9): calculate_doubling_progress(), compute_compound_position_size(), Any, Max Compound Acceleration & +100% Target Doubling Engine for Sentilyze.…, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Unit tests for Max Compound Acceleration Engine., test_calculate_doubling_progress() (+1 more)

### Community 50 - "get_news"
Cohesion: 0.12
Nodes (19): _fetch_marketaux_news(), get_news(), Fetches financial news from Marketaux API., Enterprise Multi-Source News Router: Cascades through Google News RSS -> Yahoo…, fixture, Fixture to set a temporary data directory for tests., Test that get_news fetches data from NewsAPI and saves it to a cache file., Test that get_news loads data from the cache if it's not stale. (+11 more)

### Community 51 - "get_logger"
Cohesion: 0.11
Nodes (14): ⚠️ EXPERIMENTAL / RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM PRODUCTION…, Logger, Cloud PostgreSQL Data Lake (Supabase / Neon) Connector for Sentilyze. Pillar 6…, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.…, Smart Order Routing (VWAP / TWAP Slicing Execution Engine) for Sentilyze.…, Any, Fast vectorized backtesting simulation sandbox for custom leverage, confidence,…, simulate_strategy_sandbox() (+6 more)

### Community 52 - "social_sentiment.py"
Cohesion: 0.23
Nodes (13): calculate_social_buzz_metrics(), fetch_social_sentiment_tracker(), Any, Social Sentiment Velocity & Retail Multi-Platform Scraper for Sentilyze. Pillar…, Scrapes real-time streaming retail sentiment from Stocktwits public symbol…, Scrapes tech community discussions on AI catalysts (OpenAI, Anthropic, Nvidia)…, Computes retail sentiment velocity and flow conviction metrics., High-level entry point to retrieve calibrated real-time social buzz metrics for… (+5 more)

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

### Community 57 - "quant_engine.py"
Cohesion: 0.22
Nodes (8): MasterQuantPipelineResult, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., analyze_earnings_call_transcript(), Any, Real-Time Earnings Call Transcript & Management Tone Analyzer for Sentilyze.…, Analyzes management tone and analyst sentiment across quarterly earnings calls.…, test_earnings_call_sentiment()

### Community 58 - "analyze_sec_filing_diff"
Cohesion: 0.32
Nodes (7): analyze_sec_filing_diff(), compute_text_similarity_and_diff(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Computes lexical and semantic diff metrics between consecutive filings., Retrieves and compares the most recent and prior SEC filings for a company.…, test_sec_filing_diff_analysis()

### Community 59 - "correlation_matrix.py"
Cohesion: 0.38
Nodes (6): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…

### Community 60 - ".get_closed_trades_df"
Cohesion: 0.29
Nodes (4): DataFrame, Returns a DataFrame of current open holdings with Scale-Out status., Returns a DataFrame of trade history with full company names., Returns equity history as a DatetimeIndex DataFrame.

### Community 61 - "apply_high_watermark_profit_lock"
Cohesion: 0.47
Nodes (5): apply_high_watermark_profit_lock(), Guarantees that once a trade reaches peak profit, the bot NEVER gives back >…, Unit tests for High-Watermark Peak Profit Ratchet (75% Lock Floor)., test_high_watermark_does_not_lower_sl_on_pullback(), test_high_watermark_locks_75pct_of_peak_gain()

## Knowledge Gaps
- **41 isolated node(s):** `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs`, `3. Mislabeled methodology`, `4. Results-file / README consistency` (+36 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `get_logger` to `fetch_financial_statements`, `get_price_history`, `telegram_bot.py`, `daily_scanner.py`, `run_backtest`, `ws_live_prediction.py`, `test_statistical_arbitrage.py`, `agent_committee.py`, `preprocess_data`, `ws_alternative_data.py`, `SupplyChainGraphNetwork`, `ws_portfolio.py`, `run_unified_institutional_pipeline`, `train_meta_ensemble`, `render_live_prediction_workspace`, `realtime_tracker.py`, `compute_dark_pool_sentiment`, `test_omnichannel_mobile.py`, `load_model`, `compute_lead_lag_matrix`, `black_swan_simulator.py`, `test_pillar2_alternative_data.py`, `opening_range_engine.py`, `options_surface.py`, `AICopilotEngine`, `liquidity_heatmap.py`, `ablation_study.py`, `test_rebalancer_and_tearsheet.py`, `reddit_premarket_station.py`, `safe_path_join`, `calculate_doubling_progress`, `social_sentiment.py`, `audio_briefing.py`, `quant_engine.py`, `analyze_sec_filing_diff`, `correlation_matrix.py`?**
  _High betweenness centrality (0.251) - this node is a cross-community bridge._
- **Why does `PaperBroker` connect `PaperBroker` to `AICopilotEngine`, `telegram_bot.py`, `daily_scanner.py`, `test_autonomous_trader.py`, `AutonomousTradingEngine`, `._save`, `realtime_tracker.py`, `handle_bot_command`, `.get_closed_trades_df`, `handle_telegram_command`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `SuperEnsembleClassifier` connect `SuperEnsembleClassifier` to `get_logger`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs` to the rest of the system?**
  _41 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `get_price_history` be split into smaller, more focused modules?**
  _Cohesion score 0.09176788124156546 - nodes in this community are weakly interconnected._
- **Should `daily_scanner.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11522048364153627 - nodes in this community are weakly interconnected._