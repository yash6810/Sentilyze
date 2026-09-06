# Graph Report - Sentilyze  (2026-09-06)

## Corpus Check
- 1839 files · ~2,030,964 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1901 nodes · 3923 edges · 113 communities (106 shown, 7 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 30 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d651658d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- deep_learning_model.py
- get_price_history
- test_options_flow.py
- autonomous_trader.py
- run_backtest
- get_sentiment
- test_all_14_papers.py
- test_statistical_arbitrage.py
- fetch_financial_statements
- TradingEnvironment
- AutonomousTradingEngine
- CloudDataLake
- app.py
- analyze_supply_chain_spillover
- calculate_hrp_weights
- run_temporal_fusion_forecast
- benchmark_training_paradigms.py
- meta_ensemble.py
- OnlineNewtonStepOptimizer
- smart_trader_engine.py
- price_scout.py
- compute_dark_pool_sentiment
- test_omnichannel_mobile.py
- screener_engine.py
- morning_briefing.py
- compute_lead_lag_matrix
- black_swan_simulator.py
- ipo_radar.py
- AlpacaBrokerBridge
- update_live_holdings_prices_and_alert_discord
- PaperBroker
- get_us_market_session
- load_model
- AICopilotEngine
- PolyTimeConvexOptimizer
- preprocessing.py
- SuperEnsembleClassifier
- triple_convex_engine.py
- Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform
- get_sector_for_ticker
- preprocess_data
- TickerSentinelSwarm
- RegimeMixtureOfExperts
- acpm_trainer.py
- ui/__init__.py
- Contributor Covenant Code of Conduct
- ws_alternative_data.py
- calculate_doubling_progress
- test_pillar2_alternative_data.py
- utils.py
- strategy_incubator.py
- How Can I Contribute?
- render_multi_agent_war_room
- 🧪 Experimental & Simulated Research Prototypes
- Sentilyze — Standing Audit Protocol
- webhook_dispatcher.py
- calculate_time_decayed_sentiment
- ws_live_prediction.py
- quant_engine.py
- ConformalCalibrator
- rules/graphify.md
- workflows/graphify.md
- GaussianHMMRegimeDetector
- EWMACorrelationMonitor
- OpeningRangeBreakout
- CUSUMDetector
- PageHinkleyDetector
- calculate_macro_liquidity_metrics
- run_opening_range_session
- get_news
- neutralize_features
- grossman_zhou_allocation
- ADWINDetector
- test_papers_15_24.py
- AdversarialRedTeamAgent
- agent_committee.py
- run_cppi_backtest
- DCCCorrelation
- ws_insider_radar.py
- ws_quantum_tournament.py
- risk_constrained_kelly_allocation
- train_model
- calculate_portfolio_diversity_grade
- run_universe_training.py
- generate_comprehensive_factsheet
- test_acpm_trainer.py
- correlation_matrix.py
- generate_pipeline_graph_data
- cli.py
- block_external_alerts
- audio_briefing.py
- run_full_quant_experiment
- calculate_multileg_payoff
- calculate_beneish_m_score
- sanitize_filename
- fetch_live_quote
- ablation_study.py
- live_screener.py
- train.py
- test_ui_modules.py
- get_vix_data
- get_rec_bisection
- create_candlestick_sr_chart
- ws_autonomous_trader.py
- test_api.py
- Agent Browser — Live Web Automation Skill
- Agent Memory — Persistent Multi-Session Memory Skill
- Cybersecurity Skills — Institutional MLOps Security Protocol
- Editorial Diagram Design Skill
- Harness Engineering Skill

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 97 edges
2. `get_price_history()` - 68 edges
3. `PaperBroker` - 53 edges
4. `fetch_live_quote()` - 43 edges
5. `run_unified_institutional_pipeline()` - 32 edges
6. `get_news()` - 32 edges
7. `preprocess_data()` - 29 edges
8. `render_workspace_header()` - 26 edges
9. `main()` - 24 edges
10. `convene_trading_committee()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `test_paper14_fractional_kelly_capital_growth()` --calls--> `compute_fractional_kelly_sizing()`  [EXTRACTED]
  tests/test_all_14_papers.py → src/agent_committee.py
- `test_paper6_cph_multi_agent_committee()` --calls--> `ChiefRiskOfficerAgent`  [EXTRACTED]
  tests/test_all_14_papers.py → src/agent_committee.py
- `test_paper7_quant_agents_trader()` --calls--> `AutonomousTradingEngine`  [EXTRACTED]
  tests/test_all_14_papers.py → src/autonomous_trader.py
- `test_paper9_when_agents_trade_scanner()` --calls--> `run_daily_market_scan()`  [EXTRACTED]
  tests/test_all_14_papers.py → src/daily_scanner.py
- `test_paper13_gnn_supply_chain()` --calls--> `analyze_supply_chain_spillover()`  [EXTRACTED]
  tests/test_all_14_papers.py → src/gnn_supply_chain.py

## Import Cycles
- None detected.

## Communities (113 total, 7 thin omitted)

### Community 0 - "deep_learning_model.py"
Cohesion: 0.11
Nodes (26): create_sliding_window_tensors(), DLinearTCNModel, load_dlinear_model(), predict_momentum_probability(), Any, DataFrame, Tensor, High-Efficiency Deep Learning Engine: DLinear + Temporal Convolutional Network… (+18 more)

### Community 1 - "get_price_history"
Cohesion: 0.10
Nodes (33): _fetch_alpaca_news(), _fetch_alpaca_price_history(), _fetch_direct_yahoo_chart(), _fetch_eodhd_price_history(), _fetch_finnhub_news(), _fetch_fmp_price_history(), _fetch_google_news_rss(), _fetch_marketaux_news() (+25 more)

### Community 2 - "test_options_flow.py"
Cohesion: 0.17
Nodes (21): calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain(), Any, DataFrame, Live Options Microstructure, Gamma Exposure (GEX) & Max Pain Terminal for… (+13 more)

### Community 3 - "autonomous_trader.py"
Cohesion: 0.15
Nodes (28): format_signal_card(), Any, Dispatches a crystal-clear, high-impact Discord card for live autonomous trade…, Construct a standardized trade signal data payload. Args: ticker (str): Stock…, Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord., Dispatches real-time Reddit r/wallstreetbets & Stocktwits hype spike alerts., Sends a comprehensive institutional morning macro regime, portfolio health,…, Sends a rich formatted trade alert card to a Discord channel via Webhook. (+20 more)

### Community 4 - "run_backtest"
Cohesion: 0.10
Nodes (35): _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition(), calculate_performance_metrics(), _calculate_trade_outcomes(), create_monthly_returns_heatmap() (+27 more)

### Community 5 - "get_sentiment"
Cohesion: 0.07
Nodes (37): ActorCriticPolicy, DRLTradingEnvironment, evaluate_drl_policy_action(), Any, ndarray, Tensor, Deep Reinforcement Learning (DRL) Autonomous Policy Agent for Sentilyze.…, Trains an Actor-Critic DRL policy agent on historical market returns and news… (+29 more)

### Community 6 - "test_all_14_papers.py"
Cohesion: 0.09
Nodes (24): calculate_almgren_chriss_trajectory(), Any, Paper 3: Almgren & Chriss (2000) - Optimal Execution of Portfolio Transactions.…, Computes Almgren-Chriss optimal trading trajectory. x_j = 2 * sinh(0.5 * kappa…, detect_negative_cycle_arbitrage(), Any, Finds triangular arbitrage using Bellman-Ford on log exchange rates: w =…, compute_balanced_hedge_allocation() (+16 more)

### Community 7 - "test_statistical_arbitrage.py"
Cohesion: 0.19
Nodes (27): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any, Series (+19 more)

### Community 8 - "fetch_financial_statements"
Cohesion: 0.18
Nodes (22): calculate_altman_z_score(), calculate_dcf_fair_value(), calculate_piotroski_f_score(), fetch_financial_statements(), _generate_calibrated_financials(), generate_spider_radar_profile(), Any, Institutional Fundamental Valuation & Forensic Accounting Engine for Sentilyze.… (+14 more)

### Community 9 - "TradingEnvironment"
Cohesion: 0.14
Nodes (16): optimize_rl_position_allocation(), PPOPolicyAgent, Any, ndarray, ⚠️ EXPERIMENTAL / RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM PRODUCTION…, Computes mean action (leverage) between 0.0 and 2.0., Estimates state value., Trains Actor-Critic parameters across historical episodes. (+8 more)

### Community 10 - "AutonomousTradingEngine"
Cohesion: 0.07
Nodes (34): AutonomousTradingEngine, check_daily_loss_circuit_breaker(), get_daemon_status(), is_kill_switch_active(), load_universe_tickers(), Any, Autonomous Execution Engine that integrates Live News Ingestion, 4-Agent…, Dispatches an institutional execution alert to Discord Webhook if configured. (+26 more)

### Community 11 - "CloudDataLake"
Cohesion: 0.13
Nodes (15): CloudDataLake, Any, Cloud PostgreSQL Data Lake (Supabase / Neon) Connector for Sentilyze. Pillar 6…, Supabase / PostgreSQL Cloud Data Lake Connector., Validates or generates cloud database schema., Syncs local trade executions to the cloud database., Publishes real-time portfolio snapshot to cloud WebSockets channel., generate_twap_order_schedule() (+7 more)

### Community 12 - "app.py"
Cohesion: 0.11
Nodes (27): load_universe_tickers(), main(), Sentilyze - Institutional Algorithmic Trading & MLOps Platform. Modular Master…, Loads active S&P 100 universe tickers., get_market_status(), Any, Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Wraps HTML content inside an institutional frosted glass container. (+19 more)

### Community 13 - "analyze_supply_chain_spillover"
Cohesion: 0.15
Nodes (14): analyze_supply_chain_spillover(), Any, ndarray, Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for…, Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)., Executes a Graph Convolutional Network (GCN) layer: H_new = ReLU(A_hat * H * W)…, Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab…, High-level entry point to run GNN supply chain shock propagation. (+6 more)

### Community 14 - "calculate_hrp_weights"
Cohesion: 0.15
Nodes (23): Empirical Quant Experimentation & Multi-Asset Ablation Benchmark Suite.…, build_unified_portfolio(), calculate_hrp_weights(), calculate_risk_parity_weights(), load_all_ticker_portfolios(), Any, DataFrame, Series (+15 more)

### Community 15 - "run_temporal_fusion_forecast"
Cohesion: 0.14
Nodes (15): Any, DataFrame, ndarray, Temporal Fusion Transformer (TFT) & Multi-Horizon Self-Attention Engine for…, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with… (+7 more)

### Community 16 - "benchmark_training_paradigms.py"
Cohesion: 0.21
Nodes (25): Any, DataFrame, Series, Triple-Barrier Meta-Labeling (Upper +2.0 ATR, Lower -1.2 ATR, Vertical 10d)., Regime-Conditioned Mixture of Experts (MoE) with 3 specialized sub-boosters., Online Streaming Incremental Learning with Stochastic Gradient Descent &…, Standard Rolling Walk-Forward Optimization (Fixed 500-day window)., Cross-Asset Pooled Multi-Task Learning across peer assets for high statistical… (+17 more)

### Community 17 - "meta_ensemble.py"
Cohesion: 0.10
Nodes (19): DynamicSharpeMetaEnsemble, MetaEnsembleClassifier, DataFrame, ndarray, Series, Institutional Multi-Model Meta-Ensemble Engine for Sentilyze. Pillar 1 Core…, Generates binary class prediction (0 = Hold/Sell, 1 = Buy) using soft-voting…, Instantiates and fits the Meta-Ensemble classifier. (+11 more)

### Community 18 - "OnlineNewtonStepOptimizer"
Cohesion: 0.18
Nodes (10): OnlineNewtonStepOptimizer, DataFrame, ndarray, Polynomial-Time ONS Portfolio Engine (Hazan et al.)., Processes price relatives (r_t = Close_t / Close_{t-1}) and updates weights in…, Fast O(d log d) Euclidean projection onto probability simplex., Runs ONS sequence through time and outputs daily allocations and portfolio…, test_paper1_online_newton_step() (+2 more)

### Community 19 - "smart_trader_engine.py"
Cohesion: 0.09
Nodes (38): detect_classical_chart_patterns(), generate_ai_chart_explanation(), match_historical_chart_twins(), normalize_waveform(), Any, DataFrame, ndarray, AI Chart Pattern Recognition, Geometric Wave Learning & Visual Understanding… (+30 more)

### Community 20 - "price_scout.py"
Cohesion: 0.18
Nodes (12): get_latest_scout_alerts(), PriceActionScoutAgent, PriceScoutBot, Any, Real-Time Price Action & Tape-Reading Scout Subagent (Bot) for Sentilyze.…, Continuous Background Scanner Bot that scouts the 538 universe assets, detects…, Scans given tickers for real-time volume breakout candidates., Retrieves the latest price scout breakout alerts. (+4 more)

### Community 21 - "compute_dark_pool_sentiment"
Cohesion: 0.21
Nodes (14): compute_dark_pool_sentiment(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent institutional off-exchange block trades and dark pool prints., Scans option chain contracts where daily volume significantly exceeds open…, Synthesizes dark pool prints and unusual options flow into a unified…, scan_abnormal_options_vol_oi(), scan_dark_pool_blocks() (+6 more)

### Community 22 - "test_omnichannel_mobile.py"
Cohesion: 0.15
Nodes (15): answer_financial_query(), Any, Parses natural language questions and routes them to quantitative engines.…, generate_smartwatch_glance_payload(), Any, Apple Watch & Wear OS Glance Complications API for Sentilyze. Pillar 7 Mobile &…, Generates structured complication JSON for Apple Watch (watchOS) and Wear OS., format_whatsapp_trade_alert() (+7 more)

### Community 23 - "screener_engine.py"
Cohesion: 0.08
Nodes (30): fetch_universe_live_quotes(), Fetches real-time quotes across universe with fast batching (sub-2s)., calculate_custom_rebalance(), calculate_share_allocation(), Any, Helper to calculate share allocation from latest daily signals file or universe…, Computes exact whole-share buy allocations for a given capital budget across…, evaluate_single_asset_screener() (+22 more)

### Community 24 - "morning_briefing.py"
Cohesion: 0.17
Nodes (20): generate_morning_briefing_text(), get_portfolio_intelligence(), load_universe_candidates(), Any, AI Pre-Market Audio & Executive Morning Briefing Generator for Sentilyze.…, Reads live paper portfolio state for broadcast reporting., Assembles a comprehensive, institutional Wall Street Morning Podcast and…, Synthesizes broadcast audio podcast (.mp3) using Google Text-to-Speech (gTTS)… (+12 more)

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

### Community 29 - "update_live_holdings_prices_and_alert_discord"
Cohesion: 0.14
Nodes (17): check_live_news_sentiment_shock(), evaluate_intraday_execution(), get_us_market_session_info(), Any, Checks if breaking news in the last few hours has a severe negative sentiment…, Autonomous 5-Minute Intraday Market Execution Engine: 1. Evaluates 50/50 Scale-…, Computes current US stock market session status (Pre-Market, Regular Hours,…, Sends immediate Discord alert when a new position is opened. (+9 more)

### Community 30 - "PaperBroker"
Cohesion: 0.09
Nodes (24): PaperBroker, Any, DataFrame, Executes daily quantitative scan results using the Concentrated Top-2 + Scale-…, Institutional Multi-Stage Quantitative Execution Broker ($100k Account).…, Loads existing portfolio state from JSON or initializes a fresh $100k account., Updates total equity, unrealized PnL, and win rates., Returns high-level KPI metrics for the portfolio dashboard. (+16 more)

### Community 31 - "get_us_market_session"
Cohesion: 0.12
Nodes (24): check_market_hours_preflight(), get_current_ny_time(), get_us_market_session(), Any, datetime, Unified US Stock Market (NYSE / NASDAQ) Session & Calendar Engine for…, Pre-flight sanity check for automated workflows. Returns True if execution…, Returns the current precise timestamp in America/New_York (Eastern Time). (+16 more)

### Community 32 - "load_model"
Cohesion: 0.15
Nodes (17): enrich_features_with_alpha_interactions(), execute_continuous_retrain_cycle(), Any, DataFrame, Continuous Model Self-Training & Accuracy Boosting Engine for Sentilyze. Self-…, Enriches standard feature matrix with non-linear interaction terms., Executes an end-to-end continuous learning and model boosting cycle: 1.…, handle_bot_command() (+9 more)

### Community 33 - "AICopilotEngine"
Cohesion: 0.24
Nodes (8): AICopilotEngine, Any, Conversational intelligence engine that parses queries and generates analytical…, Interprets user prompt and routes to appropriate financial analytical…, test_copilot_committee_query(), test_copilot_portfolio_query(), test_copilot_stress_query(), test_copilot_ticker_analysis_query()

### Community 34 - "PolyTimeConvexOptimizer"
Cohesion: 0.18
Nodes (8): PolyTimeConvexOptimizer, Any, DataFrame, Series, Polynomial-Time Convex Portfolio Optimizer with Market Frictions (Boyd et al.)., Solves the friction-aware convex optimization problem in polynomial time. Args:…, test_paper2_boyd_convex_optimizer(), test_polytime_convex_optimizer()

### Community 35 - "preprocessing.py"
Cohesion: 0.17
Nodes (19): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, DataFrame (+11 more)

### Community 36 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (16): Any, DataFrame, ndarray, Series, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.…, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used. (+8 more)

### Community 37 - "triple_convex_engine.py"
Cohesion: 0.12
Nodes (19): apply_triple_barrier_labeling(), calculate_deflated_sharpe_ratio(), DataFrame, Series, Computes Bailey & López de Prado's Deflated Sharpe Ratio (DSR). Adjusts for: -…, Applies López de Prado's path-dependent Triple-Barrier Method to generate trade…, Any, DataFrame (+11 more)

### Community 38 - "Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform"
Cohesion: 0.13
Nodes (14): 1. Run in 1-Click (No Installation Required), 2. Local Setup & Installation, 3. Run the 4-Agent Quantitative CLI, 4. Launch the Streamlit Mission Control, 5. Run Full Test Suite (240+ Unit Tests), 📊 Empirical Alpha Attribution & Benchmarks, 🏛️ Grounded 4-Agent Deliberation Council, 🖥️ Interactive Streamlit App & 23 Mission Control Workspaces (+6 more)

### Community 39 - "get_sector_for_ticker"
Cohesion: 0.13
Nodes (19): calculate_portfolio_correlation_matrix(), check_correlation_shield(), Any, DataFrame, Portfolio Correlation Matrix Shield. Functions: - Calculates rolling 21-day…, Computes the 21-day rolling pairwise return correlation matrix across tickers., Audits a candidate buy against currently held positions. Returns whether the…, build_pooled_sector_dataset() (+11 more)

### Community 40 - "preprocess_data"
Cohesion: 0.13
Nodes (19): clean_headline_data(), _get_api_key(), _load_sentiment_analyzer(), preprocess_data(), Any, DataFrame, Cleans a headline CSV file by removing rows with invalid stock tickers. Caches…, Safely attempts to retrieve the API key from Streamlit secrets, falling back to… (+11 more)

### Community 41 - "TickerSentinelSwarm"
Cohesion: 0.14
Nodes (15): detect_peak_crest_exhaustion(), Any, Dedicated Ticker Sentinel & Peak-Crest Volume Harvester Swarm for Sentilyze.…, Dedicated Micro-Agent assigned to monitor a single stock position 24/7., Audits live price tick and determines peak crest execution., Manages the full swarm of Dedicated Ticker Sentinels across all open positions., Synchronizes active sentinels with current portfolio open positions., Audits all active sentinels concurrently. (+7 more)

### Community 42 - "RegimeMixtureOfExperts"
Cohesion: 0.21
Nodes (9): Any, DataFrame, ndarray, Series, Regime-Conditioned Mixture of Experts (MoE) Architecture. Trains 3 specialized…, 3-Expert Gated Mixture of Experts classifier for financial regimes., Labels historical rows into 3 latent market regimes: 0 = Bull Momentum (Price >…, RegimeMixtureOfExperts (+1 more)

### Community 43 - "acpm_trainer.py"
Cohesion: 0.18
Nodes (10): Unified Alpha-Conformal Purged Multi-Task (ACPM) Quantitative Training Engine.…, compute_deflated_sharpe_ratio(), PurgedGroupTimeSeriesSplit, DataFrame, ndarray, Series, Combinatorial Purged & Embargoed Cross-Validation (CPCV) for Financial Machine…, Time-series cross-validator that purges overlapping event windows and applies… (+2 more)

### Community 47 - "Contributor Covenant Code of Conduct"
Cohesion: 0.15
Nodes (12): 1. Correction, 2. Warning, 3. Temporary Ban, 4. Permanent Ban, Attribution, Contributor Covenant Code of Conduct, Enforcement, Enforcement Guidelines (+4 more)

### Community 48 - "ws_alternative_data.py"
Cohesion: 0.14
Nodes (20): get_pre_ipo_pipeline_df(), DataFrame, Returns a formatted pandas DataFrame of all pre-IPO target assets., fetch_4station_premarket_intelligence(), _fetch_subreddit_rss_entries(), Any, Systematic 4-Station 1-Day-Prior Reddit Market Intelligence Engine. Pillar 2…, Calculates ticker mentions and sentiment within a specific Reddit station. (+12 more)

### Community 49 - "calculate_doubling_progress"
Cohesion: 0.31
Nodes (8): calculate_doubling_progress(), compute_compound_position_size(), Any, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Unit tests for Max Compound Acceleration Engine., test_calculate_doubling_progress(), test_compute_compound_position_size()

### Community 50 - "test_pillar2_alternative_data.py"
Cohesion: 0.08
Nodes (41): compute_smart_money_insider_score(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent SEC Form 4 insider transactions for a given stock., Retrieves recent Congressional STOCK Act disclosure reports for a ticker., Synthesizes SEC Form 4 and Congressional activity into an overall Smart Money…, track_congressional_stock_disclosures(), track_corporate_insider_filings() (+33 more)

### Community 51 - "utils.py"
Cohesion: 0.07
Nodes (24): ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Logger, AI Trade Copilot & Conversational Analyst for Sentilyze. Provides natural…, Max Compound Acceleration & +100% Target Doubling Engine for Sentilyze.…, Stanford Multi-Period Convex Portfolio Optimization Engine (Boyd et al.).…, Real-Time Earnings Call Transcript & Management Tone Analyzer for Sentilyze.…, Natural Language Financial Q&A Agent for Sentilyze. Pillar 7 Mobile &…, Paper 5: Negative Cycle Detection on Exchange Log-Rate Digraphs (Bellman-Ford).… (+16 more)

### Community 52 - "strategy_incubator.py"
Cohesion: 0.20
Nodes (15): breed_strategy_generation(), evaluate_3zone_robustness(), load_strategy_vault(), Any, Evolutionary Strategy Incubator & Robustness Lab for Sentilyze. Institutional…, Evaluates a Strategy Genome across 3 distinct zones: 1. In-Sample Train (70%)…, Runs evolutionary genetic algorithm across generations, breeding top survivors., Represents an algorithmic strategy rule DNA. (+7 more)

### Community 53 - "How Can I Contribute?"
Cohesion: 0.33
Nodes (5): Contributing to Sentilyze, How Can I Contribute?, Pull Requests, Reporting Bugs, Suggesting Enhancements

### Community 54 - "render_multi_agent_war_room"
Cohesion: 0.28
Nodes (7): Any, Interactive Multi-Agent War Room Visualizer Component for Streamlit. Functions:…, Renders the complete 5-Agent War Room Council deliberation chamber., render_multi_agent_war_room(), Real-Time Audio Trade Squawk Component for Streamlit. Functions: - Uses…, Renders an HTML5 Web Speech API audio squawk generator inside Streamlit., render_audio_squawk_button()

### Community 55 - "🧪 Experimental & Simulated Research Prototypes"
Cohesion: 0.50
Nodes (3): 🧪 Experimental & Simulated Research Prototypes, 🔒 Production Isolation Guarantee, 📁 Prototype Inventory

### Community 56 - "Sentilyze — Standing Audit Protocol"
Cohesion: 0.15
Nodes (12): 10. Strict Portfolio Preservation & Realized Gain Integrity, 1. Fabricated / fake data check, 2. Ticker/input-invariant bugs, 3. Mislabeled methodology, 4. Results-file / README consistency, 5. Duplicate-output smell test, 6. Safety-critical logic sanity check, 7. Silent failure check (+4 more)

### Community 57 - "webhook_dispatcher.py"
Cohesion: 0.19
Nodes (19): Workspace: Automated Broker Webhooks & Execution API Dispatcher. Configures and…, render_broker_webhooks_workspace(), _append_audit_log(), dispatch_order_webhook(), format_broker_order_payload(), generate_hmac_signature(), load_webhook_config(), Any (+11 more)

### Community 58 - "calculate_time_decayed_sentiment"
Cohesion: 0.19
Nodes (14): calculate_time_decayed_sentiment(), compute_exponential_decay_weights(), Any, DataFrame, datetime, ndarray, Series, Temporal Sentiment Half-Life Decay Engine. Functions: - Applies continuous… (+6 more)

### Community 59 - "ws_live_prediction.py"
Cohesion: 0.18
Nodes (12): Scans the entire stock universe defined in stocks.txt, generates tomorrow's…, run_daily_market_scan(), Any, Dispatches formatted HTML morning market digest email via Gmail SMTP., send_email_digest(), get_prediction_on_latest_data(), Any, DataFrame (+4 more)

### Community 60 - "quant_engine.py"
Cohesion: 0.27
Nodes (8): MasterQuantPipelineResult, Any, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), test_all_8_pillars_present_in_output(), test_run_unified_institutional_pipeline()

### Community 61 - "ConformalCalibrator"
Cohesion: 0.10
Nodes (15): ACPMTrainer, Any, DataFrame, Series, State-of-the-Art 10x Institutional Quantitative Training Engine., Executes end-to-end ACPM training for a target equity., ConformalCalibrator, ndarray (+7 more)

### Community 64 - "GaussianHMMRegimeDetector"
Cohesion: 0.13
Nodes (11): GaussianHMMRegimeDetector, Any, DataFrame, ndarray, Paper 15: Gaussian Hidden Markov Model for Market Regime Detection. Source:…, 3-state Gaussian HMM regime classifier: Bull / Normal / Crisis. Uses hand-…, Gaussian emission probability for each state., Forward algorithm step: update filtered state probabilities with one new… (+3 more)

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
Cohesion: 0.32
Nodes (6): calculate_macro_liquidity_metrics(), Any, Computes real-time macroeconomic liquidity indicators and yield curve dynamics.…, Workspace: Real-Time Macro Liquidity & Treasury Yield Curve Radar. Visualizes…, render_macro_liquidity_workspace(), test_calculate_macro_liquidity_metrics()

### Community 71 - "run_opening_range_session"
Cohesion: 0.40
Nodes (5): Any, Executes a live 5-Minute Opening Range Breakout scan across top liquid assets:…, run_opening_range_session(), Verify that ORB live session executes, filters stocks in play, and saves latest…, test_run_opening_range_session()

### Community 72 - "get_news"
Cohesion: 0.13
Nodes (17): get_news(), Enterprise Multi-Source News Router: Cascades through Google News RSS -> Yahoo…, fixture, Fixture to set a temporary data directory for tests., Test that get_news fetches data from NewsAPI and saves it to a cache file., Test that get_news loads data from the cache if it's not stale., Test that get_news re-fetches data if the cache is stale., Test that get_price_history fetches data from yfinance and saves it to a cache… (+9 more)

### Community 73 - "neutralize_features"
Cohesion: 0.22
Nodes (9): neutralize_features(), neutralize_predictions(), DataFrame, ndarray, Series, Feature & Factor Neutralization for Quantitative Machine Learning.…, Neutralizes target feature columns with respect to factor columns (e.g. SPY…, Removes linear factor exposure from raw prediction scores. (+1 more)

### Community 74 - "grossman_zhou_allocation"
Cohesion: 0.28
Nodes (5): grossman_zhou_allocation(), Any, Paper 18: Grossman-Zhou Optimal Drawdown-Constrained Strategy. Source: Grossman…, Compute the Grossman-Zhou optimal risky allocation under a drawdown constraint…, TestGrossmanZhou

### Community 75 - "ADWINDetector"
Cohesion: 0.19
Nodes (7): ADWINDetector, Any, ndarray, Paper 21: ADWIN (Adaptive Windowing) Drift Detector. Source: Bifet & Gavaldà…, ADWIN drift detector with Hoeffding bound. Maintains a variable-length window…, Add one observation. Returns whether drift was detected. If drift is detected,…, TestADWIN

### Community 76 - "test_papers_15_24.py"
Cohesion: 0.20
Nodes (10): calculate_cdar(), optimize_cdar_portfolio(), Any, DataFrame, ndarray, Paper 19: Conditional Drawdown-at-Risk (CDaR) Portfolio Optimization. Source:…, Calculate Conditional Drawdown-at-Risk: the expected drawdown in the worst…, Optimize portfolio weights to minimize CDaR. Simplified approach: compute CDaR… (+2 more)

### Community 77 - "AdversarialRedTeamAgent"
Cohesion: 0.24
Nodes (8): AdversarialRedTeamAgent, Any, Agent 5: Adversarial Red-Team / Devil's Advocate Specialist. Actively hunts for…, Conducts a rigorous adversarial audit of the target asset., Tests for Adversarial Red-Team Specialist Agent (src/red_team_agent.py).…, test_red_team_agent_initialization(), test_red_team_evaluation_structure(), test_red_team_stress_scenario()

### Community 78 - "agent_committee.py"
Cohesion: 0.10
Nodes (32): Any, Master Academic Research Papers Empirical Benchmark Suite (All 14 Papers).…, Executes empirical backtests comparing all 14 academic paper methodologies., run_all_14_papers_benchmark(), audit_full_universe_committee(), ChiefRiskOfficerAgent, compute_fractional_kelly_sizing(), convene_trading_committee() (+24 more)

### Community 79 - "run_cppi_backtest"
Cohesion: 0.23
Nodes (8): calculate_cppi_allocation(), Any, ndarray, Paper 20: Constant Proportion Portfolio Insurance (CPPI). Source: Black & Jones…, CPPI allocation: Exposure = M * (Portfolio - Floor). Args: portfolio_value:…, Run a full CPPI backtest over a return series. Args: returns: Array of daily…, run_cppi_backtest(), TestCPPI

### Community 80 - "DCCCorrelation"
Cohesion: 0.20
Nodes (7): DCCCorrelation, Any, DataFrame, Paper 24: Dynamic Conditional Correlation (DCC-GARCH). Source: Engle (2002) —…, Simplified DCC model using EWMA-GARCH(1,1) for individual volatilities and DCC…, Fit the DCC model to a returns DataFrame. Returns time-varying correlation…, TestDCC

### Community 81 - "ws_insider_radar.py"
Cohesion: 0.25
Nodes (13): calculate_insider_conviction_score(), fetch_insider_transactions(), Any, Smart-Money Executive & Institutional Insider Radar for Sentilyze.…, Computes the Quantitative Insider Conviction Index (0 to 100 Score) and detects…, Screens a universe of tickers and returns the highest-ranking insider buying…, Fetches recent SEC Form 4 insider transactions for a specific ticker. Includes…, scan_universe_insider_catalysts() (+5 more)

### Community 82 - "ws_quantum_tournament.py"
Cohesion: 0.19
Nodes (13): generate_institutional_pdf_tearsheet(), Institutional Quantitative Tearsheet & Factsheet Generator for Sentilyze.…, Generates a publication-grade institutional quantitative tearsheet PDF. Returns…, get_market_timestamp(), load_safety_benchmarks(), load_tournament_results(), Any, Workspace 14: 25-Paper Quantum Tournament, Live Omni-Hybrid Pipeline & Risk… (+5 more)

### Community 83 - "risk_constrained_kelly_allocation"
Cohesion: 0.24
Nodes (6): Any, ndarray, Paper 23: Risk-Constrained Kelly Gambling. Source: Busseti, Ryu, Boyd —…, Compute optimal Kelly allocation with drawdown probability constraint.…, risk_constrained_kelly_allocation(), TestRiskKelly

### Community 84 - "train_model"
Cohesion: 0.21
Nodes (14): Series, Save the trained model to a file using XGBoost's native format. This is safer…, Train the XGBoost model using Combinatorial Purged & Embargoed Cross-Validation…, save_model(), train_model(), Generates sample data for testing. Needs at least 520 rows for WFO…, Tests the train_model function., Tests that a model can be saved and loaded correctly. (+6 more)

### Community 85 - "calculate_portfolio_diversity_grade"
Cohesion: 0.27
Nodes (9): calculate_portfolio_diversity_grade(), Any, DataFrame, Workspace: Portfolio Diversity & Correlation Health Grader. Institutional…, render_portfolio_diversity_workspace(), test_custom_returns_correlated(), test_custom_returns_diverse(), test_empty_portfolio() (+1 more)

### Community 86 - "run_universe_training.py"
Cohesion: 0.21
Nodes (13): load_universe_from_file(), main(), prefetch_single_ticker(), prefetch_universe_data(), Any, Universal Parallel Multi-Core Model Training & High-Resolution Benchmark.…, Loads cleaned ticker symbols from stocks.txt., Worker function to train a single asset. (+5 more)

### Community 87 - "generate_comprehensive_factsheet"
Cohesion: 0.27
Nodes (8): generate_comprehensive_factsheet(), Any, Series, Computes over 30 institutional hedge-fund risk, performance, and drawdown…, Workspace: Institutional Risk & Alpha Performance Factsheet. Quantitative…, render_performance_factsheet_workspace(), test_generate_comprehensive_factsheet_custom_series(), test_generate_comprehensive_factsheet_default()

### Community 88 - "test_acpm_trainer.py"
Cohesion: 0.28
Nodes (11): find_optimal_d(), fractional_differentiation_ffd(), get_weights_ffd(), ndarray, Series, Fixed-Width Window Fractional Differentiation (FFD) for Financial Time Series.…, Generate weights for Fixed-Width Window Fractional Differentiation. w_0 = 1 w_k…, Applies Fixed-Width Window Fractional Differentiation to a price series. (+3 more)

### Community 89 - "correlation_matrix.py"
Cohesion: 0.38
Nodes (6): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…

### Community 90 - "generate_pipeline_graph_data"
Cohesion: 0.27
Nodes (9): generate_pipeline_graph_data(), Any, Interactive Multi-Agent & Pipeline Architecture Canvas Component for Streamlit.…, Renders the interactive Vis.js animated node network canvas inside Streamlit., Constructs the nodes and edges representation for the pipeline graph canvas., render_pipeline_topology_canvas(), Tests for Interactive Multi-Agent & Pipeline Architecture Canvas…, test_generate_pipeline_graph_data_defaults() (+1 more)

### Community 91 - "cli.py"
Cohesion: 0.29
Nodes (10): Root entry point for Sentilyze CLI. Run directly with: python sentilyze.py NVDA…, cmd_audit(), cmd_briefing(), cmd_portfolio(), main(), print_banner(), Sentilyze Command-Line Interface (CLI). Interactive terminal tool for 4-Agent…, Displays current portfolio metrics and live holdings. (+2 more)

### Community 92 - "block_external_alerts"
Cohesion: 0.50
Nodes (3): block_external_alerts(), fixture, Autouse fixture that prevents tests from sending real outbound network calls to…

### Community 93 - "audio_briefing.py"
Cohesion: 0.47
Nodes (5): generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses gTTS if available, or…, synthesize_morning_audio()

### Community 94 - "run_full_quant_experiment"
Cohesion: 0.22
Nodes (9): compute_performance_metrics(), Any, DataFrame, Series, Executes empirical ablation benchmark across the full asset universe., Simulates walk-forward strategy execution with or without advanced quant…, Computes key quant performance metrics., run_full_quant_experiment() (+1 more)

### Community 95 - "calculate_multileg_payoff"
Cohesion: 0.31
Nodes (9): calculate_multileg_payoff(), generate_volatility_surface_mesh(), Any, Constructs a 3D Implied Volatility Surface across strike prices and expiration…, Calculates profit and loss (P&L) curves at expiration for institutional multi-…, test_calculate_multileg_payoff_bull_call_spread(), test_calculate_multileg_payoff_iron_condor(), test_calculate_multileg_payoff_long_straddle() (+1 more)

### Community 96 - "calculate_beneish_m_score"
Cohesion: 0.27
Nodes (9): analyze_debt_maturity_wall(), calculate_beneish_m_score(), Any, DataFrame, Beneish M-Score Forensic Analyzer & Debt Maturity Wall Radar for Sentilyze.…, Evaluates corporate interest coverage and debt maturity wall runway., Computes the 8-Ratio Beneish M-Score from 2-year comparative SEC financial…, test_beneish_m_score() (+1 more)

### Community 97 - "sanitize_filename"
Cohesion: 0.29
Nodes (10): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+2 more)

### Community 98 - "fetch_live_quote"
Cohesion: 0.16
Nodes (15): compute_order_book_depth_and_clusters(), compute_volume_profile_and_poc(), Any, Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap for…, Simulates Level 2 market depth and identifies institutional buy/sell liquidity…, Computes Point of Control (POC), Value Area High (VAH), and Value Area Low…, fetch_live_quote(), _get_browser_session() (+7 more)

### Community 99 - "ablation_study.py"
Cohesion: 0.33
Nodes (8): _persist_ablation_results(), Any, 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), test_committee_ablation_study_execution()

### Community 100 - "live_screener.py"
Cohesion: 0.32
Nodes (6): Real-Time Market Anomaly Screener Component for Streamlit. Functions: - Renders…, Renders the Real-Time Market Anomaly Screener UI., render_live_screener_section(), Workspace 24: Real-Time Market Anomaly Screener., Renders the Real-Time Market Anomaly Screener Workspace., render_screener_workspace()

### Community 101 - "train.py"
Cohesion: 0.31
Nodes (7): main(), Batch Universe Trainer for Remaining S&P 100 Tickers., run_single(), main(), Trains institutional ACPM Sector-Pooled Multi-Task models across the universe.…, Main function to run the training pipeline for a given stock ticker. Args:…, train_sector_pooled_models()

### Community 102 - "test_ui_modules.py"
Cohesion: 0.40
Nodes (3): inject_custom_theme(), Dynamic Bespoke Theme Engine for Sentilyze. Supports 3 Institutional Presets:…, Injects high-performance, bespoke CSS styling into the Streamlit app.

### Community 103 - "get_vix_data"
Cohesion: 0.36
Nodes (6): get_vix_data(), Fetches historical data for the CBOE Volatility Index (VIX). Args: period…, Nightly sync script that pre-fetches and caches 10-year OHLCV prices, VIX macro…, sync_all_market_data(), Test that sync_all_market_data executes and returns summary metadata., test_sync_all_market_data()

### Community 104 - "get_rec_bisection"
Cohesion: 0.33
Nodes (7): get_cluster_var(), get_quasi_diag(), get_rec_bisection(), ndarray, Compute risk variance of a sub-cluster under inverse-variance weighting., Recursively bisect clusters and compute inverse-cluster-variance weights., Sort clustered items by hierarchical tree order.

### Community 105 - "create_candlestick_sr_chart"
Cohesion: 0.33
Nodes (5): create_candlestick_sr_chart(), DataFrame, Figure, Automated Dynamic Pivot Support & Resistance Charting Component for Streamlit.…, Constructs an institutional-grade interactive Plotly chart with automated S/R…

### Community 106 - "ws_autonomous_trader.py"
Cohesion: 0.40
Nodes (5): ensure_background_daemon_thread_running(), Ensures a single background autonomous trading daemon thread is permanently…, Workspace 3: 24/7 Autonomous Broker, Kelly Sizing & Staged Profit Scaler.…, Renders the 24/7 Autonomous Live Trading & News Agent interface., render_autonomous_trader_workspace()

### Community 108 - "Agent Browser — Live Web Automation Skill"
Cohesion: 0.50
Nodes (3): 1. Core Principles, 2. Use Cases in Sentilyze, Agent Browser — Live Web Automation Skill

### Community 109 - "Agent Memory — Persistent Multi-Session Memory Skill"
Cohesion: 0.50
Nodes (3): 1. Storage Architecture, 2. Integration Pattern, Agent Memory — Persistent Multi-Session Memory Skill

## Knowledge Gaps
- **47 isolated node(s):** `graphify`, `1. Core Principles`, `2. Use Cases in Sentilyze`, `1. Storage Architecture`, `2. Integration Pattern` (+42 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `utils.py` to `deep_learning_model.py`, `get_price_history`, `test_options_flow.py`, `autonomous_trader.py`, `run_backtest`, `get_sentiment`, `test_all_14_papers.py`, `test_statistical_arbitrage.py`, `fetch_financial_statements`, `TradingEnvironment`, `CloudDataLake`, `analyze_supply_chain_spillover`, `calculate_hrp_weights`, `run_temporal_fusion_forecast`, `benchmark_training_paradigms.py`, `meta_ensemble.py`, `smart_trader_engine.py`, `price_scout.py`, `compute_dark_pool_sentiment`, `test_omnichannel_mobile.py`, `screener_engine.py`, `morning_briefing.py`, `compute_lead_lag_matrix`, `black_swan_simulator.py`, `ipo_radar.py`, `AlpacaBrokerBridge`, `get_us_market_session`, `load_model`, `preprocessing.py`, `SuperEnsembleClassifier`, `triple_convex_engine.py`, `get_sector_for_ticker`, `TickerSentinelSwarm`, `ws_alternative_data.py`, `test_pillar2_alternative_data.py`, `strategy_incubator.py`, `webhook_dispatcher.py`, `calculate_time_decayed_sentiment`, `ws_live_prediction.py`, `quant_engine.py`, `agent_committee.py`, `ws_insider_radar.py`, `ws_quantum_tournament.py`, `calculate_portfolio_diversity_grade`, `run_universe_training.py`, `correlation_matrix.py`, `cli.py`, `audio_briefing.py`, `calculate_beneish_m_score`, `sanitize_filename`, `fetch_live_quote`, `ablation_study.py`, `train.py`, `get_vix_data`?**
  _High betweenness centrality (0.195) - this node is a cross-community bridge._
- **Why does `get_price_history()` connect `get_price_history` to `run_backtest`, `get_sentiment`, `test_statistical_arbitrage.py`, `app.py`, `calculate_hrp_weights`, `price_scout.py`, `screener_engine.py`, `morning_briefing.py`, `update_live_holdings_prices_and_alert_discord`, `load_model`, `preprocessing.py`, `get_sector_for_ticker`, `preprocess_data`, `utils.py`, `strategy_incubator.py`, `calculate_macro_liquidity_metrics`, `run_opening_range_session`, `get_news`, `AdversarialRedTeamAgent`, `agent_committee.py`, `calculate_portfolio_diversity_grade`, `run_universe_training.py`, `correlation_matrix.py`, `run_full_quant_experiment`, `sanitize_filename`, `ablation_study.py`, `get_vix_data`, `ws_autonomous_trader.py`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `PaperBroker` connect `PaperBroker` to `load_model`, `AICopilotEngine`, `fetch_live_quote`, `autonomous_trader.py`, `run_opening_range_session`, `ws_live_prediction.py`, `AutonomousTradingEngine`, `app.py`, `ws_insider_radar.py`, `utils.py`, `strategy_incubator.py`, `calculate_portfolio_diversity_grade`, `generate_comprehensive_factsheet`, `morning_briefing.py`, `cli.py`, `update_live_holdings_prices_and_alert_discord`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `1. Core Principles`, `2. Use Cases in Sentilyze` to the rest of the system?**
  _47 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `deep_learning_model.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10887096774193548 - nodes in this community are weakly interconnected._
- **Should `get_price_history` be split into smaller, more focused modules?**
  _Cohesion score 0.0957983193277311 - nodes in this community are weakly interconnected._