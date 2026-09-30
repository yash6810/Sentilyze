# Graph Report - Sentilyze  (2026-10-01)

## Corpus Check
- 3423 files · ~2,734,110 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3030 nodes · 5951 edges · 174 communities (165 shown, 9 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 63 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1df5cb67`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- deep_learning_model.py
- get_price_history
- quant_engine.py
- OpeningRangeBreakout
- run_backtest
- test_statistical_arbitrage.py
- utils.py
- drl_policy_agent.py
- render_workspace_header
- TradingEnvironment
- real_market_analyzer.py
- CloudDataLake
- app.py
- analyze_supply_chain_spillover
- calculate_hrp_weights
- run_temporal_fusion_forecast
- daily_scanner.py
- train_meta_ensemble
- OnlineNewtonStepOptimizer
- ws_live_prediction.py
- draw.io XML Style Reference
- get_sentiment
- test_omnichannel_mobile.py
- pfd-engineering-expert.md
- UltraQuantEngine
- compute_lead_lag_matrix
- cli.py
- social_sentiment.py
- AlpacaBrokerBridge
- realtime_tracker.py
- acpm_trainer.py
- get_us_market_session
- ws_alternative_data.py
- 3. Cross-Functional Flowcharts (Actor × Phase Grid)
- test_all_14_papers.py
- Edge Routing Decision Guide
- apply_triple_barrier_labeling
- triple_convex_engine.py
- Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform
- VolScaledMomentumEngine
- [Validator Rules Reference & Troubleshooting]
- TickerSentinelSwarm
- RegimeMixtureOfExperts
- test_data_ingestion.py
- ui/__init__.py
- Sentilyze Community Code of Conduct
- universal_web_scraper.py
- DLinearTCNModel
- .execute_daily_signals
- price_scout.py
- test_cross_disciplinary_alphas.py
- Contributing to Sentilyze
- test_papers_15_24.py
- 🧪 Experimental & Simulated Research Prototypes
- Sentilyze — Standing Audit Protocol
- webhook_dispatcher.py
- calculate_time_decayed_sentiment
- erd-database-expert.md
- generate_morning_briefing_text
- 6. CROSS-DISCIPLINARY & FRONTIER QUANTITATIVE IDEAS
- rules/graphify.md
- workflows/graphify.md
- GaussianHMMRegimeDetector
- EWMACorrelationMonitor
- PaperBroker
- CUSUMDetector
- PageHinkleyDetector
- QuantAlphaDAGEngine
- test_stock_image_provider.py
- run_monte_carlo_stress_test
- [Validator Rules Reference & Troubleshooting]
- grossman_zhou_allocation
- ADWINDetector
- black_swan_simulator.py
- compute_gamma_exposure_profile
- TradePostMortemLearner
- ws_screener.py
- .is_connected
- test_reddit_premarket_station.py
- InstitutionalDataGateway
- risk_constrained_kelly_allocation
- fetch_financial_statements
- test_tri_model_pipeline.py
- test_agent_committee.py
- DuckDBMarketEngine
- ws_portfolio.py
- correlation_matrix.py
- AgentMemoryStore
- preprocess_data
- block_external_alerts
- test_agent_memory.py
- AICopilotEngine
- options_surface.py
- drawio/SKILL.md
- AdversarialRedTeamAgent
- _load_sentiment_analyzer
- test_institutional_weekend_suite.py
- get_ultra_quant_engine
- compute_dark_pool_sentiment
- Domain Expert Reference Template
- create_technical_indicators
- 3. Equipment Mapping (Draw.io Native Shapes)
- smart_trader_engine.py
- calculate_insider_conviction_score
- test_pyramiding_cppi_alpha158.py
- Agent Browser — Live Web Automation Skill
- Agent Memory — Persistent Multi-Session Memory Skill
- Cybersecurity Skills — Institutional MLOps Security Protocol
- Editorial Diagram Design Skill
- Harness Engineering Skill
- macro_liquidity.py
- aws-well-architected-reviewer.md
- gcp-well-architected-reviewer.md
- run_universe_training.py
- macro_calendar.py
- evaluate_ticker_toxicity
- ws_committee.py
- liquidity_heatmap.py
- compile_biotech_catalyst_radar
- cockpit_backtest.py
- cockpit_trading.py
- test_pillar2_alternative_data.py
- calculate_15min_opening_range
- FastFinBERTEngine
- ultra_quant_engine.py
- MasterTradingLoop
- ws_deep_quant.py
- generate_comprehensive_factsheet
- TriBrainGatingLayer
- calculate_cppi_allocation
- analyze_earnings_surprises
- test_zero_giveback_profit_lock.py
- futures_sentinel.py
- FastNeuralEventClassifier
- apply_high_watermark_profit_lock
- audio_briefing.py
- print_formatted_trade_report
- test_acpm_trainer.py
- EventClassifierModel
- generate_pipeline_graph_data
- autonomous_trader.py
- calculate_portfolio_diversity_grade
- test_api.py
- check_correlation_shield
- .get_closed_trades_df
- generate_institutional_pdf_tearsheet
- neutralize_features
- .train_ticker
- SuperEnsembleClassifier
- simulate_market_opening
- advanced_quant_experiments.py
- RedditCommentMiner
- calculate_doubling_progress
- backtest_autopsy.py
- mock_return_series
- DynamicSharpeMetaEnsemble
- run_unified_institutional_pipeline
- deduplicate_news_stream
- components.py
- render_multi_agent_war_room
- ConformalCalibrator
- ws_portfolio_diversity.py
- get_sector_for_ticker
- cross_asset_pooling.py
- _get_browser_session
- .optimize_allocation
- .fuse
- SeriesDecomposition
- test_ultra_quant_engine.py
- .split
- temp_data_dir

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 129 edges
2. `get_price_history()` - 80 edges
3. `PaperBroker` - 75 edges
4. `fetch_live_quote()` - 47 edges
5. `get_news()` - 40 edges
6. `UltraQuantEngine` - 31 edges
7. `run_unified_institutional_pipeline()` - 30 edges
8. `convene_trading_committee()` - 30 edges
9. `AgentMemoryStore` - 29 edges
10. `preprocess_data()` - 29 edges

## Surprising Connections (you probably didn't know these)
- `test_technical_alpha_agent()` --calls--> `TechnicalAlphaAgent`  [EXTRACTED]
  tests/test_agent_committee.py → src/agent_committee.py
- `test_convene_trading_committee()` --calls--> `convene_trading_committee()`  [EXTRACTED]
  tests/test_agent_committee.py → src/agent_committee.py
- `mock_memory()` --uses--> `AgentMemoryStore`  [INFERRED]
  tests/test_agent_memory.py → src/agent_memory.py
- `test_agent_memory_softmax_dirichlet_recalibration()` --calls--> `AgentMemoryStore`  [EXTRACTED]
  tests/test_pyramiding_cppi_alpha158.py → src/agent_memory.py
- `test_paper7_quant_agents_trader()` --calls--> `AutonomousTradingEngine`  [EXTRACTED]
  tests/test_all_14_papers.py → src/autonomous_trader.py

## Import Cycles
- None detected.

## Communities (174 total, 9 thin omitted)

### Community 0 - "deep_learning_model.py"
Cohesion: 0.13
Nodes (20): create_sliding_window_tensors(), load_dlinear_model(), predict_momentum_probability(), Any, DataFrame, Tensor, High-Efficiency Deep Learning Engine: DLinear + Temporal Convolutional Network…, Converts a pandas DataFrame into sliding window sequence tensors for Deep… (+12 more)

### Community 1 - "get_price_history"
Cohesion: 0.06
Nodes (58): 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, Autonomous Multi-Agent Trading Committee & Deliberation Engine for Sentilyze.…, Agent 1: Evaluates Technical Price Action, Momentum, RSI, and Trend Alignment., TechnicalAlphaAgent, _fetch_alpaca_news(), _fetch_alpaca_price_history(), _fetch_benzinga_news(), _fetch_direct_yahoo_chart() (+50 more)

### Community 2 - "quant_engine.py"
Cohesion: 0.14
Nodes (25): MasterQuantPipelineResult, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain() (+17 more)

### Community 3 - "OpeningRangeBreakout"
Cohesion: 0.16
Nodes (12): OpeningRangeBreakout, Any, DataFrame, Paper 25: Opening Range Breakout (ORB) with Stocks-in-Play Filter. Source:…, Simulate multi-year daily ORB strategy returns., Opening Range Breakout (ORB) trading engine with Stocks-in-Play filter. Paper…, Rank universe and select top 'Stocks in Play' based on volume, ATR, and…, Evaluate ORB signal using daily OHLC + volatility approximation. (+4 more)

### Community 4 - "run_backtest"
Cohesion: 0.07
Nodes (60): _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition(), calculate_performance_metrics(), _calculate_trade_outcomes(), create_monthly_returns_heatmap() (+52 more)

### Community 5 - "test_statistical_arbitrage.py"
Cohesion: 0.08
Nodes (49): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), calibrate_avellaneda_lee_stat_arb(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any (+41 more)

### Community 6 - "utils.py"
Cohesion: 0.05
Nodes (34): ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Logger, main(), Batch Universe Trainer for Remaining S&P 100 Tickers., run_single(), Master Academic Research Papers Empirical Benchmark Suite (All 14 Papers).…, Agent Memory — Persistent Multi-Session Memory Engine for Sentilyze.…, Paper 3: Almgren & Chriss (2000) - Optimal Execution of Portfolio Transactions.… (+26 more)

### Community 7 - "drl_policy_agent.py"
Cohesion: 0.13
Nodes (20): ActorCriticPolicy, DRLTradingEnvironment, evaluate_drl_policy_action(), Any, ndarray, Tensor, Deep Reinforcement Learning (DRL) Autonomous Policy Agent for Sentilyze.…, Trains an Actor-Critic DRL policy agent on historical market returns and news… (+12 more)

### Community 8 - "render_workspace_header"
Cohesion: 0.09
Nodes (28): FactorAttributionEngine, generate_institutional_factsheet_for_ticker(), InstitutionalFactsheet, Any, DataFrame, Series, Institutional Factor Attribution & Performance Factsheet Engine Computes…, Compute execution statistics from trade log dataframe. (+20 more)

### Community 9 - "TradingEnvironment"
Cohesion: 0.14
Nodes (16): optimize_rl_position_allocation(), PPOPolicyAgent, Any, ndarray, ⚠️ EXPERIMENTAL / RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM PRODUCTION…, Computes mean action (leverage) between 0.0 and 2.0., Estimates state value., Trains Actor-Critic parameters across historical episodes. (+8 more)

### Community 10 - "real_market_analyzer.py"
Cohesion: 0.23
Nodes (13): calculate_real_market_metrics(), compute_institutional_cost_function(), evaluate_channel_comment_intelligence(), generate_most_increasable_set(), load_real_price_history(), Any, DataFrame, Real Market Microstructure, Cost Function & Multi-Channel Intelligence Engine.… (+5 more)

### Community 11 - "CloudDataLake"
Cohesion: 0.14
Nodes (14): CloudDataLake, Any, Cloud PostgreSQL Data Lake (Supabase / Neon) Connector for Sentilyze. Pillar 6…, Supabase / PostgreSQL Cloud Data Lake Connector., Validates or generates cloud database schema., Syncs local trade executions to the cloud database., Publishes real-time portfolio snapshot to cloud WebSockets channel., generate_twap_order_schedule() (+6 more)

### Community 12 - "app.py"
Cohesion: 0.18
Nodes (15): get_model_universe_indicator(), load_universe_tickers(), main(), cache_data, Sentilyze - Institutional Algorithmic Trading & MLOps Platform. High-Speed…, Loads active S&P 100 universe tickers., Computes universe model training coverage and mean accuracy using compiled…, render_backtest_cockpit() (+7 more)

### Community 13 - "analyze_supply_chain_spillover"
Cohesion: 0.14
Nodes (15): analyze_supply_chain_spillover(), Any, ndarray, Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for…, Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)., Executes a Graph Convolutional Network (GCN) layer: H_new = ReLU(A_hat * H * W)…, Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab…, High-level entry point to run GNN supply chain shock propagation. (+7 more)

### Community 14 - "calculate_hrp_weights"
Cohesion: 0.15
Nodes (19): calculate_hrp_weights(), calculate_risk_parity_weights(), get_cluster_var(), get_quasi_diag(), get_rec_bisection(), Any, ndarray, Series (+11 more)

### Community 15 - "run_temporal_fusion_forecast"
Cohesion: 0.14
Nodes (15): Any, DataFrame, ndarray, Temporal Fusion Transformer (TFT) & Multi-Horizon Self-Attention Engine for…, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with… (+7 more)

### Community 16 - "daily_scanner.py"
Cohesion: 0.09
Nodes (42): format_signal_card(), _is_duplicate_alert(), Any, Sends a rich formatted trade alert card to a Discord channel via Webhook., Checks if an alert fingerprint was already dispatched within ttl_seconds., Dispatches a crystal-clear, high-impact Discord card for live autonomous trade…, Records an alert fingerprint into results/discord_alert_ledger.json., Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord. (+34 more)

### Community 17 - "train_meta_ensemble"
Cohesion: 0.18
Nodes (13): MetaEnsembleClassifier, DataFrame, ndarray, Series, Generates binary class prediction (0 = Hold/Sell, 1 = Buy) using soft-voting…, Instantiates and fits the Meta-Ensemble classifier., Multi-Model Meta-Ensemble stacking XGBoost, Random Forest, and Calibrated…, Trains all component models on the training dataset. (+5 more)

### Community 18 - "OnlineNewtonStepOptimizer"
Cohesion: 0.18
Nodes (10): OnlineNewtonStepOptimizer, DataFrame, ndarray, Polynomial-Time ONS Portfolio Engine (Hazan et al.)., Processes price relatives (r_t = Close_t / Close_{t-1}) and updates weights in…, Fast O(d log d) Euclidean projection onto probability simplex., Runs ONS sequence through time and outputs daily allocations and portfolio…, test_paper1_online_newton_step() (+2 more)

### Community 19 - "ws_live_prediction.py"
Cohesion: 0.17
Nodes (20): detect_classical_chart_patterns(), generate_ai_chart_explanation(), match_historical_chart_twins(), normalize_waveform(), Any, DataFrame, ndarray, AI Chart Pattern Recognition, Geometric Wave Learning & Visual Understanding… (+12 more)

### Community 20 - "draw.io XML Style Reference"
Cohesion: 0.08
Nodes (23): Color Properties, Common Style Properties, Container Types, Critical Edge Rules, draw.io XML Style Reference, Edge Attributes, Edge Styles, Escaping Rules (XML attribute context) (+15 more)

### Community 21 - "get_sentiment"
Cohesion: 0.12
Nodes (24): assign_source_weight(), Assigns institutional source weight and priority tier based on publisher…, analyze_sentiment(), calculate_swift_sentiment_summary(), clean_financial_text(), get_sentiment(), _parse_analyzer_output(), Any (+16 more)

### Community 22 - "test_omnichannel_mobile.py"
Cohesion: 0.16
Nodes (15): answer_financial_query(), Any, Parses natural language questions and routes them to quantitative engines.…, generate_smartwatch_glance_payload(), Any, Generates structured complication JSON for Apple Watch (watchOS) and Wear OS., format_whatsapp_trade_alert(), Any (+7 more)

### Community 23 - "pfd-engineering-expert.md"
Cohesion: 0.10
Nodes (20): 1. Distillation Column (`distillation_column`), 1. `PHASE_PORT_VIOLATION`, 2. `DEAD_END_STREAM`, 2. Pump (`pump`), 3. Compressor (`compressor`), 3. `OPPOSING_FLOW`, 4. `GRAVITY_VIOLATION`, 4. Separator (`separator`) (+12 more)

### Community 24 - "UltraQuantEngine"
Cohesion: 0.10
Nodes (19): Any, DataFrame, Evaluates SEC Form 4 insider purchases and executive cluster buys., Mines SEC Form 8-K filings for material corporate events., Calculates retail mention velocity and sentiment from Reddit & StockTwits., Queries OpenFDA & ClinicalTrials.gov for biotech binary events., Calculates optimal execution trajectory and market impact., Calculates optimal risky allocation under dynamic drawdown constraint. (+11 more)

### Community 25 - "compute_lead_lag_matrix"
Cohesion: 0.21
Nodes (13): compute_lead_lag_matrix(), _granger_f_test(), Any, DataFrame, ndarray, Series, rank_market_price_leaders(), Ranks stocks by their predictive influence (number of peers they statistically… (+5 more)

### Community 26 - "cli.py"
Cohesion: 0.22
Nodes (14): Root entry point for Sentilyze CLI. Run directly with: python sentilyze.py NVDA…, cmd_audit(), cmd_briefing(), cmd_portfolio(), cmd_scan(), cmd_trades(), main(), print_banner() (+6 more)

### Community 27 - "social_sentiment.py"
Cohesion: 0.23
Nodes (13): calculate_social_buzz_metrics(), fetch_social_sentiment_tracker(), Any, Social Sentiment Velocity & Retail Multi-Platform Scraper for Sentilyze. Pillar…, Scrapes real-time streaming retail sentiment from Stocktwits public symbol…, Scrapes tech community discussions on AI catalysts (OpenAI, Anthropic, Nvidia)…, Computes retail sentiment velocity and flow conviction metrics., High-level entry point to retrieve calibrated real-time social buzz metrics for… (+5 more)

### Community 28 - "AlpacaBrokerBridge"
Cohesion: 0.27
Nodes (11): AlpacaBrokerBridge, Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.…, patch, Real integration test: connects to Alpaca's actual paper API endpoint using…, test_alpaca_broker_close_all_positions(), test_alpaca_broker_close_position(), test_alpaca_broker_get_account_summary(), test_alpaca_broker_initialization() (+3 more)

### Community 29 - "realtime_tracker.py"
Cohesion: 0.08
Nodes (35): convene_trading_committee(), Orchestrates a full round-table deliberation of the 5-Agent Trading Committee…, AI Trade Copilot & Conversational Analyst for Sentilyze. Provides natural…, Gathers overnight macro VIX volatility regime, paper portfolio balance, and top…, Any, Paper 25 Live Runner: Opening Range Breakout (ORB) on Top Stocks in Play.…, Executes a live 5-Minute Opening Range Breakout scan across top liquid assets:…, run_opening_range_session() (+27 more)

### Community 30 - "acpm_trainer.py"
Cohesion: 0.21
Nodes (8): Unified Alpha-Conformal Purged Multi-Task (ACPM) Quantitative Training Engine.…, compute_deflated_sharpe_ratio(), PurgedGroupTimeSeriesSplit, Combinatorial Purged & Embargoed Cross-Validation (CPCV) for Financial Machine…, Time-series cross-validator that purges overlapping event windows and applies…, Calculates the Deflated Sharpe Ratio (DSR) to test for backtest overfitting., Regime-Conditioned Mixture of Experts (MoE) Architecture. Trains 3 specialized…, test_purged_cv_and_dsr()

### Community 31 - "get_us_market_session"
Cohesion: 0.16
Nodes (15): check_market_hours_preflight(), get_current_ny_time(), get_us_market_session(), Any, datetime, Unified US Stock Market (NYSE / NASDAQ) Session & Calendar Engine for…, Pre-flight sanity check for automated workflows. Returns True if execution…, Returns the current precise timestamp in America/New_York (Eastern Time). (+7 more)

### Community 32 - "ws_alternative_data.py"
Cohesion: 0.12
Nodes (23): auto_register_ipo_ticker(), fetch_pre_ipo_radar_summary(), fetch_sec_edgar_ipo_filings(), get_pre_ipo_pipeline_df(), Any, DataFrame, IPO & Pre-IPO Intelligence Radar for Sentilyze. Pillar 9 Alternative Asset…, Fetches real-time SEC Form S-1 / S-1/A IPO registration statements from SEC… (+15 more)

### Community 33 - "3. Cross-Functional Flowcharts (Actor × Phase Grid)"
Cohesion: 0.10
Nodes (19): 1. Flat Swimlanes (BPMN-style Flowcharts), 2. Nested Architecture Containers, 3. Cross-Functional Flowcharts (Actor × Phase Grid), 4. When to Use What, Container Key Rules Summary, Critical Rules, Critical Rules, Decision Guide (+11 more)

### Community 34 - "test_all_14_papers.py"
Cohesion: 0.10
Nodes (24): Any, Executes empirical backtests comparing all 14 academic paper methodologies., run_all_14_papers_benchmark(), calculate_almgren_chriss_trajectory(), Any, Computes Almgren-Chriss optimal trading trajectory. x_j = 2 * sinh(0.5 * kappa…, detect_negative_cycle_arbitrage(), Any (+16 more)

### Community 35 - "Edge Routing Decision Guide"
Cohesion: 0.12
Nodes (16): Critical Rules, Decision Flowchart, Default Behavior (Built-in Router), Don't add waypoints, Don't hand-route, Edge labels, Edge Routing Decision Guide, Every edge needs an mxGeometry child (+8 more)

### Community 36 - "apply_triple_barrier_labeling"
Cohesion: 0.14
Nodes (15): apply_triple_barrier_labeling(), calculate_deflated_sharpe_ratio(), DataFrame, Series, Marcos López de Prado's Triple-Barrier Method & Deflated Sharpe Ratio (DSR).…, Computes Bailey & López de Prado's Deflated Sharpe Ratio (DSR). Adjusts for: -…, Applies López de Prado's path-dependent Triple-Barrier Method to generate trade…, Any (+7 more)

### Community 37 - "triple_convex_engine.py"
Cohesion: 0.19
Nodes (10): PolyTimeConvexOptimizer, Stanford Multi-Period Convex Portfolio Optimization Engine (Boyd et al.).…, Polynomial-Time Convex Portfolio Optimizer with Market Frictions (Boyd et al.)., Triple-Convex Quantum Execution Engine. Fuses the top quantitative research…, Unified High-Expectancy, Minimum-Drawdown, Sub-15ms Execution Engine., TripleConvexEngine, test_paper2_boyd_convex_optimizer(), test_polytime_convex_optimizer() (+2 more)

### Community 38 - "Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform"
Cohesion: 0.12
Nodes (16): 1. Run in 1-Click (No Installation Required), 2. Local Setup & Installation, 3. Run the Quantitative CLI, 4. Launch the Streamlit Mission Control, 5. Run Full Test Suite (274+ Unit Tests), Active Holdings (Entered 2026-09-08), 📊 Empirical Alpha Attribution & Benchmarks, 🏛️ Grounded 5-Agent Deliberation Council (+8 more)

### Community 39 - "VolScaledMomentumEngine"
Cohesion: 0.16
Nodes (12): DataFrame, Series, Daniel & Moskowitz (JFE 2016) Volatility-Scaled Multi-Horizon Momentum Engine.…, Implements Daniel & Moskowitz (2016) Volatility-Scaled Multi-Horizon Momentum., Computes RiskMetrics EWMA realized volatility with zero lookahead bias., Calculates the composite volatility-scaled momentum position for a given ticker., VolScaledMomentumEngine, VolScaledMomentumResult (+4 more)

### Community 40 - "[Validator Rules Reference & Troubleshooting]"
Cohesion: 0.12
Nodes (16): 1. `DIRECT_WAN_TO_LAN`, 2. `ORPHAN_DEVICE`, 3. `VLAN_LEAK`, 4. `REDUNDANCY_WARNING`, 5. `MISSING_DMZ`, 6. `FLAT_NETWORK`, 7. `FIREWALL_BYPASS`, 8. `SINGLE_TRUNK` (+8 more)

### Community 41 - "TickerSentinelSwarm"
Cohesion: 0.13
Nodes (16): detect_peak_crest_exhaustion(), Any, Dedicated Micro-Agent assigned to monitor a single stock position 24/7. Tracks…, Audits live price tick, applies Zero-Giveback profit ratchets, and detects peak…, Manages the full swarm of Dedicated Ticker Sentinels across all open positions.…, Synchronizes active sentinels with current portfolio open positions without…, Audits all active sentinels concurrently. Optionally persists ratcheted SL and…, Detects if a stock has reached the crest/peak of its 15-minute momentum wave… (+8 more)

### Community 42 - "RegimeMixtureOfExperts"
Cohesion: 0.26
Nodes (8): Any, DataFrame, ndarray, Series, 3-Expert Gated Mixture of Experts classifier for financial regimes., Labels historical rows into 3 latent market regimes: 0 = Bull Momentum (Price >…, RegimeMixtureOfExperts, test_regime_moe()

### Community 43 - "test_data_ingestion.py"
Cohesion: 0.12
Nodes (15): fixture, Fixture to set a temporary data directory for tests., Test that get_news fetches data from NewsAPI and saves it to a cache file., Test that get_news loads data from the cache if it's not stale., Test that get_news re-fetches data if the cache is stale., Test that get_price_history fetches data from yfinance and saves it to a cache…, Test that get_price_history loads data from the cache if it's not stale., Test that get_price_history re-fetches data if the cache is stale. (+7 more)

### Community 47 - "Sentilyze Community Code of Conduct"
Cohesion: 0.25
Nodes (7): 📜 Attribution, ⚖️ Enforcement Escalation, 🛡️ Enforcement & Responsibilities, 🌟 Our Pledge, 🎯 Our Standards, 📬 Reporting Guidelines, Sentilyze Community Code of Conduct

### Community 48 - "universal_web_scraper.py"
Cohesion: 0.13
Nodes (24): batch_scrape_urls(), detect_stock_tickers(), extract_financial_tables_from_html(), _extract_summary_sentences(), Any, DataFrame, Universal Web Scraper & Financial Intelligence Extractor for Sentilyze.…, Extracts stock ticker symbols from unstructured text using cashtags ($TICKER)… (+16 more)

### Community 49 - "DLinearTCNModel"
Cohesion: 0.18
Nodes (9): DLinearTCNModel, Dual-Branch Deep Learning Architecture: - Branch 1 (Trend): Linear projection…, FastAgentPipeline, Any, DataFrame, Sub-second Multi-Model Agent Decision Pipeline., Executes complete Tri-Model evaluation on incoming headline & market structure.…, Verify that DLinear-TCN forward pass produces correct [batch, num_classes]… (+1 more)

### Community 50 - ".execute_daily_signals"
Cohesion: 0.15
Nodes (11): Any, Executes an immediate manual live/simulated BUY order from UI., Executes a 0.5N ATR Turtle Pyramiding scale-in order (Unit 2 or Unit 3).…, Executes an immediate manual live/simulated exit of an open position., Executes a 50% scale-out on an open position and moves stop to break-even., Appends trade to closed_trades and records an episodic post-mortem in…, Executes daily quantitative scan results using the Concentrated Top-2 + Scale-…, Loads existing portfolio state from JSON or initializes a fresh account if file… (+3 more)

### Community 51 - "price_scout.py"
Cohesion: 0.18
Nodes (12): get_latest_scout_alerts(), PriceActionScoutAgent, PriceScoutBot, Any, Real-Time Price Action & Tape-Reading Scout Subagent (Bot) for Sentilyze.…, Continuous Background Scanner Bot that scouts the 538 universe assets, detects…, Scans given tickers for real-time volume breakout candidates., Retrieves the latest price scout breakout alerts. (+4 more)

### Community 52 - "test_cross_disciplinary_alphas.py"
Cohesion: 0.08
Nodes (63): compute_ant_colony_venue_allocation(), compute_bayesian_dark_pool_equilibrium(), compute_carnot_profit_efficiency(), compute_circadian_seasonal_risk_scalar(), compute_conformal_safety_bands(), compute_doppler_order_flow_shift(), compute_epigenetic_factor_methylation(), compute_feynman_path_least_action() (+55 more)

### Community 53 - "Contributing to Sentilyze"
Cohesion: 0.15
Nodes (12): 1. Code Formatting (Black), 1. Fork & Clone, 2. Linting (Flake8), 2. Virtual Environment & Dependencies, 3. Security Auditing (Bandit), 4. Unit Test Suite (Pytest), Contributing to Sentilyze, 🏛️ Core Project Architecture & Contribution Rules (+4 more)

### Community 54 - "test_papers_15_24.py"
Cohesion: 0.10
Nodes (17): calculate_cdar(), optimize_cdar_portfolio(), Any, DataFrame, ndarray, Paper 19: Conditional Drawdown-at-Risk (CDaR) Portfolio Optimization. Source:…, Calculate Conditional Drawdown-at-Risk: the expected drawdown in the worst…, Optimize portfolio weights to minimize CDaR. Simplified approach: compute CDaR… (+9 more)

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

### Community 59 - "erd-database-expert.md"
Cohesion: 0.13
Nodes (14): 1. `FK_WITHOUT_TARGET`, 2. `ORPHAN_TABLE`, 3. `DUPLICATE_PK`, 4. `SELF_REFERENCE_MISSING`, 5. `MISSING_JUNCTION`, 6. `CIRCULAR_FK`, [Anti-Patterns], [Docs Index] (+6 more)

### Community 60 - "generate_morning_briefing_text"
Cohesion: 0.23
Nodes (14): generate_morning_briefing_text(), get_portfolio_intelligence(), load_universe_candidates(), Any, Reads live paper portfolio state for broadcast reporting., Assembles a comprehensive, institutional Wall Street Morning Podcast and…, Loads clean ticker list from stocks.txt or falls back to core liquid universe., Scans candidate universe to score and rank the Top Alpha Stocks in Play for… (+6 more)

### Community 61 - "6. CROSS-DISCIPLINARY & FRONTIER QUANTITATIVE IDEAS"
Cohesion: 0.07
Nodes (27): 1. EXECUTIVE SUMMARY & CORE PLATFORM VISION, 2. THE 4-PILLAR WINNING QUANTITATIVE ALPHA FORMULA, 3. EMPIRICAL MULTI-DECADE HISTORICAL BENCHMARKS (1980–2026), 4. MASTER 5-COCKPIT MISSION CONTROL ARCHITECTURE, 5. THE 300 INSTITUTIONAL RESOURCES SYNTHESIS, 6. CROSS-DISCIPLINARY & FRONTIER QUANTITATIVE IDEAS, 7. IMPLEMENTATION & TECHNOLOGY STACK BLUEPRINT, 🐺 A. Evolutionary Biology: Lotka-Volterra Predator-Prey Market Dynamics (+19 more)

### Community 64 - "GaussianHMMRegimeDetector"
Cohesion: 0.13
Nodes (11): GaussianHMMRegimeDetector, Any, DataFrame, ndarray, Paper 15: Gaussian Hidden Markov Model for Market Regime Detection. Source:…, 3-state Gaussian HMM regime classifier: Bull / Normal / Crisis. Uses hand-…, Gaussian emission probability for each state., Forward algorithm step: update filtered state probabilities with one new… (+3 more)

### Community 65 - "EWMACorrelationMonitor"
Cohesion: 0.14
Nodes (10): EWMACorrelationMonitor, Any, DataFrame, ndarray, Paper 17: RiskMetrics EWMA Volatility & Correlation Monitor. Source: J.P.…, Real-time EWMA-based correlation and volatility monitor. Tracks time-varying…, Initialize EWMA state from a seed window of returns., Update EWMA state with one day's returns across all assets. Returns current… (+2 more)

### Community 67 - "PaperBroker"
Cohesion: 0.11
Nodes (21): PaperBroker, Alias for _save to ensure 100% backward compatibility., Institutional Multi-Stage Quantitative Execution Broker ($100k Account).…, Reloads latest state from disk if file exists., Returns high-level KPI metrics for the portfolio dashboard., fixture, Test that a signal=SELL with confidence=0.50 does NOT prematurely exit a fresh…, Test that a signal=SELL with confidence < 0.40 triggers MODEL_SELL exit. (+13 more)

### Community 68 - "CUSUMDetector"
Cohesion: 0.15
Nodes (9): CUSUMDetector, Any, ndarray, Paper 16: CUSUM Sequential Change-Point Detection. Source: E.S. Page (1954) —…, Cumulative Sum detector for online mean-shift detection. Monitors a numeric…, Process one observation. Returns alarm status., Process a batch of observations., Reset detector state. (+1 more)

### Community 69 - "PageHinkleyDetector"
Cohesion: 0.15
Nodes (9): PageHinkleyDetector, Any, ndarray, Paper 22: Page-Hinkley Sequential Test for Concept Drift Detection. Source:…, Page-Hinkley test for detecting changes in the mean of a stream. Monitors…, Process one observation. Returns drift status., Process a batch of observations., Reset detector state. (+1 more)

### Community 70 - "QuantAlphaDAGEngine"
Cohesion: 0.14
Nodes (17): AlphaDAGStageContract, Any, DataFrame, Series, QuantAlphaDAGEngine, Calculates time-aligned returns, normalized volumes, and volatility bands., Generates Gen-3 Multi-Horizon Feature Library: - Factor 1: Triple-Horizon Trend…, Decoupled Alpha Generators with Gen-3 FinBERT Crash Shield & Volatility Sizing. (+9 more)

### Community 71 - "test_stock_image_provider.py"
Cohesion: 0.13
Nodes (32): clean_ticker_symbol(), clear_logo_cache(), generate_fallback_svg_badge(), get_asset_avatar_html(), get_asset_logo_base64(), get_sector_gradient(), is_crypto_symbol(), Sentilyze Stock & Crypto Image & Brand Identity Engine.… (+24 more)

### Community 72 - "run_monte_carlo_stress_test"
Cohesion: 0.11
Nodes (25): calculate_custom_rebalance(), calculate_share_allocation(), Any, Helper to calculate share allocation from latest daily signals file or universe…, Computes exact whole-share buy allocations for a given capital budget across…, Any, DataFrame, Helper wrapper for Monte Carlo VaR simulation. (+17 more)

### Community 73 - "[Validator Rules Reference & Troubleshooting]"
Cohesion: 0.13
Nodes (14): 1. `ORPHAN_POD`, 2. `SERVICE_WITHOUT_TARGET`, 3. `INGRESS_BYPASS`, 4. `PVC_WITHOUT_PV`, 5. `NAMESPACE_LEAK`, 6. `MISSING_RESOURCE_LIMITS`, 7. `PRIVILEGED_CONTAINER`, [Anti-Patterns] (+6 more)

### Community 74 - "grossman_zhou_allocation"
Cohesion: 0.28
Nodes (5): grossman_zhou_allocation(), Any, Paper 18: Grossman-Zhou Optimal Drawdown-Constrained Strategy. Source: Grossman…, Compute the Grossman-Zhou optimal risky allocation under a drawdown constraint…, TestGrossmanZhou

### Community 75 - "ADWINDetector"
Cohesion: 0.19
Nodes (7): ADWINDetector, Any, ndarray, Paper 21: ADWIN (Adaptive Windowing) Drift Detector. Source: Bifet & Gavaldà…, ADWIN drift detector with Hoeffding bound. Maintains a variable-length window…, Add one observation. Returns whether drift was detected. If drift is detected,…, TestADWIN

### Community 76 - "black_swan_simulator.py"
Cohesion: 0.23
Nodes (11): calculate_kelly_sizing(), estimate_market_impact_slippage(), Any, Historical Black Swan Crisis Simulator & Kelly Position Sizing for Sentilyze.…, Calculates optimal position sizing using the Kelly Criterion: Kelly % = W - (1…, Estimates market execution slippage using the Almgren-Chriss square-root impact…, Stress-tests the current portfolio against major historical market crashes.…, simulate_portfolio_crises() (+3 more)

### Community 77 - "compute_gamma_exposure_profile"
Cohesion: 0.05
Nodes (49): calculate_black_scholes_gamma(), calculate_options_gex(), compute_gamma_exposure_profile(), fetch_options_chain_data(), Any, DataFrame, Options Gamma Exposure (GEX), Volatility Surface & Strike Wall Radar for…, Computes institutional Net Gamma Exposure (GEX), Call Wall, Put Wall, and Gamma… (+41 more)

### Community 78 - "TradePostMortemLearner"
Cohesion: 0.09
Nodes (22): Any, DataFrame, Series, Autonomous Trade Post-Mortem & Bayesian Council Weight Calibration Engine for…, Categorizes trade outcome and assigns attribution scores., Computes performance feedback scores for each specialist agent based on…, Updates weights using a Softmax Bayesian update with Dirichlet regularization:…, Executes an end-to-end post-mortem learning cycle: 1. Analyzes trade history.… (+14 more)

### Community 79 - "ws_screener.py"
Cohesion: 0.50
Nodes (3): Workspace 24: Real-Time Market Anomaly Screener., Renders the Real-Time Market Anomaly Screener Workspace., render_screener_workspace()

### Community 80 - ".is_connected"
Cohesion: 0.21
Nodes (7): Any, Fetches active positions from Alpaca brokerage., Liquidates an active position on Alpaca and cancels all open child orders., Emergency liquidator: closes all positions and cancels all open orders., Verifies active connection to Alpaca Brokerage API., Fetches live Alpaca account equity, buying power, and cash., Submits an institutional Bracket Order: - Entry: Market order - Exit 1: Limit…

### Community 81 - "test_reddit_premarket_station.py"
Cohesion: 0.12
Nodes (24): fetch_4station_premarket_intelligence(), fetch_8station_premarket_intelligence(), fetch_all_reddit_headlines_for_ticker(), _fetch_subreddit_rss_entries(), Any, DataFrame, Systematic 8-Station Multi-Channel Reddit Market Intelligence Engine. Pillar 2…, Fetches real-time Atom RSS feed for a subreddit using safe defusedxml. (+16 more)

### Community 82 - "InstitutionalDataGateway"
Cohesion: 0.12
Nodes (12): InsiderFlowSnapshot, InstitutionalDataGateway, MacroRegimeSnapshot, OptionsSentimentSnapshot, Fetches Federal Reserve Macro Indicators from St. Louis Fed (FRED) or falls…, Classifies macroeconomic regime and calculates sizing multiplier., Fetches SEC Form 4 insider trading transactions (CEOs, CFOs, 10% Owners)., Computes Options Put/Call ratios and Max Pain strike. (+4 more)

### Community 83 - "risk_constrained_kelly_allocation"
Cohesion: 0.24
Nodes (6): Any, ndarray, Paper 23: Risk-Constrained Kelly Gambling. Source: Busseti, Ryu, Boyd —…, Compute optimal Kelly allocation with drawdown probability constraint.…, risk_constrained_kelly_allocation(), TestRiskKelly

### Community 84 - "fetch_financial_statements"
Cohesion: 0.14
Nodes (27): _persist_ablation_results(), Any, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), calculate_altman_z_score(), calculate_dcf_fair_value() (+19 more)

### Community 85 - "test_tri_model_pipeline.py"
Cohesion: 0.16
Nodes (15): Financial Event & Emotion Companion Model (Model 2). Companion model to FinBERT…, get_fast_agent_pipeline(), Fast Sub-Second Agent Decision Router. Coordinates the Tri-Model Neural Brain…, Singleton getter for FastAgentPipeline., get_fast_finbert_engine(), Accelerated Low-Latency FinBERT Engine (Model 1). Optimized for high-speed…, Singleton getter for FastFinBERTEngine., NeuralGatingNetwork (+7 more)

### Community 86 - "test_agent_committee.py"
Cohesion: 0.13
Nodes (19): audit_full_universe_committee(), ChiefRiskOfficerAgent, compute_fractional_kelly_sizing(), ForensicFundamentalAgent, Runs committee deliberation across the provided universe of tickers., Agent 2: Evaluates FinBERT Deep NLP Sentiment across Live News Streams., Computes true mathematical fractional Kelly Criterion position sizing: f* = (p…, Agent 3: Evaluates Real Financial Statements, Piotroski F-Score, and DCF… (+11 more)

### Community 87 - "DuckDBMarketEngine"
Cohesion: 0.06
Nodes (30): DuckDBMarketEngine, get_duckdb_scanner(), Any, DataFrame, Inserts or replaces daily bars for a given ticker., Populates the DuckDB lake from pre-computed backtest portfolio curves in…, Screens for high-probability momentum setups across the entire lake., Screens for stocks where SMA50 is breaking out above SMA200 (Long-Term Bullish… (+22 more)

### Community 88 - "ws_portfolio.py"
Cohesion: 0.23
Nodes (12): build_unified_portfolio(), load_all_ticker_portfolios(), DataFrame, Load individual backtest portfolio CSVs for all available tickers. Args:…, Combines individual stock strategies into a single managed multi-asset fund.…, _get_cached_portfolios(), load_ticker_sectors(), cache_data (+4 more)

### Community 89 - "correlation_matrix.py"
Cohesion: 0.38
Nodes (6): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…

### Community 90 - "AgentMemoryStore"
Cohesion: 0.12
Nodes (13): AgentMemoryStore, Any, Calculates historical win rate, trade count, and recent trajectory for a…, Synthesizes episodic trade memory to provide risk adjustments to…, Synchronizes executed_trades.csv into trade_postmortems.jsonl and recalibrates…, Retrieves the most recent post-mortems in reverse chronological order., Calibrates committee voting weights using the Softmax Dirichlet RLFF Algorithm.…, Returns current dynamic voting weights. (+5 more)

### Community 91 - "preprocess_data"
Cohesion: 0.06
Nodes (64): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+56 more)

### Community 92 - "block_external_alerts"
Cohesion: 0.50
Nodes (3): block_external_alerts(), fixture, Autouse fixture that prevents tests from sending real outbound network calls to…

### Community 93 - "test_agent_memory.py"
Cohesion: 0.12
Nodes (15): mock_memory(), fixture, Unit Tests for AgentMemoryStore & Episodic Trade Memory.…, Verifies that high-expectancy winners receive alpha champion boosts., Verifies synchronizing executed trades from an external CSV file., Verifies that PaperBroker._record_trade_closure automatically records post-…, Verifies that baseline rules and default weights are created upon…, Verifies appending post-mortems and updating empirical weights. (+7 more)

### Community 94 - "AICopilotEngine"
Cohesion: 0.24
Nodes (8): AICopilotEngine, Any, Conversational intelligence engine that parses queries and generates analytical…, Interprets user prompt and routes to appropriate financial analytical…, test_copilot_committee_query(), test_copilot_portfolio_query(), test_copilot_stress_query(), test_copilot_ticker_analysis_query()

### Community 95 - "options_surface.py"
Cohesion: 0.27
Nodes (10): calculate_multileg_payoff(), generate_volatility_surface_mesh(), Any, 3D Implied Volatility Surface & Multi-Leg Options Strategy Desk for Sentilyze.…, Constructs a 3D Implied Volatility Surface across strike prices and expiration…, Calculates profit and loss (P&L) curves at expiration for institutional multi-…, test_calculate_multileg_payoff_bull_call_spread(), test_calculate_multileg_payoff_iron_condor() (+2 more)

### Community 96 - "drawio/SKILL.md"
Cohesion: 0.13
Nodes (14): 1.Reference-Docs(AI-Knowledge), 2.Validator-Scripts(Programmatic-Enforcement), 3.Topological-Corrections(Auto-Fix), [Architectural Constraints], [Batch Diagram Generation (Highly Recommended)], Containers, [Diagram Builder Tools], [Docs Index] (+6 more)

### Community 97 - "AdversarialRedTeamAgent"
Cohesion: 0.24
Nodes (8): AdversarialRedTeamAgent, Any, Agent 5: Adversarial Red-Team / Devil's Advocate Specialist. Actively hunts for…, Conducts a rigorous adversarial audit of the target asset., Tests for Adversarial Red-Team Specialist Agent (src/red_team_agent.py).…, test_red_team_agent_initialization(), test_red_team_evaluation_structure(), test_red_team_stress_scenario()

### Community 98 - "_load_sentiment_analyzer"
Cohesion: 0.21
Nodes (12): clean_headline_data(), _load_sentiment_analyzer(), Any, Cleans a headline CSV file by removing rows with invalid stock tickers. Caches…, Thread-safely loads the FinBERT sentiment analysis model and tokenizer once…, fixture, temp_data_dirs(), test_clean_headline_data_invalid_tickers_no_cache() (+4 more)

### Community 99 - "test_institutional_weekend_suite.py"
Cohesion: 0.11
Nodes (20): Any, Sentilyze 10,000-Path Monte Carlo Portfolio Stress-Testing Engine. Simulates…, Executes a vectorized 10,000-path Monte Carlo simulation over forecast_days., run_portfolio_monte_carlo_simulation(), Any, Sentilyze Deep Trade Post-Mortem & Autonomous Rule Inducer. Mines historical…, Performs comprehensive autopsy across historical trades and extracts behavioral…, run_trade_memory_autopsy() (+12 more)

### Community 100 - "get_ultra_quant_engine"
Cohesion: 0.33
Nodes (5): _persist_committee_resolution(), Any, Saves the committee resolution into results/committee_resolutions.json., get_ultra_quant_engine(), Returns the singleton UltraQuantEngine instance.

### Community 101 - "compute_dark_pool_sentiment"
Cohesion: 0.24
Nodes (13): compute_dark_pool_sentiment(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent institutional off-exchange block trades and dark pool prints., Scans option chain contracts where daily volume significantly exceeds open…, Synthesizes dark pool prints and unusual options flow into a unified…, scan_abnormal_options_vol_oi(), scan_dark_pool_blocks() (+5 more)

### Community 102 - "Domain Expert Reference Template"
Cohesion: 0.15
Nodes (12): [Anti-Patterns], [Docs Index], Domain Expert Reference Template, [Domain Rules + Patterns], [Domain-Specific Catalog / Patterns] (OPTIONAL), [Project Context], [Project Conventions], Section Checklist (+4 more)

### Community 103 - "create_technical_indicators"
Cohesion: 0.17
Nodes (19): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Aggregate sentiment scores per day using Swift-Weighted FinBERT authority…, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, DataFrame (+11 more)

### Community 104 - "3. Equipment Mapping (Draw.io Native Shapes)"
Cohesion: 0.15
Nodes (12): 1. ISA 5.1 Instrument Tagging Conventions, 2. Line Styles, 3. Equipment Mapping (Draw.io Native Shapes), 4. Diagram Layout Strategy (PFDs), Common Tag Identifiers, Heat Exchangers, Instrument Shapes (`mxgraph.pid2inst.*`), Mining & Minerals Processing (+4 more)

### Community 105 - "smart_trader_engine.py"
Cohesion: 0.21
Nodes (16): calculate_smart_money_zones(), calculate_structural_trailing_stop(), evaluate_multi_timeframe_confluence(), find_swing_pivots(), Any, DataFrame, Institutional Smart Money Market Structure & Price-Action Engine for Sentilyze.…, Ratchets the Stop-Loss up structurally behind higher swing lows. Rules: 1.… (+8 more)

### Community 106 - "calculate_insider_conviction_score"
Cohesion: 0.22
Nodes (15): calculate_insider_conviction_score(), fetch_insider_transactions(), Any, Smart-Money Executive & Institutional Insider Radar for Sentilyze.…, Computes the Quantitative Insider Conviction Index (0 to 100 Score) and detects…, Screens a universe of tickers and returns the highest-ranking insider buying…, Fetches recent SEC Form 4 insider transactions for a specific ticker. Includes…, scan_universe_insider_catalysts() (+7 more)

### Community 107 - "test_pyramiding_cppi_alpha158.py"
Cohesion: 0.12
Nodes (20): compute_alpha158_top25(), get_alpha158_feature_names(), DataFrame, Microsoft Qlib Alpha158 Top 25 Orthogonal Alpha Factors. Down-selected from the…, Computes the Top 25 Orthogonal Alpha158 factors from standard daily OHLCV…, Returns the names of all 25 orthogonal Alpha158 factors., calculate_turtle_pyramid_plan(), evaluate_pyramiding_step() (+12 more)

### Community 108 - "Agent Browser — Live Web Automation Skill"
Cohesion: 0.50
Nodes (3): 1. Core Principles, 2. Use Cases in Sentilyze, Agent Browser — Live Web Automation Skill

### Community 109 - "Agent Memory — Persistent Multi-Session Memory Skill"
Cohesion: 0.50
Nodes (3): 1. Storage Architecture, 2. Integration Pattern, Agent Memory — Persistent Multi-Session Memory Skill

### Community 113 - "macro_liquidity.py"
Cohesion: 0.19
Nodes (12): Generation 3: Multi-Horizon AI Courtroom & Regime-Leveraged Quant DAG Engine.…, calculate_macro_liquidity_metrics(), fetch_fred_series(), _get_fred_api_key(), Real-Time Macro Liquidity & Treasury Yield Curve Radar for Sentilyze. Analyzes…, Retrieves the FRED API key from the environment or .env file., Fetches raw observation entries from the St. Louis Fed FRED API., Computes real-time macroeconomic liquidity indicators and yield curve dynamics.… (+4 more)

### Community 114 - "aws-well-architected-reviewer.md"
Cohesion: 0.22
Nodes (8): [Anti-Patterns], [Docs Index], [Domain Rules + Patterns], [Project Context], [Project Conventions], [Sequential Execution Pairing (MANDATORY)], [Visual Styling], [XML Graph DOM Rules]

### Community 115 - "gcp-well-architected-reviewer.md"
Cohesion: 0.22
Nodes (8): [Anti-Patterns], [Docs Index], [Domain Rules + Patterns], [Project Context], [Project Conventions], [Sequential Execution Pairing (MANDATORY)], [Visual Styling], [XML Graph DOM Rules]

### Community 116 - "run_universe_training.py"
Cohesion: 0.16
Nodes (17): load_universe_from_file(), main(), prefetch_single_ticker(), prefetch_universe_data(), prevent_windows_sleep(), Any, Universal Parallel Multi-Core Model Training & High-Resolution Benchmark.…, Pre-fetches price history and news data concurrently across fast I/O threads. (+9 more)

### Community 117 - "macro_calendar.py"
Cohesion: 0.25
Nodes (10): Enum, evaluate_macro_volatility_blackout(), get_upcoming_macro_events(), MacroImpactTier, Any, datetime, Macroeconomic Calendar & Central Bank Event Engine for Sentilyze. Module:…, Returns list of upcoming macroeconomic events within lookahead window. (+2 more)

### Community 118 - "evaluate_ticker_toxicity"
Cohesion: 0.26
Nodes (11): calculate_vpin(), evaluate_ticker_toxicity(), Any, DataFrame, Pulls price history and evaluates institutional order flow toxicity for a given…, Computes Volume-Synchronized Probability of Toxicity (VPIN) from price and…, Unit tests for Volume-Synchronized Probability of Toxicity (VPIN)…, test_vpin_bounded_between_zero_and_one() (+3 more)

### Community 119 - "ws_committee.py"
Cohesion: 0.24
Nodes (9): create_candlestick_sr_chart(), DataFrame, Figure, Automated Dynamic Pivot Support & Resistance Charting Component for Streamlit.…, Constructs an institutional-grade interactive Plotly chart with automated S/R…, _get_cached_sr_chart(), _load_cached_committee_resolution(), cache_data (+1 more)

### Community 120 - "liquidity_heatmap.py"
Cohesion: 0.31
Nodes (8): compute_order_book_depth_and_clusters(), compute_volume_profile_and_poc(), Any, Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap for…, Simulates Level 2 market depth and identifies institutional buy/sell liquidity…, Computes Point of Control (POC), Value Area High (VAH), and Value Area Low…, test_compute_order_book_depth_and_clusters(), test_compute_volume_profile_and_poc()

### Community 121 - "compile_biotech_catalyst_radar"
Cohesion: 0.27
Nodes (10): compile_biotech_catalyst_radar(), fetch_clinical_trials_catalysts(), fetch_fda_regulatory_catalysts(), is_biotech_or_healthcare(), Any, BioTech & Pharma Catalyst Radar for Sentilyze.…, Queries openFDA API for recent approvals, drug labels, or regulatory safety…, Compiles complete regulatory and clinical trials intelligence for a given stock. (+2 more)

### Community 122 - "cockpit_backtest.py"
Cohesion: 0.18
Nodes (13): Cockpit 4: Multi-Decade Backtest & Performance Factsheet…, Renders verification card confirming 12 production models are freshly retrained., _render_retrained_models_badge(), Workspace 6: Walk-Forward Backtesting & Performance Tearsheet., Renders the Walk-Forward Backtesting and Strategy Tearsheet workspace., render_backtesting_workspace(), get_market_timestamp(), load_safety_benchmarks() (+5 more)

### Community 123 - "cockpit_trading.py"
Cohesion: 0.19
Nodes (13): Synthesizes broadcast audio podcast (.mp3) using Google Text-to-Speech (gTTS)…, synthesize_briefing_audio(), Cockpit 1: Live Trading & Alpha Execution…, Renders Monday Opening Intelligence, Overnight Futures Pulse, and Almgren-…, _render_monday_premarket_intel(), render_trading_cockpit(), Renders the 5-Agent Trading Committee deliberation panel, dynamic S/R charting,…, render_committee_workspace() (+5 more)

### Community 124 - "test_pillar2_alternative_data.py"
Cohesion: 0.15
Nodes (22): compute_smart_money_insider_score(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent SEC Form 4 insider transactions for a given stock., Retrieves recent Congressional STOCK Act disclosure reports for a ticker., Synthesizes SEC Form 4 and Congressional activity into an overall Smart Money…, track_congressional_stock_disclosures(), track_corporate_insider_filings() (+14 more)

### Community 125 - "calculate_15min_opening_range"
Cohesion: 0.31
Nodes (9): calculate_15min_opening_range(), find_low_of_day_pullback_entry(), Any, DataFrame, Calculates the 15-minute Opening Range (High, Low, Midpoint) established…, Evaluates whether a stock is in the optimal 'Low-of-Day Pullback & Volume…, Unit tests for 15-Minute Opening Volatility Shield & Low-of-Day Demand Engine., test_calculate_15min_opening_range() (+1 more)

### Community 126 - "FastFinBERTEngine"
Cohesion: 0.27
Nodes (5): FastFinBERTEngine, Any, Vectorized micro-batch inference for multiple headlines., Accelerated FinBERT inference engine delivering 2.5x - 4x speedups., Runs accelerated sentiment scoring on a single headline. Includes Companion…

### Community 127 - "ultra_quant_engine.py"
Cohesion: 0.18
Nodes (14): compute_credit_and_liquidity_radar(), Any, Computes High Yield vs Investment Grade Credit Spread (HYG/LQD) leading…, build_orderflow_candlestick_chart(), calculate_volume_profile(), detect_fair_value_gaps(), Any, DataFrame (+6 more)

### Community 128 - "MasterTradingLoop"
Cohesion: 0.11
Nodes (16): get_master_trading_loop(), MasterTradingLoop, Any, Executes Opening 15-Minute Whipsaw Shield: 1. Checks whether current market…, Executes Post-Market Trade Autopsy & Agent Weight Recalibration: 1. Runs…, Runs the complete 4-phase trading lifecycle in sequence: Phase 1 (Pre-Market)…, Returns the current operational status of the master loop for the dashboard., Returns the singleton instance of the MasterTradingLoop. (+8 more)

### Community 129 - "ws_deep_quant.py"
Cohesion: 0.18
Nodes (17): analyze_debt_maturity_wall(), calculate_beneish_m_score(), Any, DataFrame, Beneish M-Score Forensic Analyzer & Debt Maturity Wall Radar for Sentilyze.…, Evaluates corporate interest coverage and debt maturity wall runway., Computes the 8-Ratio Beneish M-Score from 2-year comparative SEC financial…, _get_cached_forensic() (+9 more)

### Community 130 - "generate_comprehensive_factsheet"
Cohesion: 0.31
Nodes (7): generate_comprehensive_factsheet(), Any, Series, Institutional Risk & Alpha Performance Factsheet Engine for Sentilyze.…, Computes over 30 institutional hedge-fund risk, performance, and drawdown…, test_generate_comprehensive_factsheet_custom_series(), test_generate_comprehensive_factsheet_default()

### Community 131 - "TriBrainGatingLayer"
Cohesion: 0.40
Nodes (3): Tensor, Neural Softmax Gating layer to dynamically allocate weights among the 3 models.…, TriBrainGatingLayer

### Community 132 - "calculate_cppi_allocation"
Cohesion: 0.23
Nodes (8): calculate_cppi_allocation(), Any, ndarray, Paper 20: Constant Proportion Portfolio Insurance (CPPI). Source: Black & Jones…, CPPI allocation: Exposure = M * (Portfolio - Floor). Args: portfolio_value:…, Run a full CPPI backtest over a return series. Args: returns: Array of daily…, run_cppi_backtest(), TestCPPI

### Community 133 - "analyze_earnings_surprises"
Cohesion: 0.36
Nodes (7): analyze_earnings_surprises(), Any, Retrieves and computes recent earnings surprise metrics, SUE score, and PEAD…, Unit tests for Post-Earnings Announcement Drift (PEAD) & SUE Earnings Surprise…, test_earnings_surprises_bearish_miss(), test_earnings_surprises_bullish_beat(), test_earnings_surprises_fallback_on_empty()

### Community 134 - "test_zero_giveback_profit_lock.py"
Cohesion: 0.20
Nodes (9): Unit tests for Institutional Zero-Giveback Profit Protection, 80% Peak Gain…, Verify that a gain of +0.50% immediately locks stop-loss to Breakeven + buffer., Verify that a gain of +1.00% locks stop-loss to at least +0.50% net profit., Verify that any stop loss wider than -2.50% is elevated to the capital shield…, Verify that PaperBroker saves atomically, maintains a backup, and guards…, test_atomic_broker_save_and_backup(), test_capital_shield_stop_floor_enforcement(), test_micro_breakeven_lock_at_0_5_pct() (+1 more)

### Community 135 - "futures_sentinel.py"
Cohesion: 0.29
Nodes (9): evaluate_overnight_futures_pulse(), fetch_futures_quote(), Any, Sentilyze Overnight Futures & Macro Sentinel Engine. Monitors US equity index…, Fetches real-time quote for a futures symbol with yfinance history and proxy…, Evaluates overnight futures sentiment, computes implied market gap, and…, Unit Tests for Overnight Futures & Macro Sentinel Engine. STRICT PORTFOLIO…, test_evaluate_futures_bearish_gap() (+1 more)

### Community 136 - "FastNeuralEventClassifier"
Cohesion: 0.29
Nodes (4): FastNeuralEventClassifier, Tensor, Builds a lightweight internal token dictionary for financial catalysts., Lightweight 2-layer Neural Event & Emotion Head for sub-millisecond…

### Community 137 - "apply_high_watermark_profit_lock"
Cohesion: 0.32
Nodes (7): apply_high_watermark_profit_lock(), Guarantees that once a trade reaches peak profit, the bot NEVER gives back >…, Unit tests for High-Watermark Peak Profit Ratchet (75% Lock Floor)., test_high_watermark_does_not_lower_sl_on_pullback(), test_high_watermark_locks_75pct_of_peak_gain(), Verify that a peak surge to $110 (+10%) locks at least $108 (+8.0%) into the…, test_high_watermark_80_pct_retention()

### Community 138 - "audio_briefing.py"
Cohesion: 0.47
Nodes (5): generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses Microsoft Edge Neural TTS…, synthesize_morning_audio()

### Community 139 - "print_formatted_trade_report"
Cohesion: 0.23
Nodes (11): get_trade_telemetry(), print_formatted_trade_report(), Any, Instant Trade Telemetry & Results Engine. Provides direct, zero-overhead access…, Fetches real-time trade execution summary, partitioning by today, yesterday,…, Prints a clean, concise institutional trade report to the terminal., Unit tests for Instant Trade Telemetry & Results Engine., Verify formatted trade report prints without raising exceptions. (+3 more)

### Community 140 - "test_acpm_trainer.py"
Cohesion: 0.28
Nodes (11): find_optimal_d(), fractional_differentiation_ffd(), get_weights_ffd(), ndarray, Series, Fixed-Width Window Fractional Differentiation (FFD) for Financial Time Series.…, Generate weights for Fixed-Width Window Fractional Differentiation. w_0 = 1 w_k…, Applies Fixed-Width Window Fractional Differentiation to a price series. (+3 more)

### Community 141 - "EventClassifierModel"
Cohesion: 0.24
Nodes (8): EventClassifierModel, Any, Translates raw social slang, emojis, and cashtags into standardized financial…, Classifies a single headline or social post with sub-millisecond speed.…, Batch-processes multiple headlines with vectorized execution., Production-grade Financial Event & Emotion Companion Model. Provides sub-…, test_event_classifier_fraud_and_dilution(), test_event_classifier_model()

### Community 142 - "generate_pipeline_graph_data"
Cohesion: 0.27
Nodes (9): generate_pipeline_graph_data(), Any, Interactive Multi-Agent & Pipeline Architecture Canvas Component for Streamlit.…, Renders the interactive Vis.js animated node network canvas inside Streamlit., Constructs the nodes and edges representation for the pipeline graph canvas., render_pipeline_topology_canvas(), Tests for Interactive Multi-Agent & Pipeline Architecture Canvas…, test_generate_pipeline_graph_data_defaults() (+1 more)

### Community 143 - "autonomous_trader.py"
Cohesion: 0.06
Nodes (50): execute_committee_order(), Executes a committee-approved buy order into the virtual paper broker ledger., AutonomousTradingEngine, check_daily_loss_circuit_breaker(), ensure_background_daemon_thread_running(), get_daemon_status(), is_kill_switch_active(), load_universe_tickers() (+42 more)

### Community 144 - "calculate_portfolio_diversity_grade"
Cohesion: 0.39
Nodes (7): calculate_portfolio_diversity_grade(), Any, DataFrame, test_custom_returns_correlated(), test_custom_returns_diverse(), test_empty_portfolio(), test_single_asset_portfolio()

### Community 146 - "check_correlation_shield"
Cohesion: 0.25
Nodes (10): calculate_portfolio_correlation_matrix(), check_correlation_shield(), Any, DataFrame, Computes the 21-day rolling pairwise return correlation matrix across tickers., Audits a candidate buy against currently held positions. Returns whether the…, Tests for Portfolio Correlation Matrix Shield (src/correlation_shield.py).…, test_calculate_portfolio_correlation_matrix() (+2 more)

### Community 147 - ".get_closed_trades_df"
Cohesion: 0.29
Nodes (4): DataFrame, Returns a DataFrame of current open holdings with Scale-Out status., Returns a DataFrame of trade history with full company names., Returns equity history as a DatetimeIndex DataFrame.

### Community 148 - "generate_institutional_pdf_tearsheet"
Cohesion: 0.31
Nodes (7): generate_institutional_pdf_tearsheet(), Institutional Quantitative Tearsheet & Factsheet Generator for Sentilyze.…, Generates a publication-grade institutional quantitative tearsheet PDF. Returns…, Verify that PDF file is saved correctly to a specified path., Verify that PDF generation creates valid, non-empty binary content., test_generate_institutional_pdf_tearsheet_bytes(), test_generate_institutional_pdf_tearsheet_file()

### Community 149 - "neutralize_features"
Cohesion: 0.22
Nodes (9): neutralize_features(), neutralize_predictions(), DataFrame, ndarray, Series, Feature & Factor Neutralization for Quantitative Machine Learning.…, Neutralizes target feature columns with respect to factor columns (e.g. SPY…, Removes linear factor exposure from raw prediction scores. (+1 more)

### Community 150 - ".train_ticker"
Cohesion: 0.22
Nodes (7): ACPMTrainer, Any, DataFrame, Series, State-of-the-Art 10x Institutional Quantitative Training Engine., Executes end-to-end ACPM training for a target equity., test_acpm_trainer_end_to_end()

### Community 151 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (16): Any, DataFrame, ndarray, Series, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.…, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used. (+8 more)

### Community 152 - "simulate_market_opening"
Cohesion: 0.40
Nodes (5): Any, Simulates Monday 9:30 AM EDT market-open execution across existing positions…, simulate_market_opening(), Unit Tests for Market-Open Order Execution Simulator. STRICT PORTFOLIO…, test_market_open_simulation_execution()

### Community 153 - "advanced_quant_experiments.py"
Cohesion: 0.22
Nodes (10): compute_performance_metrics(), Any, DataFrame, Series, Empirical Quant Experimentation & Multi-Asset Ablation Benchmark Suite.…, Executes empirical ablation benchmark across the full asset universe., Simulates walk-forward strategy execution with or without advanced quant…, Computes key quant performance metrics. (+2 more)

### Community 154 - "RedditCommentMiner"
Cohesion: 0.32
Nodes (5): Any, Executes complete Google web mining across cashflow, dark pool relays, and…, Executes Google Web Mining across Reddit posts, comments, and specialized dark…, Mines Reddit posts and comments indexed via Google News RSS without 403 blocks., RedditCommentMiner

### Community 155 - "calculate_doubling_progress"
Cohesion: 0.31
Nodes (8): calculate_doubling_progress(), compute_compound_position_size(), Any, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Computes dynamic equity-scaled position sizing so trade sizes grow…, Unit tests for Max Compound Acceleration Engine., test_calculate_doubling_progress(), test_compute_compound_position_size()

### Community 156 - "backtest_autopsy.py"
Cohesion: 0.33
Nodes (6): compute_walk_forward_efficiency(), Any, Autonomous Backtest Autopsy, WFO Efficiency & Tear-Sheet Factsheet Engine for…, Computes Pardo (2008) Walk-Forward Efficiency (WFE) Ratio: WFE = OOS_Sharpe /…, Mines executed trade history (strictly read-only) to produce an autopsy…, run_autonomous_trade_autopsy()

### Community 157 - "mock_return_series"
Cohesion: 0.40
Nodes (5): mock_return_series(), mock_trade_dataframe(), fixture, Generate 100 days of strategy and benchmark returns., Generate sample closed trade executions.

### Community 158 - "DynamicSharpeMetaEnsemble"
Cohesion: 0.25
Nodes (5): DynamicSharpeMetaEnsemble, Dynamically re-weights multi-agent specialist and sub-model signals based on…, Record historical daily returns for a submodel., Calculate annualized Sharpe ratio for each submodel over lookback window., Compute Softmax allocation weights over rolling Sharpe ratios.

### Community 159 - "run_unified_institutional_pipeline"
Cohesion: 0.21
Nodes (11): Any, Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), analyze_sec_filing_diff(), compute_text_similarity_and_diff(), Any, Computes lexical and semantic diff metrics between consecutive filings., Retrieves and compares the most recent and prior SEC filings for a company.… (+3 more)

### Community 160 - "deduplicate_news_stream"
Cohesion: 0.28
Nodes (8): calculate_jaccard_similarity(), deduplicate_news_stream(), normalize_headline_tokens(), DataFrame, Institutional News Filter, Semantic Deduplicator & Catalyst Classifier for…, Fast Jaccard-based semantic deduplication that eliminates syndicated press…, Extracts clean alphanumeric tokens for fast semantic similarity., Calculates Jaccard token overlap between two headlines.

### Community 161 - "components.py"
Cohesion: 0.18
Nodes (10): Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Wraps HTML content inside an institutional frosted glass container., render_glass_card(), load_cached_tournament_results(), cache_data, Workspace 25: Autonomous Quant Alpha DAG Engine (Gen-3 Champion). Institutional…, Loads pre-computed 3-generation tournament results if available., Renders the comprehensive Quant Alpha DAG Engine workspace. (+2 more)

### Community 162 - "render_multi_agent_war_room"
Cohesion: 0.28
Nodes (7): Any, Interactive Multi-Agent War Room Visualizer Component for Streamlit. Functions:…, Renders the complete 5-Agent War Room Council deliberation chamber., render_multi_agent_war_room(), Real-Time Audio Trade Squawk Component for Streamlit. Functions: - Uses…, Renders an HTML5 Web Speech API audio squawk generator inside Streamlit., render_audio_squawk_button()

### Community 163 - "ConformalCalibrator"
Cohesion: 0.17
Nodes (8): ConformalCalibrator, ndarray, Conformal Probability Calibration & Quantile Uncertainty Layer. Maps raw tree…, Non-parametric monotonic Isotonic Conformal Calibrator., Calibrates on an independent holdout calibration split., Returns empirical calibrated probabilities., Checks if a trade signal has statistically significant empirical edge beyond…, test_conformal_calibrator()

### Community 164 - "ws_portfolio_diversity.py"
Cohesion: 0.33
Nodes (7): Cockpit 3: Portfolio & Risk Optimization…, render_portfolio_cockpit(), render_macro_liquidity_workspace(), _get_cached_diversity_grade(), cache_data, Workspace: Portfolio Diversity & Correlation Health Grader. Institutional…, render_portfolio_diversity_workspace()

### Community 165 - "get_sector_for_ticker"
Cohesion: 0.15
Nodes (17): Real-Time Market Anomaly Screener Component for Streamlit. Functions: - Renders…, Renders the Real-Time Market Anomaly Screener UI., render_live_screener_section(), get_sector_for_ticker(), Returns the sector cluster for a given ticker or General default., fetch_universe_live_quotes(), Fetches real-time quotes across universe with fast batching (sub-2s)., evaluate_single_asset_screener() (+9 more)

### Community 166 - "cross_asset_pooling.py"
Cohesion: 0.29
Nodes (6): build_pooled_sector_dataset(), DataFrame, Series, Cross-Asset Sector-Pooled Multi-Task Dataset Engine. Maps the 538-stock…, Pools cross-sectional DataFrames across sector peers and creates a unified…, test_cross_asset_pooling()

### Community 167 - "_get_browser_session"
Cohesion: 0.33
Nodes (5): _get_browser_session(), Session, Universal Adaptive Web Scraper supporting Scrapling, Trafilatura, and…, Creates a configured requests session with realistic browser headers., UniversalWebScraper

### Community 168 - ".optimize_allocation"
Cohesion: 0.40
Nodes (4): Any, DataFrame, Series, Solves the friction-aware convex optimization problem in polynomial time. Args:…

### Community 171 - "test_ultra_quant_engine.py"
Cohesion: 0.12
Nodes (16): fixture, Unit tests for the UltraQuantEngine: Ultra-Low-Latency Unified Execution Core.…, Verify CUSUM detector triggers alarm on sudden downward price breakdown., Generates 60 bars of synthetic OHLCV data., Verify Gaussian HMM forward update executes in microseconds., Verify singleton pattern returns identical instance., Verify Volume Profile PoC, VAH, and VAL execution and latency in microseconds., Verify full multi-model quantum evaluation pipeline and nanosecond telemetry. (+8 more)

### Community 172 - ".split"
Cohesion: 0.50
Nodes (3): DataFrame, ndarray, Series

### Community 173 - "temp_data_dir"
Cohesion: 0.67
Nodes (3): fixture, Fixture to set a temporary data directory for tests., temp_data_dir()

## Knowledge Gaps
- **224 isolated node(s):** `graphify`, `1. Core Principles`, `2. Use Cases in Sentilyze`, `1. Storage Architecture`, `2. Integration Pattern` (+219 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `utils.py` to `deep_learning_model.py`, `get_price_history`, `quant_engine.py`, `ws_deep_quant.py`, `run_backtest`, `generate_comprehensive_factsheet`, `test_statistical_arbitrage.py`, `drl_policy_agent.py`, `futures_sentinel.py`, `TradingEnvironment`, `audio_briefing.py`, `CloudDataLake`, `real_market_analyzer.py`, `analyze_supply_chain_spillover`, `calculate_hrp_weights`, `autonomous_trader.py`, `daily_scanner.py`, `run_temporal_fusion_forecast`, `ws_live_prediction.py`, `generate_institutional_pdf_tearsheet`, `get_sentiment`, `test_omnichannel_mobile.py`, `SuperEnsembleClassifier`, `advanced_quant_experiments.py`, `cli.py`, `social_sentiment.py`, `AlpacaBrokerBridge`, `backtest_autopsy.py`, `realtime_tracker.py`, `get_us_market_session`, `ws_alternative_data.py`, `deduplicate_news_stream`, `test_all_14_papers.py`, `apply_triple_barrier_labeling`, `triple_convex_engine.py`, `get_sector_for_ticker`, `VolScaledMomentumEngine`, `universal_web_scraper.py`, `price_scout.py`, `webhook_dispatcher.py`, `calculate_time_decayed_sentiment`, `test_stock_image_provider.py`, `run_monte_carlo_stress_test`, `black_swan_simulator.py`, `compute_gamma_exposure_profile`, `TradePostMortemLearner`, `test_reddit_premarket_station.py`, `fetch_financial_statements`, `test_tri_model_pipeline.py`, `correlation_matrix.py`, `preprocess_data`, `options_surface.py`, `test_institutional_weekend_suite.py`, `compute_dark_pool_sentiment`, `smart_trader_engine.py`, `calculate_insider_conviction_score`, `macro_liquidity.py`, `run_universe_training.py`, `liquidity_heatmap.py`, `compile_biotech_catalyst_radar`, `test_pillar2_alternative_data.py`, `ultra_quant_engine.py`?**
  _High betweenness centrality (0.212) - this node is a cross-community bridge._
- **Why does `get_price_history()` connect `get_price_history` to `run_backtest`, `test_statistical_arbitrage.py`, `utils.py`, `drl_policy_agent.py`, `autonomous_trader.py`, `daily_scanner.py`, `calculate_portfolio_diversity_grade`, `check_correlation_shield`, `UltraQuantEngine`, `advanced_quant_experiments.py`, `realtime_tracker.py`, `test_all_14_papers.py`, `ws_portfolio_diversity.py`, `get_sector_for_ticker`, `test_data_ingestion.py`, `price_scout.py`, `generate_morning_briefing_text`, `QuantAlphaDAGEngine`, `test_reddit_premarket_station.py`, `fetch_financial_statements`, `correlation_matrix.py`, `preprocess_data`, `AdversarialRedTeamAgent`, `get_ultra_quant_engine`, `macro_liquidity.py`, `run_universe_training.py`, `evaluate_ticker_toxicity`, `ws_committee.py`, `ultra_quant_engine.py`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `PaperBroker` connect `PaperBroker` to `MasterTradingLoop`, `get_price_history`, `test_statistical_arbitrage.py`, `test_zero_giveback_profit_lock.py`, `drl_policy_agent.py`, `render_workspace_header`, `print_formatted_trade_report`, `autonomous_trader.py`, `daily_scanner.py`, `.get_closed_trades_df`, `cli.py`, `AlpacaBrokerBridge`, `realtime_tracker.py`, `ws_portfolio_diversity.py`, `.execute_daily_signals`, `generate_morning_briefing_text`, `ws_portfolio.py`, `AgentMemoryStore`, `preprocess_data`, `test_agent_memory.py`, `AICopilotEngine`, `calculate_insider_conviction_score`, `test_pyramiding_cppi_alpha158.py`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `1. Core Principles`, `2. Use Cases in Sentilyze` to the rest of the system?**
  _224 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `deep_learning_model.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13043478260869565 - nodes in this community are weakly interconnected._
- **Should `get_price_history` be split into smaller, more focused modules?**
  _Cohesion score 0.05583972719522592 - nodes in this community are weakly interconnected._