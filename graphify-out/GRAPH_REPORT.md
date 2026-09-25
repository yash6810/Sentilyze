# Graph Report - Sentilyze  (2026-09-25)

## Corpus Check
- 3401 files · ~2,656,008 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2929 nodes · 5772 edges · 174 communities (161 shown, 13 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 61 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f67b25ab`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- deep_learning_model.py
- get_price_history
- test_options_flow.py
- OpeningRangeBreakout
- run_backtest
- test_statistical_arbitrage.py
- test_all_14_papers.py
- drl_policy_agent.py
- FactorAttributionEngine
- TradingEnvironment
- test_preprocessing.py
- CloudDataLake
- app.py
- analyze_supply_chain_spillover
- calculate_hrp_weights
- run_temporal_fusion_forecast
- AutonomousTradingEngine
- train_meta_ensemble
- OnlineNewtonStepOptimizer
- ws_live_prediction.py
- draw.io XML Style Reference
- test_pyramiding_cppi_alpha158.py
- test_omnichannel_mobile.py
- pfd-engineering-expert.md
- ultra_quant_engine.py
- compute_lead_lag_matrix
- test_sentiment_analysis.py
- social_sentiment.py
- AlpacaBrokerBridge
- agent_committee.py
- .run_full_daily_cycle
- get_us_market_session
- ipo_radar.py
- 3. Cross-Functional Flowcharts (Actor × Phase Grid)
- utils.py
- Edge Routing Decision Guide
- StrategyGenome
- triple_convex_engine.py
- Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform
- VolScaledMomentumEngine
- [Validator Rules Reference & Troubleshooting]
- TickerSentinelSwarm
- RegimeMixtureOfExperts
- .split
- ui/__init__.py
- Sentilyze Community Code of Conduct
- universal_web_scraper.py
- DLinearTCNModel
- .execute_daily_signals
- price_scout.py
- test_cross_disciplinary_alphas.py
- Contributing to Sentilyze
- render_committee_workspace
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
- stress_tester.py
- [Validator Rules Reference & Troubleshooting]
- grossman_zhou_allocation
- ADWINDetector
- test_papers_15_24.py
- MetaRegimeAllocator
- TradePostMortemLearner
- ws_options_surface.py
- Any
- test_reddit_premarket_station.py
- InstitutionalDataGateway
- risk_constrained_kelly_allocation
- fetch_financial_statements
- test_tri_model_pipeline.py
- test_agent_committee.py
- test_duckdb_engine.py
- apply_triple_barrier_labeling
- compute_cross_asset_correlation
- AgentMemoryStore
- preprocess_data
- block_external_alerts
- test_agent_memory.py
- AICopilotEngine
- calculate_multileg_payoff
- drawio/SKILL.md
- AdversarialRedTeamAgent
- liquidity_heatmap.py
- temp_data_dir
- DuckDBMarketEngine
- compute_dark_pool_sentiment
- Domain Expert Reference Template
- create_technical_indicators
- 3. Equipment Mapping (Draw.io Native Shapes)
- fractional_differentiation_ffd
- calculate_insider_conviction_score
- compound_engine.py
- Agent Browser — Live Web Automation Skill
- Agent Memory — Persistent Multi-Session Memory Skill
- Cybersecurity Skills — Institutional MLOps Security Protocol
- Editorial Diagram Design Skill
- Harness Engineering Skill
- compute_gamma_exposure_profile
- aws-well-architected-reviewer.md
- gcp-well-architected-reviewer.md
- run_universe_training.py
- macro_calendar.py
- calculate_vpin
- acpm_trainer.py
- .query
- compile_biotech_catalyst_radar
- render_workspace_header
- screener_engine.py
- test_pillar2_alternative_data.py
- get_sentiment
- FastFinBERTEngine
- orderflow_chart.py
- MasterTradingLoop
- quant_engine.py
- generate_comprehensive_factsheet
- TriBrainGatingLayer
- ws_insider_radar.py
- analyze_earnings_surprises
- main
- calculate_15min_opening_range
- FastNeuralEventClassifier
- compute_advanced_performance_ratios
- audio_briefing.py
- print_formatted_trade_report
- test_acpm_trainer.py
- EventClassifierModel
- generate_pipeline_graph_data
- test_autonomous_trader.py
- ws_alternative_data.py
- test_api.py
- check_correlation_shield
- .get_closed_trades_df
- ws_quantum_tournament.py
- .is_connected
- run_daily_market_scan
- SuperEnsembleClassifier
- test_regime_allocator.py
- run_full_quant_experiment
- DynamicSharpeMetaEnsemble
- cockpit_trading.py
- run_committee_ablation_backtest
- render_multi_agent_war_room
- ws_deep_quant.py
- analyze_sec_filing_diff
- compute_alpha158_top25
- test_ui_modules.py
- .train_ticker
- ConformalCalibrator
- cockpit_deep_quant.py
- _get_browser_session
- memory_engine
- .get_portfolio_summary
- .get_coverage_summary
- .fuse
- ._load_or_initialize
- sample_ohlcv_data
- ._save_state
- sample_portfolio_returns

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 118 edges
2. `get_price_history()` - 79 edges
3. `PaperBroker` - 75 edges
4. `fetch_live_quote()` - 46 edges
5. `get_news()` - 35 edges
6. `UltraQuantEngine` - 31 edges
7. `run_unified_institutional_pipeline()` - 30 edges
8. `convene_trading_committee()` - 30 edges
9. `AgentMemoryStore` - 29 edges
10. `preprocess_data()` - 29 edges

## Surprising Connections (you probably didn't know these)
- `test_paper6_cph_multi_agent_committee()` --calls--> `ChiefRiskOfficerAgent`  [EXTRACTED]
  tests/test_all_14_papers.py → src/agent_committee.py
- `test_convene_trading_committee()` --calls--> `convene_trading_committee()`  [EXTRACTED]
  tests/test_agent_committee.py → src/agent_committee.py
- `mock_memory()` --uses--> `AgentMemoryStore`  [INFERRED]
  tests/test_agent_memory.py → src/agent_memory.py
- `test_agent_memory_softmax_dirichlet_recalibration()` --calls--> `AgentMemoryStore`  [EXTRACTED]
  tests/test_pyramiding_cppi_alpha158.py → src/agent_memory.py
- `test_load_universe_tickers()` --calls--> `load_universe_tickers()`  [EXTRACTED]
  tests/test_autonomous_trader.py → src/autonomous_trader.py

## Import Cycles
- None detected.

## Communities (174 total, 13 thin omitted)

### Community 0 - "deep_learning_model.py"
Cohesion: 0.14
Nodes (19): create_sliding_window_tensors(), load_dlinear_model(), Any, DataFrame, High-Efficiency Deep Learning Engine: DLinear + Temporal Convolutional Network…, Converts a pandas DataFrame into sliding window sequence tensors for Deep…, Trains the DLinear-TCN model on CPU using AdamW optimizer with learning rate…, Saves model weights safely using PyTorch state_dict. (+11 more)

### Community 1 - "get_price_history"
Cohesion: 0.05
Nodes (59): _fetch_alpaca_news(), _fetch_alpaca_price_history(), _fetch_direct_yahoo_chart(), _fetch_eodhd_price_history(), _fetch_finnhub_news(), _fetch_fmp_price_history(), _fetch_google_news_rss(), _fetch_marketaux_news() (+51 more)

### Community 2 - "test_options_flow.py"
Cohesion: 0.16
Nodes (22): calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain(), Any, DataFrame, Live Options Microstructure, Gamma Exposure (GEX) & Max Pain Terminal for… (+14 more)

### Community 3 - "OpeningRangeBreakout"
Cohesion: 0.16
Nodes (12): OpeningRangeBreakout, Any, DataFrame, Paper 25: Opening Range Breakout (ORB) with Stocks-in-Play Filter. Source:…, Simulate multi-year daily ORB strategy returns., Opening Range Breakout (ORB) trading engine with Stocks-in-Play filter. Paper…, Rank universe and select top 'Stocks in Play' based on volume, ATR, and…, Evaluate ORB signal using daily OHLC + volatility approximation. (+4 more)

### Community 4 - "run_backtest"
Cohesion: 0.07
Nodes (59): _persist_attribution_results(), Any, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition(), calculate_performance_metrics(), _calculate_trade_outcomes(), create_monthly_returns_heatmap(), Any (+51 more)

### Community 5 - "test_statistical_arbitrage.py"
Cohesion: 0.17
Nodes (29): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), calibrate_avellaneda_lee_stat_arb(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any (+21 more)

### Community 6 - "test_all_14_papers.py"
Cohesion: 0.10
Nodes (24): Any, Executes empirical backtests comparing all 14 academic paper methodologies., run_all_14_papers_benchmark(), calculate_almgren_chriss_trajectory(), Any, Computes Almgren-Chriss optimal trading trajectory. x_j = 2 * sinh(0.5 * kappa…, detect_negative_cycle_arbitrage(), Any (+16 more)

### Community 7 - "drl_policy_agent.py"
Cohesion: 0.13
Nodes (20): ActorCriticPolicy, DRLTradingEnvironment, evaluate_drl_policy_action(), Any, ndarray, Tensor, Deep Reinforcement Learning (DRL) Autonomous Policy Agent for Sentilyze.…, Trains an Actor-Critic DRL policy agent on historical market returns and news… (+12 more)

### Community 8 - "FactorAttributionEngine"
Cohesion: 0.09
Nodes (26): run_comparative_simulation(), FactorAttributionEngine, generate_institutional_factsheet_for_ticker(), InstitutionalFactsheet, Any, DataFrame, Series, Institutional Factor Attribution & Performance Factsheet Engine Computes… (+18 more)

### Community 9 - "TradingEnvironment"
Cohesion: 0.14
Nodes (16): optimize_rl_position_allocation(), PPOPolicyAgent, Any, ndarray, ⚠️ EXPERIMENTAL / RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM PRODUCTION…, Computes mean action (leverage) between 0.0 and 2.0., Estimates state value., Trains Actor-Critic parameters across historical episodes. (+8 more)

### Community 10 - "test_preprocessing.py"
Cohesion: 0.27
Nodes (9): clean_headline_data(), Cleans a headline CSV file by removing rows with invalid stock tickers. Caches…, fixture, temp_data_dirs(), test_clean_headline_data_invalid_tickers_no_cache(), test_clean_headline_data_loads_from_cache(), test_clean_headline_data_valid_tickers_no_cache(), test_load_sentiment_analyzer_caches() (+1 more)

### Community 11 - "CloudDataLake"
Cohesion: 0.14
Nodes (14): CloudDataLake, Any, Supabase / PostgreSQL Cloud Data Lake Connector., Validates or generates cloud database schema., Syncs local trade executions to the cloud database., Publishes real-time portfolio snapshot to cloud WebSockets channel., generate_twap_order_schedule(), generate_vwap_order_schedule() (+6 more)

### Community 12 - "app.py"
Cohesion: 0.19
Nodes (13): get_model_universe_indicator(), load_universe_tickers(), main(), cache_data, Sentilyze - Institutional Algorithmic Trading & MLOps Platform. High-Speed…, Loads active S&P 100 universe tickers., Computes universe model training coverage and mean accuracy using compiled…, get_market_status() (+5 more)

### Community 13 - "analyze_supply_chain_spillover"
Cohesion: 0.14
Nodes (15): analyze_supply_chain_spillover(), Any, ndarray, Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for…, Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)., Executes a Graph Convolutional Network (GCN) layer: H_new = ReLU(A_hat * H * W)…, Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab…, High-level entry point to run GNN supply chain shock propagation. (+7 more)

### Community 14 - "calculate_hrp_weights"
Cohesion: 0.12
Nodes (29): build_unified_portfolio(), calculate_hrp_weights(), calculate_risk_parity_weights(), get_cluster_var(), get_quasi_diag(), get_rec_bisection(), load_all_ticker_portfolios(), Any (+21 more)

### Community 15 - "run_temporal_fusion_forecast"
Cohesion: 0.14
Nodes (15): Any, DataFrame, ndarray, Temporal Fusion Transformer (TFT) & Multi-Horizon Self-Attention Engine for…, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with… (+7 more)

### Community 16 - "AutonomousTradingEngine"
Cohesion: 0.12
Nodes (13): AutonomousTradingEngine, Autonomous Execution Engine that integrates Live News Ingestion, 4-Agent…, Runs the Autonomous Trading Engine continuously on an interval., Dispatches an institutional execution alert to Discord Webhook if configured., Executes one full autonomous decision and execution cycle with: - Task 6:…, Core cycle execution body., Executes the Self-Improving Feedback Loop: 1. Analyzes trade autopsies on…, run_autonomous_daemon() (+5 more)

### Community 17 - "train_meta_ensemble"
Cohesion: 0.18
Nodes (13): MetaEnsembleClassifier, DataFrame, ndarray, Series, Generates binary class prediction (0 = Hold/Sell, 1 = Buy) using soft-voting…, Instantiates and fits the Meta-Ensemble classifier., Multi-Model Meta-Ensemble stacking XGBoost, Random Forest, and Calibrated…, Trains all component models on the training dataset. (+5 more)

### Community 18 - "OnlineNewtonStepOptimizer"
Cohesion: 0.16
Nodes (11): OnlineNewtonStepOptimizer, DataFrame, ndarray, Online Newton Step (ONS) Portfolio Engine (Agarwal, Hazan, Kale). A polynomial-…, Polynomial-Time ONS Portfolio Engine (Hazan et al.)., Processes price relatives (r_t = Close_t / Close_{t-1}) and updates weights in…, Fast O(d log d) Euclidean projection onto probability simplex., Runs ONS sequence through time and outputs daily allocations and portfolio… (+3 more)

### Community 19 - "ws_live_prediction.py"
Cohesion: 0.06
Nodes (53): detect_classical_chart_patterns(), generate_ai_chart_explanation(), match_historical_chart_twins(), normalize_waveform(), Any, DataFrame, ndarray, AI Chart Pattern Recognition, Geometric Wave Learning & Visual Understanding… (+45 more)

### Community 20 - "draw.io XML Style Reference"
Cohesion: 0.08
Nodes (23): Color Properties, Common Style Properties, Container Types, Critical Edge Rules, draw.io XML Style Reference, Edge Attributes, Edge Styles, Escaping Rules (XML attribute context) (+15 more)

### Community 21 - "test_pyramiding_cppi_alpha158.py"
Cohesion: 0.24
Nodes (10): evaluate_pyramiding_step(), Evaluates whether an active open position should add a pyramid unit and…, get_cppi_cushion_multiplier(), Computes CPPI (Constant Proportion Portfolio Insurance) scaling factor for…, Unit Tests for: 1. CPPI Dynamic Cushion Scaling (src/cppi_insurance.py) 2.…, test_agent_memory_softmax_dirichlet_recalibration(), test_cppi_cushion_multiplier_at_floor(), test_cppi_cushion_multiplier_full_capacity() (+2 more)

### Community 22 - "test_omnichannel_mobile.py"
Cohesion: 0.16
Nodes (15): answer_financial_query(), Any, Parses natural language questions and routes them to quantitative engines.…, generate_smartwatch_glance_payload(), Any, Generates structured complication JSON for Apple Watch (watchOS) and Wear OS., format_whatsapp_trade_alert(), Any (+7 more)

### Community 23 - "pfd-engineering-expert.md"
Cohesion: 0.10
Nodes (20): 1. Distillation Column (`distillation_column`), 1. `PHASE_PORT_VIOLATION`, 2. `DEAD_END_STREAM`, 2. Pump (`pump`), 3. Compressor (`compressor`), 3. `OPPOSING_FLOW`, 4. `GRAVITY_VIOLATION`, 4. Separator (`separator`) (+12 more)

### Community 24 - "ultra_quant_engine.py"
Cohesion: 0.07
Nodes (30): Any, DataFrame, Sentilyze Ultra-Low-Latency Quantitative Execution Core.…, Evaluates SEC Form 4 insider purchases and executive cluster buys., Mines SEC Form 8-K filings for material corporate events., Calculates retail mention velocity and sentiment from Reddit & StockTwits., Calculates optimal execution trajectory and market impact., Calculates optimal risky allocation under dynamic drawdown constraint. (+22 more)

### Community 25 - "compute_lead_lag_matrix"
Cohesion: 0.20
Nodes (14): compute_lead_lag_matrix(), _granger_f_test(), Any, DataFrame, ndarray, Series, rank_market_price_leaders(), Lead-Lag Granger Causality & Supply Chain Price Discovery Engine for Sentilyze.… (+6 more)

### Community 26 - "test_sentiment_analysis.py"
Cohesion: 0.14
Nodes (13): fixture, Fixture to set a temporary data directory for tests., Test that get_sentiment handles DataFrames with missing text columns gracefully., Test that get_sentiment bypasses the cache when no ticker is provided., Test that get_sentiment correctly analyzes sentiment and adds the right columns., Test that get_sentiment correctly parses full 3-class probability distributions…, Test that get_sentiment loads data from the cache if it's not stale., temp_data_dir() (+5 more)

### Community 27 - "social_sentiment.py"
Cohesion: 0.23
Nodes (13): calculate_social_buzz_metrics(), fetch_social_sentiment_tracker(), Any, Social Sentiment Velocity & Retail Multi-Platform Scraper for Sentilyze. Pillar…, Scrapes real-time streaming retail sentiment from Stocktwits public symbol…, Scrapes tech community discussions on AI catalysts (OpenAI, Anthropic, Nvidia)…, Computes retail sentiment velocity and flow conviction metrics., High-level entry point to retrieve calibrated real-time social buzz metrics for… (+5 more)

### Community 28 - "AlpacaBrokerBridge"
Cohesion: 0.29
Nodes (11): AlpacaBrokerBridge, Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.…, patch, Real integration test: connects to Alpaca's actual paper API endpoint using…, test_alpaca_broker_close_all_positions(), test_alpaca_broker_close_position(), test_alpaca_broker_get_account_summary(), test_alpaca_broker_initialization() (+3 more)

### Community 29 - "agent_committee.py"
Cohesion: 0.06
Nodes (48): convene_trading_committee(), execute_committee_order(), Autonomous Multi-Agent Trading Committee & Deliberation Engine for Sentilyze.…, Orchestrates a full round-table deliberation of the 5-Agent Trading Committee…, Executes a committee-approved buy order into the virtual paper broker ledger., AI Trade Copilot & Conversational Analyst for Sentilyze. Provides natural…, is_kill_switch_active(), load_universe_tickers() (+40 more)

### Community 30 - ".run_full_daily_cycle"
Cohesion: 0.24
Nodes (6): Any, Executes Opening 15-Minute Whipsaw Shield: 1. Checks whether current market…, Executes Post-Market Trade Autopsy & Agent Weight Recalibration: 1. Runs…, Runs the complete 4-phase trading lifecycle in sequence: Phase 1 (Pre-Market)…, Returns the current operational status of the master loop for the dashboard., Executes Pre-Market Intelligence: 1. Macro yield curve & macro regime via…

### Community 31 - "get_us_market_session"
Cohesion: 0.19
Nodes (12): check_market_hours_preflight(), get_current_ny_time(), get_us_market_session(), Any, datetime, Unified US Stock Market (NYSE / NASDAQ) Session & Calendar Engine for…, Pre-flight sanity check for automated workflows. Returns True if execution…, Returns the current precise timestamp in America/New_York (Eastern Time). (+4 more)

### Community 32 - "ipo_radar.py"
Cohesion: 0.21
Nodes (12): auto_register_ipo_ticker(), fetch_pre_ipo_radar_summary(), fetch_sec_edgar_ipo_filings(), Any, IPO & Pre-IPO Intelligence Radar for Sentilyze. Pillar 9 Alternative Asset…, Fetches real-time SEC Form S-1 / S-1/A IPO registration statements from SEC…, Appends a newly public IPO ticker to stocks.txt to initiate model ingestion., High-level entry point returning the complete Pre-IPO and SEC S-1 pipeline. (+4 more)

### Community 33 - "3. Cross-Functional Flowcharts (Actor × Phase Grid)"
Cohesion: 0.10
Nodes (19): 1. Flat Swimlanes (BPMN-style Flowcharts), 2. Nested Architecture Containers, 3. Cross-Functional Flowcharts (Actor × Phase Grid), 4. When to Use What, Container Key Rules Summary, Critical Rules, Critical Rules, Decision Guide (+11 more)

### Community 34 - "utils.py"
Cohesion: 0.05
Nodes (33): Logger, main(), Batch Universe Trainer for Remaining S&P 100 Tickers., run_single(), 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, Master Academic Research Papers Empirical Benchmark Suite (All 14 Papers).…, Empirical Quant Experimentation & Multi-Asset Ablation Benchmark Suite.…, Agent Memory — Persistent Multi-Session Memory Engine for Sentilyze.… (+25 more)

### Community 35 - "Edge Routing Decision Guide"
Cohesion: 0.12
Nodes (16): Critical Rules, Decision Flowchart, Default Behavior (Built-in Router), Don't add waypoints, Don't hand-route, Edge labels, Edge Routing Decision Guide, Every edge needs an mxGeometry child (+8 more)

### Community 36 - "StrategyGenome"
Cohesion: 0.20
Nodes (15): breed_strategy_generation(), evaluate_3zone_robustness(), load_strategy_vault(), Any, Evolutionary Strategy Incubator & Robustness Lab for Sentilyze. Institutional…, Evaluates a Strategy Genome across 3 distinct zones: 1. In-Sample Train (70%)…, Runs evolutionary genetic algorithm across generations, breeding top survivors., Represents an algorithmic strategy rule DNA. (+7 more)

### Community 37 - "triple_convex_engine.py"
Cohesion: 0.14
Nodes (13): PolyTimeConvexOptimizer, Any, DataFrame, Series, Polynomial-Time Convex Portfolio Optimizer with Market Frictions (Boyd et al.)., Solves the friction-aware convex optimization problem in polynomial time. Args:…, Triple-Convex Quantum Execution Engine. Fuses the top quantitative research…, Unified High-Expectancy, Minimum-Drawdown, Sub-15ms Execution Engine. (+5 more)

### Community 38 - "Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform"
Cohesion: 0.12
Nodes (16): 1. Run in 1-Click (No Installation Required), 2. Local Setup & Installation, 3. Run the Quantitative CLI, 4. Launch the Streamlit Mission Control, 5. Run Full Test Suite (274+ Unit Tests), Active Holdings (Entered 2026-09-08), 📊 Empirical Alpha Attribution & Benchmarks, 🏛️ Grounded 5-Agent Deliberation Council (+8 more)

### Community 39 - "VolScaledMomentumEngine"
Cohesion: 0.18
Nodes (12): DataFrame, Series, Daniel & Moskowitz (JFE 2016) Volatility-Scaled Multi-Horizon Momentum Engine.…, Implements Daniel & Moskowitz (2016) Volatility-Scaled Multi-Horizon Momentum., Computes RiskMetrics EWMA realized volatility with zero lookahead bias., Calculates the composite volatility-scaled momentum position for a given ticker., VolScaledMomentumEngine, VolScaledMomentumResult (+4 more)

### Community 40 - "[Validator Rules Reference & Troubleshooting]"
Cohesion: 0.12
Nodes (16): 1. `DIRECT_WAN_TO_LAN`, 2. `ORPHAN_DEVICE`, 3. `VLAN_LEAK`, 4. `REDUNDANCY_WARNING`, 5. `MISSING_DMZ`, 6. `FLAT_NETWORK`, 7. `FIREWALL_BYPASS`, 8. `SINGLE_TRUNK` (+8 more)

### Community 41 - "TickerSentinelSwarm"
Cohesion: 0.13
Nodes (16): detect_peak_crest_exhaustion(), Any, Dedicated Micro-Agent assigned to monitor a single stock position 24/7. Tracks…, Audits live price tick, applies Zero-Giveback profit ratchets, and detects peak…, Manages the full swarm of Dedicated Ticker Sentinels across all open positions.…, Synchronizes active sentinels with current portfolio open positions without…, Audits all active sentinels concurrently. Optionally persists ratcheted SL and…, Detects if a stock has reached the crest/peak of its 15-minute momentum wave… (+8 more)

### Community 42 - "RegimeMixtureOfExperts"
Cohesion: 0.21
Nodes (9): Any, DataFrame, ndarray, Series, Regime-Conditioned Mixture of Experts (MoE) Architecture. Trains 3 specialized…, 3-Expert Gated Mixture of Experts classifier for financial regimes., Labels historical rows into 3 latent market regimes: 0 = Bull Momentum (Price >…, RegimeMixtureOfExperts (+1 more)

### Community 43 - ".split"
Cohesion: 0.50
Nodes (3): DataFrame, ndarray, Series

### Community 47 - "Sentilyze Community Code of Conduct"
Cohesion: 0.25
Nodes (7): 📜 Attribution, ⚖️ Enforcement Escalation, 🛡️ Enforcement & Responsibilities, 🌟 Our Pledge, 🎯 Our Standards, 📬 Reporting Guidelines, Sentilyze Community Code of Conduct

### Community 48 - "universal_web_scraper.py"
Cohesion: 0.11
Nodes (29): _parse_analyzer_output(), Any, Normalizes pipeline output whether given full multi-class probabilities (list…, batch_scrape_urls(), detect_stock_tickers(), extract_financial_tables_from_html(), _extract_summary_sentences(), Any (+21 more)

### Community 49 - "DLinearTCNModel"
Cohesion: 0.13
Nodes (12): DLinearTCNModel, predict_momentum_probability(), Tensor, Evaluates a single or batched sequence tensor and returns calibrated momentum…, Decomposes a time-series sequence into Trend and Seasonal/Residual components…, Dual-Branch Deep Learning Architecture: - Branch 1 (Trend): Linear projection…, SeriesDecomposition, FastAgentPipeline (+4 more)

### Community 50 - ".execute_daily_signals"
Cohesion: 0.18
Nodes (10): Any, Executes an immediate manual live/simulated BUY order from UI., Executes a 0.5N ATR Turtle Pyramiding scale-in order (Unit 2 or Unit 3).…, Executes an immediate manual live/simulated exit of an open position., Executes a 50% scale-out on an open position and moves stop to break-even., Appends trade to closed_trades and records an episodic post-mortem in…, Executes daily quantitative scan results using the Concentrated Top-2 + Scale-…, Updates total equity, unrealized PnL, and win rates. (+2 more)

### Community 51 - "price_scout.py"
Cohesion: 0.18
Nodes (12): get_latest_scout_alerts(), PriceActionScoutAgent, PriceScoutBot, Any, Real-Time Price Action & Tape-Reading Scout Subagent (Bot) for Sentilyze.…, Continuous Background Scanner Bot that scouts the 538 universe assets, detects…, Scans given tickers for real-time volume breakout candidates., Retrieves the latest price scout breakout alerts. (+4 more)

### Community 52 - "test_cross_disciplinary_alphas.py"
Cohesion: 0.08
Nodes (64): Comparative Institutional Backtest & Teardown Simulator: Strategy A: Baseline…, compute_ant_colony_venue_allocation(), compute_bayesian_dark_pool_equilibrium(), compute_carnot_profit_efficiency(), compute_circadian_seasonal_risk_scalar(), compute_conformal_safety_bands(), compute_doppler_order_flow_shift(), compute_epigenetic_factor_methylation() (+56 more)

### Community 53 - "Contributing to Sentilyze"
Cohesion: 0.15
Nodes (12): 1. Code Formatting (Black), 1. Fork & Clone, 2. Linting (Flake8), 2. Virtual Environment & Dependencies, 3. Security Auditing (Bandit), 4. Unit Test Suite (Pytest), Contributing to Sentilyze, 🏛️ Core Project Architecture & Contribution Rules (+4 more)

### Community 54 - "render_committee_workspace"
Cohesion: 0.20
Nodes (10): create_candlestick_sr_chart(), DataFrame, Figure, Automated Dynamic Pivot Support & Resistance Charting Component for Streamlit.…, Constructs an institutional-grade interactive Plotly chart with automated S/R…, _get_cached_sr_chart(), _load_cached_committee_resolution(), cache_data (+2 more)

### Community 55 - "🧪 Experimental & Simulated Research Prototypes"
Cohesion: 0.50
Nodes (3): 🧪 Experimental & Simulated Research Prototypes, 🔒 Production Isolation Guarantee, 📁 Prototype Inventory

### Community 56 - "Sentilyze — Standing Audit Protocol"
Cohesion: 0.15
Nodes (12): 10. Strict Portfolio Preservation & Realized Gain Integrity, 1. Fabricated / fake data check, 2. Ticker/input-invariant bugs, 3. Mislabeled methodology, 4. Results-file / README consistency, 5. Duplicate-output smell test, 6. Safety-critical logic sanity check, 7. Silent failure check (+4 more)

### Community 57 - "webhook_dispatcher.py"
Cohesion: 0.22
Nodes (15): _append_audit_log(), dispatch_order_webhook(), format_broker_order_payload(), generate_hmac_signature(), load_webhook_config(), Any, Automated Broker Webhooks & API Order Dispatcher for Sentilyze. Enables secure…, Dispatches order payload to the configured external webhook endpoint. (+7 more)

### Community 58 - "calculate_time_decayed_sentiment"
Cohesion: 0.19
Nodes (14): calculate_time_decayed_sentiment(), compute_exponential_decay_weights(), Any, DataFrame, datetime, ndarray, Series, Temporal Sentiment Half-Life Decay Engine. Functions: - Applies continuous… (+6 more)

### Community 59 - "erd-database-expert.md"
Cohesion: 0.13
Nodes (14): 1. `FK_WITHOUT_TARGET`, 2. `ORPHAN_TABLE`, 3. `DUPLICATE_PK`, 4. `SELF_REFERENCE_MISSING`, 5. `MISSING_JUNCTION`, 6. `CIRCULAR_FK`, [Anti-Patterns], [Docs Index] (+6 more)

### Community 60 - "generate_morning_briefing_text"
Cohesion: 0.16
Nodes (19): generate_morning_briefing_text(), get_portfolio_intelligence(), load_universe_candidates(), Any, Reads live paper portfolio state for broadcast reporting., Assembles a comprehensive, institutional Wall Street Morning Podcast and…, Synthesizes broadcast audio podcast (.mp3) using Google Text-to-Speech (gTTS)…, Loads clean ticker list from stocks.txt or falls back to core liquid universe. (+11 more)

### Community 61 - "6. CROSS-DISCIPLINARY & FRONTIER QUANTITATIVE IDEAS"
Cohesion: 0.07
Nodes (27): 1. EXECUTIVE SUMMARY & CORE PLATFORM VISION, 2. THE 4-PILLAR WINNING QUANTITATIVE ALPHA FORMULA, 3. EMPIRICAL MULTI-DECADE HISTORICAL BENCHMARKS (1980–2026), 4. MASTER 5-COCKPIT MISSION CONTROL ARCHITECTURE, 5. THE 300 INSTITUTIONAL RESOURCES SYNTHESIS, 6. CROSS-DISCIPLINARY & FRONTIER QUANTITATIVE IDEAS, 7. IMPLEMENTATION & TECHNOLOGY STACK BLUEPRINT, 🐺 A. Evolutionary Biology: Lotka-Volterra Predator-Prey Market Dynamics (+19 more)

### Community 64 - "GaussianHMMRegimeDetector"
Cohesion: 0.12
Nodes (11): GaussianHMMRegimeDetector, Any, DataFrame, ndarray, Paper 15: Gaussian Hidden Markov Model for Market Regime Detection. Source:…, 3-state Gaussian HMM regime classifier: Bull / Normal / Crisis. Uses hand-…, Gaussian emission probability for each state., Forward algorithm step: update filtered state probabilities with one new… (+3 more)

### Community 65 - "EWMACorrelationMonitor"
Cohesion: 0.14
Nodes (10): EWMACorrelationMonitor, Any, DataFrame, ndarray, Paper 17: RiskMetrics EWMA Volatility & Correlation Monitor. Source: J.P.…, Real-time EWMA-based correlation and volatility monitor. Tracks time-varying…, Initialize EWMA state from a seed window of returns., Update EWMA state with one day's returns across all assets. Returns current… (+2 more)

### Community 67 - "PaperBroker"
Cohesion: 0.14
Nodes (20): PaperBroker, Institutional Multi-Stage Quantitative Execution Broker ($100k Account).…, fixture, Test that a signal=SELL with confidence=0.50 does NOT prematurely exit a fresh…, Test that a signal=SELL with confidence < 0.40 triggers MODEL_SELL exit., Test that a ticker closed within the last 3 days is quarantined from immediate…, Test that an asset up +3% ratchets stop loss to lock in +1% profit., temp_portfolio_file() (+12 more)

### Community 68 - "CUSUMDetector"
Cohesion: 0.15
Nodes (9): CUSUMDetector, Any, ndarray, Paper 16: CUSUM Sequential Change-Point Detection. Source: E.S. Page (1954) —…, Cumulative Sum detector for online mean-shift detection. Monitors a numeric…, Process one observation. Returns alarm status., Process a batch of observations., Reset detector state. (+1 more)

### Community 69 - "PageHinkleyDetector"
Cohesion: 0.15
Nodes (9): PageHinkleyDetector, Any, ndarray, Paper 22: Page-Hinkley Sequential Test for Concept Drift Detection. Source:…, Page-Hinkley test for detecting changes in the mean of a stream. Monitors…, Process one observation. Returns drift status., Process a batch of observations., Reset detector state. (+1 more)

### Community 70 - "QuantAlphaDAGEngine"
Cohesion: 0.06
Nodes (46): AlphaDAGStageContract, Any, DataFrame, Series, QuantAlphaDAGEngine, Generation 3: Multi-Horizon AI Courtroom & Regime-Leveraged Quant DAG Engine.…, Calculates time-aligned returns, normalized volumes, and volatility bands., Generates Gen-3 Multi-Horizon Feature Library: - Factor 1: Triple-Horizon Trend… (+38 more)

### Community 71 - "test_stock_image_provider.py"
Cohesion: 0.06
Nodes (70): format_signal_card(), _is_duplicate_alert(), Any, Sends a rich formatted trade alert card to a Discord channel via Webhook., Checks if an alert fingerprint was already dispatched within ttl_seconds., Dispatches a crystal-clear, high-impact Discord card for live autonomous trade…, Records an alert fingerprint into results/discord_alert_ledger.json., Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord. (+62 more)

### Community 72 - "stress_tester.py"
Cohesion: 0.08
Nodes (36): calculate_kelly_sizing(), estimate_market_impact_slippage(), Any, Historical Black Swan Crisis Simulator & Kelly Position Sizing for Sentilyze.…, Calculates optimal position sizing using the Kelly Criterion: Kelly % = W - (1…, Estimates market execution slippage using the Almgren-Chriss square-root impact…, Stress-tests the current portfolio against major historical market crashes.…, simulate_portfolio_crises() (+28 more)

### Community 73 - "[Validator Rules Reference & Troubleshooting]"
Cohesion: 0.13
Nodes (14): 1. `ORPHAN_POD`, 2. `SERVICE_WITHOUT_TARGET`, 3. `INGRESS_BYPASS`, 4. `PVC_WITHOUT_PV`, 5. `NAMESPACE_LEAK`, 6. `MISSING_RESOURCE_LIMITS`, 7. `PRIVILEGED_CONTAINER`, [Anti-Patterns] (+6 more)

### Community 74 - "grossman_zhou_allocation"
Cohesion: 0.28
Nodes (5): grossman_zhou_allocation(), Any, Paper 18: Grossman-Zhou Optimal Drawdown-Constrained Strategy. Source: Grossman…, Compute the Grossman-Zhou optimal risky allocation under a drawdown constraint…, TestGrossmanZhou

### Community 75 - "ADWINDetector"
Cohesion: 0.19
Nodes (7): ADWINDetector, Any, ndarray, Paper 21: ADWIN (Adaptive Windowing) Drift Detector. Source: Bifet & Gavaldà…, ADWIN drift detector with Hoeffding bound. Maintains a variable-length window…, Add one observation. Returns whether drift was detected. If drift is detected,…, TestADWIN

### Community 76 - "test_papers_15_24.py"
Cohesion: 0.08
Nodes (25): calculate_cdar(), optimize_cdar_portfolio(), Any, DataFrame, ndarray, Paper 19: Conditional Drawdown-at-Risk (CDaR) Portfolio Optimization. Source:…, Calculate Conditional Drawdown-at-Risk: the expected drawdown in the worst…, Optimize portfolio weights to minimize CDaR. Simplified approach: compute CDaR… (+17 more)

### Community 77 - "MetaRegimeAllocator"
Cohesion: 0.25
Nodes (8): MetaRegimeAllocator, DataFrame, Fetch historical price data for the benchmark asset., Generate statistically grounded market proxy data if network or feed is…, Fit Gaussian Mixture and estimate empirical transition probabilities., Infer the current regime, state probabilities, and dynamic allocation weight., 3-State Gaussian Markov Regime Switching Allocator. Discovers latent market…, Compute daily log returns and rolling 20-day annualized realized volatility.

### Community 78 - "TradePostMortemLearner"
Cohesion: 0.09
Nodes (24): Any, DataFrame, Series, Autonomous Trade Post-Mortem & Bayesian Council Weight Calibration Engine for…, Categorizes trade outcome and assigns attribution scores., Computes performance feedback scores for each specialist agent based on…, Updates weights using a Softmax Bayesian update with Dirichlet regularization:…, Executes an end-to-end post-mortem learning cycle: 1. Analyzes trade history.… (+16 more)

### Community 79 - "ws_options_surface.py"
Cohesion: 0.23
Nodes (11): get_current_macro_regime(), MetaRegimeReport, 3-State Gaussian Hidden Markov Model & Meta-Regime Capital Allocator…, Convenience helper to compute macro regime report on demand., RegimeState, _get_cached_gex(), _get_cached_macro_regime(), cache_data (+3 more)

### Community 80 - "Any"
Cohesion: 0.24
Nodes (6): Any, Calculates historical win rate, trade count, and recent trajectory for a…, Synthesizes episodic trade memory to provide risk adjustments to…, Retrieves the most recent post-mortems in reverse chronological order., Returns all persistent semantic trading heuristics., Adds a newly discovered semantic rule to memory.

### Community 81 - "test_reddit_premarket_station.py"
Cohesion: 0.18
Nodes (19): fetch_4station_premarket_intelligence(), fetch_8station_premarket_intelligence(), fetch_all_reddit_headlines_for_ticker(), _fetch_subreddit_rss_entries(), Any, DataFrame, Systematic 8-Station Multi-Channel Reddit Market Intelligence Engine. Pillar 2…, Fetches real-time Atom RSS feed for a subreddit using safe defusedxml. (+11 more)

### Community 82 - "InstitutionalDataGateway"
Cohesion: 0.12
Nodes (12): InsiderFlowSnapshot, InstitutionalDataGateway, MacroRegimeSnapshot, OptionsSentimentSnapshot, Fetches Federal Reserve Macro Indicators from St. Louis Fed (FRED) or falls…, Classifies macroeconomic regime and calculates sizing multiplier., Fetches SEC Form 4 insider trading transactions (CEOs, CFOs, 10% Owners)., Computes Options Put/Call ratios and Max Pain strike. (+4 more)

### Community 83 - "risk_constrained_kelly_allocation"
Cohesion: 0.24
Nodes (6): Any, ndarray, Paper 23: Risk-Constrained Kelly Gambling. Source: Busseti, Ryu, Boyd —…, Compute optimal Kelly allocation with drawdown probability constraint.…, risk_constrained_kelly_allocation(), TestRiskKelly

### Community 84 - "fetch_financial_statements"
Cohesion: 0.20
Nodes (20): calculate_altman_z_score(), calculate_dcf_fair_value(), calculate_piotroski_f_score(), fetch_financial_statements(), _generate_calibrated_financials(), generate_spider_radar_profile(), Any, Institutional Fundamental Valuation & Forensic Accounting Engine for Sentilyze.… (+12 more)

### Community 85 - "test_tri_model_pipeline.py"
Cohesion: 0.16
Nodes (15): Financial Event & Emotion Companion Model (Model 2). Companion model to FinBERT…, get_fast_agent_pipeline(), Fast Sub-Second Agent Decision Router. Coordinates the Tri-Model Neural Brain…, Singleton getter for FastAgentPipeline., get_fast_finbert_engine(), Accelerated Low-Latency FinBERT Engine (Model 1). Optimized for high-speed…, Singleton getter for FastFinBERTEngine., NeuralGatingNetwork (+7 more)

### Community 86 - "test_agent_committee.py"
Cohesion: 0.09
Nodes (27): audit_full_universe_committee(), ChiefRiskOfficerAgent, compute_fractional_kelly_sizing(), ForensicFundamentalAgent, _persist_committee_resolution(), Any, Saves the committee resolution into results/committee_resolutions.json., Runs committee deliberation across the provided universe of tickers. (+19 more)

### Community 87 - "test_duckdb_engine.py"
Cohesion: 0.17
Nodes (11): Unit tests for DuckDB Columnar Data Lake & High-Speed Scanner Engine., Verify tables are created in the database., Verify bar ingestion and fast SQL querying., Verify vectorized momentum breakout scanner., Verify golden cross scanner detection., Verify coverage summary reporting., test_duckdb_coverage_summary(), test_duckdb_golden_cross_scanner() (+3 more)

### Community 88 - "apply_triple_barrier_labeling"
Cohesion: 0.15
Nodes (14): apply_triple_barrier_labeling(), calculate_deflated_sharpe_ratio(), DataFrame, Series, Computes Bailey & López de Prado's Deflated Sharpe Ratio (DSR). Adjusts for: -…, Applies López de Prado's path-dependent Triple-Barrier Method to generate trade…, Any, DataFrame (+6 more)

### Community 89 - "compute_cross_asset_correlation"
Cohesion: 0.40
Nodes (6): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…

### Community 90 - "AgentMemoryStore"
Cohesion: 0.19
Nodes (7): AgentMemoryStore, Synchronizes executed_trades.csv into trade_postmortems.jsonl and recalibrates…, Calibrates committee voting weights using the Softmax Dirichlet RLFF Algorithm.…, Returns current dynamic voting weights., Manages persistent episodic post-mortems, semantic rules, and dynamic voting…, Creates storage directories and baseline semantic files if missing., Appends an episodic trade post-mortem to the JSONL log. Args: ticker: Asset…

### Community 91 - "preprocess_data"
Cohesion: 0.07
Nodes (56): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+48 more)

### Community 92 - "block_external_alerts"
Cohesion: 0.50
Nodes (3): block_external_alerts(), fixture, Autouse fixture that prevents tests from sending real outbound network calls to…

### Community 93 - "test_agent_memory.py"
Cohesion: 0.12
Nodes (15): mock_memory(), fixture, Unit Tests for AgentMemoryStore & Episodic Trade Memory.…, Verifies that high-expectancy winners receive alpha champion boosts., Verifies synchronizing executed trades from an external CSV file., Verifies that PaperBroker._record_trade_closure automatically records post-…, Verifies that baseline rules and default weights are created upon…, Verifies appending post-mortems and updating empirical weights. (+7 more)

### Community 94 - "AICopilotEngine"
Cohesion: 0.24
Nodes (8): AICopilotEngine, Any, Conversational intelligence engine that parses queries and generates analytical…, Interprets user prompt and routes to appropriate financial analytical…, test_copilot_committee_query(), test_copilot_portfolio_query(), test_copilot_stress_query(), test_copilot_ticker_analysis_query()

### Community 95 - "calculate_multileg_payoff"
Cohesion: 0.31
Nodes (9): calculate_multileg_payoff(), generate_volatility_surface_mesh(), Any, Constructs a 3D Implied Volatility Surface across strike prices and expiration…, Calculates profit and loss (P&L) curves at expiration for institutional multi-…, test_calculate_multileg_payoff_bull_call_spread(), test_calculate_multileg_payoff_iron_condor(), test_calculate_multileg_payoff_long_straddle() (+1 more)

### Community 96 - "drawio/SKILL.md"
Cohesion: 0.13
Nodes (14): 1.Reference-Docs(AI-Knowledge), 2.Validator-Scripts(Programmatic-Enforcement), 3.Topological-Corrections(Auto-Fix), [Architectural Constraints], [Batch Diagram Generation (Highly Recommended)], Containers, [Diagram Builder Tools], [Docs Index] (+6 more)

### Community 97 - "AdversarialRedTeamAgent"
Cohesion: 0.24
Nodes (8): AdversarialRedTeamAgent, Any, Agent 5: Adversarial Red-Team / Devil's Advocate Specialist. Actively hunts for…, Conducts a rigorous adversarial audit of the target asset., Tests for Adversarial Red-Team Specialist Agent (src/red_team_agent.py).…, test_red_team_agent_initialization(), test_red_team_evaluation_structure(), test_red_team_stress_scenario()

### Community 98 - "liquidity_heatmap.py"
Cohesion: 0.31
Nodes (8): compute_order_book_depth_and_clusters(), compute_volume_profile_and_poc(), Any, Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap for…, Simulates Level 2 market depth and identifies institutional buy/sell liquidity…, Computes Point of Control (POC), Value Area High (VAH), and Value Area Low…, test_compute_order_book_depth_and_clusters(), test_compute_volume_profile_and_poc()

### Community 99 - "temp_data_dir"
Cohesion: 0.67
Nodes (3): fixture, Fixture to set a temporary data directory for tests., temp_data_dir()

### Community 100 - "DuckDBMarketEngine"
Cohesion: 0.22
Nodes (7): DuckDBMarketEngine, get_duckdb_scanner(), High-Performance DuckDB Columnar Data Lake & Market-Wide Scanner Engine.…, Embedded Columnar Analytical Engine powered by DuckDB., Closes connection cleanly., Convenience factory helper., Initializes high-performance columnar tables with index constraints.

### Community 101 - "compute_dark_pool_sentiment"
Cohesion: 0.21
Nodes (14): compute_dark_pool_sentiment(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent institutional off-exchange block trades and dark pool prints., Scans option chain contracts where daily volume significantly exceeds open…, Synthesizes dark pool prints and unusual options flow into a unified…, scan_abnormal_options_vol_oi(), scan_dark_pool_blocks() (+6 more)

### Community 102 - "Domain Expert Reference Template"
Cohesion: 0.15
Nodes (12): [Anti-Patterns], [Docs Index], Domain Expert Reference Template, [Domain Rules + Patterns], [Domain-Specific Catalog / Patterns] (OPTIONAL), [Project Context], [Project Conventions], Section Checklist (+4 more)

### Community 103 - "create_technical_indicators"
Cohesion: 0.17
Nodes (19): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, DataFrame (+11 more)

### Community 104 - "3. Equipment Mapping (Draw.io Native Shapes)"
Cohesion: 0.15
Nodes (12): 1. ISA 5.1 Instrument Tagging Conventions, 2. Line Styles, 3. Equipment Mapping (Draw.io Native Shapes), 4. Diagram Layout Strategy (PFDs), Common Tag Identifiers, Heat Exchangers, Instrument Shapes (`mxgraph.pid2inst.*`), Mining & Minerals Processing (+4 more)

### Community 105 - "fractional_differentiation_ffd"
Cohesion: 0.29
Nodes (10): find_optimal_d(), fractional_differentiation_ffd(), get_weights_ffd(), ndarray, Series, Fixed-Width Window Fractional Differentiation (FFD) for Financial Time Series.…, Generate weights for Fixed-Width Window Fractional Differentiation. w_0 = 1 w_k…, Applies Fixed-Width Window Fractional Differentiation to a price series. (+2 more)

### Community 106 - "calculate_insider_conviction_score"
Cohesion: 0.28
Nodes (11): calculate_insider_conviction_score(), fetch_insider_transactions(), Any, Smart-Money Executive & Institutional Insider Radar for Sentilyze.…, Computes the Quantitative Insider Conviction Index (0 to 100 Score) and detects…, Screens a universe of tickers and returns the highest-ranking insider buying…, Fetches recent SEC Form 4 insider transactions for a specific ticker. Includes…, scan_universe_insider_catalysts() (+3 more)

### Community 107 - "compound_engine.py"
Cohesion: 0.21
Nodes (12): calculate_doubling_progress(), calculate_turtle_pyramid_plan(), compute_compound_position_size(), Any, Max Compound Acceleration, Turtle 0.5N Pyramiding & +100% Target Doubling…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes a 3-stage Turtle ATR pyramiding plan (Seykota & Covel, Leung & Zhan… (+4 more)

### Community 108 - "Agent Browser — Live Web Automation Skill"
Cohesion: 0.50
Nodes (3): 1. Core Principles, 2. Use Cases in Sentilyze, Agent Browser — Live Web Automation Skill

### Community 109 - "Agent Memory — Persistent Multi-Session Memory Skill"
Cohesion: 0.50
Nodes (3): 1. Storage Architecture, 2. Integration Pattern, Agent Memory — Persistent Multi-Session Memory Skill

### Community 113 - "compute_gamma_exposure_profile"
Cohesion: 0.14
Nodes (20): calculate_black_scholes_gamma(), calculate_options_gex(), compute_gamma_exposure_profile(), fetch_options_chain_data(), Any, DataFrame, Options Gamma Exposure (GEX), Volatility Surface & Strike Wall Radar for…, Computes institutional Net Gamma Exposure (GEX), Call Wall, Put Wall, and Gamma… (+12 more)

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

### Community 118 - "calculate_vpin"
Cohesion: 0.27
Nodes (9): calculate_vpin(), Any, DataFrame, Computes Volume-Synchronized Probability of Toxicity (VPIN) from price and…, Unit tests for Volume-Synchronized Probability of Toxicity (VPIN)…, test_vpin_bounded_between_zero_and_one(), test_vpin_detects_severe_toxic_dumping(), test_vpin_ticker_fallback() (+1 more)

### Community 119 - "acpm_trainer.py"
Cohesion: 0.21
Nodes (8): Unified Alpha-Conformal Purged Multi-Task (ACPM) Quantitative Training Engine.…, Conformal Probability Calibration & Quantile Uncertainty Layer. Maps raw tree…, compute_deflated_sharpe_ratio(), PurgedGroupTimeSeriesSplit, Combinatorial Purged & Embargoed Cross-Validation (CPCV) for Financial Machine…, Time-series cross-validator that purges overlapping event windows and applies…, Calculates the Deflated Sharpe Ratio (DSR) to test for backtest overfitting., test_purged_cv_and_dsr()

### Community 120 - ".query"
Cohesion: 0.22
Nodes (6): DataFrame, Inserts or replaces daily bars for a given ticker., Populates the DuckDB lake from pre-computed backtest portfolio curves in…, Screens for high-probability momentum setups across the entire lake., Screens for stocks where SMA50 is breaking out above SMA200 (Long-Term Bullish…, Executes arbitrary SQL query and returns result as a Pandas DataFrame.

### Community 121 - "compile_biotech_catalyst_radar"
Cohesion: 0.23
Nodes (11): compile_biotech_catalyst_radar(), fetch_clinical_trials_catalysts(), fetch_fda_regulatory_catalysts(), is_biotech_or_healthcare(), Any, BioTech & Pharma Catalyst Radar for Sentilyze.…, Queries openFDA API for recent approvals, drug labels, or regulatory safety…, Compiles complete regulatory and clinical trials intelligence for a given stock. (+3 more)

### Community 122 - "render_workspace_header"
Cohesion: 0.16
Nodes (16): Cockpit 4: Multi-Decade Backtest & Performance Factsheet…, render_backtest_cockpit(), Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Renders an executive header banner with live status badge and market clock., Wraps HTML content inside an institutional frosted glass container., render_glass_card(), render_workspace_header(), Workspace 6: Walk-Forward Backtesting & Performance Tearsheet. (+8 more)

### Community 123 - "screener_engine.py"
Cohesion: 0.16
Nodes (14): Real-Time Market Anomaly Screener Component for Streamlit. Functions: - Renders…, Renders the Real-Time Market Anomaly Screener UI., render_live_screener_section(), Cross-Asset Sector-Pooled Multi-Task Dataset Engine. Maps the 538-stock…, evaluate_single_asset_screener(), Any, DataFrame, Real-Time Market Anomaly Screener Engine. Functions: - Rapid multi-condition… (+6 more)

### Community 124 - "test_pillar2_alternative_data.py"
Cohesion: 0.19
Nodes (18): compute_smart_money_insider_score(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent SEC Form 4 insider transactions for a given stock., Retrieves recent Congressional STOCK Act disclosure reports for a ticker., Synthesizes SEC Form 4 and Congressional activity into an overall Smart Money…, track_congressional_stock_disclosures(), track_corporate_insider_filings() (+10 more)

### Community 125 - "get_sentiment"
Cohesion: 0.16
Nodes (15): Any, Executes a live 5-Minute Opening Range Breakout scan across top liquid assets:…, run_opening_range_session(), _load_sentiment_analyzer(), Any, Thread-safely loads the FinBERT sentiment analysis model and tokenizer once…, analyze_sentiment(), clean_financial_text() (+7 more)

### Community 126 - "FastFinBERTEngine"
Cohesion: 0.27
Nodes (5): FastFinBERTEngine, Any, Vectorized micro-batch inference for multiple headlines., Accelerated FinBERT inference engine delivering 2.5x - 4x speedups., Runs accelerated sentiment scoring on a single headline. Includes Companion…

### Community 127 - "orderflow_chart.py"
Cohesion: 0.29
Nodes (10): build_orderflow_candlestick_chart(), calculate_volume_profile(), detect_fair_value_gaps(), Any, DataFrame, Figure, Institutional TradingView-Style Interactive Candlestick & Order Flow Engine.…, Detects 3-candle Fair Value Gaps (FVG) / institutional liquidity imbalances. (+2 more)

### Community 128 - "MasterTradingLoop"
Cohesion: 0.15
Nodes (12): get_master_trading_loop(), MasterTradingLoop, Returns the singleton instance of the MasterTradingLoop., Unified Master Daily Trading Loop Orchestrator for Sentilyze. Coordinates all 4…, get_recent_sec_catalysts(), Any, Fetches Form 8-K material filings for a given company within the lookback…, Fast heuristic catalyst sentiment classifier. Yields positive score for… (+4 more)

### Community 129 - "quant_engine.py"
Cohesion: 0.11
Nodes (22): MasterQuantPipelineResult, Any, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), analyze_earnings_call_transcript(), Any (+14 more)

### Community 130 - "generate_comprehensive_factsheet"
Cohesion: 0.38
Nodes (6): generate_comprehensive_factsheet(), Any, Series, Computes over 30 institutional hedge-fund risk, performance, and drawdown…, test_generate_comprehensive_factsheet_custom_series(), test_generate_comprehensive_factsheet_default()

### Community 131 - "TriBrainGatingLayer"
Cohesion: 0.40
Nodes (3): Tensor, Neural Softmax Gating layer to dynamically allocate weights among the 3 models.…, TriBrainGatingLayer

### Community 132 - "ws_insider_radar.py"
Cohesion: 0.24
Nodes (9): Cockpit 2: Smart Money & Alternative Data…, render_smart_money_cockpit(), _get_cached_insider_data(), cache_data, Workspace: Smart-Money Executive & Institutional Insider Radar. SEC Form 4…, render_insider_radar_workspace(), Workspace 24: Real-Time Market Anomaly Screener., Renders the Real-Time Market Anomaly Screener Workspace. (+1 more)

### Community 133 - "analyze_earnings_surprises"
Cohesion: 0.23
Nodes (10): analyze_earnings_surprises(), calculate_sue_metric(), Any, Post-Earnings Announcement Drift (PEAD) & SUE Earnings Surprise Radar for…, Computes surprise percentage and Standardized Unexpected Earnings (SUE)., Retrieves and computes recent earnings surprise metrics, SUE score, and PEAD…, Unit tests for Post-Earnings Announcement Drift (PEAD) & SUE Earnings Surprise…, test_earnings_surprises_bearish_miss() (+2 more)

### Community 134 - "main"
Cohesion: 0.21
Nodes (11): Root entry point for Sentilyze CLI. Run directly with: python sentilyze.py NVDA…, cmd_briefing(), cmd_portfolio(), cmd_scan(), cmd_trades(), main(), print_banner(), Displays current portfolio metrics and live holdings. (+3 more)

### Community 135 - "calculate_15min_opening_range"
Cohesion: 0.24
Nodes (12): calculate_15min_opening_range(), find_low_of_day_pullback_entry(), is_opening_15min_whipsaw_period(), Any, DataFrame, 15-Minute Opening Bell Volatility Shield & Low-of-Day Demand Pullback Engine…, Checks if current Eastern Time is within the hectic 09:30 - 09:45 EDT opening…, Calculates the 15-minute Opening Range (High, Low, Midpoint) established… (+4 more)

### Community 136 - "FastNeuralEventClassifier"
Cohesion: 0.29
Nodes (4): FastNeuralEventClassifier, Tensor, Builds a lightweight internal token dictionary for financial catalysts., Lightweight 2-layer Neural Event & Emotion Head for sub-millisecond…

### Community 137 - "compute_advanced_performance_ratios"
Cohesion: 0.33
Nodes (6): compute_advanced_performance_ratios(), compute_walk_forward_efficiency(), Any, Series, Computes Pardo (2008) Walk-Forward Efficiency (WFE) Ratio: WFE = OOS_Sharpe /…, Computes institutional performance ratios: Calmar, Sortino, Omega, Max DD…

### Community 138 - "audio_briefing.py"
Cohesion: 0.47
Nodes (5): generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses Microsoft Edge Neural TTS…, synthesize_morning_audio()

### Community 139 - "print_formatted_trade_report"
Cohesion: 0.23
Nodes (11): get_trade_telemetry(), print_formatted_trade_report(), Any, Instant Trade Telemetry & Results Engine. Provides direct, zero-overhead access…, Fetches real-time trade execution summary, partitioning by today, yesterday,…, Prints a clean, concise institutional trade report to the terminal., Unit tests for Instant Trade Telemetry & Results Engine., Verify formatted trade report prints without raising exceptions. (+3 more)

### Community 140 - "test_acpm_trainer.py"
Cohesion: 0.15
Nodes (15): build_pooled_sector_dataset(), DataFrame, Series, Pools cross-sectional DataFrames across sector peers and creates a unified…, neutralize_features(), neutralize_predictions(), DataFrame, ndarray (+7 more)

### Community 141 - "EventClassifierModel"
Cohesion: 0.24
Nodes (8): EventClassifierModel, Any, Translates raw social slang, emojis, and cashtags into standardized financial…, Classifies a single headline or social post with sub-millisecond speed.…, Batch-processes multiple headlines with vectorized execution., Production-grade Financial Event & Emotion Companion Model. Provides sub-…, test_event_classifier_fraud_and_dilution(), test_event_classifier_model()

### Community 142 - "generate_pipeline_graph_data"
Cohesion: 0.27
Nodes (9): generate_pipeline_graph_data(), Any, Interactive Multi-Agent & Pipeline Architecture Canvas Component for Streamlit.…, Renders the interactive Vis.js animated node network canvas inside Streamlit., Constructs the nodes and edges representation for the pipeline graph canvas., render_pipeline_topology_canvas(), Tests for Interactive Multi-Agent & Pipeline Architecture Canvas…, test_generate_pipeline_graph_data_defaults() (+1 more)

### Community 143 - "test_autonomous_trader.py"
Cohesion: 0.12
Nodes (19): check_daily_loss_circuit_breaker(), ensure_background_daemon_thread_running(), get_daemon_status(), Any, Returns current live daemon running state and last pulse timestamp., Ensures a single background autonomous trading daemon thread is permanently…, Task 8: Independent Max-Daily-Loss Circuit Breaker. Compares current total…, clean_lock_file() (+11 more)

### Community 144 - "ws_alternative_data.py"
Cohesion: 0.24
Nodes (10): get_pre_ipo_pipeline_df(), DataFrame, Returns a formatted pandas DataFrame of all pre-IPO target assets., Renders a progress meter with dynamic color coding., render_conviction_gauge(), _get_cached_reddit_intel(), cache_data, Workspace 4: Systematic 9-Station 1-Day-Prior Reddit News & SEC S-1 Pre-IPO… (+2 more)

### Community 146 - "check_correlation_shield"
Cohesion: 0.25
Nodes (10): calculate_portfolio_correlation_matrix(), check_correlation_shield(), Any, DataFrame, Computes the 21-day rolling pairwise return correlation matrix across tickers., Audits a candidate buy against currently held positions. Returns whether the…, Tests for Portfolio Correlation Matrix Shield (src/correlation_shield.py).…, test_calculate_portfolio_correlation_matrix() (+2 more)

### Community 147 - ".get_closed_trades_df"
Cohesion: 0.29
Nodes (4): DataFrame, Returns a DataFrame of current open holdings with Scale-Out status., Returns a DataFrame of trade history with full company names., Returns equity history as a DatetimeIndex DataFrame.

### Community 148 - "ws_quantum_tournament.py"
Cohesion: 0.43
Nodes (7): get_market_timestamp(), load_safety_benchmarks(), load_tournament_results(), Any, cache_data, Workspace 14: 25-Paper Quantum Tournament, Live Omni-Hybrid Pipeline & Risk…, render_quantum_tournament_workspace()

### Community 149 - ".is_connected"
Cohesion: 0.21
Nodes (7): Any, Fetches active positions from Alpaca brokerage., Liquidates an active position on Alpaca and cancels all open child orders., Emergency liquidator: closes all positions and cancels all open orders., Verifies active connection to Alpaca Brokerage API., Fetches live Alpaca account equity, buying power, and cash., Submits an institutional Bracket Order: - Entry: Market order - Exit 1: Limit…

### Community 150 - "run_daily_market_scan"
Cohesion: 0.19
Nodes (11): Scans the entire stock universe defined in stocks.txt, generates tomorrow's…, run_daily_market_scan(), Any, Dispatches formatted HTML morning market digest email via Gmail SMTP., send_email_digest(), Test that high VPIN toxicity vetoes a trade and downgrades signal to HOLD., Test that confidence < 0.40 triggers explicit SELL signal., Test that run_daily_market_scan produces BUY when all institutional gates… (+3 more)

### Community 151 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (15): Any, DataFrame, ndarray, Series, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used., Loads all 3 models natively. (+7 more)

### Community 152 - "test_regime_allocator.py"
Cohesion: 0.18
Nodes (10): mock_price_history(), fixture, Unit tests for 3-State Gaussian Hidden Markov Model & Meta-Regime Allocator., Generate 100 days of price history with bull and volatile phases., Verify HMM discovers 3 regimes and sorts them properly., Verify analysis report structure and allocation bounds., Verify fallback generation works if remote API fails., test_regime_allocator_analysis_report() (+2 more)

### Community 153 - "run_full_quant_experiment"
Cohesion: 0.22
Nodes (9): compute_performance_metrics(), Any, DataFrame, Series, Executes empirical ablation benchmark across the full asset universe., Simulates walk-forward strategy execution with or without advanced quant…, Computes key quant performance metrics., run_full_quant_experiment() (+1 more)

### Community 154 - "DynamicSharpeMetaEnsemble"
Cohesion: 0.25
Nodes (5): DynamicSharpeMetaEnsemble, Dynamically re-weights multi-agent specialist and sub-model signals based on…, Record historical daily returns for a submodel., Calculate annualized Sharpe ratio for each submodel over lookback window., Compute Softmax allocation weights over rolling Sharpe ratios.

### Community 155 - "cockpit_trading.py"
Cohesion: 0.24
Nodes (10): Cockpit 1: Live Trading & Alpha Execution…, render_trading_cockpit(), Renders the 24/7 Autonomous Live Trading & News Agent interface., render_autonomous_trader_workspace(), Workspace: Automated Broker Webhooks & Execution API Dispatcher. Configures and…, render_broker_webhooks_workspace(), Renders the high-speed live inference and directional prediction workspace., render_live_prediction_workspace() (+2 more)

### Community 156 - "run_committee_ablation_backtest"
Cohesion: 0.36
Nodes (7): _persist_ablation_results(), Any, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), test_committee_ablation_study_execution()

### Community 157 - "render_multi_agent_war_room"
Cohesion: 0.28
Nodes (7): Any, Interactive Multi-Agent War Room Visualizer Component for Streamlit. Functions:…, Renders the complete 5-Agent War Room Council deliberation chamber., render_multi_agent_war_room(), Real-Time Audio Trade Squawk Component for Streamlit. Functions: - Uses…, Renders an HTML5 Web Speech API audio squawk generator inside Streamlit., render_audio_squawk_button()

### Community 158 - "ws_deep_quant.py"
Cohesion: 0.42
Nodes (8): _get_cached_forensic(), _get_cached_gnn(), _get_cached_pairs(), _get_cached_stress(), cache_data, Workspaces 9-13: Deep Quantitative Modeling, GNN Supply Chain, Stress Tests &…, Renders specialized institutional quant workspaces., render_deep_quant_workspace()

### Community 159 - "analyze_sec_filing_diff"
Cohesion: 0.32
Nodes (7): analyze_sec_filing_diff(), compute_text_similarity_and_diff(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Computes lexical and semantic diff metrics between consecutive filings., Retrieves and compares the most recent and prior SEC filings for a company.…, test_sec_filing_diff_analysis()

### Community 160 - "compute_alpha158_top25"
Cohesion: 0.29
Nodes (7): compute_alpha158_top25(), get_alpha158_feature_names(), DataFrame, Microsoft Qlib Alpha158 Top 25 Orthogonal Alpha Factors. Down-selected from the…, Computes the Top 25 Orthogonal Alpha158 factors from standard daily OHLCV…, Returns the names of all 25 orthogonal Alpha158 factors., test_alpha158_top25_calculation()

### Community 161 - "test_ui_modules.py"
Cohesion: 0.27
Nodes (7): load_cached_tournament_results(), cache_data, Workspace 25: Autonomous Quant Alpha DAG Engine (Gen-3 Champion). Institutional…, Loads pre-computed 3-generation tournament results if available., Renders the comprehensive Quant Alpha DAG Engine workspace., render_alpha_dag_workspace(), test_ws_alpha_dag_module_import()

### Community 162 - ".train_ticker"
Cohesion: 0.40
Nodes (4): Any, DataFrame, Series, Executes end-to-end ACPM training for a target equity.

### Community 163 - "ConformalCalibrator"
Cohesion: 0.15
Nodes (10): ACPMTrainer, State-of-the-Art 10x Institutional Quantitative Training Engine., ConformalCalibrator, ndarray, Non-parametric monotonic Isotonic Conformal Calibrator., Calibrates on an independent holdout calibration split., Returns empirical calibrated probabilities., Checks if a trade signal has statistically significant empirical edge beyond… (+2 more)

### Community 164 - "cockpit_deep_quant.py"
Cohesion: 0.38
Nodes (5): Cockpit 5: Deep Quant & Explainability (XAI)…, render_deep_quant_cockpit(), Workspace 7: SHAP Explainability & Dynamic Game-Theoretic Decision Trees., Renders SHAP explainability plots, dynamic Plotly waterfall charts, and feature…, render_xai_workspace()

### Community 165 - "_get_browser_session"
Cohesion: 0.33
Nodes (5): _get_browser_session(), Session, Universal Adaptive Web Scraper supporting Scrapling, Trafilatura, and…, Creates a configured requests session with realistic browser headers., UniversalWebScraper

### Community 166 - "memory_engine"
Cohesion: 0.40
Nodes (5): memory_engine(), fixture, Provides an isolated in-memory DuckDB market engine., Generates synthetic daily bars for testing., sample_bars_df()

### Community 171 - "sample_ohlcv_data"
Cohesion: 0.67
Nodes (3): fixture, Generates 60 bars of synthetic OHLCV data., sample_ohlcv_data()

## Knowledge Gaps
- **224 isolated node(s):** `graphify`, `1. Core Principles`, `2. Use Cases in Sentilyze`, `1. Storage Architecture`, `2. Integration Pattern` (+219 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `utils.py` to `deep_learning_model.py`, `quant_engine.py`, `get_price_history`, `test_options_flow.py`, `run_backtest`, `analyze_earnings_surprises`, `test_all_14_papers.py`, `drl_policy_agent.py`, `calculate_15min_opening_range`, `TradingEnvironment`, `audio_briefing.py`, `CloudDataLake`, `test_statistical_arbitrage.py`, `analyze_supply_chain_spillover`, `calculate_hrp_weights`, `run_temporal_fusion_forecast`, `OnlineNewtonStepOptimizer`, `ws_live_prediction.py`, `run_daily_market_scan`, `test_omnichannel_mobile.py`, `ultra_quant_engine.py`, `compute_lead_lag_matrix`, `social_sentiment.py`, `agent_committee.py`, `analyze_sec_filing_diff`, `ipo_radar.py`, `get_us_market_session`, `StrategyGenome`, `triple_convex_engine.py`, `VolScaledMomentumEngine`, `universal_web_scraper.py`, `price_scout.py`, `webhook_dispatcher.py`, `calculate_time_decayed_sentiment`, `QuantAlphaDAGEngine`, `test_stock_image_provider.py`, `stress_tester.py`, `TradePostMortemLearner`, `test_reddit_premarket_station.py`, `fetch_financial_statements`, `test_tri_model_pipeline.py`, `preprocess_data`, `liquidity_heatmap.py`, `DuckDBMarketEngine`, `compute_dark_pool_sentiment`, `calculate_insider_conviction_score`, `compound_engine.py`, `compute_gamma_exposure_profile`, `run_universe_training.py`, `compile_biotech_catalyst_radar`, `screener_engine.py`, `test_pillar2_alternative_data.py`, `get_sentiment`, `orderflow_chart.py`?**
  _High betweenness centrality (0.191) - this node is a cross-community bridge._
- **Why does `get_price_history()` connect `get_price_history` to `run_backtest`, `test_statistical_arbitrage.py`, `test_all_14_papers.py`, `drl_policy_agent.py`, `FactorAttributionEngine`, `check_correlation_shield`, `run_daily_market_scan`, `ultra_quant_engine.py`, `run_full_quant_experiment`, `run_committee_ablation_backtest`, `agent_committee.py`, `utils.py`, `StrategyGenome`, `price_scout.py`, `test_cross_disciplinary_alphas.py`, `render_committee_workspace`, `generate_morning_briefing_text`, `QuantAlphaDAGEngine`, `test_agent_committee.py`, `compute_cross_asset_correlation`, `preprocess_data`, `AdversarialRedTeamAgent`, `run_universe_training.py`, `screener_engine.py`, `get_sentiment`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `PaperBroker` connect `PaperBroker` to `MasterTradingLoop`, `ws_insider_radar.py`, `main`, `drl_policy_agent.py`, `print_formatted_trade_report`, `calculate_hrp_weights`, `test_autonomous_trader.py`, `AutonomousTradingEngine`, `.get_closed_trades_df`, `ws_live_prediction.py`, `test_pyramiding_cppi_alpha158.py`, `run_daily_market_scan`, `AlpacaBrokerBridge`, `agent_committee.py`, `StrategyGenome`, `.get_portfolio_summary`, `._load_or_initialize`, `._save_state`, `.execute_daily_signals`, `generate_morning_briefing_text`, `QuantAlphaDAGEngine`, `AgentMemoryStore`, `preprocess_data`, `test_agent_memory.py`, `AICopilotEngine`, `render_workspace_header`, `get_sentiment`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `1. Core Principles`, `2. Use Cases in Sentilyze` to the rest of the system?**
  _224 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `deep_learning_model.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1380952380952381 - nodes in this community are weakly interconnected._
- **Should `get_price_history` be split into smaller, more focused modules?**
  _Cohesion score 0.05443371378402107 - nodes in this community are weakly interconnected._