# Graph Report - Sentilyze  (2026-09-11)

## Corpus Check
- 1910 files · ~1,175,896 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2407 nodes · 4743 edges · 133 communities (125 shown, 8 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 36 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `52c79fb2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- deep_learning_model.py
- get_price_history
- quant_engine.py
- alerts.py
- run_backtest
- test_statistical_arbitrage.py
- utils.py
- drl_policy_agent.py
- fetch_financial_statements
- TradingEnvironment
- OpeningRangeBreakout
- CloudDataLake
- app.py
- analyze_supply_chain_spillover
- ws_portfolio.py
- run_temporal_fusion_forecast
- autonomous_trader.py
- meta_ensemble.py
- OnlineNewtonStepOptimizer
- ws_live_prediction.py
- draw.io XML Style Reference
- compute_dark_pool_sentiment
- test_omnichannel_mobile.py
- pfd-engineering-expert.md
- preprocess_data
- compute_lead_lag_matrix
- black_swan_simulator.py
- render_workspace_header
- AlpacaBrokerBridge
- PaperBroker
- ._save
- get_us_market_session
- ws_alternative_data.py
- 3. Cross-Functional Flowcharts (Actor × Phase Grid)
- calculate_hrp_weights
- Edge Routing Decision Guide
- config.py
- triple_convex_engine.py
- Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform
- get_sector_for_ticker
- [Validator Rules Reference & Troubleshooting]
- test_zero_giveback_profit_lock.py
- RegimeMixtureOfExperts
- .split
- ui/__init__.py
- Sentilyze Community Code of Conduct
- universal_web_scraper.py
- calculate_doubling_progress
- test_pillar2_alternative_data.py
- price_scout.py
- test_cross_disciplinary_alphas.py
- Contributing to Sentilyze
- ws_committee.py
- 🧪 Experimental & Simulated Research Prototypes
- Sentilyze — Standing Audit Protocol
- generate_morning_briefing_text
- calculate_time_decayed_sentiment
- erd-database-expert.md
- daily_scanner.py
- 6. CROSS-DISCIPLINARY & FRONTIER QUANTITATIVE IDEAS
- rules/graphify.md
- workflows/graphify.md
- GaussianHMMRegimeDetector
- EWMACorrelationMonitor
- get_sentiment
- CUSUMDetector
- PageHinkleyDetector
- QuantAlphaDAGEngine
- preprocessing.py
- screener_engine.py
- [Validator Rules Reference & Troubleshooting]
- grossman_zhou_allocation
- ADWINDetector
- test_papers_15_24.py
- test_api.py
- sanitize_filename
- run_cppi_backtest
- DCCCorrelation
- test_reddit_premarket_station.py
- test_rebalancer_and_tearsheet.py
- risk_constrained_kelly_allocation
- benchmark_training_paradigms.py
- test_tri_model_pipeline.py
- agent_committee.py
- ws_quantum_tournament.py
- calculate_beneish_m_score
- correlation_matrix.py
- generate_pipeline_graph_data
- train_model
- block_external_alerts
- temp_data_dir
- AICopilotEngine
- calculate_multileg_payoff
- drawio/SKILL.md
- AdversarialRedTeamAgent
- liquidity_heatmap.py
- .get_closed_trades_df
- calculate_portfolio_diversity_grade
- cli.py
- Domain Expert Reference Template
- social_sentiment.py
- 3. Equipment Mapping (Draw.io Native Shapes)
- test_acpm_trainer.py
- ws_insider_radar.py
- ablation_study.py
- Agent Browser — Live Web Automation Skill
- Agent Memory — Persistent Multi-Session Memory Skill
- Cybersecurity Skills — Institutional MLOps Security Protocol
- Editorial Diagram Design Skill
- Harness Engineering Skill
- generate_institutional_pdf_tearsheet
- aws-well-architected-reviewer.md
- gcp-well-architected-reviewer.md
- run_universe_training.py
- generate_comprehensive_factsheet
- FastNeuralEventClassifier
- ConformalCalibrator
- EventClassifierModel
- simulate_strategy_sandbox
- fractional_differentiation_ffd
- run_unified_institutional_pipeline
- FastFinBERTEngine
- test_ui_modules.py
- render_multi_agent_war_room
- TriBrainGatingLayer
- DLinearTCNModel
- .fuse
- audio_briefing.py
- apply_triple_barrier_labeling
- _get_browser_session

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 103 edges
2. `get_price_history()` - 71 edges
3. `PaperBroker` - 56 edges
4. `fetch_live_quote()` - 44 edges
5. `get_news()` - 35 edges
6. `run_unified_institutional_pipeline()` - 32 edges
7. `preprocess_data()` - 29 edges
8. `render_workspace_header()` - 26 edges
9. `convene_trading_committee()` - 24 edges
10. `sanitize_filename()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `test_paper9_when_agents_trade_scanner()` --calls--> `run_daily_market_scan()`  [EXTRACTED]
  tests/test_all_14_papers.py → src/daily_scanner.py
- `test_autonomous_cycle_execution()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_idempotency_lock_prevents_overlap()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_paper12_hierarchical_risk_parity()` --calls--> `calculate_hrp_weights()`  [EXTRACTED]
  tests/test_all_14_papers.py → src/portfolio.py
- `test_calculate_hrp_weights()` --calls--> `calculate_hrp_weights()`  [EXTRACTED]
  tests/test_portfolio.py → src/portfolio.py

## Import Cycles
- None detected.

## Communities (133 total, 8 thin omitted)

### Community 0 - "deep_learning_model.py"
Cohesion: 0.14
Nodes (19): create_sliding_window_tensors(), load_dlinear_model(), Any, DataFrame, High-Efficiency Deep Learning Engine: DLinear + Temporal Convolutional Network…, Converts a pandas DataFrame into sliding window sequence tensors for Deep…, Trains the DLinear-TCN model on CPU using AdamW optimizer with learning rate…, Saves model weights safely using PyTorch state_dict. (+11 more)

### Community 1 - "get_price_history"
Cohesion: 0.06
Nodes (57): prefetch_single_ticker(), Pre-fetches price history and news data concurrently across fast I/O threads., _fetch_alpaca_news(), _fetch_alpaca_price_history(), _fetch_direct_yahoo_chart(), _fetch_eodhd_price_history(), _fetch_finnhub_news(), _fetch_fmp_price_history() (+49 more)

### Community 2 - "quant_engine.py"
Cohesion: 0.15
Nodes (24): MasterQuantPipelineResult, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain() (+16 more)

### Community 3 - "alerts.py"
Cohesion: 0.15
Nodes (25): format_signal_card(), _is_duplicate_alert(), Any, Checks if an alert fingerprint was already dispatched within ttl_seconds., Dispatches a crystal-clear, high-impact Discord card for live autonomous trade…, Records an alert fingerprint into results/discord_alert_ledger.json., Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord., Dispatches real-time Reddit r/wallstreetbets & Stocktwits hype spike alerts. (+17 more)

### Community 4 - "run_backtest"
Cohesion: 0.08
Nodes (42): main(), Batch Universe Trainer for Remaining S&P 100 Tickers., run_single(), _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition() (+34 more)

### Community 5 - "test_statistical_arbitrage.py"
Cohesion: 0.19
Nodes (27): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any, Series (+19 more)

### Community 6 - "utils.py"
Cohesion: 0.06
Nodes (41): ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Logger, Any, Master Academic Research Papers Empirical Benchmark Suite (All 14 Papers).…, Executes empirical backtests comparing all 14 academic paper methodologies., run_all_14_papers_benchmark(), calculate_almgren_chriss_trajectory(), Any (+33 more)

### Community 7 - "drl_policy_agent.py"
Cohesion: 0.13
Nodes (20): ActorCriticPolicy, DRLTradingEnvironment, evaluate_drl_policy_action(), Any, ndarray, Tensor, Deep Reinforcement Learning (DRL) Autonomous Policy Agent for Sentilyze.…, Trains an Actor-Critic DRL policy agent on historical market returns and news… (+12 more)

### Community 8 - "fetch_financial_statements"
Cohesion: 0.20
Nodes (20): calculate_altman_z_score(), calculate_dcf_fair_value(), calculate_piotroski_f_score(), fetch_financial_statements(), _generate_calibrated_financials(), generate_spider_radar_profile(), Any, Institutional Fundamental Valuation & Forensic Accounting Engine for Sentilyze.… (+12 more)

### Community 9 - "TradingEnvironment"
Cohesion: 0.14
Nodes (16): optimize_rl_position_allocation(), PPOPolicyAgent, Any, ndarray, ⚠️ EXPERIMENTAL / RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM PRODUCTION…, Computes mean action (leverage) between 0.0 and 2.0., Estimates state value., Trains Actor-Critic parameters across historical episodes. (+8 more)

### Community 10 - "OpeningRangeBreakout"
Cohesion: 0.16
Nodes (12): OpeningRangeBreakout, Any, DataFrame, Paper 25: Opening Range Breakout (ORB) with Stocks-in-Play Filter. Source:…, Simulate multi-year daily ORB strategy returns., Opening Range Breakout (ORB) trading engine with Stocks-in-Play filter. Paper…, Rank universe and select top 'Stocks in Play' based on volume, ATR, and…, Evaluate ORB signal using daily OHLC + volatility approximation. (+4 more)

### Community 11 - "CloudDataLake"
Cohesion: 0.14
Nodes (14): CloudDataLake, Any, Supabase / PostgreSQL Cloud Data Lake Connector., Validates or generates cloud database schema., Syncs local trade executions to the cloud database., Publishes real-time portfolio snapshot to cloud WebSockets channel., generate_twap_order_schedule(), generate_vwap_order_schedule() (+6 more)

### Community 12 - "app.py"
Cohesion: 0.18
Nodes (15): get_model_universe_indicator(), load_universe_tickers(), main(), cache_data, Sentilyze - Institutional Algorithmic Trading & MLOps Platform. High-Speed…, Loads active S&P 100 universe tickers., Computes universe model training coverage and mean accuracy using compiled…, render_deep_quant_cockpit() (+7 more)

### Community 13 - "analyze_supply_chain_spillover"
Cohesion: 0.14
Nodes (15): analyze_supply_chain_spillover(), Any, ndarray, Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for…, Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)., Executes a Graph Convolutional Network (GCN) layer: H_new = ReLU(A_hat * H * W)…, Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab…, High-level entry point to run GNN supply chain shock propagation. (+7 more)

### Community 14 - "ws_portfolio.py"
Cohesion: 0.14
Nodes (21): build_unified_portfolio(), calculate_risk_parity_weights(), load_all_ticker_portfolios(), Any, DataFrame, Load individual backtest portfolio CSVs for all available tickers. Args:…, Combines individual stock strategies into a single managed multi-asset fund.…, Calculate Inverse-Volatility (Naive Risk Parity) weights for each asset. Assets… (+13 more)

### Community 15 - "run_temporal_fusion_forecast"
Cohesion: 0.14
Nodes (15): Any, DataFrame, ndarray, Temporal Fusion Transformer (TFT) & Multi-Horizon Self-Attention Engine for…, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with… (+7 more)

### Community 16 - "autonomous_trader.py"
Cohesion: 0.06
Nodes (45): execute_committee_order(), Executes a committee-approved buy order into the virtual paper broker ledger., AutonomousTradingEngine, check_daily_loss_circuit_breaker(), ensure_background_daemon_thread_running(), get_daemon_status(), is_kill_switch_active(), load_universe_tickers() (+37 more)

### Community 17 - "meta_ensemble.py"
Cohesion: 0.11
Nodes (19): DynamicSharpeMetaEnsemble, MetaEnsembleClassifier, DataFrame, ndarray, Series, Institutional Multi-Model Meta-Ensemble Engine for Sentilyze. Pillar 1 Core…, Generates binary class prediction (0 = Hold/Sell, 1 = Buy) using soft-voting…, Instantiates and fits the Meta-Ensemble classifier. (+11 more)

### Community 18 - "OnlineNewtonStepOptimizer"
Cohesion: 0.18
Nodes (10): OnlineNewtonStepOptimizer, DataFrame, ndarray, Polynomial-Time ONS Portfolio Engine (Hazan et al.)., Processes price relatives (r_t = Close_t / Close_{t-1}) and updates weights in…, Fast O(d log d) Euclidean projection onto probability simplex., Runs ONS sequence through time and outputs daily allocations and portfolio…, test_paper1_online_newton_step() (+2 more)

### Community 19 - "ws_live_prediction.py"
Cohesion: 0.15
Nodes (22): detect_classical_chart_patterns(), generate_ai_chart_explanation(), match_historical_chart_twins(), normalize_waveform(), Any, DataFrame, ndarray, AI Chart Pattern Recognition, Geometric Wave Learning & Visual Understanding… (+14 more)

### Community 20 - "draw.io XML Style Reference"
Cohesion: 0.08
Nodes (23): Color Properties, Common Style Properties, Container Types, Critical Edge Rules, draw.io XML Style Reference, Edge Attributes, Edge Styles, Escaping Rules (XML attribute context) (+15 more)

### Community 21 - "compute_dark_pool_sentiment"
Cohesion: 0.21
Nodes (14): compute_dark_pool_sentiment(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent institutional off-exchange block trades and dark pool prints., Scans option chain contracts where daily volume significantly exceeds open…, Synthesizes dark pool prints and unusual options flow into a unified…, scan_abnormal_options_vol_oi(), scan_dark_pool_blocks() (+6 more)

### Community 22 - "test_omnichannel_mobile.py"
Cohesion: 0.15
Nodes (16): answer_financial_query(), Any, Natural Language Financial Q&A Agent for Sentilyze. Pillar 7 Mobile &…, Parses natural language questions and routes them to quantitative engines.…, generate_smartwatch_glance_payload(), Any, Generates structured complication JSON for Apple Watch (watchOS) and Wear OS., format_whatsapp_trade_alert() (+8 more)

### Community 23 - "pfd-engineering-expert.md"
Cohesion: 0.10
Nodes (20): 1. Distillation Column (`distillation_column`), 1. `PHASE_PORT_VIOLATION`, 2. `DEAD_END_STREAM`, 2. Pump (`pump`), 3. Compressor (`compressor`), 3. `OPPOSING_FLOW`, 4. `GRAVITY_VIOLATION`, 4. Separator (`separator`) (+12 more)

### Community 24 - "preprocess_data"
Cohesion: 0.13
Nodes (19): clean_headline_data(), _get_api_key(), _load_sentiment_analyzer(), preprocess_data(), Any, DataFrame, Cleans a headline CSV file by removing rows with invalid stock tickers. Caches…, Safely attempts to retrieve the API key from Streamlit secrets, falling back to… (+11 more)

### Community 25 - "compute_lead_lag_matrix"
Cohesion: 0.20
Nodes (14): compute_lead_lag_matrix(), _granger_f_test(), Any, DataFrame, ndarray, Series, rank_market_price_leaders(), Lead-Lag Granger Causality & Supply Chain Price Discovery Engine for Sentilyze.… (+6 more)

### Community 26 - "black_swan_simulator.py"
Cohesion: 0.23
Nodes (11): calculate_kelly_sizing(), estimate_market_impact_slippage(), Any, Historical Black Swan Crisis Simulator & Kelly Position Sizing for Sentilyze.…, Calculates optimal position sizing using the Kelly Criterion: Kelly % = W - (1…, Estimates market execution slippage using the Almgren-Chriss square-root impact…, Stress-tests the current portfolio against major historical market crashes.…, simulate_portfolio_crises() (+3 more)

### Community 27 - "render_workspace_header"
Cohesion: 0.13
Nodes (19): Cockpit 5: Deep Quant & Explainability (XAI)…, Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Renders an executive header banner with live status badge and market clock., Wraps HTML content inside an institutional frosted glass container., render_glass_card(), render_workspace_header(), _get_cached_forensic(), _get_cached_gnn() (+11 more)

### Community 28 - "AlpacaBrokerBridge"
Cohesion: 0.16
Nodes (14): AlpacaBrokerBridge, Any, Fetches active positions from Alpaca brokerage., Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.…, Verifies active connection to Alpaca Brokerage API., Fetches live Alpaca account equity, buying power, and cash., Submits an institutional Bracket Order: - Entry: Market order - Exit 1: Limit…, patch (+6 more)

### Community 29 - "PaperBroker"
Cohesion: 0.06
Nodes (50): AI Trade Copilot & Conversational Analyst for Sentilyze. Provides natural…, handle_bot_command(), Any, Executes command and posts formatted embed reply to Discord., Parses and processes interactive bot commands: - `/signal <ticker>` -…, send_bot_command_reply(), Any, Paper 25 Live Runner: Opening Range Breakout (ORB) on Top Stocks in Play.… (+42 more)

### Community 30 - "._save"
Cohesion: 0.12
Nodes (12): Any, Executes daily quantitative scan results using the Concentrated Top-2 + Scale-…, Loads existing portfolio state from JSON or initializes a fresh account if file…, Updates total equity, unrealized PnL, and win rates., Reloads latest state from disk if file exists., Returns high-level KPI metrics for the portfolio dashboard., Executes an institutional BUY order into the virtual paper broker ledger.…, Executes an immediate manual live/simulated BUY order from UI. (+4 more)

### Community 31 - "get_us_market_session"
Cohesion: 0.12
Nodes (24): check_market_hours_preflight(), get_current_ny_time(), get_us_market_session(), Any, datetime, Unified US Stock Market (NYSE / NASDAQ) Session & Calendar Engine for…, Pre-flight sanity check for automated workflows. Returns True if execution…, Returns the current precise timestamp in America/New_York (Eastern Time). (+16 more)

### Community 32 - "ws_alternative_data.py"
Cohesion: 0.13
Nodes (22): auto_register_ipo_ticker(), fetch_pre_ipo_radar_summary(), fetch_sec_edgar_ipo_filings(), get_pre_ipo_pipeline_df(), Any, DataFrame, IPO & Pre-IPO Intelligence Radar for Sentilyze. Pillar 9 Alternative Asset…, Fetches real-time SEC Form S-1 / S-1/A IPO registration statements from SEC… (+14 more)

### Community 33 - "3. Cross-Functional Flowcharts (Actor × Phase Grid)"
Cohesion: 0.10
Nodes (19): 1. Flat Swimlanes (BPMN-style Flowcharts), 2. Nested Architecture Containers, 3. Cross-Functional Flowcharts (Actor × Phase Grid), 4. When to Use What, Container Key Rules Summary, Critical Rules, Critical Rules, Decision Guide (+11 more)

### Community 34 - "calculate_hrp_weights"
Cohesion: 0.13
Nodes (20): compute_performance_metrics(), Any, DataFrame, Series, Empirical Quant Experimentation & Multi-Asset Ablation Benchmark Suite.…, Executes empirical ablation benchmark across the full asset universe., Simulates walk-forward strategy execution with or without advanced quant…, Computes key quant performance metrics. (+12 more)

### Community 35 - "Edge Routing Decision Guide"
Cohesion: 0.12
Nodes (16): Critical Rules, Decision Flowchart, Default Behavior (Built-in Router), Don't add waypoints, Don't hand-route, Edge labels, Edge Routing Decision Guide, Every edge needs an mxGeometry child (+8 more)

### Community 36 - "config.py"
Cohesion: 0.07
Nodes (31): Any, DataFrame, ndarray, Series, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.…, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used. (+23 more)

### Community 37 - "triple_convex_engine.py"
Cohesion: 0.14
Nodes (13): PolyTimeConvexOptimizer, Any, DataFrame, Series, Polynomial-Time Convex Portfolio Optimizer with Market Frictions (Boyd et al.)., Solves the friction-aware convex optimization problem in polynomial time. Args:…, Triple-Convex Quantum Execution Engine. Fuses the top quantitative research…, Unified High-Expectancy, Minimum-Drawdown, Sub-15ms Execution Engine. (+5 more)

### Community 38 - "Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform"
Cohesion: 0.12
Nodes (16): 1. Run in 1-Click (No Installation Required), 2. Local Setup & Installation, 3. Run the Quantitative CLI, 4. Launch the Streamlit Mission Control, 5. Run Full Test Suite (274+ Unit Tests), Active Holdings (Entered 2026-09-08), 📊 Empirical Alpha Attribution & Benchmarks, 🏛️ Grounded 5-Agent Deliberation Council (+8 more)

### Community 39 - "get_sector_for_ticker"
Cohesion: 0.13
Nodes (19): calculate_portfolio_correlation_matrix(), check_correlation_shield(), Any, DataFrame, Portfolio Correlation Matrix Shield. Functions: - Calculates rolling 21-day…, Computes the 21-day rolling pairwise return correlation matrix across tickers., Audits a candidate buy against currently held positions. Returns whether the…, build_pooled_sector_dataset() (+11 more)

### Community 40 - "[Validator Rules Reference & Troubleshooting]"
Cohesion: 0.12
Nodes (16): 1. `DIRECT_WAN_TO_LAN`, 2. `ORPHAN_DEVICE`, 3. `VLAN_LEAK`, 4. `REDUNDANCY_WARNING`, 5. `MISSING_DMZ`, 6. `FLAT_NETWORK`, 7. `FIREWALL_BYPASS`, 8. `SINGLE_TRUNK` (+8 more)

### Community 41 - "test_zero_giveback_profit_lock.py"
Cohesion: 0.06
Nodes (49): apply_high_watermark_profit_lock(), calculate_smart_money_zones(), calculate_structural_trailing_stop(), evaluate_multi_timeframe_confluence(), find_swing_pivots(), Any, DataFrame, Institutional Smart Money Market Structure & Price-Action Engine for Sentilyze.… (+41 more)

### Community 42 - "RegimeMixtureOfExperts"
Cohesion: 0.26
Nodes (8): Any, DataFrame, ndarray, Series, 3-Expert Gated Mixture of Experts classifier for financial regimes., Labels historical rows into 3 latent market regimes: 0 = Bull Momentum (Price >…, RegimeMixtureOfExperts, test_regime_moe()

### Community 43 - ".split"
Cohesion: 0.50
Nodes (3): DataFrame, ndarray, Series

### Community 47 - "Sentilyze Community Code of Conduct"
Cohesion: 0.25
Nodes (7): 📜 Attribution, ⚖️ Enforcement Escalation, 🛡️ Enforcement & Responsibilities, 🌟 Our Pledge, 🎯 Our Standards, 📬 Reporting Guidelines, Sentilyze Community Code of Conduct

### Community 48 - "universal_web_scraper.py"
Cohesion: 0.13
Nodes (26): batch_scrape_urls(), detect_stock_tickers(), extract_financial_tables_from_html(), _extract_summary_sentences(), Any, DataFrame, Universal Web Scraper & Financial Intelligence Extractor for Sentilyze.…, Extracts stock ticker symbols from unstructured text using cashtags ($TICKER)… (+18 more)

### Community 49 - "calculate_doubling_progress"
Cohesion: 0.27
Nodes (9): calculate_doubling_progress(), compute_compound_position_size(), Any, Max Compound Acceleration & +100% Target Doubling Engine for Sentilyze.…, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Unit tests for Max Compound Acceleration Engine., test_calculate_doubling_progress() (+1 more)

### Community 50 - "test_pillar2_alternative_data.py"
Cohesion: 0.15
Nodes (22): compute_smart_money_insider_score(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent SEC Form 4 insider transactions for a given stock., Retrieves recent Congressional STOCK Act disclosure reports for a ticker., Synthesizes SEC Form 4 and Congressional activity into an overall Smart Money…, track_congressional_stock_disclosures(), track_corporate_insider_filings() (+14 more)

### Community 51 - "price_scout.py"
Cohesion: 0.20
Nodes (11): get_latest_scout_alerts(), PriceActionScoutAgent, PriceScoutBot, Any, Real-Time Price Action & Tape-Reading Scout Subagent (Bot) for Sentilyze.…, Continuous Background Scanner Bot that scouts the 538 universe assets, detects…, Scans given tickers for real-time volume breakout candidates., Retrieves the latest price scout breakout alerts. (+3 more)

### Community 52 - "test_cross_disciplinary_alphas.py"
Cohesion: 0.08
Nodes (63): compute_ant_colony_venue_allocation(), compute_bayesian_dark_pool_equilibrium(), compute_carnot_profit_efficiency(), compute_circadian_seasonal_risk_scalar(), compute_conformal_safety_bands(), compute_doppler_order_flow_shift(), compute_epigenetic_factor_methylation(), compute_feynman_path_least_action() (+55 more)

### Community 53 - "Contributing to Sentilyze"
Cohesion: 0.15
Nodes (12): 1. Code Formatting (Black), 1. Fork & Clone, 2. Linting (Flake8), 2. Virtual Environment & Dependencies, 3. Security Auditing (Bandit), 4. Unit Test Suite (Pytest), Contributing to Sentilyze, 🏛️ Core Project Architecture & Contribution Rules (+4 more)

### Community 54 - "ws_committee.py"
Cohesion: 0.22
Nodes (11): create_candlestick_sr_chart(), DataFrame, Figure, Automated Dynamic Pivot Support & Resistance Charting Component for Streamlit.…, Constructs an institutional-grade interactive Plotly chart with automated S/R…, _get_cached_sr_chart(), _load_cached_committee_resolution(), cache_data (+3 more)

### Community 55 - "🧪 Experimental & Simulated Research Prototypes"
Cohesion: 0.50
Nodes (3): 🧪 Experimental & Simulated Research Prototypes, 🔒 Production Isolation Guarantee, 📁 Prototype Inventory

### Community 56 - "Sentilyze — Standing Audit Protocol"
Cohesion: 0.15
Nodes (12): 10. Strict Portfolio Preservation & Realized Gain Integrity, 1. Fabricated / fake data check, 2. Ticker/input-invariant bugs, 3. Mislabeled methodology, 4. Results-file / README consistency, 5. Duplicate-output smell test, 6. Safety-critical logic sanity check, 7. Silent failure check (+4 more)

### Community 57 - "generate_morning_briefing_text"
Cohesion: 0.08
Nodes (40): generate_morning_briefing_text(), get_portfolio_intelligence(), load_universe_candidates(), Any, Reads live paper portfolio state for broadcast reporting., Assembles a comprehensive, institutional Wall Street Morning Podcast and…, Synthesizes broadcast audio podcast (.mp3) using Google Text-to-Speech (gTTS)…, Loads clean ticker list from stocks.txt or falls back to core liquid universe. (+32 more)

### Community 58 - "calculate_time_decayed_sentiment"
Cohesion: 0.19
Nodes (14): calculate_time_decayed_sentiment(), compute_exponential_decay_weights(), Any, DataFrame, datetime, ndarray, Series, Temporal Sentiment Half-Life Decay Engine. Functions: - Applies continuous… (+6 more)

### Community 59 - "erd-database-expert.md"
Cohesion: 0.13
Nodes (14): 1. `FK_WITHOUT_TARGET`, 2. `ORPHAN_TABLE`, 3. `DUPLICATE_PK`, 4. `SELF_REFERENCE_MISSING`, 5. `MISSING_JUNCTION`, 6. `CIRCULAR_FK`, [Anti-Patterns], [Docs Index] (+6 more)

### Community 60 - "daily_scanner.py"
Cohesion: 0.19
Nodes (13): Sends a rich formatted trade alert card to a Discord channel via Webhook., Sends a consolidated master market digest card containing all universe signals., send_discord_alert(), send_discord_digest(), Scans the entire stock universe defined in stocks.txt, generates tomorrow's…, run_daily_market_scan(), Any, Dispatches formatted HTML morning market digest email via Gmail SMTP. (+5 more)

### Community 61 - "6. CROSS-DISCIPLINARY & FRONTIER QUANTITATIVE IDEAS"
Cohesion: 0.07
Nodes (27): 1. EXECUTIVE SUMMARY & CORE PLATFORM VISION, 2. THE 4-PILLAR WINNING QUANTITATIVE ALPHA FORMULA, 3. EMPIRICAL MULTI-DECADE HISTORICAL BENCHMARKS (1980–2026), 4. MASTER 5-COCKPIT MISSION CONTROL ARCHITECTURE, 5. THE 300 INSTITUTIONAL RESOURCES SYNTHESIS, 6. CROSS-DISCIPLINARY & FRONTIER QUANTITATIVE IDEAS, 7. IMPLEMENTATION & TECHNOLOGY STACK BLUEPRINT, 🐺 A. Evolutionary Biology: Lotka-Volterra Predator-Prey Market Dynamics (+19 more)

### Community 64 - "GaussianHMMRegimeDetector"
Cohesion: 0.13
Nodes (11): GaussianHMMRegimeDetector, Any, DataFrame, ndarray, Paper 15: Gaussian Hidden Markov Model for Market Regime Detection. Source:…, 3-state Gaussian HMM regime classifier: Bull / Normal / Crisis. Uses hand-…, Gaussian emission probability for each state., Forward algorithm step: update filtered state probabilities with one new… (+3 more)

### Community 65 - "EWMACorrelationMonitor"
Cohesion: 0.14
Nodes (10): EWMACorrelationMonitor, Any, DataFrame, ndarray, Paper 17: RiskMetrics EWMA Volatility & Correlation Monitor. Source: J.P.…, Real-time EWMA-based correlation and volatility monitor. Tracks time-varying…, Initialize EWMA state from a seed window of returns., Update EWMA state with one day's returns across all assets. Returns current… (+2 more)

### Community 67 - "get_sentiment"
Cohesion: 0.14
Nodes (20): analyze_sentiment(), clean_financial_text(), get_sentiment(), _parse_analyzer_output(), Any, DataFrame, Analyzes the sentiment of news articles using high-precision FinBERT pipeline,…, Cleans raw financial headlines/descriptions by stripping boilerplate publisher… (+12 more)

### Community 68 - "CUSUMDetector"
Cohesion: 0.15
Nodes (9): CUSUMDetector, Any, ndarray, Paper 16: CUSUM Sequential Change-Point Detection. Source: E.S. Page (1954) —…, Cumulative Sum detector for online mean-shift detection. Monitors a numeric…, Process one observation. Returns alarm status., Process a batch of observations., Reset detector state. (+1 more)

### Community 69 - "PageHinkleyDetector"
Cohesion: 0.15
Nodes (9): PageHinkleyDetector, Any, ndarray, Paper 22: Page-Hinkley Sequential Test for Concept Drift Detection. Source:…, Page-Hinkley test for detecting changes in the mean of a stream. Monitors…, Process one observation. Returns drift status., Process a batch of observations., Reset detector state. (+1 more)

### Community 70 - "QuantAlphaDAGEngine"
Cohesion: 0.08
Nodes (30): AlphaDAGStageContract, Any, DataFrame, Series, QuantAlphaDAGEngine, Calculates time-aligned returns, normalized volumes, and volatility bands., Generates Gen-3 Multi-Horizon Feature Library: - Factor 1: Triple-Horizon Trend…, Decoupled Alpha Generators with Gen-3 FinBERT Crash Shield & Volatility Sizing. (+22 more)

### Community 71 - "preprocessing.py"
Cohesion: 0.17
Nodes (19): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, DataFrame (+11 more)

### Community 72 - "screener_engine.py"
Cohesion: 0.18
Nodes (13): Real-Time Market Anomaly Screener Component for Streamlit. Functions: - Renders…, fetch_universe_live_quotes(), Fetches real-time quotes across universe with fast batching (sub-2s)., evaluate_single_asset_screener(), Any, DataFrame, Real-Time Market Anomaly Screener Engine. Functions: - Rapid multi-condition…, Executes parallel multi-condition anomaly screening across a universe of… (+5 more)

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
Cohesion: 0.20
Nodes (10): calculate_cdar(), optimize_cdar_portfolio(), Any, DataFrame, ndarray, Paper 19: Conditional Drawdown-at-Risk (CDaR) Portfolio Optimization. Source:…, Calculate Conditional Drawdown-at-Risk: the expected drawdown in the worst…, Optimize portfolio weights to minimize CDaR. Simplified approach: compute CDaR… (+2 more)

### Community 78 - "sanitize_filename"
Cohesion: 0.11
Nodes (27): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+19 more)

### Community 79 - "run_cppi_backtest"
Cohesion: 0.23
Nodes (8): calculate_cppi_allocation(), Any, ndarray, Paper 20: Constant Proportion Portfolio Insurance (CPPI). Source: Black & Jones…, CPPI allocation: Exposure = M * (Portfolio - Floor). Args: portfolio_value:…, Run a full CPPI backtest over a return series. Args: returns: Array of daily…, run_cppi_backtest(), TestCPPI

### Community 80 - "DCCCorrelation"
Cohesion: 0.20
Nodes (7): DCCCorrelation, Any, DataFrame, Paper 24: Dynamic Conditional Correlation (DCC-GARCH). Source: Engle (2002) —…, Simplified DCC model using EWMA-GARCH(1,1) for individual volatilities and DCC…, Fit the DCC model to a returns DataFrame. Returns time-varying correlation…, TestDCC

### Community 81 - "test_reddit_premarket_station.py"
Cohesion: 0.18
Nodes (19): fetch_4station_premarket_intelligence(), fetch_8station_premarket_intelligence(), fetch_all_reddit_headlines_for_ticker(), _fetch_subreddit_rss_entries(), Any, DataFrame, Systematic 8-Station Multi-Channel Reddit Market Intelligence Engine. Pillar 2…, Fetches real-time Atom RSS feed for a subreddit using safe defusedxml. (+11 more)

### Community 82 - "test_rebalancer_and_tearsheet.py"
Cohesion: 0.13
Nodes (18): calculate_custom_rebalance(), calculate_share_allocation(), Any, Helper to calculate share allocation from latest daily signals file or universe…, Computes exact whole-share buy allocations for a given capital budget across…, Any, DataFrame, Helper wrapper for Monte Carlo VaR simulation. (+10 more)

### Community 83 - "risk_constrained_kelly_allocation"
Cohesion: 0.24
Nodes (6): Any, ndarray, Paper 23: Risk-Constrained Kelly Gambling. Source: Busseti, Ryu, Boyd —…, Compute optimal Kelly allocation with drawdown probability constraint.…, risk_constrained_kelly_allocation(), TestRiskKelly

### Community 84 - "benchmark_training_paradigms.py"
Cohesion: 0.21
Nodes (25): Any, DataFrame, Series, Triple-Barrier Meta-Labeling (Upper +2.0 ATR, Lower -1.2 ATR, Vertical 10d)., Regime-Conditioned Mixture of Experts (MoE) with 3 specialized sub-boosters., Online Streaming Incremental Learning with Stochastic Gradient Descent &…, Standard Rolling Walk-Forward Optimization (Fixed 500-day window)., Cross-Asset Pooled Multi-Task Learning across peer assets for high statistical… (+17 more)

### Community 85 - "test_tri_model_pipeline.py"
Cohesion: 0.16
Nodes (15): Financial Event & Emotion Companion Model (Model 2). Companion model to FinBERT…, get_fast_agent_pipeline(), Fast Sub-Second Agent Decision Router. Coordinates the Tri-Model Neural Brain…, Singleton getter for FastAgentPipeline., get_fast_finbert_engine(), Accelerated Low-Latency FinBERT Engine (Model 1). Optimized for high-speed…, Singleton getter for FastFinBERTEngine., NeuralGatingNetwork (+7 more)

### Community 86 - "agent_committee.py"
Cohesion: 0.11
Nodes (28): audit_full_universe_committee(), ChiefRiskOfficerAgent, compute_fractional_kelly_sizing(), convene_trading_committee(), ForensicFundamentalAgent, _persist_committee_resolution(), Any, Autonomous Multi-Agent Trading Committee & Deliberation Engine for Sentilyze.… (+20 more)

### Community 87 - "ws_quantum_tournament.py"
Cohesion: 0.16
Nodes (16): Cockpit 4: Multi-Decade Backtest & Performance Factsheet…, render_backtest_cockpit(), Workspace 6: Walk-Forward Backtesting & Performance Tearsheet., Renders the Walk-Forward Backtesting and Strategy Tearsheet workspace., render_backtesting_workspace(), _get_cached_factsheet(), cache_data, Workspace: Institutional Risk & Alpha Performance Factsheet. Quantitative… (+8 more)

### Community 88 - "calculate_beneish_m_score"
Cohesion: 0.27
Nodes (9): analyze_debt_maturity_wall(), calculate_beneish_m_score(), Any, DataFrame, Beneish M-Score Forensic Analyzer & Debt Maturity Wall Radar for Sentilyze.…, Evaluates corporate interest coverage and debt maturity wall runway., Computes the 8-Ratio Beneish M-Score from 2-year comparative SEC financial…, test_beneish_m_score() (+1 more)

### Community 89 - "correlation_matrix.py"
Cohesion: 0.38
Nodes (6): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…

### Community 90 - "generate_pipeline_graph_data"
Cohesion: 0.27
Nodes (9): generate_pipeline_graph_data(), Any, Interactive Multi-Agent & Pipeline Architecture Canvas Component for Streamlit.…, Renders the interactive Vis.js animated node network canvas inside Streamlit., Constructs the nodes and edges representation for the pipeline graph canvas., render_pipeline_topology_canvas(), Tests for Interactive Multi-Agent & Pipeline Architecture Canvas…, test_generate_pipeline_graph_data_defaults() (+1 more)

### Community 91 - "train_model"
Cohesion: 0.20
Nodes (15): get_prediction_on_latest_data(), Any, DataFrame, Series, Gets a prediction from the model for the latest available data point. Args:…, Train the XGBoost model using Combinatorial Purged & Embargoed Cross-Validation…, train_model(), Generates sample data for testing. Needs at least 520 rows for WFO… (+7 more)

### Community 92 - "block_external_alerts"
Cohesion: 0.50
Nodes (3): block_external_alerts(), fixture, Autouse fixture that prevents tests from sending real outbound network calls to…

### Community 93 - "temp_data_dir"
Cohesion: 0.67
Nodes (3): fixture, Fixture to set a temporary data directory for tests., temp_data_dir()

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

### Community 99 - ".get_closed_trades_df"
Cohesion: 0.29
Nodes (4): DataFrame, Returns a DataFrame of current open holdings with Scale-Out status., Returns a DataFrame of trade history with full company names., Returns equity history as a DatetimeIndex DataFrame.

### Community 100 - "calculate_portfolio_diversity_grade"
Cohesion: 0.20
Nodes (12): calculate_portfolio_diversity_grade(), Any, DataFrame, Cockpit 3: Portfolio & Risk Optimization…, _get_cached_diversity_grade(), cache_data, Workspace: Portfolio Diversity & Correlation Health Grader. Institutional…, render_portfolio_diversity_workspace() (+4 more)

### Community 101 - "cli.py"
Cohesion: 0.29
Nodes (10): Root entry point for Sentilyze CLI. Run directly with: python sentilyze.py NVDA…, cmd_audit(), cmd_briefing(), cmd_portfolio(), main(), print_banner(), Sentilyze Command-Line Interface (CLI). Interactive terminal tool for 4-Agent…, Displays current portfolio metrics and live holdings. (+2 more)

### Community 102 - "Domain Expert Reference Template"
Cohesion: 0.15
Nodes (12): [Anti-Patterns], [Docs Index], Domain Expert Reference Template, [Domain Rules + Patterns], [Domain-Specific Catalog / Patterns] (OPTIONAL), [Project Context], [Project Conventions], Section Checklist (+4 more)

### Community 103 - "social_sentiment.py"
Cohesion: 0.23
Nodes (13): calculate_social_buzz_metrics(), fetch_social_sentiment_tracker(), Any, Social Sentiment Velocity & Retail Multi-Platform Scraper for Sentilyze. Pillar…, Scrapes real-time streaming retail sentiment from Stocktwits public symbol…, Scrapes tech community discussions on AI catalysts (OpenAI, Anthropic, Nvidia)…, Computes retail sentiment velocity and flow conviction metrics., High-level entry point to retrieve calibrated real-time social buzz metrics for… (+5 more)

### Community 104 - "3. Equipment Mapping (Draw.io Native Shapes)"
Cohesion: 0.15
Nodes (12): 1. ISA 5.1 Instrument Tagging Conventions, 2. Line Styles, 3. Equipment Mapping (Draw.io Native Shapes), 4. Diagram Layout Strategy (PFDs), Common Tag Identifiers, Heat Exchangers, Instrument Shapes (`mxgraph.pid2inst.*`), Mining & Minerals Processing (+4 more)

### Community 105 - "test_acpm_trainer.py"
Cohesion: 0.13
Nodes (19): Unified Alpha-Conformal Purged Multi-Task (ACPM) Quantitative Training Engine.…, Conformal Probability Calibration & Quantile Uncertainty Layer. Maps raw tree…, neutralize_features(), neutralize_predictions(), DataFrame, ndarray, Series, Feature & Factor Neutralization for Quantitative Machine Learning.… (+11 more)

### Community 106 - "ws_insider_radar.py"
Cohesion: 0.13
Nodes (23): Renders the Real-Time Market Anomaly Screener UI., render_live_screener_section(), calculate_insider_conviction_score(), fetch_insider_transactions(), Any, Smart-Money Executive & Institutional Insider Radar for Sentilyze.…, Computes the Quantitative Insider Conviction Index (0 to 100 Score) and detects…, Screens a universe of tickers and returns the highest-ranking insider buying… (+15 more)

### Community 107 - "ablation_study.py"
Cohesion: 0.33
Nodes (8): _persist_ablation_results(), Any, 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), test_committee_ablation_study_execution()

### Community 108 - "Agent Browser — Live Web Automation Skill"
Cohesion: 0.50
Nodes (3): 1. Core Principles, 2. Use Cases in Sentilyze, Agent Browser — Live Web Automation Skill

### Community 109 - "Agent Memory — Persistent Multi-Session Memory Skill"
Cohesion: 0.50
Nodes (3): 1. Storage Architecture, 2. Integration Pattern, Agent Memory — Persistent Multi-Session Memory Skill

### Community 113 - "generate_institutional_pdf_tearsheet"
Cohesion: 0.31
Nodes (7): generate_institutional_pdf_tearsheet(), Institutional Quantitative Tearsheet & Factsheet Generator for Sentilyze.…, Generates a publication-grade institutional quantitative tearsheet PDF. Returns…, Verify that PDF file is saved correctly to a specified path., Verify that PDF generation creates valid, non-empty binary content., test_generate_institutional_pdf_tearsheet_bytes(), test_generate_institutional_pdf_tearsheet_file()

### Community 114 - "aws-well-architected-reviewer.md"
Cohesion: 0.22
Nodes (8): [Anti-Patterns], [Docs Index], [Domain Rules + Patterns], [Project Context], [Project Conventions], [Sequential Execution Pairing (MANDATORY)], [Visual Styling], [XML Graph DOM Rules]

### Community 115 - "gcp-well-architected-reviewer.md"
Cohesion: 0.22
Nodes (8): [Anti-Patterns], [Docs Index], [Domain Rules + Patterns], [Project Context], [Project Conventions], [Sequential Execution Pairing (MANDATORY)], [Visual Styling], [XML Graph DOM Rules]

### Community 116 - "run_universe_training.py"
Cohesion: 0.24
Nodes (11): load_universe_from_file(), main(), prefetch_universe_data(), Any, Universal Parallel Multi-Core Model Training & High-Resolution Benchmark.…, Loads cleaned ticker symbols from stocks.txt., Worker function to train a single asset., Pre-fetches market and news data in parallel across fast I/O threads. (+3 more)

### Community 117 - "generate_comprehensive_factsheet"
Cohesion: 0.38
Nodes (6): generate_comprehensive_factsheet(), Any, Series, Computes over 30 institutional hedge-fund risk, performance, and drawdown…, test_generate_comprehensive_factsheet_custom_series(), test_generate_comprehensive_factsheet_default()

### Community 118 - "FastNeuralEventClassifier"
Cohesion: 0.29
Nodes (4): FastNeuralEventClassifier, Tensor, Builds a lightweight internal token dictionary for financial catalysts., Lightweight 2-layer Neural Event & Emotion Head for sub-millisecond…

### Community 119 - "ConformalCalibrator"
Cohesion: 0.12
Nodes (14): ACPMTrainer, Any, DataFrame, Series, State-of-the-Art 10x Institutional Quantitative Training Engine., Executes end-to-end ACPM training for a target equity., ConformalCalibrator, ndarray (+6 more)

### Community 120 - "EventClassifierModel"
Cohesion: 0.24
Nodes (8): EventClassifierModel, Any, Translates raw social slang, emojis, and cashtags into standardized financial…, Classifies a single headline or social post with sub-millisecond speed.…, Batch-processes multiple headlines with vectorized execution., Production-grade Financial Event & Emotion Companion Model. Provides sub-…, test_event_classifier_fraud_and_dilution(), test_event_classifier_model()

### Community 121 - "simulate_strategy_sandbox"
Cohesion: 0.67
Nodes (3): Any, Fast vectorized backtesting simulation sandbox for custom leverage, confidence,…, simulate_strategy_sandbox()

### Community 122 - "fractional_differentiation_ffd"
Cohesion: 0.29
Nodes (10): find_optimal_d(), fractional_differentiation_ffd(), get_weights_ffd(), ndarray, Series, Fixed-Width Window Fractional Differentiation (FFD) for Financial Time Series.…, Generate weights for Fixed-Width Window Fractional Differentiation. w_0 = 1 w_k…, Applies Fixed-Width Window Fractional Differentiation to a price series. (+2 more)

### Community 125 - "run_unified_institutional_pipeline"
Cohesion: 0.21
Nodes (11): Any, Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), analyze_sec_filing_diff(), compute_text_similarity_and_diff(), Any, Computes lexical and semantic diff metrics between consecutive filings., Retrieves and compares the most recent and prior SEC filings for a company.… (+3 more)

### Community 126 - "FastFinBERTEngine"
Cohesion: 0.27
Nodes (5): FastFinBERTEngine, Any, Vectorized micro-batch inference for multiple headlines., Accelerated FinBERT inference engine delivering 2.5x - 4x speedups., Runs accelerated sentiment scoring on a single headline. Includes Companion…

### Community 128 - "test_ui_modules.py"
Cohesion: 0.27
Nodes (7): load_cached_tournament_results(), cache_data, Workspace 25: Autonomous Quant Alpha DAG Engine (Gen-3 Champion). Institutional…, Loads pre-computed 3-generation tournament results if available., Renders the comprehensive Quant Alpha DAG Engine workspace., render_alpha_dag_workspace(), test_ws_alpha_dag_module_import()

### Community 129 - "render_multi_agent_war_room"
Cohesion: 0.28
Nodes (7): Any, Interactive Multi-Agent War Room Visualizer Component for Streamlit. Functions:…, Renders the complete 5-Agent War Room Council deliberation chamber., render_multi_agent_war_room(), Real-Time Audio Trade Squawk Component for Streamlit. Functions: - Uses…, Renders an HTML5 Web Speech API audio squawk generator inside Streamlit., render_audio_squawk_button()

### Community 131 - "TriBrainGatingLayer"
Cohesion: 0.40
Nodes (3): Tensor, Neural Softmax Gating layer to dynamically allocate weights among the 3 models.…, TriBrainGatingLayer

### Community 134 - "DLinearTCNModel"
Cohesion: 0.13
Nodes (12): DLinearTCNModel, predict_momentum_probability(), Tensor, Evaluates a single or batched sequence tensor and returns calibrated momentum…, Decomposes a time-series sequence into Trend and Seasonal/Residual components…, Dual-Branch Deep Learning Architecture: - Branch 1 (Trend): Linear projection…, SeriesDecomposition, FastAgentPipeline (+4 more)

### Community 138 - "audio_briefing.py"
Cohesion: 0.47
Nodes (5): generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses Microsoft Edge Neural TTS…, synthesize_morning_audio()

### Community 139 - "apply_triple_barrier_labeling"
Cohesion: 0.15
Nodes (14): apply_triple_barrier_labeling(), calculate_deflated_sharpe_ratio(), DataFrame, Series, Computes Bailey & López de Prado's Deflated Sharpe Ratio (DSR). Adjusts for: -…, Applies López de Prado's path-dependent Triple-Barrier Method to generate trade…, Any, DataFrame (+6 more)

### Community 140 - "_get_browser_session"
Cohesion: 0.33
Nodes (5): _get_browser_session(), Session, Universal Adaptive Web Scraper supporting Scrapling, Trafilatura, and…, Creates a configured requests session with realistic browser headers., UniversalWebScraper

## Knowledge Gaps
- **224 isolated node(s):** `graphify`, `1. Core Principles`, `2. Use Cases in Sentilyze`, `1. Storage Architecture`, `2. Integration Pattern` (+219 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `utils.py` to `deep_learning_model.py`, `get_price_history`, `quant_engine.py`, `alerts.py`, `run_backtest`, `test_statistical_arbitrage.py`, `drl_policy_agent.py`, `fetch_financial_statements`, `TradingEnvironment`, `audio_briefing.py`, `CloudDataLake`, `analyze_supply_chain_spillover`, `run_temporal_fusion_forecast`, `autonomous_trader.py`, `meta_ensemble.py`, `ws_live_prediction.py`, `compute_dark_pool_sentiment`, `test_omnichannel_mobile.py`, `compute_lead_lag_matrix`, `black_swan_simulator.py`, `AlpacaBrokerBridge`, `PaperBroker`, `get_us_market_session`, `ws_alternative_data.py`, `calculate_hrp_weights`, `config.py`, `triple_convex_engine.py`, `get_sector_for_ticker`, `test_zero_giveback_profit_lock.py`, `universal_web_scraper.py`, `calculate_doubling_progress`, `test_pillar2_alternative_data.py`, `price_scout.py`, `generate_morning_briefing_text`, `calculate_time_decayed_sentiment`, `daily_scanner.py`, `get_sentiment`, `QuantAlphaDAGEngine`, `preprocessing.py`, `screener_engine.py`, `sanitize_filename`, `test_reddit_premarket_station.py`, `benchmark_training_paradigms.py`, `test_tri_model_pipeline.py`, `agent_committee.py`, `calculate_beneish_m_score`, `correlation_matrix.py`, `liquidity_heatmap.py`, `cli.py`, `social_sentiment.py`, `test_acpm_trainer.py`, `ws_insider_radar.py`, `ablation_study.py`, `generate_institutional_pdf_tearsheet`, `run_universe_training.py`?**
  _High betweenness centrality (0.178) - this node is a cross-community bridge._
- **Why does `get_price_history()` connect `get_price_history` to `run_backtest`, `test_statistical_arbitrage.py`, `utils.py`, `drl_policy_agent.py`, `autonomous_trader.py`, `preprocess_data`, `PaperBroker`, `calculate_hrp_weights`, `config.py`, `get_sector_for_ticker`, `price_scout.py`, `ws_committee.py`, `generate_morning_briefing_text`, `QuantAlphaDAGEngine`, `preprocessing.py`, `screener_engine.py`, `sanitize_filename`, `agent_committee.py`, `correlation_matrix.py`, `AdversarialRedTeamAgent`, `calculate_portfolio_diversity_grade`, `ablation_study.py`, `run_universe_training.py`, `simulate_strategy_sandbox`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `PaperBroker` connect `PaperBroker` to `.get_closed_trades_df`, `calculate_portfolio_diversity_grade`, `cli.py`, `config.py`, `drl_policy_agent.py`, `test_zero_giveback_profit_lock.py`, `ws_insider_radar.py`, `autonomous_trader.py`, `agent_committee.py`, `._save`, `ws_quantum_tournament.py`, `generate_morning_briefing_text`, `daily_scanner.py`, `AICopilotEngine`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `1. Core Principles`, `2. Use Cases in Sentilyze` to the rest of the system?**
  _224 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `deep_learning_model.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1380952380952381 - nodes in this community are weakly interconnected._
- **Should `get_price_history` be split into smaller, more focused modules?**
  _Cohesion score 0.060109289617486336 - nodes in this community are weakly interconnected._