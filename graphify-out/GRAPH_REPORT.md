# Graph Report - Sentilyze  (2026-09-17)

## Corpus Check
- 1943 files · ~2,495,534 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2712 nodes · 5304 edges · 147 communities (140 shown, 7 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 50 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `3492ae93`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- deep_learning_model.py
- get_price_history
- test_options_flow.py
- daily_scanner.py
- run_backtest
- test_statistical_arbitrage.py
- get_sentiment
- drl_policy_agent.py
- FactorAttributionEngine
- TradingEnvironment
- OpeningRangeBreakout
- CloudDataLake
- app.py
- analyze_supply_chain_spillover
- calculate_hrp_weights
- run_temporal_fusion_forecast
- is_kill_switch_active
- meta_ensemble.py
- OnlineNewtonStepOptimizer
- test_all_14_papers.py
- draw.io XML Style Reference
- compute_dark_pool_sentiment
- test_omnichannel_mobile.py
- pfd-engineering-expert.md
- .query
- compute_lead_lag_matrix
- ws_live_prediction.py
- social_sentiment.py
- AlpacaBrokerBridge
- realtime_tracker.py
- test_onnx_and_orderflow.py
- get_us_market_session
- ipo_radar.py
- 3. Cross-Functional Flowcharts (Actor × Phase Grid)
- triple_convex_engine.py
- Edge Routing Decision Guide
- SuperEnsembleClassifier
- SECCatalystCrawler
- Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform
- correlation_shield.py
- [Validator Rules Reference & Troubleshooting]
- TickerSentinelSwarm
- RegimeMixtureOfExperts
- test_acpm_trainer.py
- ui/__init__.py
- Sentilyze Community Code of Conduct
- universal_web_scraper.py
- DuckDBMarketEngine
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
- ws_portfolio.py
- 6. CROSS-DISCIPLINARY & FRONTIER QUANTITATIVE IDEAS
- rules/graphify.md
- workflows/graphify.md
- GaussianHMMRegimeDetector
- EWMACorrelationMonitor
- analyze_sec_filing_diff
- CUSUMDetector
- PageHinkleyDetector
- QuantAlphaDAGEngine
- test_duckdb_engine.py
- preprocess_data
- [Validator Rules Reference & Troubleshooting]
- grossman_zhou_allocation
- ADWINDetector
- test_papers_15_24.py
- api.py
- chart_pattern_learning.py
- run_cppi_backtest
- DCCCorrelation
- test_reddit_premarket_station.py
- fetch_universe_live_quotes
- risk_constrained_kelly_allocation
- ws_alternative_data.py
- test_tri_model_pipeline.py
- agent_committee.py
- test_zero_giveback_profit_lock.py
- generate_comprehensive_factsheet
- correlation_matrix.py
- generate_pipeline_graph_data
- SecurityMaster
- block_external_alerts
- advanced_quant_experiments.py
- AICopilotEngine
- options_surface.py
- drawio/SKILL.md
- red_team_agent.py
- liquidity_heatmap.py
- PaperBroker
- calculate_portfolio_diversity_grade
- cli.py
- Domain Expert Reference Template
- preprocessing.py
- 3. Equipment Mapping (Draw.io Native Shapes)
- main
- ws_options_surface.py
- _get_browser_session
- Agent Browser — Live Web Automation Skill
- Agent Memory — Persistent Multi-Session Memory Skill
- Cybersecurity Skills — Institutional MLOps Security Protocol
- Editorial Diagram Design Skill
- Harness Engineering Skill
- MetaRegimeAllocator
- aws-well-architected-reviewer.md
- gcp-well-architected-reviewer.md
- run_universe_training.py
- modeling.py
- compute_gamma_exposure_profile
- ConformalCalibrator
- EventClassifierModel
- execute_continuous_retrain_cycle
- fractional_differentiation_ffd
- temp_data_dir
- render_live_prediction_workspace
- ws_deep_quant.py
- FastFinBERTEngine
- test_full_market_funnel.py
- components.py
- .split
- calculate_doubling_progress
- TriBrainGatingLayer
- black_swan_simulator.py
- autonomous_trader.py
- FastNeuralEventClassifier
- .fuse
- apply_high_watermark_profit_lock
- render_workspace_header
- utils.py
- print_formatted_trade_report
- screener_engine.py
- ws_insider_radar.py
- quant_engine.py
- render_multi_agent_war_room
- audio_briefing.py
- PolyTimeConvexOptimizer
- .evaluate_catalyst_and_market

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 110 edges
2. `get_price_history()` - 71 edges
3. `PaperBroker` - 63 edges
4. `fetch_live_quote()` - 44 edges
5. `get_news()` - 35 edges
6. `run_unified_institutional_pipeline()` - 32 edges
7. `AutonomousTradingEngine` - 30 edges
8. `preprocess_data()` - 29 edges
9. `render_workspace_header()` - 29 edges
10. `DuckDBMarketEngine` - 28 edges

## Surprising Connections (you probably didn't know these)
- `test_paper7_quant_agents_trader()` --calls--> `AutonomousTradingEngine`  [EXTRACTED]
  tests/test_all_14_papers.py → src/autonomous_trader.py
- `test_full_funnel_cycle_execution_safe()` --uses--> `AutonomousTradingEngine`  [INFERRED]
  tests/test_full_market_funnel.py → src/autonomous_trader.py
- `memory_engine()` --uses--> `DuckDBMarketEngine`  [INFERRED]
  tests/test_duckdb_engine.py → src/duckdb_engine.py
- `test_scan_top_alpha_stocks()` --calls--> `scan_top_alpha_stocks()`  [EXTRACTED]
  tests/test_morning_briefing.py → src/morning_briefing.py
- `test_autonomous_cycle_execution()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py

## Import Cycles
- None detected.

## Communities (147 total, 7 thin omitted)

### Community 0 - "deep_learning_model.py"
Cohesion: 0.11
Nodes (26): create_sliding_window_tensors(), DLinearTCNModel, load_dlinear_model(), predict_momentum_probability(), Any, DataFrame, Tensor, High-Efficiency Deep Learning Engine: DLinear + Temporal Convolutional Network… (+18 more)

### Community 1 - "get_price_history"
Cohesion: 0.06
Nodes (56): prefetch_single_ticker(), Pre-fetches price history and news data concurrently across fast I/O threads., _fetch_alpaca_news(), _fetch_alpaca_price_history(), _fetch_direct_yahoo_chart(), _fetch_eodhd_price_history(), _fetch_finnhub_news(), _fetch_fmp_price_history() (+48 more)

### Community 2 - "test_options_flow.py"
Cohesion: 0.17
Nodes (21): calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain(), Any, DataFrame, Live Options Microstructure, Gamma Exposure (GEX) & Max Pain Terminal for… (+13 more)

### Community 3 - "daily_scanner.py"
Cohesion: 0.08
Nodes (46): format_signal_card(), _is_duplicate_alert(), Any, Sends a rich formatted trade alert card to a Discord channel via Webhook., Checks if an alert fingerprint was already dispatched within ttl_seconds., Dispatches a crystal-clear, high-impact Discord card for live autonomous trade…, Records an alert fingerprint into results/discord_alert_ledger.json., Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord. (+38 more)

### Community 4 - "run_backtest"
Cohesion: 0.07
Nodes (60): _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition(), calculate_performance_metrics(), _calculate_trade_outcomes(), create_monthly_returns_heatmap() (+52 more)

### Community 5 - "test_statistical_arbitrage.py"
Cohesion: 0.09
Nodes (43): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any, Series (+35 more)

### Community 6 - "get_sentiment"
Cohesion: 0.13
Nodes (19): get_sentiment(), _parse_analyzer_output(), Any, DataFrame, Analyzes the sentiment of news articles using high-precision FinBERT pipeline,…, Normalizes pipeline output whether given full multi-class probabilities (list…, fixture, Fixture to set a temporary data directory for tests. (+11 more)

### Community 7 - "drl_policy_agent.py"
Cohesion: 0.13
Nodes (20): ActorCriticPolicy, DRLTradingEnvironment, evaluate_drl_policy_action(), Any, ndarray, Tensor, Deep Reinforcement Learning (DRL) Autonomous Policy Agent for Sentilyze.…, Trains an Actor-Critic DRL policy agent on historical market returns and news… (+12 more)

### Community 8 - "FactorAttributionEngine"
Cohesion: 0.09
Nodes (23): FactorAttributionEngine, InstitutionalFactsheet, Any, DataFrame, Series, Institutional Factor Attribution & Performance Factsheet Engine Computes…, Compute execution statistics from trade log dataframe., Compute rolling 30-day rolling beta, alpha, and sharpe. (+15 more)

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
Nodes (14): get_model_universe_indicator(), load_universe_tickers(), main(), cache_data, Sentilyze - Institutional Algorithmic Trading & MLOps Platform. High-Speed…, Loads active S&P 100 universe tickers., Computes universe model training coverage and mean accuracy using compiled…, render_deep_quant_cockpit() (+6 more)

### Community 13 - "analyze_supply_chain_spillover"
Cohesion: 0.14
Nodes (15): analyze_supply_chain_spillover(), Any, ndarray, Graph Neural Networks (GNN) & Supply Chain Shock Spillover Engine for…, Computes symmetric normalized Laplacian: D^(-1/2) * A * D^(-1/2)., Executes a Graph Convolutional Network (GCN) layer: H_new = ReLU(A_hat * H * W)…, Simulates an upstream supply/production shock (e.g. Taiwan earthquake or fab…, High-level entry point to run GNN supply chain shock propagation. (+7 more)

### Community 14 - "calculate_hrp_weights"
Cohesion: 0.13
Nodes (25): build_unified_portfolio(), calculate_hrp_weights(), calculate_risk_parity_weights(), get_cluster_var(), get_quasi_diag(), get_rec_bisection(), load_all_ticker_portfolios(), Any (+17 more)

### Community 15 - "run_temporal_fusion_forecast"
Cohesion: 0.14
Nodes (15): Any, DataFrame, ndarray, Temporal Fusion Transformer (TFT) & Multi-Horizon Self-Attention Engine for…, High-level entry point for Temporal Fusion Transformer multi-horizon…, Computes scaled dot-product attention weights and context vectors., Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V Args: Q, K, V: Matrices…, Lightweight, high-performance Temporal Fusion Transformer architecture with… (+7 more)

### Community 16 - "is_kill_switch_active"
Cohesion: 0.50
Nodes (4): is_kill_switch_active(), Task 7: Master Kill Switch Check. Returns True if SENTILYZE_KILL_SWITCH…, Task 7: Verify master kill switch disables order placement., test_master_kill_switch_blocks_buys()

### Community 17 - "meta_ensemble.py"
Cohesion: 0.10
Nodes (19): DynamicSharpeMetaEnsemble, MetaEnsembleClassifier, DataFrame, ndarray, Series, Institutional Multi-Model Meta-Ensemble Engine for Sentilyze. Pillar 1 Core…, Generates binary class prediction (0 = Hold/Sell, 1 = Buy) using soft-voting…, Instantiates and fits the Meta-Ensemble classifier. (+11 more)

### Community 18 - "OnlineNewtonStepOptimizer"
Cohesion: 0.16
Nodes (11): OnlineNewtonStepOptimizer, DataFrame, ndarray, Online Newton Step (ONS) Portfolio Engine (Agarwal, Hazan, Kale). A polynomial-…, Polynomial-Time ONS Portfolio Engine (Hazan et al.)., Processes price relatives (r_t = Close_t / Close_{t-1}) and updates weights in…, Fast O(d log d) Euclidean projection onto probability simplex., Runs ONS sequence through time and outputs daily allocations and portfolio… (+3 more)

### Community 19 - "test_all_14_papers.py"
Cohesion: 0.09
Nodes (30): Any, Master Academic Research Papers Empirical Benchmark Suite (All 14 Papers).…, Executes empirical backtests comparing all 14 academic paper methodologies., run_all_14_papers_benchmark(), compute_fractional_kelly_sizing(), Computes true mathematical fractional Kelly Criterion position sizing: f* = (p…, calculate_almgren_chriss_trajectory(), Any (+22 more)

### Community 20 - "draw.io XML Style Reference"
Cohesion: 0.08
Nodes (23): Color Properties, Common Style Properties, Container Types, Critical Edge Rules, draw.io XML Style Reference, Edge Attributes, Edge Styles, Escaping Rules (XML attribute context) (+15 more)

### Community 21 - "compute_dark_pool_sentiment"
Cohesion: 0.24
Nodes (13): compute_dark_pool_sentiment(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent institutional off-exchange block trades and dark pool prints., Scans option chain contracts where daily volume significantly exceeds open…, Synthesizes dark pool prints and unusual options flow into a unified…, scan_abnormal_options_vol_oi(), scan_dark_pool_blocks() (+5 more)

### Community 22 - "test_omnichannel_mobile.py"
Cohesion: 0.15
Nodes (16): answer_financial_query(), Any, Natural Language Financial Q&A Agent for Sentilyze. Pillar 7 Mobile &…, Parses natural language questions and routes them to quantitative engines.…, generate_smartwatch_glance_payload(), Any, Generates structured complication JSON for Apple Watch (watchOS) and Wear OS., format_whatsapp_trade_alert() (+8 more)

### Community 23 - "pfd-engineering-expert.md"
Cohesion: 0.10
Nodes (20): 1. Distillation Column (`distillation_column`), 1. `PHASE_PORT_VIOLATION`, 2. `DEAD_END_STREAM`, 2. Pump (`pump`), 3. Compressor (`compressor`), 3. `OPPOSING_FLOW`, 4. `GRAVITY_VIOLATION`, 4. Separator (`separator`) (+12 more)

### Community 24 - ".query"
Cohesion: 0.15
Nodes (10): Any, DataFrame, Inserts or replaces daily bars for a given ticker., Populates the DuckDB lake from pre-computed backtest portfolio curves in…, Screens for high-probability momentum setups across the entire lake., Screens for stocks where SMA50 is breaking out above SMA200 (Long-Term Bullish…, Screens for oversold mean-reversion dip opportunities during crisis/pullback…, Stage 1 Broad Screener: Returns top ranked candidates conditioned on Macro… (+2 more)

### Community 25 - "compute_lead_lag_matrix"
Cohesion: 0.20
Nodes (14): compute_lead_lag_matrix(), _granger_f_test(), Any, DataFrame, ndarray, Series, rank_market_price_leaders(), Lead-Lag Granger Causality & Supply Chain Price Discovery Engine for Sentilyze.… (+6 more)

### Community 26 - "ws_live_prediction.py"
Cohesion: 0.17
Nodes (19): calculate_smart_money_zones(), calculate_structural_trailing_stop(), evaluate_multi_timeframe_confluence(), find_swing_pivots(), Any, DataFrame, Institutional Smart Money Market Structure & Price-Action Engine for Sentilyze.…, Ratchets the Stop-Loss up structurally behind higher swing lows. Rules: 1.… (+11 more)

### Community 27 - "social_sentiment.py"
Cohesion: 0.23
Nodes (13): calculate_social_buzz_metrics(), fetch_social_sentiment_tracker(), Any, Social Sentiment Velocity & Retail Multi-Platform Scraper for Sentilyze. Pillar…, Scrapes real-time streaming retail sentiment from Stocktwits public symbol…, Scrapes tech community discussions on AI catalysts (OpenAI, Anthropic, Nvidia)…, Computes retail sentiment velocity and flow conviction metrics., High-level entry point to retrieve calibrated real-time social buzz metrics for… (+5 more)

### Community 28 - "AlpacaBrokerBridge"
Cohesion: 0.16
Nodes (14): AlpacaBrokerBridge, Any, Fetches active positions from Alpaca brokerage., Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.…, Verifies active connection to Alpaca Brokerage API., Fetches live Alpaca account equity, buying power, and cash., Submits an institutional Bracket Order: - Entry: Market order - Exit 1: Limit…, patch (+6 more)

### Community 29 - "realtime_tracker.py"
Cohesion: 0.07
Nodes (42): AI Trade Copilot & Conversational Analyst for Sentilyze. Provides natural…, handle_bot_command(), Any, Executes command and posts formatted embed reply to Discord., Parses and processes interactive bot commands: - `/signal <ticker>` -…, send_bot_command_reply(), AI Pre-Market Audio & Executive Morning Briefing Generator for Sentilyze.…, Any (+34 more)

### Community 30 - "test_onnx_and_orderflow.py"
Cohesion: 0.09
Nodes (27): ONNXFinBERTEngine, Any, Initializes the existing PyTorch FastFinBERTEngine as graceful fallback., Runs accelerated sentiment scoring on a single headline., Batched sentiment inference., Accelerated FinBERT inference engine powered by ONNX Runtime with automatic…, Attempts to initialize ONNX Runtime and load the optimized model., build_orderflow_candlestick_chart() (+19 more)

### Community 31 - "get_us_market_session"
Cohesion: 0.12
Nodes (24): check_market_hours_preflight(), get_current_ny_time(), get_us_market_session(), Any, datetime, Unified US Stock Market (NYSE / NASDAQ) Session & Calendar Engine for…, Pre-flight sanity check for automated workflows. Returns True if execution…, Returns the current precise timestamp in America/New_York (Eastern Time). (+16 more)

### Community 32 - "ipo_radar.py"
Cohesion: 0.21
Nodes (12): auto_register_ipo_ticker(), fetch_pre_ipo_radar_summary(), fetch_sec_edgar_ipo_filings(), Any, IPO & Pre-IPO Intelligence Radar for Sentilyze. Pillar 9 Alternative Asset…, Fetches real-time SEC Form S-1 / S-1/A IPO registration statements from SEC…, Appends a newly public IPO ticker to stocks.txt to initiate model ingestion., High-level entry point returning the complete Pre-IPO and SEC S-1 pipeline. (+4 more)

### Community 33 - "3. Cross-Functional Flowcharts (Actor × Phase Grid)"
Cohesion: 0.10
Nodes (19): 1. Flat Swimlanes (BPMN-style Flowcharts), 2. Nested Architecture Containers, 3. Cross-Functional Flowcharts (Actor × Phase Grid), 4. When to Use What, Container Key Rules Summary, Critical Rules, Critical Rules, Decision Guide (+11 more)

### Community 34 - "triple_convex_engine.py"
Cohesion: 0.12
Nodes (20): apply_triple_barrier_labeling(), calculate_deflated_sharpe_ratio(), DataFrame, Series, Marcos López de Prado's Triple-Barrier Method & Deflated Sharpe Ratio (DSR).…, Computes Bailey & López de Prado's Deflated Sharpe Ratio (DSR). Adjusts for: -…, Applies López de Prado's path-dependent Triple-Barrier Method to generate trade…, Any (+12 more)

### Community 35 - "Edge Routing Decision Guide"
Cohesion: 0.12
Nodes (16): Critical Rules, Decision Flowchart, Default Behavior (Built-in Router), Don't add waypoints, Don't hand-route, Edge labels, Edge Routing Decision Guide, Every edge needs an mxGeometry child (+8 more)

### Community 36 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (16): Any, DataFrame, ndarray, Series, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.…, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used. (+8 more)

### Community 37 - "SECCatalystCrawler"
Cohesion: 0.17
Nodes (13): get_recent_sec_catalysts(), Any, SEC EDGAR 8-K Material Corporate Event Crawler & Catalyst Miner. Directly…, Fetches Form 8-K material filings for a given company within the lookback…, Fast heuristic catalyst sentiment classifier. Yields positive score for…, Convenience helper to retrieve SEC 8-K catalysts., Mines material corporate events from the SEC EDGAR system., Loads the official mapping from stock ticker to 10-digit CIK number. (+5 more)

### Community 38 - "Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform"
Cohesion: 0.12
Nodes (16): 1. Run in 1-Click (No Installation Required), 2. Local Setup & Installation, 3. Run the Quantitative CLI, 4. Launch the Streamlit Mission Control, 5. Run Full Test Suite (274+ Unit Tests), Active Holdings (Entered 2026-09-08), 📊 Empirical Alpha Attribution & Benchmarks, 🏛️ Grounded 5-Agent Deliberation Council (+8 more)

### Community 39 - "correlation_shield.py"
Cohesion: 0.23
Nodes (11): calculate_portfolio_correlation_matrix(), check_correlation_shield(), Any, DataFrame, Portfolio Correlation Matrix Shield. Functions: - Calculates rolling 21-day…, Computes the 21-day rolling pairwise return correlation matrix across tickers., Audits a candidate buy against currently held positions. Returns whether the…, Tests for Portfolio Correlation Matrix Shield (src/correlation_shield.py).… (+3 more)

### Community 40 - "[Validator Rules Reference & Troubleshooting]"
Cohesion: 0.12
Nodes (16): 1. `DIRECT_WAN_TO_LAN`, 2. `ORPHAN_DEVICE`, 3. `VLAN_LEAK`, 4. `REDUNDANCY_WARNING`, 5. `MISSING_DMZ`, 6. `FLAT_NETWORK`, 7. `FIREWALL_BYPASS`, 8. `SINGLE_TRUNK` (+8 more)

### Community 41 - "TickerSentinelSwarm"
Cohesion: 0.12
Nodes (17): detect_peak_crest_exhaustion(), Any, Dedicated Ticker Sentinel & Peak-Crest Volume Harvester Swarm for Sentilyze.…, Dedicated Micro-Agent assigned to monitor a single stock position 24/7. Tracks…, Audits live price tick, applies Zero-Giveback profit ratchets, and detects peak…, Manages the full swarm of Dedicated Ticker Sentinels across all open positions.…, Synchronizes active sentinels with current portfolio open positions without…, Audits all active sentinels concurrently. Optionally persists ratcheted SL and… (+9 more)

### Community 42 - "RegimeMixtureOfExperts"
Cohesion: 0.26
Nodes (8): Any, DataFrame, ndarray, Series, 3-Expert Gated Mixture of Experts classifier for financial regimes., Labels historical rows into 3 latent market regimes: 0 = Bull Momentum (Price >…, RegimeMixtureOfExperts, test_regime_moe()

### Community 43 - "test_acpm_trainer.py"
Cohesion: 0.12
Nodes (19): Unified Alpha-Conformal Purged Multi-Task (ACPM) Quantitative Training Engine.…, Conformal Probability Calibration & Quantile Uncertainty Layer. Maps raw tree…, neutralize_features(), neutralize_predictions(), DataFrame, ndarray, Series, Feature & Factor Neutralization for Quantitative Machine Learning.… (+11 more)

### Community 47 - "Sentilyze Community Code of Conduct"
Cohesion: 0.25
Nodes (7): 📜 Attribution, ⚖️ Enforcement Escalation, 🛡️ Enforcement & Responsibilities, 🌟 Our Pledge, 🎯 Our Standards, 📬 Reporting Guidelines, Sentilyze Community Code of Conduct

### Community 48 - "universal_web_scraper.py"
Cohesion: 0.13
Nodes (25): clean_financial_text(), Cleans raw financial headlines/descriptions by stripping boilerplate publisher…, batch_scrape_urls(), detect_stock_tickers(), extract_financial_tables_from_html(), _extract_summary_sentences(), Any, Universal Web Scraper & Financial Intelligence Extractor for Sentilyze.… (+17 more)

### Community 49 - "DuckDBMarketEngine"
Cohesion: 0.15
Nodes (10): DuckDBMarketEngine, get_duckdb_scanner(), High-Performance DuckDB Columnar Data Lake & Market-Wide Scanner Engine.…, Embedded Columnar Analytical Engine powered by DuckDB., Closes connection cleanly., Convenience factory helper., Initializes high-performance columnar tables with index constraints., test_run_duckdb_benchmark_in_memory() (+2 more)

### Community 50 - "test_pillar2_alternative_data.py"
Cohesion: 0.19
Nodes (18): compute_smart_money_insider_score(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent SEC Form 4 insider transactions for a given stock., Retrieves recent Congressional STOCK Act disclosure reports for a ticker., Synthesizes SEC Form 4 and Congressional activity into an overall Smart Money…, track_congressional_stock_disclosures(), track_corporate_insider_filings() (+10 more)

### Community 51 - "price_scout.py"
Cohesion: 0.18
Nodes (12): get_latest_scout_alerts(), PriceActionScoutAgent, PriceScoutBot, Any, Real-Time Price Action & Tape-Reading Scout Subagent (Bot) for Sentilyze.…, Continuous Background Scanner Bot that scouts the 538 universe assets, detects…, Scans given tickers for real-time volume breakout candidates., Retrieves the latest price scout breakout alerts. (+4 more)

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
Cohesion: 0.09
Nodes (38): generate_morning_briefing_text(), get_portfolio_intelligence(), load_universe_candidates(), Any, Reads live paper portfolio state for broadcast reporting., Assembles a comprehensive, institutional Wall Street Morning Podcast and…, Synthesizes broadcast audio podcast (.mp3) using Google Text-to-Speech (gTTS)…, Loads clean ticker list from stocks.txt or falls back to core liquid universe. (+30 more)

### Community 58 - "calculate_time_decayed_sentiment"
Cohesion: 0.19
Nodes (14): calculate_time_decayed_sentiment(), compute_exponential_decay_weights(), Any, DataFrame, datetime, ndarray, Series, Temporal Sentiment Half-Life Decay Engine. Functions: - Applies continuous… (+6 more)

### Community 59 - "erd-database-expert.md"
Cohesion: 0.13
Nodes (14): 1. `FK_WITHOUT_TARGET`, 2. `ORPHAN_TABLE`, 3. `DUPLICATE_PK`, 4. `SELF_REFERENCE_MISSING`, 5. `MISSING_JUNCTION`, 6. `CIRCULAR_FK`, [Anti-Patterns], [Docs Index] (+6 more)

### Community 60 - "ws_portfolio.py"
Cohesion: 0.14
Nodes (17): Cockpit 3: Portfolio & Risk Optimization…, render_portfolio_cockpit(), _get_cached_macro_metrics(), cache_data, Workspace: Real-Time Macro Liquidity & Treasury Yield Curve Radar. Visualizes…, render_macro_liquidity_workspace(), _get_cached_diversity_grade(), cache_data (+9 more)

### Community 61 - "6. CROSS-DISCIPLINARY & FRONTIER QUANTITATIVE IDEAS"
Cohesion: 0.07
Nodes (27): 1. EXECUTIVE SUMMARY & CORE PLATFORM VISION, 2. THE 4-PILLAR WINNING QUANTITATIVE ALPHA FORMULA, 3. EMPIRICAL MULTI-DECADE HISTORICAL BENCHMARKS (1980–2026), 4. MASTER 5-COCKPIT MISSION CONTROL ARCHITECTURE, 5. THE 300 INSTITUTIONAL RESOURCES SYNTHESIS, 6. CROSS-DISCIPLINARY & FRONTIER QUANTITATIVE IDEAS, 7. IMPLEMENTATION & TECHNOLOGY STACK BLUEPRINT, 🐺 A. Evolutionary Biology: Lotka-Volterra Predator-Prey Market Dynamics (+19 more)

### Community 64 - "GaussianHMMRegimeDetector"
Cohesion: 0.13
Nodes (11): GaussianHMMRegimeDetector, Any, DataFrame, ndarray, Paper 15: Gaussian Hidden Markov Model for Market Regime Detection. Source:…, 3-state Gaussian HMM regime classifier: Bull / Normal / Crisis. Uses hand-…, Gaussian emission probability for each state., Forward algorithm step: update filtered state probabilities with one new… (+3 more)

### Community 65 - "EWMACorrelationMonitor"
Cohesion: 0.14
Nodes (10): EWMACorrelationMonitor, Any, DataFrame, ndarray, Paper 17: RiskMetrics EWMA Volatility & Correlation Monitor. Source: J.P.…, Real-time EWMA-based correlation and volatility monitor. Tracks time-varying…, Initialize EWMA state from a seed window of returns., Update EWMA state with one day's returns across all assets. Returns current… (+2 more)

### Community 67 - "analyze_sec_filing_diff"
Cohesion: 0.32
Nodes (7): analyze_sec_filing_diff(), compute_text_similarity_and_diff(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Computes lexical and semantic diff metrics between consecutive filings., Retrieves and compares the most recent and prior SEC filings for a company.…, test_sec_filing_diff_analysis()

### Community 68 - "CUSUMDetector"
Cohesion: 0.15
Nodes (9): CUSUMDetector, Any, ndarray, Paper 16: CUSUM Sequential Change-Point Detection. Source: E.S. Page (1954) —…, Cumulative Sum detector for online mean-shift detection. Monitors a numeric…, Process one observation. Returns alarm status., Process a batch of observations., Reset detector state. (+1 more)

### Community 69 - "PageHinkleyDetector"
Cohesion: 0.15
Nodes (9): PageHinkleyDetector, Any, ndarray, Paper 22: Page-Hinkley Sequential Test for Concept Drift Detection. Source:…, Page-Hinkley test for detecting changes in the mean of a stream. Monitors…, Process one observation. Returns drift status., Process a batch of observations., Reset detector state. (+1 more)

### Community 70 - "QuantAlphaDAGEngine"
Cohesion: 0.10
Nodes (26): AlphaDAGStageContract, Any, DataFrame, Series, QuantAlphaDAGEngine, Generation 3: Multi-Horizon AI Courtroom & Regime-Leveraged Quant DAG Engine.…, Calculates time-aligned returns, normalized volumes, and volatility bands., Generates Gen-3 Multi-Horizon Feature Library: - Factor 1: Triple-Horizon Trend… (+18 more)

### Community 71 - "test_duckdb_engine.py"
Cohesion: 0.12
Nodes (16): memory_engine(), fixture, Unit tests for DuckDB Columnar Data Lake & High-Speed Scanner Engine., Provides an isolated in-memory DuckDB market engine., Generates synthetic daily bars for testing., Verify tables are created in the database., Verify bar ingestion and fast SQL querying., Verify vectorized momentum breakout scanner. (+8 more)

### Community 72 - "preprocess_data"
Cohesion: 0.15
Nodes (17): clean_headline_data(), _get_api_key(), _load_sentiment_analyzer(), preprocess_data(), Any, DataFrame, Cleans a headline CSV file by removing rows with invalid stock tickers. Caches…, Safely attempts to retrieve the API key from Streamlit secrets, falling back to… (+9 more)

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

### Community 77 - "api.py"
Cohesion: 0.18
Nodes (10): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+2 more)

### Community 78 - "chart_pattern_learning.py"
Cohesion: 0.19
Nodes (17): detect_classical_chart_patterns(), generate_ai_chart_explanation(), match_historical_chart_twins(), normalize_waveform(), Any, DataFrame, ndarray, AI Chart Pattern Recognition, Geometric Wave Learning & Visual Understanding… (+9 more)

### Community 79 - "run_cppi_backtest"
Cohesion: 0.23
Nodes (8): calculate_cppi_allocation(), Any, ndarray, Paper 20: Constant Proportion Portfolio Insurance (CPPI). Source: Black & Jones…, CPPI allocation: Exposure = M * (Portfolio - Floor). Args: portfolio_value:…, Run a full CPPI backtest over a return series. Args: returns: Array of daily…, run_cppi_backtest(), TestCPPI

### Community 80 - "DCCCorrelation"
Cohesion: 0.20
Nodes (7): DCCCorrelation, Any, DataFrame, Paper 24: Dynamic Conditional Correlation (DCC-GARCH). Source: Engle (2002) —…, Simplified DCC model using EWMA-GARCH(1,1) for individual volatilities and DCC…, Fit the DCC model to a returns DataFrame. Returns time-varying correlation…, TestDCC

### Community 81 - "test_reddit_premarket_station.py"
Cohesion: 0.18
Nodes (19): fetch_4station_premarket_intelligence(), fetch_8station_premarket_intelligence(), fetch_all_reddit_headlines_for_ticker(), _fetch_subreddit_rss_entries(), Any, DataFrame, Systematic 8-Station Multi-Channel Reddit Market Intelligence Engine. Pillar 2…, Fetches real-time Atom RSS feed for a subreddit using safe defusedxml. (+11 more)

### Community 82 - "fetch_universe_live_quotes"
Cohesion: 0.10
Nodes (27): fetch_universe_live_quotes(), Fetches real-time quotes across universe with fast batching (sub-2s)., calculate_custom_rebalance(), calculate_share_allocation(), Any, Helper to calculate share allocation from latest daily signals file or universe…, Computes exact whole-share buy allocations for a given capital budget across…, Any (+19 more)

### Community 83 - "risk_constrained_kelly_allocation"
Cohesion: 0.24
Nodes (6): Any, ndarray, Paper 23: Risk-Constrained Kelly Gambling. Source: Busseti, Ryu, Boyd —…, Compute optimal Kelly allocation with drawdown probability constraint.…, risk_constrained_kelly_allocation(), TestRiskKelly

### Community 84 - "ws_alternative_data.py"
Cohesion: 0.21
Nodes (11): get_pre_ipo_pipeline_df(), DataFrame, Returns a formatted pandas DataFrame of all pre-IPO target assets., _get_cached_reddit_intel(), cache_data, Workspace 4: Systematic 9-Station 1-Day-Prior Reddit News & SEC S-1 Pre-IPO…, Renders the 9-Station Reddit Intelligence and Pre-IPO Valuation Radar., render_alternative_data_workspace() (+3 more)

### Community 85 - "test_tri_model_pipeline.py"
Cohesion: 0.15
Nodes (17): Financial Event & Emotion Companion Model (Model 2). Companion model to FinBERT…, FastAgentPipeline, get_fast_agent_pipeline(), Fast Sub-Second Agent Decision Router. Coordinates the Tri-Model Neural Brain…, Singleton getter for FastAgentPipeline., Sub-second Multi-Model Agent Decision Pipeline., get_fast_finbert_engine(), Accelerated Low-Latency FinBERT Engine (Model 1). Optimized for high-speed… (+9 more)

### Community 86 - "agent_committee.py"
Cohesion: 0.12
Nodes (25): audit_full_universe_committee(), ChiefRiskOfficerAgent, convene_trading_committee(), execute_committee_order(), ForensicFundamentalAgent, _persist_committee_resolution(), Any, Autonomous Multi-Agent Trading Committee & Deliberation Engine for Sentilyze.… (+17 more)

### Community 87 - "test_zero_giveback_profit_lock.py"
Cohesion: 0.18
Nodes (11): enforce_capital_shield_stop_floor(), Enforces a strict downside capital shield ceiling (max 2.50% loss from entry).…, Unit tests for Institutional Zero-Giveback Profit Protection, 80% Peak Gain…, Verify that a gain of +0.50% immediately locks stop-loss to Breakeven + buffer., Verify that a gain of +1.00% locks stop-loss to at least +0.50% net profit., Verify that any stop loss wider than -2.50% is elevated to the capital shield…, Verify that PaperBroker saves atomically, maintains a backup, and guards…, test_atomic_broker_save_and_backup() (+3 more)

### Community 88 - "generate_comprehensive_factsheet"
Cohesion: 0.24
Nodes (9): generate_comprehensive_factsheet(), Any, Series, Institutional Risk & Alpha Performance Factsheet Engine for Sentilyze.…, Computes over 30 institutional hedge-fund risk, performance, and drawdown…, optimize_dataframe_memory(), Downcasts numeric types in a pandas DataFrame to reduce memory footprint by…, test_generate_comprehensive_factsheet_custom_series() (+1 more)

### Community 89 - "correlation_matrix.py"
Cohesion: 0.38
Nodes (6): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…

### Community 90 - "generate_pipeline_graph_data"
Cohesion: 0.27
Nodes (9): generate_pipeline_graph_data(), Any, Interactive Multi-Agent & Pipeline Architecture Canvas Component for Streamlit.…, Renders the interactive Vis.js animated node network canvas inside Streamlit., Constructs the nodes and edges representation for the pipeline graph canvas., render_pipeline_topology_canvas(), Tests for Interactive Multi-Agent & Pipeline Architecture Canvas…, test_generate_pipeline_graph_data_defaults() (+1 more)

### Community 91 - "SecurityMaster"
Cohesion: 0.06
Nodes (51): get_security_master(), DataFrame, US Security Master & Universal Ticker Registry. Automates official discovery,…, Manages the global universe of actively traded US equities and ETFs., Downloads and parses the official NASDAQ/NYSE consolidated traded directory.…, Loads cached security master or bootstraps from stocks.txt., Returns the GICS sector for a given ticker with intelligent heuristics., Classifies an asset into an institutional execution tier based on daily dollar… (+43 more)

### Community 92 - "block_external_alerts"
Cohesion: 0.50
Nodes (3): block_external_alerts(), fixture, Autouse fixture that prevents tests from sending real outbound network calls to…

### Community 93 - "advanced_quant_experiments.py"
Cohesion: 0.22
Nodes (10): compute_performance_metrics(), Any, DataFrame, Series, Empirical Quant Experimentation & Multi-Asset Ablation Benchmark Suite.…, Executes empirical ablation benchmark across the full asset universe., Simulates walk-forward strategy execution with or without advanced quant…, Computes key quant performance metrics. (+2 more)

### Community 94 - "AICopilotEngine"
Cohesion: 0.24
Nodes (8): AICopilotEngine, Any, Conversational intelligence engine that parses queries and generates analytical…, Interprets user prompt and routes to appropriate financial analytical…, test_copilot_committee_query(), test_copilot_portfolio_query(), test_copilot_stress_query(), test_copilot_ticker_analysis_query()

### Community 95 - "options_surface.py"
Cohesion: 0.27
Nodes (10): calculate_multileg_payoff(), generate_volatility_surface_mesh(), Any, 3D Implied Volatility Surface & Multi-Leg Options Strategy Desk for Sentilyze.…, Constructs a 3D Implied Volatility Surface across strike prices and expiration…, Calculates profit and loss (P&L) curves at expiration for institutional multi-…, test_calculate_multileg_payoff_bull_call_spread(), test_calculate_multileg_payoff_iron_condor() (+2 more)

### Community 96 - "drawio/SKILL.md"
Cohesion: 0.13
Nodes (14): 1.Reference-Docs(AI-Knowledge), 2.Validator-Scripts(Programmatic-Enforcement), 3.Topological-Corrections(Auto-Fix), [Architectural Constraints], [Batch Diagram Generation (Highly Recommended)], Containers, [Diagram Builder Tools], [Docs Index] (+6 more)

### Community 97 - "red_team_agent.py"
Cohesion: 0.21
Nodes (9): AdversarialRedTeamAgent, Any, Adversarial Red-Team & Devil's Advocate Specialist Agent. Functions: - Stress-…, Agent 5: Adversarial Red-Team / Devil's Advocate Specialist. Actively hunts for…, Conducts a rigorous adversarial audit of the target asset., Tests for Adversarial Red-Team Specialist Agent (src/red_team_agent.py).…, test_red_team_agent_initialization(), test_red_team_evaluation_structure() (+1 more)

### Community 98 - "liquidity_heatmap.py"
Cohesion: 0.31
Nodes (8): compute_order_book_depth_and_clusters(), compute_volume_profile_and_poc(), Any, Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap for…, Simulates Level 2 market depth and identifies institutional buy/sell liquidity…, Computes Point of Control (POC), Value Area High (VAH), and Value Area Low…, test_compute_order_book_depth_and_clusters(), test_compute_volume_profile_and_poc()

### Community 99 - "PaperBroker"
Cohesion: 0.08
Nodes (26): PaperBroker, Any, DataFrame, Institutional Multi-Stage Quantitative Execution Broker ($100k Account).…, Executes daily quantitative scan results using the Concentrated Top-2 + Scale-…, Loads existing portfolio state from JSON or initializes a fresh account if file…, Updates total equity, unrealized PnL, and win rates., Reloads latest state from disk if file exists. (+18 more)

### Community 100 - "calculate_portfolio_diversity_grade"
Cohesion: 0.36
Nodes (7): calculate_portfolio_diversity_grade(), Any, DataFrame, test_custom_returns_correlated(), test_custom_returns_diverse(), test_empty_portfolio(), test_single_asset_portfolio()

### Community 101 - "cli.py"
Cohesion: 0.19
Nodes (16): Root entry point for Sentilyze CLI. Run directly with: python sentilyze.py NVDA…, cmd_audit(), cmd_briefing(), cmd_experiment(), cmd_portfolio(), cmd_scan(), cmd_trades(), main() (+8 more)

### Community 102 - "Domain Expert Reference Template"
Cohesion: 0.15
Nodes (12): [Anti-Patterns], [Docs Index], Domain Expert Reference Template, [Domain Rules + Patterns], [Domain-Specific Catalog / Patterns] (OPTIONAL), [Project Context], [Project Conventions], Section Checklist (+4 more)

### Community 103 - "preprocessing.py"
Cohesion: 0.17
Nodes (19): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, DataFrame (+11 more)

### Community 104 - "3. Equipment Mapping (Draw.io Native Shapes)"
Cohesion: 0.15
Nodes (12): 1. ISA 5.1 Instrument Tagging Conventions, 2. Line Styles, 3. Equipment Mapping (Draw.io Native Shapes), 4. Diagram Layout Strategy (PFDs), Common Tag Identifiers, Heat Exchangers, Instrument Shapes (`mxgraph.pid2inst.*`), Mining & Minerals Processing (+4 more)

### Community 105 - "main"
Cohesion: 0.29
Nodes (7): main(), Batch Universe Trainer for Remaining S&P 100 Tickers., run_single(), main(), Trains institutional ACPM Sector-Pooled Multi-Task models across the universe.…, Main function to run the training pipeline for a given stock ticker. Args:…, train_sector_pooled_models()

### Community 106 - "ws_options_surface.py"
Cohesion: 0.14
Nodes (17): Real-Time Market Anomaly Screener Component for Streamlit. Functions: - Renders…, Renders the Real-Time Market Anomaly Screener UI., render_live_screener_section(), load_universe_tickers(), Loads universe of tickers from stocks.txt., Cockpit 2: Smart Money & Alternative Data…, render_smart_money_cockpit(), _get_cached_gex() (+9 more)

### Community 107 - "_get_browser_session"
Cohesion: 0.33
Nodes (5): _get_browser_session(), Session, Universal Adaptive Web Scraper supporting Scrapling, Trafilatura, and…, Creates a configured requests session with realistic browser headers., UniversalWebScraper

### Community 108 - "Agent Browser — Live Web Automation Skill"
Cohesion: 0.50
Nodes (3): 1. Core Principles, 2. Use Cases in Sentilyze, Agent Browser — Live Web Automation Skill

### Community 109 - "Agent Memory — Persistent Multi-Session Memory Skill"
Cohesion: 0.50
Nodes (3): 1. Storage Architecture, 2. Integration Pattern, Agent Memory — Persistent Multi-Session Memory Skill

### Community 113 - "MetaRegimeAllocator"
Cohesion: 0.11
Nodes (23): get_current_macro_regime(), MetaRegimeAllocator, MetaRegimeReport, DataFrame, 3-State Gaussian Hidden Markov Model & Meta-Regime Capital Allocator…, Fetch historical price data for the benchmark asset., Generate statistically grounded market proxy data if network or feed is…, Fit Gaussian Mixture and estimate empirical transition probabilities. (+15 more)

### Community 114 - "aws-well-architected-reviewer.md"
Cohesion: 0.22
Nodes (8): [Anti-Patterns], [Docs Index], [Domain Rules + Patterns], [Project Context], [Project Conventions], [Sequential Execution Pairing (MANDATORY)], [Visual Styling], [XML Graph DOM Rules]

### Community 115 - "gcp-well-architected-reviewer.md"
Cohesion: 0.22
Nodes (8): [Anti-Patterns], [Docs Index], [Domain Rules + Patterns], [Project Context], [Project Conventions], [Sequential Execution Pairing (MANDATORY)], [Visual Styling], [XML Graph DOM Rules]

### Community 116 - "run_universe_training.py"
Cohesion: 0.24
Nodes (11): load_universe_from_file(), main(), prefetch_universe_data(), Any, Universal Parallel Multi-Core Model Training & High-Resolution Benchmark.…, Loads cleaned ticker symbols from stocks.txt., Worker function to train a single asset., Pre-fetches market and news data in parallel across fast I/O threads. (+3 more)

### Community 117 - "modeling.py"
Cohesion: 0.15
Nodes (21): Continuous Model Self-Training & Accuracy Boosting Engine for Sentilyze. Self-…, get_prediction_on_latest_data(), load_model(), Any, DataFrame, Series, Save the trained model to a file using XGBoost's native format. This is safer…, Load a trained model from a file using XGBoost's native format. Args: filepath… (+13 more)

### Community 118 - "compute_gamma_exposure_profile"
Cohesion: 0.14
Nodes (20): calculate_black_scholes_gamma(), calculate_options_gex(), compute_gamma_exposure_profile(), fetch_options_chain_data(), Any, DataFrame, Options Gamma Exposure (GEX), Volatility Surface & Strike Wall Radar for…, Computes institutional Net Gamma Exposure (GEX), Call Wall, Put Wall, and Gamma… (+12 more)

### Community 119 - "ConformalCalibrator"
Cohesion: 0.12
Nodes (14): ACPMTrainer, Any, DataFrame, Series, State-of-the-Art 10x Institutional Quantitative Training Engine., Executes end-to-end ACPM training for a target equity., ConformalCalibrator, ndarray (+6 more)

### Community 120 - "EventClassifierModel"
Cohesion: 0.21
Nodes (8): EventClassifierModel, Any, Translates raw social slang, emojis, and cashtags into standardized financial…, Classifies a single headline or social post with sub-millisecond speed.…, Batch-processes multiple headlines with vectorized execution., Production-grade Financial Event & Emotion Companion Model. Provides sub-…, test_event_classifier_fraud_and_dilution(), test_event_classifier_model()

### Community 121 - "execute_continuous_retrain_cycle"
Cohesion: 0.32
Nodes (7): enrich_features_with_alpha_interactions(), execute_continuous_retrain_cycle(), Any, DataFrame, Enriches standard feature matrix with non-linear interaction terms., Executes an end-to-end continuous learning and model boosting cycle: 1.…, test_enrich_features_with_alpha_interactions()

### Community 122 - "fractional_differentiation_ffd"
Cohesion: 0.29
Nodes (10): find_optimal_d(), fractional_differentiation_ffd(), get_weights_ffd(), ndarray, Series, Fixed-Width Window Fractional Differentiation (FFD) for Financial Time Series.…, Generate weights for Fixed-Width Window Fractional Differentiation. w_0 = 1 w_k…, Applies Fixed-Width Window Fractional Differentiation to a price series. (+2 more)

### Community 123 - "temp_data_dir"
Cohesion: 0.67
Nodes (3): fixture, Fixture to set a temporary data directory for tests., temp_data_dir()

### Community 124 - "render_live_prediction_workspace"
Cohesion: 0.18
Nodes (11): Renders a progress meter with dynamic color coding., render_conviction_gauge(), load_cached_tournament_results(), cache_data, Workspace 25: Autonomous Quant Alpha DAG Engine (Gen-3 Champion). Institutional…, Loads pre-computed 3-generation tournament results if available., Renders the comprehensive Quant Alpha DAG Engine workspace., render_alpha_dag_workspace() (+3 more)

### Community 125 - "ws_deep_quant.py"
Cohesion: 0.09
Nodes (45): _persist_ablation_results(), Any, 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), analyze_debt_maturity_wall() (+37 more)

### Community 126 - "FastFinBERTEngine"
Cohesion: 0.31
Nodes (5): FastFinBERTEngine, Any, Vectorized micro-batch inference for multiple headlines., Accelerated FinBERT inference engine delivering 2.5x - 4x speedups., Runs accelerated sentiment scoring on a single headline. Includes Companion…

### Community 127 - "test_full_market_funnel.py"
Cohesion: 0.29
Nodes (7): clean_lock_file(), fixture, patch, Verifies that Stage 1 broad screener queries DuckDB and returns ranked…, End-to-end verification of AutonomousTradingEngine with Two-Stage Funnel., test_full_funnel_cycle_execution_safe(), test_get_screened_candidates_duckdb_integration()

### Community 128 - "components.py"
Cohesion: 0.25
Nodes (6): Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Wraps HTML content inside an institutional frosted glass container., render_glass_card(), Workspace 7: SHAP Explainability & Dynamic Game-Theoretic Decision Trees., Renders SHAP explainability plots, dynamic Plotly waterfall charts, and feature…, render_xai_workspace()

### Community 129 - ".split"
Cohesion: 0.50
Nodes (3): DataFrame, ndarray, Series

### Community 130 - "calculate_doubling_progress"
Cohesion: 0.27
Nodes (9): calculate_doubling_progress(), compute_compound_position_size(), Any, Max Compound Acceleration & +100% Target Doubling Engine for Sentilyze.…, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Unit tests for Max Compound Acceleration Engine., test_calculate_doubling_progress() (+1 more)

### Community 131 - "TriBrainGatingLayer"
Cohesion: 0.40
Nodes (3): Tensor, Neural Softmax Gating layer to dynamically allocate weights among the 3 models.…, TriBrainGatingLayer

### Community 132 - "black_swan_simulator.py"
Cohesion: 0.23
Nodes (11): calculate_kelly_sizing(), estimate_market_impact_slippage(), Any, Historical Black Swan Crisis Simulator & Kelly Position Sizing for Sentilyze.…, Calculates optimal position sizing using the Kelly Criterion: Kelly % = W - (1…, Estimates market execution slippage using the Almgren-Chriss square-root impact…, Stress-tests the current portfolio against major historical market crashes.…, simulate_portfolio_crises() (+3 more)

### Community 133 - "autonomous_trader.py"
Cohesion: 0.06
Nodes (43): AutonomousTradingEngine, check_daily_loss_circuit_breaker(), ensure_background_daemon_thread_running(), get_daemon_status(), Any, Autonomous Live Trading & News Intelligence Engine for Sentilyze. Institutional…, Returns current live daemon running state and last pulse timestamp., Ensures a single background autonomous trading daemon thread is permanently… (+35 more)

### Community 134 - "FastNeuralEventClassifier"
Cohesion: 0.29
Nodes (4): FastNeuralEventClassifier, Tensor, Builds a lightweight internal token dictionary for financial catalysts., Lightweight 2-layer Neural Event & Emotion Head for sub-millisecond…

### Community 136 - "apply_high_watermark_profit_lock"
Cohesion: 0.32
Nodes (7): apply_high_watermark_profit_lock(), Guarantees that once a trade reaches peak profit, the bot NEVER gives back >…, Unit tests for High-Watermark Peak Profit Ratchet (75% Lock Floor)., test_high_watermark_does_not_lower_sl_on_pullback(), test_high_watermark_locks_75pct_of_peak_gain(), Verify that a peak surge to $110 (+10%) locks at least $108 (+8.0%) into the…, test_high_watermark_80_pct_retention()

### Community 137 - "render_workspace_header"
Cohesion: 0.14
Nodes (22): generate_institutional_factsheet_for_ticker(), Reads ticker backtest portfolio curve and builds full institutional factsheet., Cockpit 4: Multi-Decade Backtest & Performance Factsheet…, render_backtest_cockpit(), Renders an executive header banner with live status badge, market clock, and…, render_workspace_header(), Workspace 6: Walk-Forward Backtesting & Performance Tearsheet., Renders the Walk-Forward Backtesting and Strategy Tearsheet workspace. (+14 more)

### Community 138 - "utils.py"
Cohesion: 0.12
Nodes (14): Logger, Cloud PostgreSQL Data Lake (Supabase / Neon) Connector for Sentilyze. Pillar 6…, Real-Time Earnings Call Transcript & Management Tone Analyzer for Sentilyze.…, Real-Time Macro Liquidity & Treasury Yield Curve Radar for Sentilyze. Analyzes…, ONNX Runtime Accelerated FinBERT Engine (Model 1 & 2).…, Apple Watch & Wear OS Glance Complications API for Sentilyze. Pillar 7 Mobile &…, Any, Fast vectorized backtesting simulation sandbox for custom leverage, confidence,… (+6 more)

### Community 139 - "print_formatted_trade_report"
Cohesion: 0.23
Nodes (11): get_trade_telemetry(), print_formatted_trade_report(), Any, Instant Trade Telemetry & Results Engine. Provides direct, zero-overhead access…, Fetches real-time trade execution summary, partitioning by today, yesterday,…, Prints a clean, concise institutional trade report to the terminal., Unit tests for Instant Trade Telemetry & Results Engine., Verify formatted trade report prints without raising exceptions. (+3 more)

### Community 140 - "screener_engine.py"
Cohesion: 0.14
Nodes (18): build_pooled_sector_dataset(), get_sector_for_ticker(), DataFrame, Series, Cross-Asset Sector-Pooled Multi-Task Dataset Engine. Maps the 538-stock…, Returns the sector cluster for a given ticker or General default., Pools cross-sectional DataFrames across sector peers and creates a unified…, evaluate_single_asset_screener() (+10 more)

### Community 142 - "ws_insider_radar.py"
Cohesion: 0.22
Nodes (15): calculate_insider_conviction_score(), fetch_insider_transactions(), Any, Smart-Money Executive & Institutional Insider Radar for Sentilyze.…, Computes the Quantitative Insider Conviction Index (0 to 100 Score) and detects…, Screens a universe of tickers and returns the highest-ranking insider buying…, Fetches recent SEC Form 4 insider transactions for a specific ticker. Includes…, scan_universe_insider_catalysts() (+7 more)

### Community 143 - "quant_engine.py"
Cohesion: 0.19
Nodes (12): MasterQuantPipelineResult, Any, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), analyze_earnings_call_transcript(), Any (+4 more)

### Community 144 - "render_multi_agent_war_room"
Cohesion: 0.28
Nodes (7): Any, Interactive Multi-Agent War Room Visualizer Component for Streamlit. Functions:…, Renders the complete 5-Agent War Room Council deliberation chamber., render_multi_agent_war_room(), Real-Time Audio Trade Squawk Component for Streamlit. Functions: - Uses…, Renders an HTML5 Web Speech API audio squawk generator inside Streamlit., render_audio_squawk_button()

### Community 145 - "audio_briefing.py"
Cohesion: 0.47
Nodes (5): generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses Microsoft Edge Neural TTS…, synthesize_morning_audio()

### Community 146 - "PolyTimeConvexOptimizer"
Cohesion: 0.16
Nodes (9): PolyTimeConvexOptimizer, Any, DataFrame, Series, Stanford Multi-Period Convex Portfolio Optimization Engine (Boyd et al.).…, Polynomial-Time Convex Portfolio Optimizer with Market Frictions (Boyd et al.)., Solves the friction-aware convex optimization problem in polynomial time. Args:…, test_paper2_boyd_convex_optimizer() (+1 more)

### Community 152 - ".evaluate_catalyst_and_market"
Cohesion: 0.50
Nodes (3): Any, DataFrame, Executes complete Tri-Model evaluation on incoming headline & market structure.…

## Knowledge Gaps
- **224 isolated node(s):** `graphify`, `1. Core Principles`, `2. Use Cases in Sentilyze`, `1. Storage Architecture`, `2. Integration Pattern` (+219 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `utils.py` to `deep_learning_model.py`, `get_price_history`, `calculate_doubling_progress`, `daily_scanner.py`, `run_backtest`, `autonomous_trader.py`, `black_swan_simulator.py`, `drl_policy_agent.py`, `test_options_flow.py`, `TradingEnvironment`, `test_statistical_arbitrage.py`, `CloudDataLake`, `screener_engine.py`, `analyze_supply_chain_spillover`, `ws_insider_radar.py`, `quant_engine.py`, `calculate_hrp_weights`, `audio_briefing.py`, `PolyTimeConvexOptimizer`, `test_all_14_papers.py`, `meta_ensemble.py`, `compute_dark_pool_sentiment`, `test_omnichannel_mobile.py`, `OnlineNewtonStepOptimizer`, `run_temporal_fusion_forecast`, `compute_lead_lag_matrix`, `ws_live_prediction.py`, `social_sentiment.py`, `AlpacaBrokerBridge`, `realtime_tracker.py`, `test_onnx_and_orderflow.py`, `get_us_market_session`, `ipo_radar.py`, `triple_convex_engine.py`, `SuperEnsembleClassifier`, `SECCatalystCrawler`, `correlation_shield.py`, `TickerSentinelSwarm`, `universal_web_scraper.py`, `DuckDBMarketEngine`, `test_pillar2_alternative_data.py`, `price_scout.py`, `generate_morning_briefing_text`, `calculate_time_decayed_sentiment`, `analyze_sec_filing_diff`, `QuantAlphaDAGEngine`, `api.py`, `chart_pattern_learning.py`, `test_reddit_premarket_station.py`, `fetch_universe_live_quotes`, `test_tri_model_pipeline.py`, `agent_committee.py`, `generate_comprehensive_factsheet`, `correlation_matrix.py`, `SecurityMaster`, `advanced_quant_experiments.py`, `options_surface.py`, `red_team_agent.py`, `liquidity_heatmap.py`, `calculate_portfolio_diversity_grade`, `cli.py`, `preprocessing.py`, `main`, `run_universe_training.py`, `modeling.py`, `compute_gamma_exposure_profile`, `ws_deep_quant.py`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Why does `get_price_history()` connect `get_price_history` to `run_backtest`, `autonomous_trader.py`, `test_statistical_arbitrage.py`, `drl_policy_agent.py`, `utils.py`, `screener_engine.py`, `test_all_14_papers.py`, `realtime_tracker.py`, `correlation_shield.py`, `price_scout.py`, `ws_committee.py`, `generate_morning_briefing_text`, `ws_portfolio.py`, `QuantAlphaDAGEngine`, `preprocess_data`, `agent_committee.py`, `correlation_matrix.py`, `advanced_quant_experiments.py`, `red_team_agent.py`, `calculate_portfolio_diversity_grade`, `preprocessing.py`, `run_universe_training.py`, `ws_deep_quant.py`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `PaperBroker` connect `PaperBroker` to `daily_scanner.py`, `cli.py`, `autonomous_trader.py`, `drl_policy_agent.py`, `test_statistical_arbitrage.py`, `render_workspace_header`, `print_formatted_trade_report`, `ws_insider_radar.py`, `test_zero_giveback_profit_lock.py`, `generate_morning_briefing_text`, `ws_portfolio.py`, `realtime_tracker.py`, `AICopilotEngine`, `test_full_market_funnel.py`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `1. Core Principles`, `2. Use Cases in Sentilyze` to the rest of the system?**
  _224 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `deep_learning_model.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10887096774193548 - nodes in this community are weakly interconnected._
- **Should `get_price_history` be split into smaller, more focused modules?**
  _Cohesion score 0.062146892655367235 - nodes in this community are weakly interconnected._