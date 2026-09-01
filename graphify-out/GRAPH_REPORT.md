# Graph Report - Sentilyze  (2026-09-01)

## Corpus Check
- 485 files · ~673,186 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1284 nodes · 2722 edges · 62 communities (59 shown, 3 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `73990881`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- fetch_financial_statements
- get_price_history
- options_flow.py
- daily_scanner.py
- run_backtest
- app.py
- academic_papers_benchmark.py
- test_statistical_arbitrage.py
- agent_committee.py
- TradingEnvironment
- test_all_14_papers.py
- CloudDataLake
- ws_alternative_data.py
- SupplyChainGraphNetwork
- advanced_quant_experiments.py
- run_temporal_fusion_forecast
- ws_deep_quant.py
- meta_ensemble.py
- OnlineNewtonStepOptimizer
- ws_live_prediction.py
- social_sentiment.py
- compute_dark_pool_sentiment
- test_omnichannel_mobile.py
- PolyTimeConvexOptimizer
- utils.py
- compute_lead_lag_matrix
- black_swan_simulator.py
- test_pillar2_alternative_data.py
- AlpacaBrokerBridge
- telegram_bot.py
- PaperBroker
- datetime
- options_surface.py
- AICopilotEngine
- liquidity_heatmap.py
- ablation_study.py
- SuperEnsembleClassifier
- apply_triple_barrier_labeling
- Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform
- test_autonomous_trader.py
- render_workspace_header
- realtime_tracker.py
- AutonomousTradingEngine
- quant_engine.py
- ui/__init__.py
- Contributor Covenant Code of Conduct
- preprocessing.py
- calculate_doubling_progress
- compute_government_and_patent_index
- get_logger
- autonomous_trader.py
- How Can I Contribute?
- dispatcher.py
- 🧪 Experimental & Simulated Research Prototypes
- Sentilyze — Standing Audit Protocol
- ForensicFundamentalAgent
- run_unified_institutional_pipeline
- rules/graphify.md
- workflows/graphify.md

## God Nodes (most connected - your core abstractions)
1. `get_logger()` - 79 edges
2. `PaperBroker` - 44 edges
3. `get_price_history()` - 39 edges
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
- `test_paper9_when_agents_trade_scanner()` --calls--> `run_daily_market_scan()`  [EXTRACTED]
  tests/test_all_14_papers.py → src/daily_scanner.py
- `test_paper13_gnn_supply_chain()` --calls--> `analyze_supply_chain_spillover()`  [EXTRACTED]
  tests/test_all_14_papers.py → src/gnn_supply_chain.py
- `test_autonomous_cycle_execution()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py
- `test_idempotency_lock_prevents_overlap()` --uses--> `PaperBroker`  [INFERRED]
  tests/test_autonomous_trader.py → src/paper_broker.py

## Import Cycles
- None detected.

## Communities (62 total, 3 thin omitted)

### Community 0 - "fetch_financial_statements"
Cohesion: 0.20
Nodes (19): calculate_altman_z_score(), calculate_dcf_fair_value(), calculate_piotroski_f_score(), fetch_financial_statements(), _generate_calibrated_financials(), generate_spider_radar_profile(), Any, Institutional Fundamental Valuation & Forensic Accounting Engine for Sentilyze.… (+11 more)

### Community 1 - "get_price_history"
Cohesion: 0.05
Nodes (62): compute_correlation_matrix(), compute_cross_asset_correlation(), Any, DataFrame, Convenience wrapper returning correlation matrix and analytics dictionary., Computes cross-asset returns correlation matrix and identifies optimal hedge…, _fetch_alpaca_news(), _fetch_alpaca_price_history() (+54 more)

### Community 2 - "options_flow.py"
Cohesion: 0.17
Nodes (21): calculate_max_pain(), calculate_put_call_ratios(), estimate_gamma_exposure(), fetch_option_chain(), _generate_mock_option_chain(), Any, DataFrame, Live Options Microstructure, Gamma Exposure (GEX) & Max Pain Terminal for… (+13 more)

### Community 3 - "daily_scanner.py"
Cohesion: 0.15
Nodes (25): format_signal_card(), Any, Dispatches a high-impact Discord card for live autonomous trade lifecycle…, Construct a standardized trade signal data payload. Args: ticker (str): Stock…, Sends a comprehensive institutional morning macro regime, portfolio health,…, Sends a rich formatted trade alert card to a Discord channel via Webhook., Sends a consolidated master market digest card containing all universe signals., Sends a formatted Markdown alert to a Telegram chat or channel. (+17 more)

### Community 4 - "run_backtest"
Cohesion: 0.10
Nodes (35): Figure, _persist_attribution_results(), Any, Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine…, Runs a 4-way attribution experiment on a given asset using real out-of-sample…, run_attribution_decomposition(), calculate_performance_metrics(), _calculate_trade_outcomes() (+27 more)

### Community 5 - "app.py"
Cohesion: 0.15
Nodes (16): load_universe_tickers(), main(), Sentilyze - Institutional Algorithmic Trading & MLOps Platform. Modular Master…, Loads active S&P 100 universe tickers., inject_custom_theme(), Dynamic Bespoke Theme Engine for Sentilyze. Supports 3 Institutional Presets:…, Injects high-performance, bespoke CSS styling into the Streamlit app., Renders the 24/7 Autonomous Live Trading & News Agent interface. (+8 more)

### Community 6 - "academic_papers_benchmark.py"
Cohesion: 0.15
Nodes (14): Any, Master Academic Research Papers Empirical Benchmark Suite (All 14 Papers).…, Executes empirical backtests comparing all 14 academic paper methodologies., run_all_14_papers_benchmark(), calculate_almgren_chriss_trajectory(), Any, Paper 3: Almgren & Chriss (2000) - Optimal Execution of Portfolio Transactions.…, Computes Almgren-Chriss optimal trading trajectory. x_j = 2 * sinh(0.5 * kappa… (+6 more)

### Community 7 - "test_statistical_arbitrage.py"
Cohesion: 0.19
Nodes (25): backtest_pairs_strategy(), calculate_half_life(), calculate_hedge_ratio_and_spread(), calculate_rolling_zscore(), evaluate_cointegration_adf(), generate_pairs_trading_signals(), Any, Series (+17 more)

### Community 8 - "agent_committee.py"
Cohesion: 0.10
Nodes (30): audit_full_universe_committee(), ChiefRiskOfficerAgent, compute_fractional_kelly_sizing(), convene_trading_committee(), execute_committee_order(), _persist_committee_resolution(), Any, Autonomous Multi-Agent Trading Committee & Deliberation Engine for Sentilyze.… (+22 more)

### Community 9 - "TradingEnvironment"
Cohesion: 0.14
Nodes (16): optimize_rl_position_allocation(), PPOPolicyAgent, Any, ndarray, ⚠️ EXPERIMENTAL / RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM PRODUCTION…, Computes mean action (leverage) between 0.0 and 2.0., Estimates state value., Trains Actor-Critic parameters across historical episodes. (+8 more)

### Community 10 - "test_all_14_papers.py"
Cohesion: 0.15
Nodes (14): detect_negative_cycle_arbitrage(), Any, Paper 5: Negative Cycle Detection on Exchange Log-Rate Digraphs (Bellman-Ford).…, Finds triangular arbitrage using Bellman-Ford on log exchange rates: w =…, optimize_higher_order_moments(), Any, DataFrame, Paper 4: Xu, Deng et al. - Polynomial Portfolio Optimization (Moment-SOS).… (+6 more)

### Community 11 - "CloudDataLake"
Cohesion: 0.13
Nodes (15): CloudDataLake, Any, Cloud PostgreSQL Data Lake (Supabase / Neon) Connector for Sentilyze. Pillar 6…, Supabase / PostgreSQL Cloud Data Lake Connector., Validates or generates cloud database schema., Syncs local trade executions to the cloud database., Publishes real-time portfolio snapshot to cloud WebSockets channel., generate_twap_order_schedule() (+7 more)

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

### Community 23 - "PolyTimeConvexOptimizer"
Cohesion: 0.18
Nodes (9): PolyTimeConvexOptimizer, Any, DataFrame, Series, Stanford Multi-Period Convex Portfolio Optimization Engine (Boyd et al.).…, Polynomial-Time Convex Portfolio Optimizer with Market Frictions (Boyd et al.)., Solves the friction-aware convex optimization problem in polynomial time. Args:…, test_paper2_boyd_convex_optimizer() (+1 more)

### Community 24 - "utils.py"
Cohesion: 0.05
Nodes (64): FeatureContribution, health_check(), predict(), PredictionResponse, Fetches the latest market and sentiment data, computes technical indicators,…, root(), BaseModel, get (+56 more)

### Community 25 - "compute_lead_lag_matrix"
Cohesion: 0.20
Nodes (14): compute_lead_lag_matrix(), _granger_f_test(), Any, DataFrame, ndarray, Series, rank_market_price_leaders(), Lead-Lag Granger Causality & Supply Chain Price Discovery Engine for Sentilyze.… (+6 more)

### Community 26 - "black_swan_simulator.py"
Cohesion: 0.23
Nodes (11): calculate_kelly_sizing(), estimate_market_impact_slippage(), Any, Historical Black Swan Crisis Simulator & Kelly Position Sizing for Sentilyze.…, Calculates optimal position sizing using the Kelly Criterion: Kelly % = W - (1…, Estimates market execution slippage using the Almgren-Chriss square-root impact…, Stress-tests the current portfolio against major historical market crashes.…, simulate_portfolio_crises() (+3 more)

### Community 27 - "test_pillar2_alternative_data.py"
Cohesion: 0.21
Nodes (14): compute_smart_money_insider_score(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Retrieves recent SEC Form 4 insider transactions for a given stock., Retrieves recent Congressional STOCK Act disclosure reports for a ticker., Synthesizes SEC Form 4 and Congressional activity into an overall Smart Money…, track_congressional_stock_disclosures(), track_corporate_insider_filings() (+6 more)

### Community 28 - "AlpacaBrokerBridge"
Cohesion: 0.16
Nodes (14): AlpacaBrokerBridge, Any, Fetches active positions from Alpaca brokerage., Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.…, Verifies active connection to Alpaca Brokerage API., Fetches live Alpaca account equity, buying power, and cash., Submits an institutional Bracket Order: - Entry: Market order - Exit 1: Limit…, patch (+6 more)

### Community 29 - "telegram_bot.py"
Cohesion: 0.20
Nodes (16): build_interactive_inline_keyboard(), handle_telegram_command(), Any, 2-Way Interactive Telegram Bot Controller & Remote Execution Bridge for…, Sends a formatted markdown message with inline buttons to a Telegram chat., Builds interactive inline quick-action buttons for mobile Telegram., Parses and executes Telegram slash commands and callback buttons., send_telegram_bot_message() (+8 more)

### Community 30 - "PaperBroker"
Cohesion: 0.06
Nodes (36): Root entry point for Sentilyze CLI. Run directly with: python sentilyze.py NVDA…, cmd_audit(), cmd_briefing(), cmd_portfolio(), main(), print_banner(), Sentilyze Command-Line Interface (CLI). Interactive terminal tool for 4-Agent…, Displays current portfolio metrics and live holdings. (+28 more)

### Community 31 - "datetime"
Cohesion: 0.07
Nodes (41): datetime, generate_audio_script(), Any, Generates an institutional Wall Street morning audio briefing script., Synthesizes the morning briefing audio MP3 file. Uses gTTS if available, or…, synthesize_morning_audio(), check_market_hours_preflight(), get_current_ny_time() (+33 more)

### Community 32 - "options_surface.py"
Cohesion: 0.27
Nodes (10): calculate_multileg_payoff(), generate_volatility_surface_mesh(), Any, 3D Implied Volatility Surface & Multi-Leg Options Strategy Desk for Sentilyze.…, Constructs a 3D Implied Volatility Surface across strike prices and expiration…, Calculates profit and loss (P&L) curves at expiration for institutional multi-…, test_calculate_multileg_payoff_bull_call_spread(), test_calculate_multileg_payoff_iron_condor() (+2 more)

### Community 33 - "AICopilotEngine"
Cohesion: 0.24
Nodes (8): AICopilotEngine, Any, Conversational intelligence engine that parses queries and generates analytical…, Interprets user prompt and routes to appropriate financial analytical…, test_copilot_committee_query(), test_copilot_portfolio_query(), test_copilot_stress_query(), test_copilot_ticker_analysis_query()

### Community 34 - "liquidity_heatmap.py"
Cohesion: 0.31
Nodes (8): compute_order_book_depth_and_clusters(), compute_volume_profile_and_poc(), Any, Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap for…, Simulates Level 2 market depth and identifies institutional buy/sell liquidity…, Computes Point of Control (POC), Value Area High (VAH), and Value Area Low…, test_compute_order_book_depth_and_clusters(), test_compute_volume_profile_and_poc()

### Community 35 - "ablation_study.py"
Cohesion: 0.33
Nodes (8): _persist_ablation_results(), Any, 4-Agent Trading Committee Ablation Study Engine for Sentilyze. Evaluates the…, Runs committee ablation study across multiple assets and returns aggregated…, Runs systematic ablation backtests comparing all 5 committee configurations.…, run_committee_ablation_backtest(), run_multi_ticker_ablation_study(), test_committee_ablation_study_execution()

### Community 36 - "SuperEnsembleClassifier"
Cohesion: 0.11
Nodes (16): Any, DataFrame, ndarray, Series, Institutional 3-Way Super-Ensemble & Stacking Meta-Learner for Sentilyze.…, Predicts directional momentum class (0 or 1)., Calculates individual predictions and consensus score for transparency., Saves all 3 models natively using secure serialization. No pickle/joblib used. (+8 more)

### Community 37 - "apply_triple_barrier_labeling"
Cohesion: 0.21
Nodes (11): apply_triple_barrier_labeling(), calculate_deflated_sharpe_ratio(), DataFrame, Series, Marcos López de Prado's Triple-Barrier Method & Deflated Sharpe Ratio (DSR).…, Computes Bailey & López de Prado's Deflated Sharpe Ratio (DSR). Adjusts for: -…, Applies López de Prado's path-dependent Triple-Barrier Method to generate trade…, test_paper10_deflated_sharpe_ratio() (+3 more)

### Community 38 - "Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform"
Cohesion: 0.12
Nodes (16): 1. Run in 1-Click (No Installation Required), 2. Local Setup & Installation, 3. Run the 4-Agent Quantitative CLI, 4. Launch the Streamlit Mission Control, 5. Run Automated Multi-Agent Backtests & Full Test Suite, 🎯 Asymmetric Risk Management & Staged Profit Scaling, 📊 Empirical Alpha Attribution & Benchmarks, 🏛️ Grounded 4-Agent Deliberation Council (+8 more)

### Community 39 - "test_autonomous_trader.py"
Cohesion: 0.10
Nodes (19): is_kill_switch_active(), load_universe_tickers(), Task 7: Master Kill Switch Check. Returns True if SENTILYZE_KILL_SWITCH…, Loads universe of tickers from stocks.txt., clean_lock_file(), fixture, patch, Task 7: Verify master kill switch disables order placement. (+11 more)

### Community 40 - "render_workspace_header"
Cohesion: 0.12
Nodes (16): get_market_status(), Any, Shared Institutional UI Components & Widgets for Sentilyze. Includes Live US…, Wraps HTML content inside an institutional frosted glass container., Renders a progress meter with dynamic color coding., Calculates live US Market (NYSE/NASDAQ) status based on Eastern Time., Renders an executive header banner with live status badge and market clock., render_conviction_gauge() (+8 more)

### Community 41 - "realtime_tracker.py"
Cohesion: 0.05
Nodes (46): Sends a sleek, institutional Discord embed with live prices, PnL, and distance…, send_discord_holdings_heartbeat(), fetch_universe_live_quotes(), _get_browser_session(), get_us_market_session_info(), Any, Session, Fetches real-time quotes across the entire universe concurrently in parallel. (+38 more)

### Community 42 - "AutonomousTradingEngine"
Cohesion: 0.21
Nodes (10): AutonomousTradingEngine, check_daily_loss_circuit_breaker(), Any, Autonomous Execution Engine that integrates Live News Ingestion, 4-Agent…, Dispatches an institutional execution alert to Discord Webhook if configured., Executes one full autonomous decision and execution cycle with: - Task 6:…, Core cycle execution body., Task 8: Independent Max-Daily-Loss Circuit Breaker. Compares current total… (+2 more)

### Community 43 - "quant_engine.py"
Cohesion: 0.21
Nodes (10): MasterQuantPipelineResult, Master Institutional Quantitative Orchestrator for Sentilyze. Unifies all 8…, Strongly-typed container for end-to-end unified institutional analysis., analyze_sec_filing_diff(), compute_text_similarity_and_diff(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Computes lexical and semantic diff metrics between consecutive filings. (+2 more)

### Community 47 - "Contributor Covenant Code of Conduct"
Cohesion: 0.15
Nodes (12): 1. Correction, 2. Warning, 3. Temporary Ban, 4. Permanent Ban, Attribution, Contributor Covenant Code of Conduct, Enforcement, Enforcement Guidelines (+4 more)

### Community 48 - "preprocessing.py"
Cohesion: 0.05
Nodes (58): aggregate_sentiment_scores(), create_features(), create_technical_indicators(), DataFrame, Aggregate sentiment scores per day by resampling. Args: news_with_sentiment…, Merges price history with daily sentiment scores and VIX data to create a…, Create technical indicators from price history. Args: price_history…, clean_headline_data() (+50 more)

### Community 49 - "calculate_doubling_progress"
Cohesion: 0.27
Nodes (9): calculate_doubling_progress(), compute_compound_position_size(), Any, Max Compound Acceleration & +100% Target Doubling Engine for Sentilyze.…, Computes dynamic equity-scaled position sizing so trade sizes grow…, Computes exact mathematical progress, run-rate, and remaining cycles to reach…, Unit tests for Max Compound Acceleration Engine., test_calculate_doubling_progress() (+1 more)

### Community 50 - "compute_government_and_patent_index"
Cohesion: 0.33
Nodes (9): compute_government_and_patent_index(), Any, ⚠️ EXPERIMENTAL / SIMULATED RESEARCH PROTOTYPE STATUS: DISCONNECTED FROM…, Synthesizes federal contracting dollars and patent velocity into a single…, Retrieves recent prime federal government contract awards for a company., Tracks recent USPTO patent grants in AI/ML, Semiconductor Design, and Cloud…, track_federal_contract_awards(), track_uspto_patent_momentum() (+1 more)

### Community 51 - "get_logger"
Cohesion: 0.40
Nodes (5): Logger, get_logger(), Configures and returns a logger with a standard format and utf-8 safe…, Tests that the get_logger function returns a configured logger., test_get_logger()

### Community 52 - "autonomous_trader.py"
Cohesion: 0.25
Nodes (8): Dispatches a structured 4-Agent Committee Round-Table debate summary to Discord., Dispatches real-time Reddit r/wallstreetbets & Stocktwits hype spike alerts., send_discord_committee_alert(), send_discord_social_spike_alert(), Autonomous Live Trading & News Intelligence Engine for Sentilyze. Institutional…, Runs the Autonomous Trading Engine continuously on an interval., run_autonomous_daemon(), test_send_discord_committee_and_pulse()

### Community 53 - "How Can I Contribute?"
Cohesion: 0.33
Nodes (5): Contributing to Sentilyze, How Can I Contribute?, Pull Requests, Reporting Bugs, Suggesting Enhancements

### Community 54 - "dispatcher.py"
Cohesion: 0.40
Nodes (5): Any, Sends Telegram formatted market digest message., Dispatches formatted HTML morning market digest email via Gmail SMTP., send_email_digest(), send_telegram_digest()

### Community 55 - "🧪 Experimental & Simulated Research Prototypes"
Cohesion: 0.50
Nodes (3): 🧪 Experimental & Simulated Research Prototypes, 🔒 Production Isolation Guarantee, 📁 Prototype Inventory

### Community 56 - "Sentilyze — Standing Audit Protocol"
Cohesion: 0.17
Nodes (11): 1. Fabricated / fake data check, 2. Ticker/input-invariant bugs, 3. Mislabeled methodology, 4. Results-file / README consistency, 5. Duplicate-output smell test, 6. Safety-critical logic sanity check, 7. Silent failure check, 8. Scope creep check (+3 more)

### Community 57 - "ForensicFundamentalAgent"
Cohesion: 0.50
Nodes (3): ForensicFundamentalAgent, Agent 3: Evaluates Real Financial Statements, Piotroski F-Score, and DCF…, test_forensic_fundamental_agent()

### Community 61 - "run_unified_institutional_pipeline"
Cohesion: 0.47
Nodes (5): Any, Executes all 8 quantitative pillars in a synchronized machine flow with zero…, run_unified_institutional_pipeline(), test_all_8_pillars_present_in_output(), test_run_unified_institutional_pipeline()

## Knowledge Gaps
- **40 isolated node(s):** `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs`, `3. Mislabeled methodology`, `4. Results-file / README consistency` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `get_logger` to `fetch_financial_statements`, `get_price_history`, `options_flow.py`, `daily_scanner.py`, `run_backtest`, `academic_papers_benchmark.py`, `test_statistical_arbitrage.py`, `agent_committee.py`, `TradingEnvironment`, `test_all_14_papers.py`, `CloudDataLake`, `ws_alternative_data.py`, `SupplyChainGraphNetwork`, `advanced_quant_experiments.py`, `run_temporal_fusion_forecast`, `ws_deep_quant.py`, `meta_ensemble.py`, `OnlineNewtonStepOptimizer`, `ws_live_prediction.py`, `social_sentiment.py`, `compute_dark_pool_sentiment`, `test_omnichannel_mobile.py`, `PolyTimeConvexOptimizer`, `utils.py`, `compute_lead_lag_matrix`, `black_swan_simulator.py`, `test_pillar2_alternative_data.py`, `AlpacaBrokerBridge`, `telegram_bot.py`, `PaperBroker`, `datetime`, `options_surface.py`, `liquidity_heatmap.py`, `ablation_study.py`, `SuperEnsembleClassifier`, `apply_triple_barrier_labeling`, `realtime_tracker.py`, `quant_engine.py`, `preprocessing.py`, `calculate_doubling_progress`, `compute_government_and_patent_index`, `autonomous_trader.py`, `dispatcher.py`?**
  _High betweenness centrality (0.238) - this node is a cross-community bridge._
- **Why does `get_price_history()` connect `get_price_history` to `ablation_study.py`, `run_backtest`, `app.py`, `academic_papers_benchmark.py`, `test_statistical_arbitrage.py`, `agent_committee.py`, `realtime_tracker.py`, `render_workspace_header`, `advanced_quant_experiments.py`, `preprocessing.py`, `utils.py`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `PaperBroker` connect `PaperBroker` to `AICopilotEngine`, `daily_scanner.py`, `test_autonomous_trader.py`, `agent_committee.py`, `realtime_tracker.py`, `AutonomousTradingEngine`, `autonomous_trader.py`, `utils.py`, `telegram_bot.py`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `PaperBroker` (e.g. with `AICopilotEngine` and `AutonomousTradingEngine`) actually correct?**
  _`PaperBroker` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `1. Fabricated / fake data check`, `2. Ticker/input-invariant bugs` to the rest of the system?**
  _40 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `get_price_history` be split into smaller, more focused modules?**
  _Cohesion score 0.05267778753292362 - nodes in this community are weakly interconnected._
- **Should `run_backtest` be split into smaller, more focused modules?**
  _Cohesion score 0.09716599190283401 - nodes in this community are weakly interconnected._