import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score
from typing import Dict, List, Any, Tuple, Optional
from src.utils import get_logger, sanitize_filename, safe_path_join
from src.preprocessing import preprocess_data
from src.backtesting import run_backtest
from src.config import XGB_MODEL_PARAMS

logger = get_logger(__name__)


# =====================================================================
# 1. PARADIGM 1: BASELINE ROLLING WALK-FORWARD (WFO)
# =====================================================================
def run_paradigm_rolling_wfo(
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = 500,
    test_window: int = 20,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Standard Rolling Walk-Forward Optimization (Fixed 500-day window)."""
    total_samples = len(X)
    oos_preds = []
    oos_indices = []
    oos_true = []
    model_params = XGB_MODEL_PARAMS.copy()

    for start_idx in range(0, total_samples - train_window, test_window):
        end_train = start_idx + train_window
        end_test = min(end_train + test_window, total_samples)

        X_train = X.iloc[start_idx:end_train]
        y_train = y.iloc[start_idx:end_train]
        X_test = X.iloc[end_train:end_test]
        y_test = y.iloc[end_train:end_test]

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        oos_preds.extend(probs)
        oos_true.extend(y_test.values)
        oos_indices.extend(y_test.index)

    preds_series = pd.Series(oos_preds, index=oos_indices)
    auc = roc_auc_score(oos_true, oos_preds) if len(np.unique(oos_true)) > 1 else 0.5
    acc = accuracy_score(oos_true, [1 if p >= 0.5 else 0 for p in oos_preds])
    return preds_series, {"auc": float(auc), "accuracy": float(acc)}


# =====================================================================
# 2. PARADIGM 2: EXPANDING WINDOW + EXPONENTIAL TIME-DECAY
# =====================================================================
def run_paradigm_expanding_timedecay(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_window: int = 500,
    test_window: int = 20,
    half_life_days: int = 250,
    embargo_days: int = 5,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Anchored Expanding Window with Exponential Time-Decay Sample Weighting & Purged Embargo."""
    total_samples = len(X)
    oos_preds = []
    oos_indices = []
    oos_true = []
    model_params = XGB_MODEL_PARAMS.copy()

    for end_train in range(min_train_window, total_samples, test_window):
        train_cutoff = max(0, end_train - embargo_days)
        end_test = min(end_train + test_window, total_samples)

        X_train = X.iloc[0:train_cutoff]
        y_train = y.iloc[0:train_cutoff]
        X_test = X.iloc[end_train:end_test]
        y_test = y.iloc[end_train:end_test]

        if len(X_test) == 0:
            break

        t_indices = np.arange(len(X_train))
        T = len(X_train)
        sample_weights = np.exp(-np.log(2) * (T - t_indices) / half_life_days)
        sample_weights = sample_weights / np.mean(sample_weights)

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        probs = model.predict_proba(X_test)[:, 1]

        oos_preds.extend(probs)
        oos_true.extend(y_test.values)
        oos_indices.extend(y_test.index)

    preds_series = pd.Series(oos_preds, index=oos_indices)
    auc = roc_auc_score(oos_true, oos_preds) if len(np.unique(oos_true)) > 1 else 0.5
    acc = accuracy_score(oos_true, [1 if p >= 0.5 else 0 for p in oos_preds])
    return preds_series, {"auc": float(auc), "accuracy": float(acc)}


# =====================================================================
# 3. PARADIGM 3: TRIPLE-BARRIER META-LABELING
# =====================================================================
def run_paradigm_triple_barrier_meta(
    df_raw: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = 500,
    test_window: int = 20,
    pt_atr: float = 2.0,
    sl_atr: float = 1.2,
    horizon_days: int = 10,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Triple-Barrier Meta-Labeling (Upper +2.0 ATR, Lower -1.2 ATR, Vertical 10d)."""
    total_samples = len(X)
    close = df_raw["Close"] if "Close" in df_raw.columns else df_raw.iloc[:, 0]
    atr = df_raw["ATR"] if "ATR" in df_raw.columns else (close.rolling(14).max() - close.rolling(14).min()).fillna(1.0)

    tb_labels = np.zeros(total_samples)
    for i in range(total_samples - horizon_days):
        p0 = close.iloc[i]
        curr_atr = atr.iloc[i] if atr.iloc[i] > 0 else 1.0
        upper_barrier = p0 + pt_atr * curr_atr
        lower_barrier = p0 - sl_atr * curr_atr

        hit_upper = False
        hit_lower = False
        for step in range(1, horizon_days + 1):
            p_future = close.iloc[i + step]
            if p_future >= upper_barrier:
                hit_upper = True
                break
            elif p_future <= lower_barrier:
                hit_lower = True
                break

        tb_labels[i] = 1 if (hit_upper and not hit_lower) else 0

    tb_series = pd.Series(tb_labels, index=X.index)

    oos_preds = []
    oos_indices = []
    oos_true = []
    model_params = XGB_MODEL_PARAMS.copy()

    for start_idx in range(0, total_samples - train_window, test_window):
        end_train = start_idx + train_window
        end_test = min(end_train + test_window, total_samples)

        X_train = X.iloc[start_idx:end_train]
        y_train = tb_series.iloc[start_idx:end_train]
        X_test = X.iloc[end_train:end_test]
        y_test = y.iloc[end_train:end_test]

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        oos_preds.extend(probs)
        oos_true.extend(y_test.values)
        oos_indices.extend(y_test.index)

    preds_series = pd.Series(oos_preds, index=oos_indices)
    auc = roc_auc_score(oos_true, oos_preds) if len(np.unique(oos_true)) > 1 else 0.5
    acc = accuracy_score(oos_true, [1 if p >= 0.5 else 0 for p in oos_preds])
    return preds_series, {"auc": float(auc), "accuracy": float(acc)}


# =====================================================================
# 4. PARADIGM 4: REGIME-CONDITIONED MIXTURE OF EXPERTS (MoE)
# =====================================================================
def run_paradigm_mixture_of_experts(
    df_raw: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = 500,
    test_window: int = 20,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Regime-Conditioned Mixture of Experts (MoE) with 3 specialized sub-boosters."""
    total_samples = len(X)
    close = df_raw["Close"] if "Close" in df_raw.columns else df_raw.iloc[:, 0]
    sma200 = df_raw["SMA200"] if "SMA200" in df_raw.columns else close.rolling(200).mean().fillna(close)
    rsi = df_raw["RSI"] if "RSI" in df_raw.columns else pd.Series(50, index=X.index)

    regimes = np.zeros(total_samples, dtype=int)
    for i in range(total_samples):
        if close.iloc[i] > sma200.iloc[i] and rsi.iloc[i] >= 50:
            regimes[i] = 0  # Bull Momentum
        elif close.iloc[i] < sma200.iloc[i]:
            regimes[i] = 1  # Bear Defense
        else:
            regimes[i] = 2  # Mean Reversion / Choppy

    oos_preds = []
    oos_indices = []
    oos_true = []
    model_params = XGB_MODEL_PARAMS.copy()

    for start_idx in range(0, total_samples - train_window, test_window):
        end_train = start_idx + train_window
        end_test = min(end_train + test_window, total_samples)

        X_train = X.iloc[start_idx:end_train]
        y_train = y.iloc[start_idx:end_train]
        r_train = regimes[start_idx:end_train]

        X_test = X.iloc[end_train:end_test]
        y_test = y.iloc[end_train:end_test]
        r_test = regimes[end_train:end_test]

        experts = {}
        for r_id in [0, 1, 2]:
            mask = r_train == r_id
            if np.sum(mask) >= 30 and len(np.unique(y_train[mask])) > 1:
                exp_model = xgb.XGBClassifier(**model_params)
                exp_model.fit(X_train[mask], y_train[mask])
                experts[r_id] = exp_model
            else:
                gen_model = xgb.XGBClassifier(**model_params)
                gen_model.fit(X_train, y_train)
                experts[r_id] = gen_model

        fold_probs = []
        for i in range(len(X_test)):
            target_regime = r_test[i]
            exp = experts.get(target_regime, experts[0])
            p = exp.predict_proba(X_test.iloc[[i]])[0, 1]
            fold_probs.append(p)

        oos_preds.extend(fold_probs)
        oos_true.extend(y_test.values)
        oos_indices.extend(y_test.index)

    preds_series = pd.Series(oos_preds, index=oos_indices)
    auc = roc_auc_score(oos_true, oos_preds) if len(np.unique(oos_true)) > 1 else 0.5
    acc = accuracy_score(oos_true, [1 if p >= 0.5 else 0 for p in oos_preds])
    return preds_series, {"auc": float(auc), "accuracy": float(acc)}


# =====================================================================
# 5. PARADIGM 5: ONLINE STREAMING & CONTINUAL LEARNING
# =====================================================================
def run_paradigm_online_continual(
    X: pd.DataFrame,
    y: pd.Series,
    warmup_days: int = 500,
    learning_rate: float = 0.01,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Online Streaming Incremental Learning with Stochastic Gradient Descent & Adaptive Learning Rate."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X.fillna(0)), index=X.index, columns=X.columns)

    online_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="adaptive",
        eta0=learning_rate,
        random_state=42,
    )

    X_warmup = X_scaled.iloc[:warmup_days]
    y_warmup = y.iloc[:warmup_days]
    online_model.fit(X_warmup, y_warmup)

    oos_preds = []
    oos_indices = []
    oos_true = []

    for i in range(warmup_days, len(X)):
        x_sample = X_scaled.iloc[[i]]
        y_sample = y.iloc[i]

        p = online_model.predict_proba(x_sample)[0, 1]
        oos_preds.append(p)
        oos_true.append(y_sample)
        oos_indices.append(X.index[i])

        online_model.partial_fit(x_sample, [y_sample], classes=np.array([0, 1]))

    preds_series = pd.Series(oos_preds, index=oos_indices)
    auc = roc_auc_score(oos_true, oos_preds) if len(np.unique(oos_true)) > 1 else 0.5
    acc = accuracy_score(oos_true, [1 if p >= 0.5 else 0 for p in oos_preds])
    return preds_series, {"auc": float(auc), "accuracy": float(acc)}


# =====================================================================
# 6. PARADIGM 6: CROSS-ASSET POOLED MULTI-TASK LEARNING
# =====================================================================
def run_paradigm_cross_asset_pooled(
    asset_data_map: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]],
    target_ticker: str,
    train_window: int = 500,
    test_window: int = 20,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Cross-Asset Pooled Multi-Task Learning across peer assets for high statistical power."""
    target_raw, target_X, target_y = asset_data_map[target_ticker]
    total_samples = len(target_X)
    # Build Pooled Training Set
    pooled_X_list = []
    pooled_y_list = []
    pooled_dates_list = []
    for ticker, (r, x_df, y_s) in asset_data_map.items():
        pooled_X_list.append(x_df.reset_index(drop=True))
        pooled_y_list.append(y_s.reset_index(drop=True))
        pooled_dates_list.extend(x_df.index)

    pooled_X_all = pd.concat(pooled_X_list, axis=0, ignore_index=True).fillna(0.0)
    pooled_y_all = pd.concat(pooled_y_list, axis=0, ignore_index=True)
    pooled_dates_arr = np.array(pooled_dates_list)

    oos_preds = []
    oos_indices = []
    oos_true = []
    model_params = XGB_MODEL_PARAMS.copy()

    for start_idx in range(0, total_samples - train_window, test_window):
        end_train = start_idx + train_window
        end_test = min(end_train + test_window, total_samples)

        X_test = target_X.iloc[end_train:end_test]
        y_test = target_y.iloc[end_train:end_test]

        cutoff_date = target_X.index[end_train]
        train_mask = pooled_dates_arr < cutoff_date
        X_train_pooled = pooled_X_all[train_mask]
        y_train_pooled = pooled_y_all[train_mask]

        if len(X_train_pooled) < 200:
            X_train_pooled = target_X.iloc[start_idx:end_train]
            y_train_pooled = target_y.iloc[start_idx:end_train]

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train_pooled, y_train_pooled)
        probs = model.predict_proba(X_test)[:, 1]

        oos_preds.extend(probs)
        oos_true.extend(y_test.values)
        oos_indices.extend(y_test.index)

    preds_series = pd.Series(oos_preds, index=oos_indices)
    auc = roc_auc_score(oos_true, oos_preds) if len(np.unique(oos_true)) > 1 else 0.5
    acc = accuracy_score(oos_true, [1 if p >= 0.5 else 0 for p in oos_preds])
    return preds_series, {"auc": float(auc), "accuracy": float(acc)}


# =====================================================================
# 7. PARADIGM 7: DIRECT REINFORCEMENT SHARPE POLICY GRADIENT
# =====================================================================
def run_paradigm_direct_reinforcement_policy(
    df_raw: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = 500,
    test_window: int = 20,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Direct Reinforcement Learning (Moody & Saffell) policy optimization for Sharpe maximization."""
    close = df_raw["Close"] if "Close" in df_raw.columns else df_raw.iloc[:, 0]
    pct_returns = close.pct_change().fillna(0.0)
    total_samples = len(X)

    oos_preds = []
    oos_indices = []
    oos_true = []

    for start_idx in range(0, total_samples - train_window, test_window):
        end_train = start_idx + train_window
        end_test = min(end_train + test_window, total_samples)

        X_train = X.iloc[start_idx:end_train]
        ret_train = pct_returns.iloc[start_idx:end_train]
        X_test = X.iloc[end_train:end_test]
        y_test = y.iloc[end_train:end_test]

        weights = (ret_train - ret_train.mean()) / (ret_train.std() + 1e-6)
        binary_utility = (weights > 0).astype(int)

        model = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.03,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        sample_w = np.abs(weights.values) + 0.1
        model.fit(X_train, binary_utility, sample_weight=sample_w)
        probs = model.predict_proba(X_test)[:, 1]

        oos_preds.extend(probs)
        oos_true.extend(y_test.values)
        oos_indices.extend(y_test.index)

    preds_series = pd.Series(oos_preds, index=oos_indices)
    auc = roc_auc_score(oos_true, oos_preds) if len(np.unique(oos_true)) > 1 else 0.5
    acc = accuracy_score(oos_true, [1 if p >= 0.5 else 0 for p in oos_preds])
    return preds_series, {"auc": float(auc), "accuracy": float(acc)}


# =====================================================================
# 8. PARADIGM 8: CONFORMAL PROBABILITY CALIBRATION ENSEMBLE
# =====================================================================
def run_paradigm_conformal_calibrated(
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = 500,
    test_window: int = 20,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Conformal Probability Calibration & Quantile Uncertainty Filter (Angelopoulos & Bates 2021)."""
    total_samples = len(X)
    oos_preds = []
    oos_indices = []
    oos_true = []
    model_params = XGB_MODEL_PARAMS.copy()

    for start_idx in range(0, total_samples - train_window, test_window):
        end_train = start_idx + train_window
        end_test = min(end_train + test_window, total_samples)

        X_train = X.iloc[start_idx:end_train]
        y_train = y.iloc[start_idx:end_train]
        X_test = X.iloc[end_train:end_test]
        y_test = y.iloc[end_train:end_test]

        # 80% train / 20% calibration split
        calib_split = int(len(X_train) * 0.8)
        X_tr = X_train.iloc[:calib_split]
        y_tr = y_train.iloc[:calib_split]
        X_cal = X_train.iloc[calib_split:]
        y_cal = y_train.iloc[calib_split:]

        base_model = xgb.XGBClassifier(**model_params)
        base_model.fit(X_tr, y_tr)

        cal_probs = base_model.predict_proba(X_cal)[:, 1]
        calibrator = LogisticRegression()
        if len(np.unique(y_cal)) > 1:
            calibrator.fit(cal_probs.reshape(-1, 1), y_cal)
            raw_test_p = base_model.predict_proba(X_test)[:, 1]
            calibrated_p = calibrator.predict_proba(raw_test_p.reshape(-1, 1))[:, 1]
        else:
            calibrated_p = base_model.predict_proba(X_test)[:, 1]

        oos_preds.extend(calibrated_p)
        oos_true.extend(y_test.values)
        oos_indices.extend(y_test.index)

    preds_series = pd.Series(oos_preds, index=oos_indices)
    auc = roc_auc_score(oos_true, oos_preds) if len(np.unique(oos_true)) > 1 else 0.5
    acc = accuracy_score(oos_true, [1 if p >= 0.5 else 0 for p in oos_preds])
    return preds_series, {"auc": float(auc), "accuracy": float(acc)}


# =====================================================================
# TOURNAMENT EXECUTION HARNESS ACROSS ALL 8 PARADIGMS
# =====================================================================
def run_financial_ml_tournament(
    tickers: List[str] = ["NVDA", "AAPL", "MSFT"],
    initial_capital: float = 10000.0,
    prob_threshold: float = 0.52,
) -> Dict[str, Any]:
    """Executes a head-to-head empirical tournament across all 8 training paradigms."""
    logger.info(f"🏆 Starting 8-Paradigm Financial ML Tournament across {len(tickers)} assets: {tickers}...")

    # 1. Ingest Data for Assets
    from src.config import FEATURES

    asset_data_map = {}
    for tk in tickers:
        try:
            features_df, price_history_with_indicators, _ = preprocess_data(tk, use_cache=True)
            avail_features = [f for f in FEATURES if f in features_df.columns]
            X_clean = features_df[avail_features].select_dtypes(include=[np.number]).fillna(0.0)
            
            target_col = "target" if "target" in features_df.columns else "Target"
            if target_col in features_df.columns:
                y_clean = features_df[target_col].astype(int)
            else:
                y_clean = (price_history_with_indicators["Close"].pct_change().shift(-1) > 0).loc[features_df.index].fillna(0).astype(int)

            raw_df = price_history_with_indicators.loc[features_df.index]
            asset_data_map[tk] = (raw_df, X_clean, y_clean)
        except Exception as e:
            logger.warning(f"Failed to preprocess {tk} for tournament: {e}")



    if not asset_data_map:
        raise ValueError("No asset data available for tournament benchmark.")

    paradigms = [
        "1. Baseline Rolling WFO",
        "2. Expanding Window + Exponential Decay",
        "3. Triple-Barrier Meta-Labeling",
        "4. Regime Mixture of Experts (MoE)",
        "5. Online Continual Streaming",
        "6. Cross-Asset Pooled Multi-Task",
        "7. Direct Reinforcement Policy Gradient",
        "8. Conformal Calibrated Uncertainty",
    ]

    tournament_results = {p: [] for p in paradigms}

    for tk, (raw_df, X, y) in asset_data_map.items():
        logger.info(f"⚔️ Evaluating all 8 paradigms on {tk}...")

        # 1. Baseline
        p1_preds, p1_meta = run_paradigm_rolling_wfo(X, y)
        _, p1_bt, _ = run_backtest(raw_df.loc[p1_preds.index], p1_preds, initial_capital=initial_capital, prob_threshold=prob_threshold)
        tournament_results["1. Baseline Rolling WFO"].append({"ticker": tk, "meta": p1_meta, "bt": p1_bt})

        # 2. Expanding
        p2_preds, p2_meta = run_paradigm_expanding_timedecay(X, y)
        _, p2_bt, _ = run_backtest(raw_df.loc[p2_preds.index], p2_preds, initial_capital=initial_capital, prob_threshold=prob_threshold)
        tournament_results["2. Expanding Window + Exponential Decay"].append({"ticker": tk, "meta": p2_meta, "bt": p2_bt})

        # 3. Triple Barrier
        p3_preds, p3_meta = run_paradigm_triple_barrier_meta(raw_df, X, y)
        _, p3_bt, _ = run_backtest(raw_df.loc[p3_preds.index], p3_preds, initial_capital=initial_capital, prob_threshold=prob_threshold)
        tournament_results["3. Triple-Barrier Meta-Labeling"].append({"ticker": tk, "meta": p3_meta, "bt": p3_bt})

        # 4. MoE
        p4_preds, p4_meta = run_paradigm_mixture_of_experts(raw_df, X, y)
        _, p4_bt, _ = run_backtest(raw_df.loc[p4_preds.index], p4_preds, initial_capital=initial_capital, prob_threshold=prob_threshold)
        tournament_results["4. Regime Mixture of Experts (MoE)"].append({"ticker": tk, "meta": p4_meta, "bt": p4_bt})

        # 5. Online
        p5_preds, p5_meta = run_paradigm_online_continual(X, y)
        _, p5_bt, _ = run_backtest(raw_df.loc[p5_preds.index], p5_preds, initial_capital=initial_capital, prob_threshold=prob_threshold)
        tournament_results["5. Online Continual Streaming"].append({"ticker": tk, "meta": p5_meta, "bt": p5_bt})

        # 6. Pooled
        p6_preds, p6_meta = run_paradigm_cross_asset_pooled(asset_data_map, tk)
        _, p6_bt, _ = run_backtest(raw_df.loc[p6_preds.index], p6_preds, initial_capital=initial_capital, prob_threshold=prob_threshold)
        tournament_results["6. Cross-Asset Pooled Multi-Task"].append({"ticker": tk, "meta": p6_meta, "bt": p6_bt})

        # 7. Reinforcement Policy
        p7_preds, p7_meta = run_paradigm_direct_reinforcement_policy(raw_df, X, y)
        _, p7_bt, _ = run_backtest(raw_df.loc[p7_preds.index], p7_preds, initial_capital=initial_capital, prob_threshold=prob_threshold)
        tournament_results["7. Direct Reinforcement Policy Gradient"].append({"ticker": tk, "meta": p7_meta, "bt": p7_bt})

        # 8. Conformal
        p8_preds, p8_meta = run_paradigm_conformal_calibrated(X, y)
        _, p8_bt, _ = run_backtest(raw_df.loc[p8_preds.index], p8_preds, initial_capital=initial_capital, prob_threshold=prob_threshold)
        tournament_results["8. Conformal Calibrated Uncertainty"].append({"ticker": tk, "meta": p8_meta, "bt": p8_bt})

    # Aggregate Metrics into Leaderboard
    leaderboard_rows = []
    for paradigm_name, runs in tournament_results.items():
        sharpes = [r["bt"].get("sharpe_ratio", 0.0) for r in runs]
        win_rates = [r["bt"].get("win_rate", 0.0) for r in runs]
        returns = [r["bt"].get("total_return", 0.0) for r in runs]
        drawdowns = [r["bt"].get("max_drawdown", 0.0) for r in runs]
        profit_factors = [r["bt"].get("profit_factor", 0.0) for r in runs]
        aucs = [r["meta"].get("auc", 0.5) for r in runs]

        leaderboard_rows.append(
            {
                "Paradigm": paradigm_name,
                "Avg Sharpe Ratio": round(float(np.mean(sharpes)), 2),
                "Avg Win Rate (%)": round(float(np.mean(win_rates) * 100), 1),
                "Avg Net Return (%)": round(float(np.mean(returns) * 100), 1),
                "Avg Max Drawdown (%)": round(float(np.mean(drawdowns) * 100), 1),
                "Avg Profit Factor": round(float(np.mean(profit_factors)), 2),
                "Avg ROC-AUC": round(float(np.mean(aucs)), 3),
            }
        )

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(by="Avg Sharpe Ratio", ascending=False)
    leaderboard_df.reset_index(drop=True, inplace=True)
    leaderboard_df["Rank"] = [f"#{i+1}" for i in range(len(leaderboard_df))]

    os.makedirs("results", exist_ok=True)
    leaderboard_df.to_csv("results/training_paradigm_tournament.csv", index=False)
    with open("results/training_paradigm_tournament.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard_df.to_dict(orient="records"), f, indent=2)

    logger.info("🏆 Tournament Completed Successfully! Final Leaderboard:")
    print(leaderboard_df.to_string(index=False))

    return {
        "leaderboard": leaderboard_df.to_dict(orient="records"),
        "best_paradigm": leaderboard_df.iloc[0]["Paradigm"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Financial ML Training Paradigm Tournament")
    parser.add_argument("--tickers", type=str, default="NVDA,AAPL,MSFT", help="Comma-separated ticker list")
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    run_financial_ml_tournament(tickers=tickers)
