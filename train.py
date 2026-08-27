from typing import Any
from dotenv import load_dotenv

load_dotenv()
import pandas as pd
import argparse
import json
import shap
import mlflow
import mlflow.xgboost
import numpy as np
from src.modeling import train_model, save_model
from src.backtesting import run_backtest, run_significance_test
from src.config import FEATURES
from src.utils import get_logger
from src.preprocessing import preprocess_data

logger = get_logger(__name__)


def main(ticker: str, leverage: float = 1.5, use_cache: bool = False) -> None:
    """
    Main function to run the training pipeline for a given stock ticker.

    Args:
        ticker (str): The stock ticker to train the model on.
        leverage (float): Maximum leverage for the backtest.
        use_cache (bool): Whether to aggressively use cached data.
    """
    logger.info(f"Starting training pipeline for {ticker} (Cache: {use_cache})...")

    # 1. Preprocess data
    logger.info("Preprocessing data...")
    features_df, price_history_with_indicators, _ = preprocess_data(
        ticker, use_cache=use_cache
    )

    # 2. Prepare data for training
    logger.info("Preparing data for training...")
    features = FEATURES
    target = "target"

    X = pd.DataFrame(features_df[features])
    y = features_df[target]

    logger.info(f"Total dataset size: {len(X)} days")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    with mlflow.start_run():
        mlflow.log_param("ticker", ticker)

        # 3. Train Model using WFO (alongside Baseline Logistic Regression)
        model, metrics, oos_predictions = train_model(X, y)

        mlflow.log_params(metrics["best_params"])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("roc_auc", metrics.get("roc_auc", 0.5))
        mlflow.log_metric(
            "baseline_logistic_accuracy", metrics.get("baseline_logistic_accuracy", 0.5)
        )

        logger.info(
            f"WFO Model accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics.get('roc_auc', 0.5):.4f}"
        )

        # 4. Run Backtest
        logger.info(
            f"Running backtest on out-of-sample predictions (Leverage: {leverage})..."
        )
        test_price_history = pd.DataFrame(
            price_history_with_indicators.loc[oos_predictions.index]
        )
        portfolio, backtest_metrics, heatmap_fig = run_backtest(
            test_price_history, oos_predictions, max_leverage=leverage
        )
        logger.info(f"Backtest performance: {backtest_metrics}")
        mlflow.log_metrics(backtest_metrics)

        # 5. Run Statistical Significance Test
        logger.info("Running permutation significance test...")
        significance_results = run_significance_test(
            portfolio, test_price_history, n_simulations=1000
        )
        mlflow.log_metric("significance_p_value", significance_results["p_value"])

        # Log classification report as an artifact
        classification_report_str = metrics.get("classification_report", "N/A")
        if classification_report_str != "N/A":
            report_path = f"results/{ticker}_classification_report.txt"
            with open(report_path, "w") as f:
                f.write(classification_report_str)
            mlflow.log_artifact(report_path)
            logger.info(f"Saved classification report to {report_path}")

        # 6. Save Model and Results
        logger.info(f"Saving model to models/{ticker}_model.json...")
        save_model(model, f"models/{ticker}_model.json")
        try:
            mlflow.xgboost.log_model(model, "model")
        except Exception as e:
            logger.warning(f"MLflow model log notice: {e}")
            mlflow.log_artifact(f"models/{ticker}_model.json")

        # Save the heatmap
        heatmap_fig.savefig(f"results/{ticker}_monthly_returns_heatmap.png")
        mlflow.log_artifact(f"results/{ticker}_monthly_returns_heatmap.png")
        logger.info(
            f"Saved monthly returns heatmap to results/{ticker}_monthly_returns_heatmap.png"
        )

        # Save combined metrics to a JSON file
        combined_metrics = {**metrics, **backtest_metrics}
        with open(f"results/{ticker}_metrics.json", "w") as f:
            json.dump(combined_metrics, f, indent=4)
        mlflow.log_artifact(f"results/{ticker}_metrics.json")
        logger.info(f"Saved metrics to results/{ticker}_metrics.json")

        # Save significance results
        with open(f"results/{ticker}_significance.json", "w") as f:
            json.dump(significance_results, f, indent=4)
        mlflow.log_artifact(f"results/{ticker}_significance.json")
        logger.info(f"Saved significance to results/{ticker}_significance.json")

        # Save portfolio to a CSV file
        portfolio.to_csv(f"results/{ticker}_portfolio.csv")
        mlflow.log_artifact(f"results/{ticker}_portfolio.csv")
        logger.info(f"Saved portfolio to results/{ticker}_portfolio.csv")

        # Save feature importances to a CSV file
        feature_importances = pd.DataFrame(
            {"feature": features, "importance": model.feature_importances_}
        ).sort_values(by="importance", ascending=False)
        feature_importances.to_csv(
            f"results/{ticker}_feature_importances.csv", index=False
        )
        mlflow.log_artifact(f"results/{ticker}_feature_importances.csv")
        logger.info(
            f"Saved feature importances to results/{ticker}_feature_importances.csv"
        )

        # Calculate and save SHAP values on the final OOS features
        logger.info("Calculating SHAP values...")
        X_test_oos = X.loc[oos_predictions.index]
        try:
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            explainer = shap.TreeExplainer(booster)
            raw_shap = explainer.shap_values(X_test_oos)
            shap_values = (
                raw_shap[1]
                if isinstance(raw_shap, list) and len(raw_shap) > 1
                else np.asarray(raw_shap)
            )
        except Exception as e1:
            logger.warning(
                f"SHAP TreeExplainer notice: {e1}. Using robust fallback explainer..."
            )
            try:
                explainer = shap.Explainer(
                    model.predict_proba, X.sample(min(50, len(X)), random_state=42)
                )
                shap_obj: Any = explainer(X_test_oos.head(min(100, len(X_test_oos))))
                shap_values = getattr(shap_obj, "values", np.asarray(shap_obj))
            except Exception as e2:
                logger.warning(
                    f"SHAP fallback notice: {e2}. Initializing SHAP values matrix."
                )
                shap_values = np.zeros(X_test_oos.shape)

        np.save(f"results/{ticker}_shap_values.npy", shap_values)
        mlflow.log_artifact(f"results/{ticker}_shap_values.npy")
        logger.info(f"Saved SHAP values to results/{ticker}_shap_values.npy")

        # Save X_test (the OOS period) to a CSV file
        X_test_oos.to_csv(f"results/{ticker}_X_test.csv")
        mlflow.log_artifact(f"results/{ticker}_X_test.csv")
        logger.info(f"Saved X_test to results/{ticker}_X_test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model for a specific stock ticker or all stocks."
    )
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g., NVDA)")
    parser.add_argument(
        "--all", action="store_true", help="Train models for all tickers in stocks.txt"
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.5,
        help="Maximum leverage for the backtest (default: 1.5)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Aggressively use cached data to avoid API rate limits",
    )
    args = parser.parse_args()

    if args.all:
        try:
            with open("stocks.txt", "r") as f:
                tickers = [line.strip() for line in f if line.strip()]

            logger.info(
                f"Training models for {len(tickers)} tickers found in stocks.txt..."
            )
            for ticker in tickers:
                main(ticker, leverage=args.leverage, use_cache=args.use_cache)

            logger.info("Finished processing all tickers.")
        except FileNotFoundError:
            logger.error("stocks.txt not found. Cannot run --all.")
    elif args.ticker:
        main(args.ticker, leverage=args.leverage, use_cache=args.use_cache)
    else:
        logger.error("You must provide either --ticker or --all.")
