import os
import pandas as pd
import numpy as np
from src.utils import get_logger, create_sequences
from src.universal_modeling import UniversalLSTM
import argparse
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)






def define_model(input_size: int, hidden_size: int, num_layers: int):
    """
    Defines the architecture for the universal sequence model (e.g., LSTM).
    """
    logger.info("Defining universal model architecture...")
    model = UniversalLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    logger.info("Model definition complete.")
    return model

from src.backtesting import run_backtest

def train_universal_model(model, X_train, y_train, X_val, y_val, df_val: pd.DataFrame, input_size: int, learning_rate: float, epochs: int, batch_size: int, sequence_length: int):
    """
    Handles the main training loop for the universal model with MLflow logging.
    """
    logger.info("Starting universal model training loop...")

    with mlflow.start_run(run_name="Universal LSTM Training"):
        mlflow.log_param("hidden_size", model.hidden_size)
        mlflow.log_param("num_layers", model.num_layers)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("sequence_length", sequence_length)

        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()
        X_val_tensor = torch.from_numpy(X_val).float()
        y_val_tensor = torch.from_numpy(y_val).float()

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        for epoch in range(epochs):
            model.train()
            for i, (sequences, labels) in enumerate(train_loader):
                sequences, labels = sequences.to(device), labels.to(device).unsqueeze(1)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mlflow.log_metric("train_loss", loss.item(), step=epoch)

            model.eval()
            with torch.no_grad():
                correct, total = 0, 0
                val_outputs = []
                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(device), labels.to(device).unsqueeze(1)
                    outputs = model(sequences)
                    predicted = torch.round(torch.sigmoid(outputs))
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_outputs.extend(predicted.cpu().numpy())
                val_accuracy = (correct / total) if total > 0 else 0
                logger.info(f'Validation Accuracy: {val_accuracy*100:.2f}%')
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        logger.info("Training loop complete.")
        
        logger.info("Running backtest on validation set...")
        y_pred_val = np.array(val_outputs).flatten()
        
        index_list = []
        for ticker, group in df_val.groupby('Ticker'):
            if len(group) > sequence_length:
                for i in range(len(group) - sequence_length):
                    index_list.append(group.index[i + sequence_length])

        if len(index_list) == len(y_pred_val):
            predictions_df = pd.DataFrame({'prediction': y_pred_val}, index=pd.MultiIndex.from_tuples(index_list, names=['Ticker', 'Date']))
            
            all_metrics = []
            for ticker in predictions_df.index.get_level_values('Ticker').unique():
                ticker_predictions = predictions_df.loc[ticker]
                if not ticker_predictions.empty:
                    signals = ticker_predictions['prediction'].replace({0: -1})
                    price_history = df_val.loc[ticker_predictions.index]
                    
                    _, backtest_metrics, _ = run_backtest(price_history, signals)
                    all_metrics.append(backtest_metrics)

            if all_metrics:
                avg_metrics = pd.DataFrame(all_metrics).mean().to_dict()
                logger.info(f"Average backtest performance: {avg_metrics}")
                mlflow.log_metrics({f"avg_{k}": v for k, v in avg_metrics.items()})

        logger.info("Saving universal model...")
        model_path = "models/universal_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "universal_model")
        
        config_path = "models/universal_model_config.json"
        config = { "input_size": input_size, "hidden_size": model.hidden_size, "num_layers": model.num_layers, "output_size": 1 }
        with open(config_path, "w") as f:
            import json
            json.dump(config, f)
        mlflow.log_artifact(config_path)

def main(max_tickers: int | None = None, hidden_size: int = 50, num_layers: int = 2, learning_rate: float = 0.001, epochs: int = 10, batch_size: int = 64, sequence_length: int = 30):
    logger.info("--- Starting Universal Model Training Pipeline ---")
    merged_df_path = "data/processed/universal_merged_data.parquet"
    if not os.path.exists(merged_df_path):
        logger.error(f"Preprocessed data not found. Please run src/data_optimizer.py first.")
        return

    merged_df = pd.read_parquet(merged_df_path)
    
    unique_dates = merged_df.index.get_level_values('Date').unique().sort_values()
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    
    df_train = merged_df[merged_df.index.get_level_values('Date') < split_date]
    df_val = merged_df[merged_df.index.get_level_values('Date') >= split_date]

    X_train, y_train = create_sequences(df_train, sequence_length)
    X_val, y_val = create_sequences(df_val, sequence_length)

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        logger.error("Not enough data to create training or validation sequences. Please check the data and split date.")
        return

    input_size = X_train.shape[2]
    model = define_model(input_size, hidden_size, num_layers)
    if model is None: return

    train_universal_model(model, X_train, y_train, X_val, y_val, df_val, input_size, learning_rate, epochs, batch_size, sequence_length)
    logger.info("--- Universal Model Training Pipeline Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a universal sentiment-driven stock momentum predictor.')
    parser.add_argument('--max_tickers', type=int, default=None, help='Optional: Limit the number of tickers to process for faster training/testing.')
    parser.add_argument('--hidden_size', type=int, default=50, help='Number of features in the hidden state of the LSTM.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of recurrent layers in the LSTM.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length for the LSTM.')
    args = parser.parse_args()
    main(max_tickers=args.max_tickers, hidden_size=args.hidden_size, num_layers=args.num_layers, learning_rate=args.learning_rate, epochs=args.epochs, batch_size=args.batch_size, sequence_length=args.sequence_length)
