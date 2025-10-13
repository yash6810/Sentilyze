import os
import pandas as pd
import numpy as np
from src.utils import get_logger
from src.data_ingestion import get_price_history
from src.sentiment_analysis import get_sentiment
from src.feature_engineering import create_technical_indicators, aggregate_sentiment_scores
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

import torch
import torch.nn as nn

logger = get_logger(__name__)

class UniversalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(UniversalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out


def load_and_preprocess_data():
    """
    Loads the Kaggle dataset and performs initial preprocessing.
    This will be a major function that needs to merge news with price data for all stocks.
    """
    logger.info("Loading and preprocessing universal dataset...")
    
    # Define path to the dataset downloaded by download_dataset.py
    dataset_path = os.path.join("data", "raw", "all-data.csv")

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}. Please run download_dataset.py first.")
        return None

    # Step 1.1: Load and Clean News Data
    logger.info(f"Loading news data from {dataset_path}...")
    news_df = pd.read_csv(dataset_path, encoding='ISO-8859-1', header=None, names=["Date", "Title", "Ticker"])
    news_df.dropna(subset=["Title"], inplace=True)
    news_df["Date"] = pd.to_datetime(news_df["Date"])

    unique_tickers = news_df["Ticker"].unique()
    logger.info(f"Found {len(news_df)} articles for {len(unique_tickers)} unique tickers.")

    # Step 1.2: Fetch All Price Data
    logger.info("Fetching historical price data for all tickers...")
    all_prices = []
    for ticker in tqdm(unique_tickers, desc="Fetching price data"):
        try:
            price_history = get_price_history(ticker, period="max")
            if not price_history.empty:
                price_history['Ticker'] = ticker
                all_prices.append(price_history)
        except Exception as e:
            logger.warning(f"Could not fetch price data for {ticker}: {e}")
    
    if not all_prices:
        logger.error("Failed to fetch any price data. Aborting.")
        return None

    price_df = pd.concat(all_prices)
    price_df.set_index(['Ticker', price_df.index], inplace=True)
    logger.info(f"Successfully fetched price data for {len(price_df['Ticker'].unique())} tickers.")

    # Step 1.3: Perform Bulk Sentiment Analysis
    logger.info("Performing bulk sentiment analysis on news data...")
    tokenizer = AutoTokenizer.from_pretrained('./models/finbert-fine-tuned')
    model = AutoModelForSequenceClassification.from_pretrained('./models/finbert-fine-tuned')
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    news_df = get_sentiment(news_df, sentiment_analyzer)
    logger.info("Bulk sentiment analysis complete.")

    # Step 1.4: Merge and Engineer Features
    logger.info("Merging data and engineering features...")
    # Aggregate sentiment scores daily for each stock
    daily_sentiment = news_df.groupby(['Ticker', pd.Grouper(key='Date', freq='D')]).agg(
        mean_sentiment_score=('sentiment_score', 'mean')
    ).reset_index()
    sentiment_counts = pd.get_dummies(news_df['sentiment_label']).groupby([news_df['Ticker'], pd.Grouper(key='Date', freq='D')]).sum().reset_index()
    daily_sentiment = pd.merge(daily_sentiment, sentiment_counts, on=['Ticker', 'Date'])
    daily_sentiment.set_index(['Ticker', 'Date'], inplace=True)

    # Calculate technical indicators for each stock
    price_df = price_df.groupby('Ticker').apply(create_technical_indicators)

    # Merge price data with sentiment data
    merged_df = pd.merge(price_df, daily_sentiment, left_index=True, right_index=True, how='left')
    merged_df.ffill(inplace=True) # Forward fill sentiment data on non-news days
    merged_df.fillna(0, inplace=True)

    # Create the target variable for each stock
    merged_df['target'] = merged_df.groupby('Ticker')['Close'].shift(-1) > merged_df['Close']
    merged_df['target'] = merged_df['target'].astype(int)
    merged_df.dropna(subset=['target'], inplace=True) # Drop the last day for each stock

    logger.info("Feature engineering complete.")

    # Step 1.5: Create Sequences
    logger.info("Creating sequences for LSTM model...")
    X, y = create_sequences(merged_df, sequence_length=30)
    logger.info(f"Created {len(X)} sequences.")

    return X, y

def create_sequences(df: pd.DataFrame, sequence_length: int):
    """
    Transforms a DataFrame into sequences for an LSTM model.
    """
    sequences = []
    labels = []
    # Group by stock ticker to create sequences per stock
    for ticker, group in df.groupby('Ticker'):
        feature_columns = [col for col in df.columns if col not in ['Ticker', 'target']]
        features = group[feature_columns].values
        target = group['target'].values

        for i in range(len(features) - sequence_length):
            sequences.append(features[i:i+sequence_length])
            labels.append(target[i+sequence_length])

    return np.array(sequences), np.array(labels)

def define_model(input_size):
    """
    Defines the architecture for the universal sequence model (e.g., LSTM).
    """
    logger.info("Defining universal model architecture...")
    model = UniversalLSTM(input_size=input_size, hidden_size=50, num_layers=2, output_size=1)
    logger.info("Model definition complete.")
    return model

def train_universal_model(model, X, y):
    """
    Handles the main training loop for the universal model.
    """
    logger.info("Starting universal model training loop...")
    
    # 1. Split data into training and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Create PyTorch Datasets and DataLoaders
    from torch.utils.data import TensorDataset, DataLoader
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    # 3. Define Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss() # Better for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    num_epochs = 10 # Placeholder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Validation loop (optional, but good practice)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(sequences)
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            logger.info(f'Validation Accuracy: {(correct/total)*100:.2f}%')

    logger.info("Training loop complete.")
    # 5. Save the model
    logger.info("Saving universal model...")
    torch.save(model.state_dict(), 'models/universal_model.pth')
    logger.info("Model saved to models/universal_model.pth")

def main():
    """
    Main function to orchestrate the training of the universal model.
    """
    logger.info("--- Starting Universal Model Training Pipeline ---")

    # 1. Load and process data
    processed_data = load_and_preprocess_data()
    if processed_data is None:
        return

    # 2. Define the model architecture
    model = define_model()
    if model is None:
        return

    # 3. Run the training loop
    train_universal_model(model, processed_data)

    # 4. Save the final model
    logger.info("Saving universal model...")
    # TODO: Save the trained model (e.g., torch.save(model.state_dict(), 'models/universal_model.pth'))

    logger.info("--- Universal Model Training Pipeline Finished ---")

if __name__ == "__main__":
    main()
