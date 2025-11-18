# Sentilyze: Strategy and Application Overview

This document provides a clear overview of the trading strategy, technical indicators, and application functionality of the Sentilyze project.

## 1. The Trading Strategy

The project employs a **short-term momentum strategy** that decides whether to buy or sell a stock based on a daily prediction. Here’s how it works:

- **Signal Generation**: The machine learning model analyzes a combination of news sentiment and technical data to predict if the stock’s price will be higher (Positive) or lower (Negative) on the next trading day.
- **Execution**:
    - A "Positive" prediction generates a **buy signal (1)**.
    - A "Negative" prediction generates a **sell signal (-1)**.
- **Backtesting**: To prove its effectiveness, the strategy is simulated over the last 5 years of historical data. The backtest calculates key performance metrics like total return, win rate, and the Sharpe ratio, comparing the strategy's performance against a simple "buy-and-hold" approach. This simulation accounts for realistic factors like transaction costs and slippage.

## 2. The Technical Indicators Used

The model uses a rich set of widely-recognized technical indicators to capture different aspects of price movement and momentum. These are calculated in the `feature_engineering.py` file:

- **Moving Averages (MA)**: 7-day and 21-day moving averages to smooth out price data and identify trends.
- **Relative Strength Index (RSI)**: A momentum oscillator that measures the speed and change of price movements to identify overbought or oversold conditions.
- **Moving Average Convergence Divergence (MACD)**: A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
- **Bollinger Bands**: These measure market volatility. The bands widen when volatility increases and narrow when it decreases.
- **Stochastic Oscillator**: A momentum indicator that compares a particular closing price of a security to a range of its prices over a certain period of time to identify overbought and oversold signals.

## 3. Application Overview

The application provides a user-friendly interface built with Streamlit, organized into three main tabs:

### Tab 1: Next-Day Prediction
This is the main feature. A user can enter a stock ticker (e.g., "NVDA"), and the app will:
1.  Fetch the latest price data and news headlines.
2.  Analyze the sentiment of the news.
3.  Combine sentiment with the technical indicators listed above.
4.  Feed this data into the trained model to generate a **prediction (Positive/Negative)** and a **confidence score**.

### Tab 2: Backtest Analysis
This tab lets users evaluate the trading strategy's historical performance for a given stock. You can:
1.  Set the initial capital and transaction costs.
2.  Run a 5-year backtest simulation.
3.  View the results, including:
    - **Performance Metrics**: Strategy Return vs. Buy & Hold Return, Sharpe Ratio, Win Rate, and Max Drawdown.
    - **Portfolio Chart**: A line chart comparing the growth of your investment using the strategy versus just buying and holding the stock.
    - **Monthly Returns Heatmap**: A visual breakdown of the strategy's performance for each month.

### Tab 3: Model Performance
This tab offers a deep dive into the machine learning model's effectiveness, pulling data from the latest training run:
- **Key Metrics**: Displays the model's Accuracy, Precision, and Recall.
- **Explainable AI (XAI)**:
    - **Feature Importance**: A bar chart showing which features (e.g., RSI, sentiment score) were most influential in the model's decisions.
    - **SHAP Summary Plot**: A more advanced visualization that explains how different features push the model's output from the base value to the final prediction.
- **Detailed Reports**: Users can also view the full classification report and other data logged during model training.
