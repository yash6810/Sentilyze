
import unittest
import pandas as pd
import numpy as np
from src.backtesting import run_backtest, _calculate_trade_outcomes, calculate_performance_metrics

class TestBacktesting(unittest.TestCase):

    def setUp(self):
        # Create a sample price history
        self.price_history = pd.DataFrame({
            'Close': [100, 102, 105, 103, 106, 108, 107, 110, 112, 111]
        }, index=pd.to_datetime(pd.date_range('2025-01-01', periods=10)))

        # Create a sample signals series
        self.signals = pd.Series([1, 1, -1, -1, 1, 1, 1, -1, -1, -1], index=self.price_history.index)

    def test_calculate_trade_outcomes(self):
        # Create a sample portfolio
        portfolio = pd.DataFrame({
            'signal': [1, 1, -1, -1, 1, 1, 1, -1, -1, -1],
            'price': [100, 102, 105, 103, 106, 108, 107, 110, 112, 111]
        }, index=self.price_history.index)

        # Calculate trade outcomes
        trade_outcomes = _calculate_trade_outcomes(portfolio)

        # Check the trade outcomes
        self.assertEqual(len(trade_outcomes), 2)
        self.assertAlmostEqual(trade_outcomes[0], 5.0)
        self.assertAlmostEqual(trade_outcomes[1], 4.0)

    def test_calculate_performance_metrics(self):
        # Run the backtest to get the portfolio and metrics
        portfolio, metrics, _ = run_backtest(self.price_history, self.signals)

        # Check the performance metrics
        self.assertEqual(metrics['Total Trades'], 2)
        self.assertEqual(metrics['Win Rate'], '100.00%')
        self.assertEqual(metrics['Strategy Total Return'], '4.43%')
        self.assertEqual(metrics['Buy & Hold Total Return'], '11.00%')
        self.assertAlmostEqual(float(metrics['Sharpe Ratio']), 5.35, places=2)
        self.assertAlmostEqual(float(metrics['Sortino Ratio']), 18.99, places=2)
        self.assertEqual(metrics['Strategy Max Drawdown'], '-1.12%')

    def test_run_backtest(self):
        # Run the backtest
        portfolio, metrics, heatmap_fig = run_backtest(self.price_history, self.signals)

        # Check the portfolio
        self.assertIsInstance(portfolio, pd.DataFrame)
        self.assertEqual(len(portfolio), 10)
        self.assertIn('total', portfolio.columns)
        self.assertIn('benchmark', portfolio.columns)

        # Check the metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('Total Trades', metrics)
        self.assertIn('Win Rate', metrics)

        # Check the heatmap
        self.assertIsNotNone(heatmap_fig)

if __name__ == '__main__':
    unittest.main()
