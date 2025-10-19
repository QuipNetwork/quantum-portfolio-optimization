"""Backtesting framework for portfolio strategies."""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class Backtester:
    """Rolling window backtesting for portfolio strategies."""

    def __init__(self,
                 optimizer,
                 train_days: int = 252,
                 test_days: int = 21,
                 step_days: int = 21,
                 initial_capital: float = 100000.0):
        """
        Initialize backtester.

        Args:
            optimizer: Optimizer instance (ClassicalOptimizer or QuantumOptimizer)
            train_days: Training window size (days)
            test_days: Test/holding period (days)
            step_days: Rebalancing frequency (days)
            initial_capital: Starting portfolio value ($)
        """
        self.optimizer = optimizer
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.initial_capital = initial_capital

    def run(self,
            prices: pd.DataFrame,
            returns: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            prices: Price DataFrame (for portfolio valuation)
            returns: Returns DataFrame (optional, computed if not provided)

        Returns:
            {
                'portfolio_values': pd.Series,
                'weights_history': List[dict],
                'metrics': dict,
                'runtimes': List[float],
                'rebalance_dates': List[date]
            }
        """
        if returns is None:
            returns = prices.pct_change().dropna()

        n = len(prices)
        portfolio_values = [self.initial_capital]
        weights_history = []
        runtimes = []
        rebalance_dates = []
        dates = [prices.index[0]]

        # Track current portfolio value
        current_value = self.initial_capital

        # Rolling windows
        for start_idx in range(0, n - self.train_days - self.test_days + 1,
                              self.step_days):
            train_end = start_idx + self.train_days
            test_end = min(train_end + self.test_days, n)

            # Training data
            train_returns = returns.iloc[start_idx:train_end]

            # Optimize
            start_time = time.time()
            result = self.optimizer.optimize(train_returns)
            runtime = time.time() - start_time

            weights = result['weights']
            runtimes.append(runtime)
            rebalance_dates.append(prices.index[train_end])

            # Test period: hold portfolio
            test_prices = prices.iloc[train_end:test_end]
            test_returns = returns.iloc[train_end:test_end]

            # Track portfolio performance
            weights_history.append({
                'date': prices.index[train_end],
                'weights': weights.to_dict(),
                'metrics': result['metrics']
            })

            # Compute portfolio returns for test period
            aligned_weights = weights.reindex(test_returns.columns, fill_value=0)
            portfolio_returns = test_returns @ aligned_weights

            # Update portfolio values
            for i, (date, ret) in enumerate(portfolio_returns.items()):
                current_value *= (1 + ret)
                portfolio_values.append(current_value)
                dates.append(date)

        # Convert to Series
        portfolio_series = pd.Series(portfolio_values, index=dates)

        # Compute overall metrics
        metrics = self._compute_backtest_metrics(portfolio_series)
        metrics['avg_runtime'] = float(np.mean(runtimes))
        metrics['total_runtime'] = float(np.sum(runtimes))
        metrics['n_rebalances'] = len(runtimes)

        return {
            'portfolio_values': portfolio_series,
            'weights_history': weights_history,
            'metrics': metrics,
            'runtimes': runtimes,
            'rebalance_dates': rebalance_dates
        }

    def _compute_backtest_metrics(self,
                                  portfolio_values: pd.Series) -> Dict[str, float]:
        """
        Compute backtest performance metrics.

        Args:
            portfolio_values: Time series of portfolio values

        Returns:
            Dictionary of performance metrics
        """
        # Total return
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

        # Annualized return
        n_days = len(portfolio_values)
        years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Daily returns
        daily_returns = portfolio_values.pct_change().dropna()

        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = annualized_return / volatility if volatility > 0 else 0

        # Max drawdown
        cumulative = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate (% of positive return days)
        win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = annualized_return / downside_std if downside_std > 0 else 0

        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'final_value': float(portfolio_values.iloc[-1]),
            'n_periods': len(portfolio_values)
        }


def compare_to_benchmark(backtest_results: Dict[str, Any],
                        benchmark_prices: pd.Series) -> Dict[str, float]:
    """
    Compare backtest results to a benchmark (e.g., equal-weight or market).

    Args:
        backtest_results: Results from Backtester.run()
        benchmark_prices: Benchmark price series

    Returns:
        Dictionary of comparative metrics
    """
    portfolio_values = backtest_results['portfolio_values']

    # Align dates
    common_dates = portfolio_values.index.intersection(benchmark_prices.index)
    portfolio_aligned = portfolio_values[common_dates]
    benchmark_aligned = benchmark_prices[common_dates]

    # Normalize both to start at same value
    portfolio_norm = portfolio_aligned / portfolio_aligned.iloc[0]
    benchmark_norm = benchmark_aligned / benchmark_aligned.iloc[0]

    # Compute relative performance
    relative_return = (portfolio_norm.iloc[-1] / benchmark_norm.iloc[-1]) - 1

    # Tracking error
    portfolio_returns = portfolio_aligned.pct_change().dropna()
    benchmark_returns = benchmark_aligned.pct_change().dropna()
    tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)

    # Information ratio
    excess_return = portfolio_returns.mean() - benchmark_returns.mean()
    information_ratio = (excess_return * 252) / tracking_error if tracking_error > 0 else 0

    return {
        'relative_return': float(relative_return),
        'tracking_error': float(tracking_error),
        'information_ratio': float(information_ratio),
        'outperformance_days': int((portfolio_returns > benchmark_returns).sum()),
        'total_days': len(portfolio_returns)
    }
