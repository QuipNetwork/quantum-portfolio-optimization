"""Data preprocessing utilities for portfolio optimization."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def preprocess_portfolio(df: pd.DataFrame,
                        max_missing_pct: float = 0.05,
                        max_ffill_days: int = 3) -> pd.DataFrame:
    """
    Clean and validate portfolio data.

    Steps:
    1. Remove tickers with >max_missing_pct missing data
    2. Forward-fill remaining gaps (max max_ffill_days)
    3. Drop rows with any remaining NaNs
    4. Validate monotonic date index

    Args:
        df: Portfolio DataFrame (dates × tickers)
        max_missing_pct: Maximum allowed missing data per ticker (0.05 = 5%)
        max_ffill_days: Maximum days to forward-fill

    Returns:
        Cleaned DataFrame
    """
    # Calculate missing data percentage per ticker
    missing_pct = df.isna().sum() / len(df)

    # Remove tickers with too much missing data
    valid_tickers = missing_pct[missing_pct <= max_missing_pct].index
    df_clean = df[valid_tickers].copy()

    removed = len(df.columns) - len(df_clean.columns)
    if removed > 0:
        print(f"Removed {removed} tickers with >{max_missing_pct*100}% missing data")

    # Forward-fill gaps (limited)
    df_clean = df_clean.ffill(limit=max_ffill_days)

    # Drop rows with remaining NaNs
    df_clean = df_clean.dropna()

    # Validate date index
    if not df_clean.index.is_monotonic_increasing:
        df_clean = df_clean.sort_index()

    return df_clean


def compute_returns(prices: pd.DataFrame,
                   method: str = 'log') -> pd.DataFrame:
    """
    Compute asset returns.

    Args:
        prices: DataFrame of prices
        method: 'log' (log returns) or 'simple' (arithmetic returns)

    Returns:
        DataFrame of returns (N-1 × M)
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'")

    return returns.dropna()


class PortfolioStatistics:
    """Compute portfolio statistics from returns data."""

    def __init__(self,
                 returns: pd.DataFrame,
                 window: Optional[int] = None,
                 annualization_factor: int = 252):
        """
        Args:
            returns: Returns DataFrame
            window: Rolling window size (None = full history)
            annualization_factor: Days per year for annualization (252 for daily)
        """
        self.returns = returns
        self.window = window
        self.annualization_factor = annualization_factor

    def expected_returns(self) -> pd.Series:
        """
        Compute mean returns (annualized).

        Returns:
            Series of expected returns per asset
        """
        if self.window:
            mean_returns = self.returns.rolling(self.window).mean().iloc[-1]
        else:
            mean_returns = self.returns.mean()

        return mean_returns * self.annualization_factor

    def covariance_matrix(self) -> pd.DataFrame:
        """
        Compute covariance matrix (annualized).

        Returns:
            Covariance matrix (N × N)
        """
        if self.window:
            # Get last window of data
            window_data = self.returns.iloc[-self.window:]
            cov = window_data.cov()
        else:
            cov = self.returns.cov()

        return cov * self.annualization_factor

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Compute correlation matrix.

        Returns:
            Correlation matrix (N × N)
        """
        if self.window:
            window_data = self.returns.iloc[-self.window:]
            corr = window_data.corr()
        else:
            corr = self.returns.corr()

        return corr


class RollingWindow:
    """Rolling window iterator for backtesting."""

    def __init__(self,
                 data: pd.DataFrame,
                 train_days: int = 252,
                 test_days: int = 21,
                 step_days: int = 21):
        """
        Create rolling windows for backtesting.

        Args:
            data: Full time series data
            train_days: Training window size
            test_days: Test window size
            step_days: Step size between windows
        """
        self.data = data
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

        # Calculate number of windows
        n = len(data)
        self.n_windows = (n - train_days - test_days) // step_days + 1

    def __iter__(self):
        """Iterate over rolling windows."""
        n = len(self.data)

        for start_idx in range(0, n - self.train_days - self.test_days + 1,
                              self.step_days):
            train_end = start_idx + self.train_days
            test_end = min(train_end + self.test_days, n)

            train_data = self.data.iloc[start_idx:train_end]
            test_data = self.data.iloc[train_end:test_end]

            yield {
                'train': train_data,
                'test': test_data,
                'train_start': self.data.index[start_idx],
                'train_end': self.data.index[train_end - 1],
                'test_start': self.data.index[train_end],
                'test_end': self.data.index[test_end - 1]
            }

    def __len__(self):
        """Number of windows."""
        return self.n_windows


def split_train_test(data: pd.DataFrame,
                    train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple train/test split.

    Args:
        data: Time series data
        train_ratio: Fraction for training (0.8 = 80%)

    Returns:
        (train_data, test_data)
    """
    n = len(data)
    split_idx = int(n * train_ratio)

    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]

    return train, test
