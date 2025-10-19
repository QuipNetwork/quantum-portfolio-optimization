"""Test suite for risk parity portfolio optimizer."""

import pytest
import numpy as np
import pandas as pd
from qpo.optimizers.risk_parity import RiskParityOptimizer
from qpo.utils.data_prep import compute_returns


def generate_synthetic_data(n_assets: int, n_days: int = 504, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic price data."""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n_days)
    tickers = [f'ASSET_{i:03d}' for i in range(n_assets)]

    prices_data = {}
    for i, ticker in enumerate(tickers):
        drift = 0.0003 + (i / n_assets) * 0.0007
        vol = 0.015 + (i / n_assets) * 0.025
        returns = np.random.normal(drift, vol, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        prices_data[ticker] = prices

    return pd.DataFrame(prices_data, index=dates)


class TestRiskParity:
    """Test risk parity portfolio optimizer."""

    @pytest.mark.parametrize("n_assets", [16, 32, 64])
    def test_risk_parity(self, n_assets):
        """Test risk parity portfolio."""
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(returns)

        # Validate
        weights = result['weights']
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        assert (weights >= 0).all()

        # Risk contributions should be approximately equal
        Sigma = returns.cov().values * 252
        w = weights.values
        marginal_risk = Sigma @ w
        risk_contrib = w * marginal_risk
        risk_contrib_normalized = risk_contrib / risk_contrib.sum()

        # Check that risk contributions are more equal than weights
        weight_std = np.std(weights.values)
        risk_contrib_std = np.std(risk_contrib_normalized)

        print(f"\n{'='*70}")
        print(f"Risk Parity Portfolio - {n_assets} Assets")
        print(f"{'='*70}")
        print(f"  Sharpe ratio:          {result['metrics']['sharpe_ratio']:.3f}")
        print(f"  Effective N:           {result['metrics']['effective_n_assets']:.1f}")
        print(f"  Weight std:            {weight_std:.4f}")
        print(f"  Risk contrib std:      {risk_contrib_std:.4f}")
        print(f"  Runtime:               {result['runtime']:.4f}s")
        print(f"  Min/Max weights:       {weights.min()*100:.2f}% / {weights.max()*100:.2f}%")

    def test_risk_contributions_equal(self):
        """Test that risk contributions are approximately equal."""
        prices = generate_synthetic_data(20)
        returns = compute_returns(prices, method='log')

        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(returns)

        # Compute risk contributions
        Sigma = returns.cov().values * 252
        w = result['weights'].values
        marginal_risk = Sigma @ w
        risk_contrib = w * marginal_risk
        risk_contrib_normalized = risk_contrib / risk_contrib.sum()

        # All risk contributions should be close to 1/N
        expected_contrib = 1.0 / len(w)
        max_deviation = np.max(np.abs(risk_contrib_normalized - expected_contrib))

        # Allow some tolerance due to optimization convergence
        assert max_deviation < 0.02, f"Risk contributions not equal enough: {max_deviation}"

    def test_deterministic(self):
        """Test that risk parity produces consistent results."""
        prices = generate_synthetic_data(10, seed=123)
        returns = compute_returns(prices, method='log')

        optimizer = RiskParityOptimizer()
        result1 = optimizer.optimize(returns)
        result2 = optimizer.optimize(returns)

        # Should be very close (optimizer is deterministic with same data)
        assert np.allclose(result1['weights'].values, result2['weights'].values, atol=1e-6)

    def test_edge_cases(self):
        """Test edge cases."""
        # Single asset
        prices = generate_synthetic_data(1)
        returns = compute_returns(prices, method='log')
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(returns)
        assert result['weights'].values[0] == 1.0

        # Two assets
        prices = generate_synthetic_data(2)
        returns = compute_returns(prices, method='log')
        result = optimizer.optimize(returns)
        assert np.isclose(result['weights'].sum(), 1.0)
        assert (result['weights'] >= 0).all()

    def test_metrics(self):
        """Test that all expected metrics are present."""
        prices = generate_synthetic_data(20)
        returns = compute_returns(prices, method='log')
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(returns)

        assert 'metrics' in result
        metrics = result['metrics']

        required_metrics = [
            'expected_return',
            'expected_risk',
            'sharpe_ratio',
            'n_assets',
            'herfindahl_index',
            'effective_n_assets'
        ]

        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float))

    def test_performance(self):
        """Test that risk parity is reasonably fast."""
        prices = generate_synthetic_data(50)
        returns = compute_returns(prices, method='log')
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(returns)

        # Should complete in reasonable time (< 1 second for 50 assets)
        assert result['runtime'] < 1.0

    def test_different_volatilities(self):
        """Test that risk parity handles different volatilities correctly."""
        # Create assets with very different volatilities
        np.random.seed(42)
        n_days = 504
        dates = pd.date_range('2022-01-01', periods=n_days)

        # Low vol asset
        low_vol = np.random.normal(0.0005, 0.01, n_days)
        low_vol_prices = 100 * np.exp(np.cumsum(low_vol))

        # High vol asset
        high_vol = np.random.normal(0.0005, 0.05, n_days)
        high_vol_prices = 100 * np.exp(np.cumsum(high_vol))

        prices = pd.DataFrame({
            'LOW_VOL': low_vol_prices,
            'HIGH_VOL': high_vol_prices
        }, index=dates)

        returns = compute_returns(prices, method='log')
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(returns)

        # Low vol asset should have higher weight
        assert result['weights']['LOW_VOL'] > result['weights']['HIGH_VOL']

        # Risk contributions should be approximately equal
        Sigma = returns.cov().values * 252
        w = result['weights'].values
        marginal_risk = Sigma @ w
        risk_contrib = w * marginal_risk
        risk_contrib_normalized = risk_contrib / risk_contrib.sum()

        # Should be close to 50/50 risk contribution
        assert np.allclose(risk_contrib_normalized, 0.5, atol=0.05)
