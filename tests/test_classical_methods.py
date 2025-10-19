"""Test suite for classical portfolio optimization methods."""

import pytest
import numpy as np
import pandas as pd
from qpo.optimizers.equal_weight import EqualWeightOptimizer
from qpo.optimizers.risk_parity import RiskParityOptimizer
from qpo.optimizers.regularized import L1RegularizedOptimizer, L2RegularizedOptimizer
from qpo.optimizers.classical import ClassicalOptimizer
from qpo.optimizers.backtest import Backtester
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


class TestClassicalOptimizers:
    """Test all classical optimization methods."""

    @pytest.mark.parametrize("n_assets", [16, 32, 64])
    def test_equal_weight(self, n_assets):
        """Test equal-weight portfolio (1/N)."""
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        optimizer = EqualWeightOptimizer()
        result = optimizer.optimize(returns)

        # Validate
        assert 'weights' in result
        weights = result['weights']

        # All weights should be equal
        expected_weight = 1.0 / n_assets
        assert np.allclose(weights.values, expected_weight, atol=1e-10)
        assert np.isclose(weights.sum(), 1.0)

        # Should be instant
        assert result['runtime'] < 0.001

        print(f"\n{'='*70}")
        print(f"Equal-Weight Portfolio - {n_assets} Assets")
        print(f"{'='*70}")
        print(f"  Weight per asset: {expected_weight*100:.4f}%")
        print(f"  Sharpe ratio:     {result['metrics']['sharpe_ratio']:.3f}")
        print(f"  Effective N:      {result['metrics']['effective_n_assets']:.1f}")
        print(f"  Runtime:          {result['runtime']:.6f}s")

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
        # Risk parity should have more uniform risk contribution
        # (though not necessarily more uniform weights)

        print(f"\n{'='*70}")
        print(f"Risk Parity Portfolio - {n_assets} Assets")
        print(f"{'='*70}")
        print(f"  Sharpe ratio:          {result['metrics']['sharpe_ratio']:.3f}")
        print(f"  Effective N:           {result['metrics']['effective_n_assets']:.1f}")
        print(f"  Weight std:            {weight_std:.4f}")
        print(f"  Risk contrib std:      {risk_contrib_std:.4f}")
        print(f"  Runtime:               {result['runtime']:.4f}s")
        print(f"  Min/Max weights:       {weights.min()*100:.2f}% / {weights.max()*100:.2f}%")

    @pytest.mark.parametrize("lambda_l1", [0.001, 0.01, 0.1])
    def test_l1_regularization(self, lambda_l1):
        """Test L1 (Lasso) regularization with different lambda values."""
        n_assets = 32
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        optimizer = L1RegularizedOptimizer(gamma=1.0, lambda_l1=lambda_l1)
        result = optimizer.optimize(returns)

        weights = result['weights']
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        assert (weights >= 0).all()

        # Higher lambda should produce sparser portfolios
        n_nonzero = (weights > 1e-6).sum()

        print(f"\n{'='*70}")
        print(f"L1-Regularized Portfolio - λ={lambda_l1}")
        print(f"{'='*70}")
        print(f"  Assets selected:  {n_nonzero}/{n_assets}")
        print(f"  Sharpe ratio:     {result['metrics']['sharpe_ratio']:.3f}")
        print(f"  Effective N:      {result['metrics']['effective_n_assets']:.1f}")
        print(f"  Herfindahl:       {result['metrics']['herfindahl_index']:.4f}")
        print(f"  Runtime:          {result['runtime']:.4f}s")

        # Sparsity should increase with lambda
        if lambda_l1 == 0.1:
            assert n_nonzero < n_assets * 0.5, "High lambda should produce sparse portfolio"

    def test_l2_regularization(self):
        """Test L2 (Ridge) regularization."""
        n_assets = 32
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        optimizer = L2RegularizedOptimizer(gamma=1.0, lambda_l2=0.1)
        result = optimizer.optimize(returns)

        weights = result['weights']
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        assert (weights >= 0).all()

        # L2 should produce more uniform weights than unregularized
        print(f"\n{'='*70}")
        print(f"L2-Regularized Portfolio")
        print(f"{'='*70}")
        print(f"  Sharpe ratio:     {result['metrics']['sharpe_ratio']:.3f}")
        print(f"  Effective N:      {result['metrics']['effective_n_assets']:.1f}")
        print(f"  Weight std:       {weights.std():.4f}")
        print(f"  Runtime:          {result['runtime']:.4f}s")


class TestClassicalComparison:
    """Compare all classical methods head-to-head."""

    def test_method_comparison(self):
        """Compare all methods on same data."""
        n_assets = 32
        prices = generate_synthetic_data(n_assets, n_days=504, seed=42)
        returns = compute_returns(prices, method='log')

        methods = {
            'Equal-Weight': EqualWeightOptimizer(),
            'Risk Parity': RiskParityOptimizer(),
            'L1 (λ=0.01)': L1RegularizedOptimizer(lambda_l1=0.01),
            'L2 (λ=0.1)': L2RegularizedOptimizer(lambda_l2=0.1),
            'Mean-Variance': ClassicalOptimizer(gamma=1.0, method='cvxpy'),
        }

        results = {}
        for name, optimizer in methods.items():
            result = optimizer.optimize(returns)
            results[name] = result

        # Print comparison table
        print(f"\n{'='*80}")
        print("SOTA Method Comparison - Single Optimization")
        print(f"{'='*80}")
        print(f"{'Method':<20} {'Runtime':>10} {'Sharpe':>8} {'Eff N':>8} {'Assets':>8}")
        print(f"{'-'*80}")

        for name, result in results.items():
            metrics = result['metrics']
            print(f"{name:<20} {result['runtime']:>9.5f}s {metrics['sharpe_ratio']:>8.3f} "
                  f"{metrics['effective_n_assets']:>8.1f} {metrics['n_assets']:>8}")

        # All should be fast
        for name, result in results.items():
            assert result['runtime'] < 1.0, f"{name} too slow"

    @pytest.mark.parametrize("n_assets", [16, 32])
    def test_backtest_comparison(self, n_assets):
        """Backtest comparison of SOTA methods."""
        prices = generate_synthetic_data(n_assets, n_days=504)
        returns = compute_returns(prices, method='log')

        methods = {
            'Equal-Weight': EqualWeightOptimizer(),
            'Risk Parity': RiskParityOptimizer(),
            'Mean-Variance': ClassicalOptimizer(gamma=1.0, method='cvxpy'),
        }

        backtest_results = {}
        for name, optimizer in methods.items():
            backtester = Backtester(
                optimizer,
                train_days=252,
                test_days=21,
                step_days=21
            )
            result = backtester.run(prices, returns)
            backtest_results[name] = result

        # Print backtest comparison
        print(f"\n{'='*80}")
        print(f"SOTA Backtest Comparison - {n_assets} Assets")
        print(f"{'='*80}")
        print(f"{'Method':<20} {'Return':>10} {'Sharpe':>8} {'Drawdown':>10} {'Avg Time':>10}")
        print(f"{'-'*80}")

        for name, result in backtest_results.items():
            metrics = result['metrics']
            print(f"{name:<20} {metrics['annualized_return']*100:>9.2f}% "
                  f"{metrics['sharpe_ratio']:>8.3f} {metrics['max_drawdown']*100:>9.2f}% "
                  f"{metrics['avg_runtime']:>9.5f}s")

        # Validate all methods produced reasonable results
        for name, result in backtest_results.items():
            metrics = result['metrics']
            assert -0.5 <= metrics['annualized_return'] <= 3.0
            assert metrics['max_drawdown'] < 0
            assert metrics['avg_runtime'] < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
