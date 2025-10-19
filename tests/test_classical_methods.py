"""Test suite for comparing all classical portfolio optimization methods."""

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
        print("Classical Method Comparison - Single Optimization")
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
        """Backtest comparison of classical methods."""
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
        print(f"Classical Backtest Comparison - {n_assets} Assets")
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

    def test_all_methods_consistent(self):
        """Test that all methods produce valid portfolios."""
        n_assets = 20
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        methods = {
            'Equal-Weight': EqualWeightOptimizer(),
            'Risk Parity': RiskParityOptimizer(),
            'L1 (λ=0.01)': L1RegularizedOptimizer(lambda_l1=0.01),
            'L2 (λ=0.1)': L2RegularizedOptimizer(lambda_l2=0.1),
            'Mean-Variance': ClassicalOptimizer(gamma=1.0, method='cvxpy'),
        }

        for name, optimizer in methods.items():
            result = optimizer.optimize(returns)
            weights = result['weights']

            # Basic validation
            assert np.isclose(weights.sum(), 1.0, atol=1e-5), f"{name}: weights don't sum to 1"
            assert (weights >= -1e-10).all(), f"{name}: has negative weights"
            assert 'metrics' in result, f"{name}: missing metrics"
            assert 'runtime' in result, f"{name}: missing runtime"

            # Check all required metrics
            required_metrics = ['sharpe_ratio', 'expected_return', 'expected_risk',
                              'n_assets', 'effective_n_assets', 'herfindahl_index']
            for metric in required_metrics:
                assert metric in result['metrics'], f"{name}: missing metric {metric}"

    def test_performance_ranking(self):
        """Test that methods have expected performance characteristics on synthetic data."""
        n_assets = 32
        prices = generate_synthetic_data(n_assets, seed=42)
        returns = compute_returns(prices, method='log')

        methods = {
            'Equal-Weight': EqualWeightOptimizer(),
            'Risk Parity': RiskParityOptimizer(),
        }

        results = {}
        for name, optimizer in methods.items():
            result = optimizer.optimize(returns)
            results[name] = result

        # Both should have reasonable Sharpe ratios
        for name, result in results.items():
            sharpe = result['metrics']['sharpe_ratio']
            assert sharpe > 0, f"{name} has non-positive Sharpe ratio"
            assert sharpe < 10, f"{name} has unrealistically high Sharpe ratio"

    def test_runtime_comparison(self):
        """Test that all classical methods are fast."""
        n_assets = 50
        prices = generate_synthetic_data(n_assets)
        returns = compute_returns(prices, method='log')

        methods = {
            'Equal-Weight': EqualWeightOptimizer(),
            'Risk Parity': RiskParityOptimizer(),
            'L1 (λ=0.01)': L1RegularizedOptimizer(lambda_l1=0.01),
            'L2 (λ=0.1)': L2RegularizedOptimizer(lambda_l2=0.1),
            'Mean-Variance': ClassicalOptimizer(gamma=1.0, method='cvxpy'),
        }

        print(f"\n{'='*70}")
        print(f"Runtime Comparison - {n_assets} Assets")
        print(f"{'='*70}")

        for name, optimizer in methods.items():
            result = optimizer.optimize(returns)
            runtime = result['runtime']
            print(f"  {name:<20} {runtime:>10.6f}s")

            # All should be fast (< 1 second)
            assert runtime < 1.0, f"{name} too slow: {runtime}s"

        # Equal-weight should be fastest
        ew_time = methods['Equal-Weight'].optimize(returns)['runtime']
        assert ew_time < 0.01, "Equal-Weight should be nearly instant"
