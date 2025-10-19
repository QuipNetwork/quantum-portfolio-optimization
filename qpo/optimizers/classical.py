"""Classical portfolio optimization using mean-variance framework."""

import time
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import differential_evolution
from typing import Dict, Any, Optional


def solve_markowitz_continuous(mu: np.ndarray,
                               Sigma: np.ndarray,
                               gamma: float = 1.0) -> np.ndarray:
    """
    Solve continuous mean-variance optimization using CVXPY.

    Objective: minimize gamma * risk - return
    Subject to: sum(w) = 1, w >= 0

    Args:
        mu: Expected returns (N,)
        Sigma: Covariance matrix (N, N)
        gamma: Risk aversion parameter (higher = more conservative)

    Returns:
        Optimal weights (N,)
    """
    N = len(mu)
    w = cp.Variable(N)

    # Objective: risk - return trade-off
    risk = cp.quad_form(w, Sigma)
    ret = mu @ w
    objective = cp.Minimize(gamma * risk - ret)

    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Budget constraint
        w >= 0           # No short selling
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    return w.value


def solve_markowitz_discrete(mu: np.ndarray,
                             Sigma: np.ndarray,
                             k: int,
                             gamma: float = 1.0) -> np.ndarray:
    """
    Solve discrete mean-variance with cardinality constraint.

    Uses Genetic Algorithm (Differential Evolution) to handle
    combinatorial cardinality constraint.

    Args:
        mu: Expected returns (N,)
        Sigma: Covariance matrix (N, N)
        k: Maximum number of assets in portfolio
        gamma: Risk aversion parameter

    Returns:
        Optimal weights (N,)
    """
    N = len(mu)

    def objective(w):
        """Objective function with cardinality penalty."""
        # Count non-zero weights
        n_nonzero = np.sum(w > 1e-6)
        cardinality_penalty = max(0, n_nonzero - k) * 1e6

        # Portfolio objective
        risk = w @ Sigma @ w
        ret = mu @ w

        return gamma * risk - ret + cardinality_penalty

    def budget_constraint(w):
        """Budget constraint: sum(w) = 1."""
        return np.sum(w) - 1

    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in range(N)]

    # Solve with differential evolution
    # Note: scipy's differential_evolution uses NonlinearConstraint format
    from scipy.optimize import NonlinearConstraint

    constraints = NonlinearConstraint(budget_constraint, 0, 0)

    result = differential_evolution(
        objective,
        bounds,
        constraints=constraints,
        seed=42,
        maxiter=1000,
        polish=True,
        atol=1e-6,
        tol=1e-6
    )

    # Check if optimization succeeded (allow small constraint violations from numerical precision)
    if not result.success and result.maxcv > 1e-5:
        raise ValueError(f"Optimization failed: {result.message} (MAXCV={result.maxcv})")

    # Post-process: enforce cardinality by keeping only top k assets
    w_opt = result.x

    # Select top k assets by weight
    top_k_indices = np.argsort(w_opt)[-k:]
    w_enforced = np.zeros(N)
    w_enforced[top_k_indices] = w_opt[top_k_indices]

    # Zero out very small weights
    w_enforced[w_enforced < 1e-6] = 0

    # Renormalize
    if w_enforced.sum() > 0:
        w_enforced /= w_enforced.sum()

    return w_enforced


class ClassicalOptimizer:
    """Classical portfolio optimization using mean-variance framework."""

    def __init__(self,
                 gamma: float = 1.0,
                 k: Optional[int] = None,
                 method: str = 'cvxpy'):
        """
        Initialize classical optimizer.

        Args:
            gamma: Risk aversion parameter (higher = more conservative)
                   gamma=0: maximize return only
                   gamma=∞: minimize risk only
            k: Cardinality constraint (max number of assets, None = no limit)
            method: Optimization method
                   'cvxpy': Continuous optimization (fast, no cardinality)
                   'ga': Genetic algorithm (slower, supports cardinality)
        """
        self.gamma = gamma
        self.k = k
        self.method = method

        # Validate
        if method not in ['cvxpy', 'ga']:
            raise ValueError(f"Unknown method: {method}. Use 'cvxpy' or 'ga'")

        if k is not None and method == 'cvxpy':
            raise ValueError("CVXPY method does not support cardinality constraint. Use 'ga'")

    def optimize(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Run optimization on returns data.

        Args:
            returns: Returns DataFrame (dates × tickers)

        Returns:
            {
                'weights': pd.Series,      # Asset weights
                'metrics': dict,           # Performance metrics
                'runtime': float           # Optimization time (seconds)
            }
        """
        start_time = time.time()

        # Compute statistics
        mu = returns.mean().values * 252  # Annualized
        Sigma = returns.cov().values * 252  # Annualized

        # Solve based on method
        if self.method == 'cvxpy':
            weights = solve_markowitz_continuous(mu, Sigma, self.gamma)
        elif self.method == 'ga':
            if self.k is None:
                # Default: use all assets
                self.k = len(returns.columns)
            weights = solve_markowitz_discrete(mu, Sigma, self.k, self.gamma)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        runtime = time.time() - start_time

        # Package results
        weights_series = pd.Series(weights, index=returns.columns)

        # Compute metrics
        metrics = self._compute_metrics(weights_series, mu, Sigma)

        return {
            'weights': weights_series,
            'metrics': metrics,
            'runtime': runtime
        }

    def _compute_metrics(self,
                        w: pd.Series,
                        mu: np.ndarray,
                        Sigma: np.ndarray) -> Dict[str, float]:
        """
        Compute portfolio performance metrics.

        Args:
            w: Portfolio weights (pd.Series)
            mu: Expected returns (annualized)
            Sigma: Covariance matrix (annualized)

        Returns:
            Dictionary of metrics
        """
        w_arr = w.values

        # Expected return
        exp_return = mu @ w_arr

        # Expected risk (volatility)
        exp_risk = np.sqrt(w_arr @ Sigma @ w_arr)

        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe = exp_return / exp_risk if exp_risk > 0 else 0

        # Number of assets with non-zero weights
        n_assets = int(np.sum(w_arr > 1e-6))

        return {
            'expected_return': float(exp_return),
            'expected_risk': float(exp_risk),
            'sharpe_ratio': float(sharpe),
            'n_assets': n_assets
        }
