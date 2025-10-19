"""Risk Parity portfolio optimizer.

Reference:
    Maillard, S., Roncalli, T., & Teiletche, J. (2010).
    "The properties of equally weighted risk contribution portfolios."
    The Journal of Portfolio Management, 36(4), 60-70.

Used by institutional investors including Bridgewater's All Weather fund.
Each asset contributes equally to portfolio risk.
"""

import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any


class RiskParityOptimizer:
    """Risk Parity - equal risk contribution from each asset."""

    def __init__(self, max_iter: int = 1000, tol: float = 1e-8):
        """
        Initialize risk parity optimizer.

        Args:
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol

    def optimize(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Create risk parity portfolio.

        Args:
            returns: Returns DataFrame (dates × tickers)

        Returns:
            {
                'weights': pd.Series,
                'metrics': dict,
                'runtime': float
            }
        """
        start_time = time.time()

        # Compute covariance
        Sigma = returns.cov().values * 252  # Annualized

        # Solve for risk parity weights
        weights = self._solve_risk_parity(Sigma)

        runtime = time.time() - start_time

        # Package results
        weights_series = pd.Series(weights, index=returns.columns)

        # Compute metrics
        mu = returns.mean().values * 252
        metrics = self._compute_metrics(weights_series, mu, Sigma)

        return {
            'weights': weights_series,
            'metrics': metrics,
            'runtime': runtime
        }

    def _solve_risk_parity(self, Sigma: np.ndarray) -> np.ndarray:
        """
        Solve for risk parity weights.

        Objective: Each asset contributes equally to portfolio risk.
            w_i * (Σw)_i = constant for all i

        Args:
            Sigma: Covariance matrix (N × N)

        Returns:
            Risk parity weights (N,)
        """
        N = len(Sigma)

        def risk_contribution(w):
            """Risk contribution of each asset."""
            portfolio_var = w @ Sigma @ w
            marginal_contrib = Sigma @ w
            risk_contrib = w * marginal_contrib
            return risk_contrib

        def objective(w):
            """Minimize squared difference in risk contributions."""
            rc = risk_contribution(w)
            # Target: equal risk contribution (1/N of total risk)
            target = np.ones(N) / N
            # Normalize risk contributions to sum to 1
            rc_normalized = rc / rc.sum() if rc.sum() > 0 else rc
            return np.sum((rc_normalized - target) ** 2)

        def budget_constraint(w):
            """Budget: sum(w) = 1."""
            return np.sum(w) - 1

        # Initial guess: equal weights
        w0 = np.ones(N) / N

        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(N)]

        # Constraint
        constraints = {'type': 'eq', 'fun': budget_constraint}

        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )

        if not result.success:
            # Fallback to equal weights if optimization fails
            return w0

        # Normalize
        w_opt = result.x
        w_opt = np.maximum(w_opt, 0)
        w_opt /= w_opt.sum()

        return w_opt

    def _compute_metrics(self,
                        w: pd.Series,
                        mu: np.ndarray,
                        Sigma: np.ndarray) -> Dict[str, float]:
        """Compute portfolio performance metrics."""
        w_arr = w.values
        N = len(w_arr)

        exp_return = mu @ w_arr
        exp_risk = np.sqrt(w_arr @ Sigma @ w_arr)
        sharpe = exp_return / exp_risk if exp_risk > 0 else 0
        n_assets = int(np.sum(w_arr > 1e-6))

        # Risk contribution analysis
        marginal_risk = Sigma @ w_arr
        risk_contrib = w_arr * marginal_risk
        risk_contrib_normalized = risk_contrib / risk_contrib.sum() if risk_contrib.sum() > 0 else risk_contrib

        # Concentration metrics
        herfindahl = np.sum(w_arr ** 2)
        effective_n = 1 / herfindahl if herfindahl > 0 else N

        return {
            'expected_return': float(exp_return),
            'expected_risk': float(exp_risk),
            'sharpe_ratio': float(sharpe),
            'n_assets': n_assets,
            'herfindahl_index': float(herfindahl),
            'effective_n_assets': float(effective_n),
            'risk_contrib_std': float(np.std(risk_contrib_normalized))
        }
