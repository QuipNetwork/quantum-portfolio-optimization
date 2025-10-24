# Copyright (C) 2025 Postquant Labs Incorporated
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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

    def __init__(self, max_iter: int = 1000, tol: float = 1e-8, reg_factor: float = 1e-8):
        """
        Initialize risk parity optimizer.

        Args:
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
            reg_factor: Regularization factor for covariance matrix conditioning
        """
        self.max_iter = max_iter
        self.tol = tol
        self.reg_factor = reg_factor
        self.eps = 1e-10  # Numerical stability epsilon

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

        # Compute covariance and mean
        Sigma = returns.cov().values * 252  # Annualized
        mu = returns.mean().values * 252

        # Sanitize inputs for numerical stability
        mu, Sigma = self._sanitize_inputs(mu, Sigma)

        # Condition the covariance matrix for numerical stability
        Sigma = self._condition_covariance(Sigma)

        # Solve for risk parity weights
        weights = self._solve_risk_parity(Sigma)

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

    def _sanitize_inputs(self, mu: np.ndarray, Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Sanitize mean returns and covariance matrix for numerical stability.

        Args:
            mu: Mean returns vector
            Sigma: Covariance matrix

        Returns:
            (sanitized_mu, sanitized_Sigma)
        """
        # Make copies to avoid modifying originals
        mu = mu.copy()
        Sigma = Sigma.copy()

        # Replace inf/nan in mu with 0
        if not np.all(np.isfinite(mu)):
            mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

        # Replace inf/nan in Sigma with 0 (will be conditioned later)
        if not np.all(np.isfinite(Sigma)):
            Sigma = np.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure diagonal elements are positive
        diag = np.diag(Sigma)
        if np.any(diag <= 0):
            min_var = self.eps * 100
            diag = np.maximum(diag, min_var)
            np.fill_diagonal(Sigma, diag)

        return mu, Sigma

    def _condition_covariance(self, Sigma: np.ndarray) -> np.ndarray:
        """
        Condition covariance matrix for numerical stability.

        Args:
            Sigma: Covariance matrix (N × N)

        Returns:
            Conditioned covariance matrix
        """
        N = len(Sigma)

        # Add regularization to ensure positive definiteness
        # and prevent ill-conditioning
        Sigma_conditioned = Sigma + self.reg_factor * np.eye(N)

        # Ensure symmetry (fix numerical errors)
        Sigma_conditioned = (Sigma_conditioned + Sigma_conditioned.T) / 2

        return Sigma_conditioned

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
            """Risk contribution of each asset with numerical stability."""
            # Add epsilon to weights to prevent underflow
            w_safe = np.maximum(w, self.eps)

            # Suppress runtime warnings since we handle invalid values explicitly
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                # Compute portfolio variance with clipping
                portfolio_var = w_safe @ Sigma @ w_safe
                portfolio_var = np.maximum(portfolio_var, self.eps)

                # Compute marginal contributions
                marginal_contrib = Sigma @ w_safe

            # Ensure finite values
            if not np.isfinite(portfolio_var):
                portfolio_var = self.eps
            marginal_contrib = np.nan_to_num(marginal_contrib, nan=0.0, posinf=0.0, neginf=0.0)

            # Risk contribution = weight * marginal contribution
            risk_contrib = w_safe * marginal_contrib

            return risk_contrib

        def objective(w):
            """Minimize squared difference in risk contributions."""
            rc = risk_contribution(w)

            # Target: equal risk contribution (1/N of total risk)
            target = np.ones(N) / N

            # Normalize risk contributions to sum to 1
            rc_sum = np.maximum(rc.sum(), self.eps)
            rc_normalized = rc / rc_sum

            return np.sum((rc_normalized - target) ** 2)

        def budget_constraint(w):
            """Budget: sum(w) = 1."""
            return np.sum(w) - 1

        # Initial guess: equal weights
        w0 = np.ones(N) / N

        # Bounds: weights between small epsilon and 1 to prevent numerical issues
        # Using eps * 10 as minimum to ensure stability
        min_weight = self.eps * 10
        bounds = [(min_weight, 1) for _ in range(N)]

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

        # Normalize and ensure positivity
        w_opt = result.x
        w_opt = np.maximum(w_opt, min_weight)
        w_opt /= np.maximum(w_opt.sum(), self.eps)

        return w_opt

    def _compute_metrics(self,
                        w: pd.Series,
                        mu: np.ndarray,
                        Sigma: np.ndarray) -> Dict[str, float]:
        """Compute portfolio performance metrics with numerical stability."""
        w_arr = w.values
        N = len(w_arr)

        # Ensure weights are positive and normalized
        w_arr = np.maximum(w_arr, self.eps)
        w_arr /= np.maximum(w_arr.sum(), self.eps)

        # Expected return
        exp_return = mu @ w_arr

        # Suppress runtime warnings since we handle invalid values explicitly
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            # Expected risk with numerical stability
            portfolio_var = w_arr @ Sigma @ w_arr
            portfolio_var = np.maximum(portfolio_var, self.eps)
            exp_risk = np.sqrt(portfolio_var)

            # Risk contribution analysis with numerical stability
            marginal_risk = Sigma @ w_arr

        # Ensure finite values
        if not np.isfinite(portfolio_var):
            portfolio_var = self.eps
            exp_risk = np.sqrt(portfolio_var)
        marginal_risk = np.nan_to_num(marginal_risk, nan=0.0, posinf=0.0, neginf=0.0)

        # Sharpe ratio
        sharpe = exp_return / np.maximum(exp_risk, self.eps)

        # Number of assets with meaningful weight
        n_assets = int(np.sum(w_arr > 1e-6))

        risk_contrib = w_arr * marginal_risk

        # Normalize risk contributions
        risk_contrib_sum = np.maximum(np.abs(risk_contrib).sum(), self.eps)
        risk_contrib_normalized = risk_contrib / risk_contrib_sum

        # Concentration metrics
        herfindahl = np.sum(w_arr ** 2)
        herfindahl = np.maximum(herfindahl, self.eps)
        effective_n = 1 / herfindahl

        return {
            'expected_return': float(exp_return),
            'expected_risk': float(exp_risk),
            'sharpe_ratio': float(sharpe),
            'n_assets': n_assets,
            'herfindahl_index': float(herfindahl),
            'effective_n_assets': float(effective_n),
            'risk_contrib_std': float(np.std(risk_contrib_normalized))
        }
