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

"""Regularized portfolio optimizers (L1 and L2).

References:
    - Brodie, J., et al. (2009). "Sparse and stable portfolio selection with parameter uncertainty."
    - DeMiguel, V., et al. (2009). "A generalized approach to portfolio optimization."

Regularization addresses estimation error in expected returns and covariances.
"""

import time
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Any

from qpo.utils.constants import TRADING_DAYS_PER_YEAR
from qpo.utils.matrix_ops import sanitize_covariance_matrix


class L1RegularizedOptimizer:
    """L1-regularized (Lasso) portfolio optimizer.

    Encourages sparse portfolios by adding L1 penalty:
        minimize  γ·w^T Σ w - μ^T w + λ·||w||₁
    """

    def __init__(self, gamma: float = 1.0, lambda_l1: float = 0.01):
        """
        Initialize L1-regularized optimizer.

        Args:
            gamma: Risk aversion parameter (higher = more conservative)
            lambda_l1: L1 regularization strength (higher = more sparse)
        """
        self.gamma = gamma
        self.lambda_l1 = lambda_l1

    def optimize(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Run L1-regularized optimization.

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

        # Compute annualized statistics
        mu = returns.mean().values * TRADING_DAYS_PER_YEAR
        Sigma = returns.cov().values * TRADING_DAYS_PER_YEAR

        # Sanitize for numerical stability
        mu, Sigma = sanitize_covariance_matrix(Sigma, mu)

        # Solve
        weights = self._solve(mu, Sigma)

        runtime = time.time() - start_time

        # Package results
        weights_series = pd.Series(weights, index=returns.columns)
        metrics = self._compute_metrics(weights_series, mu, Sigma)

        return {
            'weights': weights_series,
            'metrics': metrics,
            'runtime': runtime
        }

    def _solve(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Solve L1-regularized optimization."""
        N = len(mu)
        w = cp.Variable(N)

        # Ensure matrix is a numpy array and exactly symmetric for CVXPY
        Sigma = np.asarray(Sigma, dtype=np.float64)
        Sigma = (Sigma + Sigma.T) / 2.0

        # Objective: risk - return + L1 penalty
        risk = cp.quad_form(w, Sigma)
        ret = mu @ w
        l1_penalty = cp.norm(w, 1)

        objective = cp.Minimize(self.gamma * risk - ret + self.lambda_l1 * l1_penalty)

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise ValueError(f"L1 optimization failed: {problem.status}")

        return w.value

    def _compute_metrics(self,
                        w: pd.Series,
                        mu: np.ndarray,
                        Sigma: np.ndarray) -> Dict[str, float]:
        """Compute portfolio performance metrics."""
        w_arr = w.values
        N = len(w_arr)

        exp_return = mu @ w_arr

        # Compute risk with numerical stability checks
        # Clip Sigma to prevent overflow before matmul
        Sigma_safe = np.clip(Sigma, -1e8, 1e8)

        # Suppress warnings for numerical edge cases
        with np.errstate(all='ignore'):
            risk_squared = w_arr @ Sigma_safe @ w_arr

        # Post-process result with safety checks
        risk_squared = np.clip(risk_squared, 0, 1e10)  # Prevent overflow
        exp_risk = np.sqrt(risk_squared) if np.isfinite(risk_squared) else 0.0

        sharpe = exp_return / exp_risk if exp_risk > 1e-10 else 0.0
        n_assets = int(np.sum(w_arr > 1e-6))

        herfindahl = np.sum(w_arr ** 2)
        effective_n = 1 / herfindahl if herfindahl > 0 else N

        return {
            'expected_return': float(exp_return),
            'expected_risk': float(exp_risk),
            'sharpe_ratio': float(sharpe),
            'n_assets': n_assets,
            'herfindahl_index': float(herfindahl),
            'effective_n_assets': float(effective_n),
            'sparsity': 1 - (n_assets / N)
        }


class L2RegularizedOptimizer:
    """L2-regularized (Ridge) portfolio optimizer.

    Shrinks weights toward equal-weighting by adding L2 penalty:
        minimize  γ·w^T Σ w - μ^T w + λ·||w||₂²
    """

    def __init__(self, gamma: float = 1.0, lambda_l2: float = 0.01):
        """
        Initialize L2-regularized optimizer.

        Args:
            gamma: Risk aversion parameter (higher = more conservative)
            lambda_l2: L2 regularization strength (higher = more uniform)
        """
        self.gamma = gamma
        self.lambda_l2 = lambda_l2

    def optimize(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Run L2-regularized optimization.

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

        # Compute annualized statistics
        mu = returns.mean().values * TRADING_DAYS_PER_YEAR
        Sigma = returns.cov().values * TRADING_DAYS_PER_YEAR

        # Sanitize for numerical stability
        mu, Sigma = sanitize_covariance_matrix(Sigma, mu)

        # Solve
        weights = self._solve(mu, Sigma)

        runtime = time.time() - start_time

        # Package results
        weights_series = pd.Series(weights, index=returns.columns)
        metrics = self._compute_metrics(weights_series, mu, Sigma)

        return {
            'weights': weights_series,
            'metrics': metrics,
            'runtime': runtime
        }

    def _solve(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Solve L2-regularized optimization."""
        N = len(mu)
        w = cp.Variable(N)

        # Ensure matrix is a numpy array and exactly symmetric for CVXPY
        Sigma = np.asarray(Sigma, dtype=np.float64)
        Sigma = (Sigma + Sigma.T) / 2.0

        # Objective: risk - return + L2 penalty
        risk = cp.quad_form(w, Sigma)
        ret = mu @ w
        l2_penalty = cp.sum_squares(w)

        objective = cp.Minimize(self.gamma * risk - ret + self.lambda_l2 * l2_penalty)

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise ValueError(f"L2 optimization failed: {problem.status}")

        return w.value

    def _compute_metrics(self,
                        w: pd.Series,
                        mu: np.ndarray,
                        Sigma: np.ndarray) -> Dict[str, float]:
        """Compute portfolio performance metrics."""
        w_arr = w.values
        N = len(w_arr)

        exp_return = mu @ w_arr

        # Compute risk with numerical stability checks
        # Clip Sigma to prevent overflow before matmul
        Sigma_safe = np.clip(Sigma, -1e8, 1e8)

        # Suppress warnings for numerical edge cases
        with np.errstate(all='ignore'):
            risk_squared = w_arr @ Sigma_safe @ w_arr

        # Post-process result with safety checks
        risk_squared = np.clip(risk_squared, 0, 1e10)  # Prevent overflow
        exp_risk = np.sqrt(risk_squared) if np.isfinite(risk_squared) else 0.0

        sharpe = exp_return / exp_risk if exp_risk > 1e-10 else 0.0
        n_assets = int(np.sum(w_arr > 1e-6))

        herfindahl = np.sum(w_arr ** 2)
        effective_n = 1 / herfindahl if herfindahl > 0 else N

        # Weight uniformity
        weight_std = np.std(w_arr)

        return {
            'expected_return': float(exp_return),
            'expected_risk': float(exp_risk),
            'sharpe_ratio': float(sharpe),
            'n_assets': n_assets,
            'herfindahl_index': float(herfindahl),
            'effective_n_assets': float(effective_n),
            'weight_std': float(weight_std)
        }
