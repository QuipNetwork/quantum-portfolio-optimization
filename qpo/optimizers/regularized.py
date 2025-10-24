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
from qpo.utils.portfolio_metrics import compute_portfolio_metrics


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
        metrics = compute_portfolio_metrics(weights_series, mu, Sigma, include_extra=True)

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
        metrics = compute_portfolio_metrics(weights_series, mu, Sigma, include_extra=True)

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
