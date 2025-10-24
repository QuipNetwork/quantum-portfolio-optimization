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

"""Equal-weight (1/N) portfolio optimizer.

Reference:
    DeMiguel, V., Garlappi, L., & Uppal, R. (2009).
    "Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy?"
    Review of Financial Studies, 22(5), 1915-1953.

Key finding: Equal-weight often outperforms optimized portfolios out-of-sample
due to estimation error in expected returns and covariances.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any

from qpo.utils.constants import TRADING_DAYS_PER_YEAR
from qpo.utils.matrix_ops import sanitize_covariance_matrix


class EqualWeightOptimizer:
    """Equal-weight (1/N) portfolio - the naive baseline that often wins."""

    def __init__(self):
        """Initialize equal-weight optimizer."""
        pass

    def optimize(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Create equal-weight portfolio.

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

        N = len(returns.columns)
        weights = np.ones(N) / N

        runtime = time.time() - start_time

        # Package results
        weights_series = pd.Series(weights, index=returns.columns)

        # Compute metrics
        mu = returns.mean().values * TRADING_DAYS_PER_YEAR
        Sigma = returns.cov().values * TRADING_DAYS_PER_YEAR

        # Sanitize inputs before metrics computation
        mu, Sigma = sanitize_covariance_matrix(Sigma, mu)
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

        Note: Assumes mu and Sigma have already been sanitized via _sanitize_inputs.
        """
        w_arr = w.values
        N = len(w_arr)

        exp_return = mu @ w_arr

        # Compute risk with numerical safeguards
        # Suppress runtime warnings since we handle invalid values explicitly
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            variance = w_arr @ Sigma @ w_arr

        # Handle negative or invalid variance
        if not np.isfinite(variance) or variance < 0:
            variance = 0.0
        exp_risk = np.sqrt(variance)

        sharpe = exp_return / exp_risk if exp_risk > 0 else 0.0

        # All assets have equal weight
        herfindahl = np.sum(w_arr ** 2)
        effective_n = 1 / herfindahl if herfindahl > 0 else N

        return {
            'expected_return': float(exp_return),
            'expected_risk': float(exp_risk),
            'sharpe_ratio': float(sharpe),
            'n_assets': N,
            'herfindahl_index': float(herfindahl),
            'effective_n_assets': float(effective_n)
        }
