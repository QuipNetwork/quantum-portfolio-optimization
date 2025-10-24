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

"""Portfolio performance metrics computation utilities."""

from typing import Dict
import numpy as np
import pandas as pd

from qpo.utils.constants import (
    WEIGHT_EPSILON,
    RISK_EPSILON,
    COVARIANCE_CLIP_MIN,
    COVARIANCE_CLIP_MAX,
    VARIANCE_CLIP_MAX
)


def compute_portfolio_metrics(
    weights: pd.Series,
    mu: np.ndarray,
    Sigma: np.ndarray,
    include_extra: bool = False
) -> Dict[str, float]:
    """
    Compute standard portfolio performance metrics.

    This function computes:
    - Expected return (annualized)
    - Expected risk/volatility (annualized)
    - Sharpe ratio (assuming risk-free rate = 0)
    - Number of assets with non-zero weights
    - Herfindahl index (concentration measure)
    - Effective number of assets (1 / Herfindahl)

    Args:
        weights: Portfolio weights (pd.Series with ticker index)
        mu: Expected returns vector (annualized, N,)
        Sigma: Covariance matrix (annualized, N × N)
        include_extra: If True, include additional metrics (sparsity, weight_std)

    Returns:
        Dictionary containing:
        {
            'expected_return': float,      # Annualized expected return
            'expected_risk': float,         # Annualized volatility (std dev)
            'sharpe_ratio': float,          # Return / Risk
            'n_assets': int,                # Number of non-zero positions
            'herfindahl_index': float,      # Sum of squared weights
            'effective_n_assets': float     # 1 / Herfindahl (diversification measure)
        }

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> weights = pd.Series([0.3, 0.4, 0.3], index=['AAPL', 'MSFT', 'GOOGL'])
        >>> mu = np.array([0.15, 0.12, 0.18])
        >>> Sigma = np.array([[0.04, 0.01, 0.02],
        ...                   [0.01, 0.03, 0.015],
        ...                   [0.02, 0.015, 0.05]])
        >>> metrics = compute_portfolio_metrics(weights, mu, Sigma)
        >>> print(f"Expected return: {metrics['expected_return']:.2%}")
        >>> print(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    """
    w_arr = weights.values

    # Expected return
    exp_return = mu @ w_arr

    # Expected risk (volatility) with numerical stability checks
    # Clip Sigma to prevent overflow before matrix multiplication
    Sigma_safe = np.clip(Sigma, COVARIANCE_CLIP_MIN, COVARIANCE_CLIP_MAX)

    # Suppress warnings for numerical edge cases
    with np.errstate(all='ignore'):
        risk_squared = w_arr @ Sigma_safe @ w_arr

    # Post-process result with safety checks
    risk_squared = np.clip(risk_squared, 0, VARIANCE_CLIP_MAX)  # Prevent overflow
    exp_risk = np.sqrt(risk_squared) if np.isfinite(risk_squared) else 0.0

    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe = exp_return / exp_risk if exp_risk > RISK_EPSILON else 0.0

    # Number of assets with non-zero weights
    n_assets = int(np.sum(w_arr > WEIGHT_EPSILON))

    # Effective number of assets (1 / Herfindahl index)
    herfindahl = np.sum(w_arr ** 2)
    effective_n = 1 / herfindahl if herfindahl > 0 else 0

    metrics = {
        'expected_return': float(exp_return),
        'expected_risk': float(exp_risk),
        'sharpe_ratio': float(sharpe),
        'n_assets': n_assets,
        'herfindahl_index': float(herfindahl),
        'effective_n_assets': float(effective_n)
    }

    # Add extra metrics if requested (used by regularized optimizers)
    if include_extra:
        N = len(w_arr)
        metrics['sparsity'] = 1 - (n_assets / N) if N > 0 else 0
        metrics['weight_std'] = float(np.std(w_arr))

    return metrics


def compute_portfolio_risk(
    weights: np.ndarray,
    Sigma: np.ndarray
) -> float:
    """
    Compute portfolio risk (standard deviation) with numerical stability.

    Args:
        weights: Portfolio weights (N,)
        Sigma: Covariance matrix (N × N)

    Returns:
        Portfolio standard deviation (volatility)

    Example:
        >>> weights = np.array([0.5, 0.5])
        >>> Sigma = np.array([[0.04, 0.01], [0.01, 0.03]])
        >>> risk = compute_portfolio_risk(weights, Sigma)
        >>> print(f"Risk: {risk:.2%}")
    """
    Sigma_safe = np.clip(Sigma, COVARIANCE_CLIP_MIN, COVARIANCE_CLIP_MAX)

    with np.errstate(all='ignore'):
        risk_squared = weights @ Sigma_safe @ weights

    risk_squared = np.clip(risk_squared, 0, VARIANCE_CLIP_MAX)
    return np.sqrt(risk_squared) if np.isfinite(risk_squared) else 0.0


def compute_risk_contributions(
    weights: pd.Series,
    Sigma: np.ndarray
) -> pd.Series:
    """
    Compute marginal risk contribution of each asset to portfolio risk.

    Used by risk parity optimizers to balance risk contributions.

    Args:
        weights: Portfolio weights (pd.Series with ticker index)
        Sigma: Covariance matrix (N × N)

    Returns:
        Series of risk contributions (same index as weights)

    Formula:
        RC_i = w_i * (Σ @ w)_i / σ_p
        where σ_p is the portfolio standard deviation

    Example:
        >>> weights = pd.Series([0.5, 0.5], index=['A', 'B'])
        >>> Sigma = np.array([[0.04, 0.01], [0.01, 0.03]])
        >>> rc = compute_risk_contributions(weights, Sigma)
    """
    w_arr = weights.values

    # Compute portfolio risk
    portfolio_risk = compute_portfolio_risk(w_arr, Sigma)

    if portfolio_risk < RISK_EPSILON:
        # If risk is zero, return equal contributions
        return pd.Series(0.0, index=weights.index)

    # Marginal contributions: Σ @ w
    marginal_contrib = Sigma @ w_arr

    # Risk contributions: w_i * (Σ @ w)_i / σ_p
    risk_contrib = (w_arr * marginal_contrib) / portfolio_risk

    return pd.Series(risk_contrib, index=weights.index)
