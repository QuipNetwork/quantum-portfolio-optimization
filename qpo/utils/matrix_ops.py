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

"""Matrix operations and preprocessing utilities for portfolio optimization."""

from typing import Optional, Tuple, Union
import numpy as np

from qpo.utils.constants import COVARIANCE_REGULARIZATION


def sanitize_covariance_matrix(
    Sigma: np.ndarray,
    mu: Optional[np.ndarray] = None,
    regularization: float = COVARIANCE_REGULARIZATION
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Sanitize covariance matrix and optionally mean vector for numerical stability.

    This function performs several preprocessing steps:
    1. Replaces NaN, inf, -inf values with zeros
    2. Forces exact symmetry of covariance matrix
    3. Adds regularization to diagonal for positive definiteness

    Args:
        Sigma: Covariance matrix (n × n)
        mu: Optional mean returns vector (n,)
        regularization: Diagonal regularization term (default from constants)

    Returns:
        If mu is None: Sanitized covariance matrix
        If mu is provided: Tuple of (sanitized_mu, sanitized_Sigma)

    Example:
        >>> import numpy as np
        >>> Sigma = np.array([[0.01, 0.005], [0.005, 0.02]])
        >>> mu = np.array([0.1, 0.15])
        >>> mu_clean, Sigma_clean = sanitize_covariance_matrix(Sigma, mu)
    """
    # Handle mean vector if provided
    if mu is not None:
        mu_clean = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    # Handle NaN values in covariance matrix
    Sigma_clean = np.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)

    # Force exact symmetry (eliminate floating-point errors)
    Sigma_clean = (Sigma_clean + Sigma_clean.T) / 2

    # Add regularization to diagonal for numerical stability and positive definiteness
    Sigma_clean = Sigma_clean + np.eye(len(Sigma_clean)) * regularization

    if mu is not None:
        return mu_clean, Sigma_clean
    else:
        return Sigma_clean


def annualize_statistics(
    returns_mean: np.ndarray,
    returns_cov: np.ndarray,
    periods_per_year: int = 252
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Annualize mean returns and covariance matrix.

    Args:
        returns_mean: Mean returns (typically daily)
        returns_cov: Covariance matrix (typically daily)
        periods_per_year: Number of periods per year (default 252 for daily)

    Returns:
        Tuple of (annualized_mean, annualized_covariance)

    Example:
        >>> daily_mean = np.array([0.001, 0.0015])
        >>> daily_cov = np.array([[0.0001, 0.00005], [0.00005, 0.0002]])
        >>> annual_mean, annual_cov = annualize_statistics(daily_mean, daily_cov)
    """
    mu_annual = returns_mean * periods_per_year
    Sigma_annual = returns_cov * periods_per_year
    return mu_annual, Sigma_annual
