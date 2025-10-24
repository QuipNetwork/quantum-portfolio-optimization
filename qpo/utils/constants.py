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

"""Numerical and financial constants for portfolio optimization."""

# =============================================================================
# Trading Calendar Constants
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
"""Number of trading days in a year (used for annualization)."""

# =============================================================================
# Numerical Stability Constants
# =============================================================================

COVARIANCE_REGULARIZATION = 1e-5
"""Regularization term added to covariance matrix diagonal for numerical stability."""

WEIGHT_EPSILON = 1e-6
"""Minimum weight threshold to consider an asset as having non-zero allocation."""

RISK_EPSILON = 1e-10
"""Minimum risk value for Sharpe ratio and similar calculations (avoids division by zero)."""

NUMERICAL_EPSILON = 1e-10
"""General numerical stability epsilon for floating point comparisons."""

# =============================================================================
# Matrix Conditioning Constants
# =============================================================================

COVARIANCE_CLIP_MIN = -1e8
"""Minimum value for clipping covariance matrix elements."""

COVARIANCE_CLIP_MAX = 1e8
"""Maximum value for clipping covariance matrix elements."""

VARIANCE_CLIP_MAX = 1e10
"""Maximum value for clipping variance calculations."""

# =============================================================================
# QUBO/Quantum Optimization Defaults
# =============================================================================

DEFAULT_N_LEVELS = 6
"""Default number of discrete levels for quantum optimization."""

DEFAULT_ALPHA = 10.0
"""Default return coefficient in QUBO formulation."""

DEFAULT_BETA = 2.0
"""Default risk coefficient in QUBO formulation."""

DEFAULT_THERMOMETER_PENALTY = 10.0
"""Default penalty for thermometer encoding constraint violations."""

DEFAULT_N_BITS = 8
"""Default number of bits for weight encoding in quantum optimization."""

DEFAULT_K_NORMALIZATION = 2**DEFAULT_N_BITS - 1  # 255
"""Default normalization constant (2^n - 1) for discrete weight levels."""

# =============================================================================
# Clustering Parameters
# =============================================================================

DEFAULT_MAX_CLUSTER_SIZE = 18
"""Default maximum size for asset clusters in quantum optimization."""

DEFAULT_MIN_CLUSTER_SIZE = 3
"""Default minimum size for asset clusters."""

# =============================================================================
# Backtesting Defaults
# =============================================================================

DEFAULT_INITIAL_CAPITAL = 100000.0
"""Default initial capital for backtesting ($100,000)."""

DEFAULT_TRAIN_DAYS = 252
"""Default training window (1 year of trading days)."""

DEFAULT_TEST_DAYS = 21
"""Default test/holding period (approximately 1 month)."""

DEFAULT_STEP_DAYS = 21
"""Default rebalancing frequency (approximately monthly)."""
