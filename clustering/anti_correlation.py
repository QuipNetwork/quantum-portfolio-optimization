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

"""Anti-correlation (diversification-oriented) clustering."""

import numpy as np
import pandas as pd
from .base import BaseClusterer


class AntiCorrelationClusterer(BaseClusterer):
    """
    Cluster assets to maximize intra-cluster diversification.

    Distance metric: 1 + correlation (NOT 1 - |correlation|)
    Assets with LOW/NEGATIVE correlation are "close" and grouped together.
    Opposite of CorrelationClusterer.

    Rationale:
    - CorrelationClusterer groups similar assets → creates "echo chambers" with high intra-cluster correlation
    - AntiCorrelationClusterer groups dissimilar assets → maximizes intra-cluster diversification
    - Each cluster becomes a "mini diversified portfolio"
    - Addresses the diversification loss identified in CORRELATION_ISSUES.md

    Example:
    - CorrelationClusterer: {AAPL, MSFT, GOOGL} → avg ρ=0.85 (poor diversification)
    - AntiCorrelationClusterer: {AAPL, SO (Utilities), GLD (Gold)} → avg ρ≈0.0 (good diversification)
    """

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 linkage_method: str = 'ward',
                 target_cluster_size: int = None):
        """
        Initialize anti-correlation clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster
            n_bits: Bits per weight variable
            linkage_method: 'ward', 'single', 'complete', or 'average'
            target_cluster_size: Target average cluster size (default: max_cluster_size // 2)
        """
        super().__init__(max_cluster_size, n_bits, linkage_method, target_cluster_size=target_cluster_size)

    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute anti-correlation distance matrix.

        Distance = 1 + correlation (maps [-1, +1] → [0, 2])
        - Strong negative correlation (ρ=-1) → distance=0 (very close, GOOD for diversification)
        - Zero correlation (ρ=0) → distance=1 (medium distance)
        - Strong positive correlation (ρ=+1) → distance=2 (far apart, AVOID grouping)

        This is the OPPOSITE of CorrelationClusterer which uses 1 - |ρ|.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Distance matrix (N × N)
        """
        # Compute correlation matrix
        corr = returns.corr()

        # Distance: 1 + correlation
        # This creates an INVERTED similarity measure:
        #   - Negative correlation → small distance → cluster together
        #   - Positive correlation → large distance → separate
        distance = 1 + corr.values

        # Ensure distance is symmetric and zero-diagonal
        np.fill_diagonal(distance, 0)
        distance = (distance + distance.T) / 2

        # Already in [0, 2] range from transformation
        distance = np.clip(distance, 0, 2)

        return distance
