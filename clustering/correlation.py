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

"""Correlation-based clustering."""

import numpy as np
import pandas as pd
from .base import BaseClusterer


class CorrelationClusterer(BaseClusterer):
    """
    Cluster assets by correlation similarity.

    Distance metric: 1 - |correlation|
    Highly correlated assets are "close" and clustered together.
    """

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 linkage_method: str = 'ward',
                 use_absolute: bool = True,
                 target_cluster_size: int = None):
        """
        Initialize correlation clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster
            n_bits: Bits per weight variable
            linkage_method: 'ward', 'single', 'complete', or 'average'
            use_absolute: If True, use |correlation| (ignores direction).
                         If False, negative correlation = far apart.
        """
        super().__init__(max_cluster_size, n_bits, linkage_method, target_cluster_size=target_cluster_size)
        self.use_absolute = use_absolute

    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute correlation-based distance matrix.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Distance matrix (N × N)
        """
        # Compute correlation matrix
        corr = returns.corr()

        # Distance: 1 - correlation
        if self.use_absolute:
            distance = 1 - corr.abs().values
        else:
            distance = 1 - corr.values

        # Ensure distance is symmetric and zero-diagonal
        np.fill_diagonal(distance, 0)
        distance = (distance + distance.T) / 2

        # Clip to [0, 2] (for negative correlations without abs)
        distance = np.clip(distance, 0, 2)

        return distance
