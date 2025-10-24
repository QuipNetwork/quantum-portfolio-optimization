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

"""Dynamic Time Warping (DTW) clustering."""

import numpy as np
import pandas as pd
from .base import BaseClusterer


class DTWClusterer(BaseClusterer):
    """
    Cluster assets using Dynamic Time Warping distance.

    DTW captures temporal patterns and shape similarity of time series.
    Good for finding assets that move together but with time lags.

    Requires: tslearn library (pip install tslearn)
    """

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 linkage_method: str = 'average',
                 window_size: int = None,
                 target_cluster_size: int = None,
                 max_clusters: int = None,
                 min_cluster_size: int = None):
        """
        Initialize DTW clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster
            n_bits: Bits per weight variable
            linkage_method: 'ward', 'single', 'complete', or 'average'
            window_size: Sakoe-Chiba band width (None = no constraint)
        """
        super().__init__(max_cluster_size, n_bits, linkage_method, target_cluster_size=target_cluster_size,
                        max_clusters=max_clusters,
                        min_cluster_size=min_cluster_size)
        self.window_size = window_size

    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute DTW distance matrix.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Distance matrix (N × N)
        """
        try:
            from tslearn.metrics import dtw
        except ImportError:
            raise ImportError(
                "tslearn not installed. "
                "Install with: pip install tslearn\n"
                "Note: DTW clustering is optional. Use CorrelationClusterer for basic usage."
            )

        n_assets = returns.shape[1]
        distance_matrix = np.zeros((n_assets, n_assets))

        # Compute pairwise DTW distances
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                ts1 = returns.iloc[:, i].values
                ts2 = returns.iloc[:, j].values

                # Compute DTW distance
                if self.window_size is not None:
                    dist = dtw(ts1, ts2, global_constraint='sakoe_chiba',
                             sakoe_chiba_radius=self.window_size)
                else:
                    dist = dtw(ts1, ts2)

                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix
