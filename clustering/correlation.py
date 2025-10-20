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
                 use_absolute: bool = True):
        """
        Initialize correlation clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster
            n_bits: Bits per weight variable
            linkage_method: 'ward', 'single', 'complete', or 'average'
            use_absolute: If True, use |correlation| (ignores direction).
                         If False, negative correlation = far apart.
        """
        super().__init__(max_cluster_size, n_bits, linkage_method)
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
