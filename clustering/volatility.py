"""Volatility-based clustering."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from .base import BaseClusterer


class VolatilityClusterer(BaseClusterer):
    """
    Cluster assets by volatility (risk) similarity.

    Groups low-volatility, medium-volatility, and high-volatility assets.
    Useful for risk-based portfolio construction.
    """

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 linkage_method: str = 'ward',
                 target_cluster_size: int = None):
        """
        Initialize volatility-based clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster
            n_bits: Bits per weight variable
            linkage_method: 'ward', 'single', 'complete', or 'average'
        """
        super().__init__(max_cluster_size, n_bits, linkage_method, target_cluster_size=target_cluster_size)

    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute distance matrix based on volatility.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Condensed distance array
        """
        # Compute volatility (standard deviation) for each asset
        volatilities = returns.std(axis=0).values.reshape(-1, 1)

        # Euclidean distance between volatilities
        distance = pdist(volatilities, metric='euclidean')

        return distance
