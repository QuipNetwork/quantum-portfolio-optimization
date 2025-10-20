"""Clustering methods for portfolio optimization."""

# Default: Original hierarchical correlation clustering
from .hierarchical import PortfolioClustering

# Base class for custom clusterers
from .base import BaseClusterer

# Alternative clustering methods
from .correlation import CorrelationClusterer
from .returns_based import ReturnsClusterer
from .volatility import VolatilityClusterer
from .sector import SectorClusterer
from .dtw import DTWClusterer
from .graph import GraphClusterer
from .factor import FactorClusterer

__all__ = [
    # Default (backward compatible)
    'PortfolioClustering',

    # Base class
    'BaseClusterer',

    # Alternative methods
    'CorrelationClusterer',
    'ReturnsClusterer',
    'VolatilityClusterer',
    'SectorClusterer',
    'DTWClusterer',
    'GraphClusterer',
    'FactorClusterer',
]
