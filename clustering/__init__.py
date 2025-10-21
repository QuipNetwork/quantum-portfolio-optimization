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

"""Clustering methods for portfolio optimization."""

# Default: Original hierarchical correlation clustering
from .hierarchical import PortfolioClustering

# Base class for custom clusterers
from .base import BaseClusterer

# Alternative clustering methods
from .correlation import CorrelationClusterer
from .covariance import CovarianceClusterer
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
    'CovarianceClusterer',
    'ReturnsClusterer',
    'VolatilityClusterer',
    'SectorClusterer',
    'DTWClusterer',
    'GraphClusterer',
    'FactorClusterer',
]
