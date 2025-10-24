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

"""Sector-based clustering using Yahoo Finance sector data."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
from .base import BaseClusterer
from .sector_merge import smart_sector_merge, get_sector_cluster_stats


class SectorClusterer(BaseClusterer):
    """
    Cluster assets by sector/industry classification.

    Can use:
    1. Pre-computed sector_map dictionary
    2. CSV file from 'qpo fetch-info' command
    3. Fetch from Yahoo Finance API on-demand

    Most stable and interpretable clustering method.
    """

    def __init__(self,
                 max_cluster_size: int = 18,
                 n_bits: int = 10,
                 sector_map: Optional[Union[Dict[str, str], str, Path]] = None,
                 use_industry: bool = False,
                 target_cluster_size: int = None,
                 max_clusters: int = None,
                 min_cluster_size: int = None):
        """
        Initialize sector-based clusterer.

        Args:
            max_cluster_size: Maximum assets per cluster
            n_bits: Bits per weight variable
            sector_map: Can be:
                - Dict[str, str]: Pre-computed {ticker: sector} mapping
                - str or Path: Path to CSV file from 'qpo fetch-info' command
                - None: Will fetch from Yahoo Finance API
            use_industry: If True, use industry instead of sector (more granular)
        """
        super().__init__(max_cluster_size, n_bits, linkage_method='ward', target_cluster_size=target_cluster_size,
                        max_clusters=max_clusters,
                        min_cluster_size=min_cluster_size)

        # Handle different sector_map input types
        if isinstance(sector_map, (str, Path)):
            self.sector_map = self._load_from_csv(sector_map, use_industry)
        else:
            self.sector_map = sector_map

        self.use_industry = use_industry

    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Not used for sector clustering (we override cluster() method).

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Dummy distance matrix
        """
        raise NotImplementedError("Sector clustering doesn't use distance matrix")

    def cluster(self, returns: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Cluster assets by sector/industry.

        Overrides base class method since we don't use hierarchical clustering.

        Args:
            returns: Returns DataFrame (T × N)

        Returns:
            Dictionary mapping sector names to ticker lists
        """
        tickers = returns.columns.tolist()

        # Fetch sector data if not provided
        if self.sector_map is None:
            self.sector_map = self._fetch_sector_data(tickers)

        # Group by sector
        sector_clusters = {}
        for ticker in tickers:
            sector = self.sector_map.get(ticker, 'Unknown')

            if sector not in sector_clusters:
                sector_clusters[sector] = []

            sector_clusters[sector].append(ticker)

        # Calculate how many clusters we'll have after splitting
        def count_clusters_after_split(clusters, max_size):
            """Count total clusters after splitting large ones."""
            total = 0
            for tickers in clusters.values():
                if len(tickers) <= max_size:
                    total += 1
                else:
                    total += int(np.ceil(len(tickers) / max_size))
            return total

        # Apply smart sector merging if needed (before splitting)
        if self.max_clusters is not None:
            # Check if we'll exceed max_clusters after splitting
            clusters_after_split = count_clusters_after_split(sector_clusters, self.max_cluster_size)

            if clusters_after_split > self.max_clusters:
                # Need to merge sectors to reduce cluster count
                # Calculate target: how many sectors we need before splitting
                # This is approximate - we merge until splitting yields <= max_clusters
                target_sectors = max(1, self.max_clusters // 2)  # Start conservative

                # Iteratively merge until we satisfy constraints
                merged = sector_clusters
                for target in range(target_sectors, 0, -1):
                    try:
                        merged = smart_sector_merge(
                            sector_clusters,
                            target,
                            float('inf')  # No size limit during merging
                        )

                        # Check if this works after splitting
                        if count_clusters_after_split(merged, self.max_cluster_size) <= self.max_clusters:
                            sector_clusters = merged
                            break
                    except ValueError:
                        # Merging failed, try next target
                        continue

        # Split large sectors if they exceed max_cluster_size
        final_clusters = {}
        cluster_counter = 1

        for sector, tickers_in_sector in sector_clusters.items():
            if len(tickers_in_sector) <= self.max_cluster_size:
                # Sector fits in one cluster
                cluster_id = f"{sector}"
                final_clusters[cluster_id] = tickers_in_sector
            else:
                # Split large sector into multiple clusters
                n_splits = int(np.ceil(len(tickers_in_sector) / self.max_cluster_size))

                for i in range(n_splits):
                    start_idx = i * self.max_cluster_size
                    end_idx = min((i + 1) * self.max_cluster_size, len(tickers_in_sector))

                    cluster_id = f"{sector}_{i+1}"
                    final_clusters[cluster_id] = tickers_in_sector[start_idx:end_idx]

        # Final validation
        if self.max_clusters is not None and len(final_clusters) > self.max_clusters:
            stats = get_sector_cluster_stats(sector_clusters)
            raise ValueError(
                f"Cannot satisfy topology constraints for sector clustering:\n"
                f"  Portfolio has {stats['num_sectors']} sectors (after merging)\n"
                f"  Total assets: {stats['total_assets']}\n"
                f"  Sector sizes: {stats['min_size']}-{stats['max_size']} (avg {stats['avg_size']:.1f})\n"
                f"  After splitting: {len(final_clusters)} clusters\n"
                f"  Required: max {self.max_clusters} clusters, {self.max_cluster_size} assets/cluster"
            )

        return final_clusters

    def _fetch_sector_data(self, tickers: List[str]) -> Dict[str, str]:
        """
        Fetch sector/industry data from Yahoo Finance.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker -> sector/industry
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance not installed. "
                "Install with: pip install yfinance"
            )

        sector_map = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                if self.use_industry:
                    # More granular: industry classification
                    sector_map[ticker] = info.get('industry', 'Unknown')
                else:
                    # Broader: sector classification
                    sector_map[ticker] = info.get('sector', 'Unknown')

            except Exception as e:
                print(f"Warning: Could not fetch sector for {ticker}: {e}")
                sector_map[ticker] = 'Unknown'

        return sector_map

    def _load_from_csv(self, csv_path: Union[str, Path], use_industry: bool = False) -> Dict[str, str]:
        """
        Load sector/industry data from CSV file (e.g., from 'qpo fetch-info').

        Args:
            csv_path: Path to CSV file with stock info
            use_industry: If True, use 'industry' column; else use 'sector' column

        Returns:
            Dictionary mapping ticker -> sector/industry
        """
        try:
            df = pd.read_csv(csv_path)

            # Determine which column to use
            key_column = 'industry' if use_industry else 'sector'

            if 'symbol' not in df.columns:
                raise ValueError("CSV must have 'symbol' column")

            if key_column not in df.columns:
                raise ValueError(f"CSV must have '{key_column}' column")

            # Create mapping, handling missing values
            sector_map = {}
            for _, row in df.iterrows():
                symbol = str(row['symbol']).strip().upper()
                value = row[key_column]

                # Handle NaN or empty values
                if pd.isna(value) or value == '':
                    sector_map[symbol] = 'Unknown'
                else:
                    sector_map[symbol] = str(value).strip()

            print(f"Loaded {len(sector_map)} tickers from {csv_path}")
            return sector_map

        except Exception as e:
            raise ValueError(f"Failed to load sector data from CSV: {e}")

    def get_sector_summary(self, clusters: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Get summary of sector distribution.

        Args:
            clusters: Cluster dictionary

        Returns:
            DataFrame with cluster stats
        """
        summary = []

        for cluster_id, tickers in clusters.items():
            summary.append({
                'cluster': cluster_id,
                'n_assets': len(tickers),
                'tickers': ', '.join(tickers[:5]) + ('...' if len(tickers) > 5 else '')
            })

        return pd.DataFrame(summary)
