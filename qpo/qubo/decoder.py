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

"""Decoder for QUBO solutions to portfolio weights."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class QUBODecoder:
    """Decode binary QUBO solutions to portfolio weights."""

    def __init__(self, n_bits: int = 10):
        """
        Initialize QUBO decoder.

        Args:
            n_bits: Binary discretization bits (must match formulator)
        """
        self.n_bits = n_bits
        self.K = 2**n_bits - 1  # Normalization constant (1023 for n_bits=10)

    def decode_cluster_solution(self,
                                solution: Dict[str, int],
                                tickers: List[str]) -> pd.Series:
        """
        Decode binary solution to portfolio weights for a single cluster.

        The encoding is: w_n = (1/K) * Σ_{q=0}^{n_bits-1} 2^q * x_{n,q}

        Args:
            solution: Binary variable assignments {var_name: 0|1}
                     e.g., {'AAPL_0': 1, 'AAPL_1': 0, ..., 'MSFT_0': 1, ...}
            tickers: Ticker names for this cluster

        Returns:
            pd.Series with weights for each ticker
        """
        weights = {}

        for ticker in tickers:
            # Sum binary values weighted by powers of 2
            binary_value = 0
            for q in range(self.n_bits):
                var_name = f"{ticker}_{q}"
                if var_name in solution:
                    # Convert to Python int to avoid numpy int8 overflow
                    binary_value += int(solution[var_name]) * (2**q)

            # Convert to continuous weight
            weight = binary_value / self.K
            weights[ticker] = weight

        return pd.Series(weights)

    def decode_all_clusters(self,
                           cluster_solutions: Dict[str, Dict[str, int]],
                           clusters: Dict[str, List[str]]) -> pd.Series:
        """
        Decode solutions from all clusters into a global portfolio.

        Args:
            cluster_solutions: {cluster_id: binary_solution}
            clusters: {cluster_id: [tickers]}

        Returns:
            pd.Series with weights for all tickers (unnormalized)
        """
        all_weights = {}

        for cluster_id, tickers in clusters.items():
            solution = cluster_solutions[cluster_id]
            cluster_weights = self.decode_cluster_solution(solution, tickers)

            # Add to global weights
            all_weights.update(cluster_weights.to_dict())

        return pd.Series(all_weights)

    def validate_solution(self,
                         solution: Dict[str, int],
                         tickers: List[str]) -> Tuple[bool, str]:
        """
        Validate that a binary solution is well-formed.

        Args:
            solution: Binary variable assignments
            tickers: Expected ticker names

        Returns:
            (is_valid, error_message)
        """
        # Check all expected variables are present
        expected_vars = {f"{ticker}_{q}" for ticker in tickers for q in range(self.n_bits)}
        solution_vars = set(solution.keys())

        missing = expected_vars - solution_vars
        if missing:
            return False, f"Missing variables: {sorted(list(missing))[:5]}..."

        extra = solution_vars - expected_vars
        if extra:
            return False, f"Unexpected variables: {sorted(list(extra))[:5]}..."

        # Check all values are binary
        invalid_values = {k: v for k, v in solution.items() if v not in [0, 1]}
        if invalid_values:
            return False, f"Non-binary values: {list(invalid_values.items())[:5]}..."

        return True, ""

    def get_weight_range(self) -> Tuple[float, float]:
        """
        Get the range of possible weights given discretization.

        Returns:
            (min_weight, max_weight)
        """
        min_weight = 0.0  # All bits = 0
        max_weight = (2**self.n_bits - 1) / self.K  # All bits = 1
        return min_weight, max_weight

    def get_discretization_step(self) -> float:
        """
        Get the smallest non-zero weight step.

        Returns:
            Minimum weight increment (1/K)
        """
        return 1.0 / self.K

    def encode_weight(self, weight: float) -> List[int]:
        """
        Convert a continuous weight to binary representation.

        Useful for testing or initializing solutions.

        Args:
            weight: Continuous weight in [0, 1]

        Returns:
            List of binary digits [x_0, x_1, ..., x_{n_bits-1}]
        """
        if not 0 <= weight <= 1:
            raise ValueError(f"Weight {weight} must be in [0, 1]")

        # Convert to integer representation
        integer_val = int(round(weight * self.K))
        integer_val = min(integer_val, self.K)  # Clamp to max value

        # Convert to binary
        binary = []
        for q in range(self.n_bits):
            binary.append((integer_val >> q) & 1)

        return binary

    def decode_weight(self, binary: List[int]) -> float:
        """
        Convert binary representation to continuous weight.

        Args:
            binary: List of binary digits [x_0, x_1, ..., x_{n_bits-1}]

        Returns:
            Continuous weight in [0, 1]
        """
        if len(binary) != self.n_bits:
            raise ValueError(f"Binary list must have length {self.n_bits}, got {len(binary)}")

        integer_val = sum(b * (2**q) for q, b in enumerate(binary))
        return integer_val / self.K

    def get_statistics(self, weights: pd.Series) -> Dict[str, float]:
        """
        Get statistics about decoded weights.

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary with statistics
        """
        return {
            'n_assets': len(weights),
            'n_nonzero': (weights > 0).sum(),
            'sum': weights.sum(),
            'min': weights.min(),
            'max': weights.max(),
            'mean': weights.mean(),
            'std': weights.std(),
            'sparsity': (weights == 0).sum() / len(weights)
        }
