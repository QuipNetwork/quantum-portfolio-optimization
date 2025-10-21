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

"""Tests for QUBO decoder."""

import pytest
import numpy as np
import pandas as pd
from qpo.qubo.decoder import QUBODecoder


class TestQUBODecoder:
    """Test suite for QUBODecoder."""

    @pytest.fixture
    def decoder(self):
        """Create decoder with default 10-bit encoding."""
        return QUBODecoder(n_bits=10)

    @pytest.fixture
    def decoder_4bit(self):
        """Create decoder with 4-bit encoding (for easier testing)."""
        return QUBODecoder(n_bits=4)

    def test_initialization(self):
        """Test decoder initialization."""
        decoder = QUBODecoder(n_bits=10)
        assert decoder.n_bits == 10
        assert decoder.K == 1023  # 2^10 - 1

        decoder_8 = QUBODecoder(n_bits=8)
        assert decoder_8.n_bits == 8
        assert decoder_8.K == 255  # 2^8 - 1

    def test_decode_cluster_solution_all_zeros(self, decoder):
        """Test decoding solution with all zeros."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        solution = {}

        # All bits set to 0
        for ticker in tickers:
            for q in range(10):
                solution[f"{ticker}_{q}"] = 0

        weights = decoder.decode_cluster_solution(solution, tickers)

        assert len(weights) == 3
        assert all(weights == 0.0)
        assert list(weights.index) == tickers

    def test_decode_cluster_solution_all_ones(self, decoder):
        """Test decoding solution with all ones (maximum weight)."""
        tickers = ['AAPL', 'MSFT']
        solution = {}

        # All bits set to 1
        for ticker in tickers:
            for q in range(10):
                solution[f"{ticker}_{q}"] = 1

        weights = decoder.decode_cluster_solution(solution, tickers)

        # All bits = 1 means weight = 1023/1023 = 1.0
        assert len(weights) == 2
        assert all(np.isclose(weights, 1.0))

    def test_decode_cluster_solution_mixed(self, decoder_4bit):
        """Test decoding with mixed binary values (4-bit for simplicity)."""
        # With 4 bits: K = 15
        # Binary [1,0,0,0] = 2^0 = 1 → weight = 1/15
        # Binary [0,1,0,0] = 2^1 = 2 → weight = 2/15
        # Binary [1,1,0,0] = 2^0 + 2^1 = 3 → weight = 3/15
        # Binary [1,1,1,1] = 15 → weight = 15/15 = 1.0

        tickers = ['AAPL', 'MSFT', 'GOOGL']
        solution = {
            'AAPL_0': 1, 'AAPL_1': 0, 'AAPL_2': 0, 'AAPL_3': 0,  # weight = 1/15
            'MSFT_0': 0, 'MSFT_1': 1, 'MSFT_2': 0, 'MSFT_3': 0,  # weight = 2/15
            'GOOGL_0': 1, 'GOOGL_1': 1, 'GOOGL_2': 0, 'GOOGL_3': 0  # weight = 3/15
        }

        weights = decoder_4bit.decode_cluster_solution(solution, tickers)

        assert len(weights) == 3
        assert np.isclose(weights['AAPL'], 1/15)
        assert np.isclose(weights['MSFT'], 2/15)
        assert np.isclose(weights['GOOGL'], 3/15)

    def test_decode_all_clusters(self, decoder_4bit):
        """Test decoding multiple clusters."""
        clusters = {
            'cluster_0': ['AAPL', 'MSFT'],
            'cluster_1': ['GOOGL', 'AMZN']
        }

        cluster_solutions = {
            'cluster_0': {
                'AAPL_0': 1, 'AAPL_1': 0, 'AAPL_2': 0, 'AAPL_3': 0,
                'MSFT_0': 0, 'MSFT_1': 1, 'MSFT_2': 0, 'MSFT_3': 0
            },
            'cluster_1': {
                'GOOGL_0': 1, 'GOOGL_1': 1, 'GOOGL_2': 0, 'GOOGL_3': 0,
                'AMZN_0': 1, 'AMZN_1': 1, 'AMZN_2': 1, 'AMZN_3': 0
            }
        }

        all_weights = decoder_4bit.decode_all_clusters(cluster_solutions, clusters)

        assert len(all_weights) == 4
        assert np.isclose(all_weights['AAPL'], 1/15)
        assert np.isclose(all_weights['MSFT'], 2/15)
        assert np.isclose(all_weights['GOOGL'], 3/15)
        assert np.isclose(all_weights['AMZN'], 7/15)

    def test_validate_solution_success(self, decoder_4bit):
        """Test solution validation with valid input."""
        tickers = ['AAPL', 'MSFT']
        solution = {
            'AAPL_0': 1, 'AAPL_1': 0, 'AAPL_2': 1, 'AAPL_3': 0,
            'MSFT_0': 0, 'MSFT_1': 1, 'MSFT_2': 0, 'MSFT_3': 1
        }

        is_valid, error = decoder_4bit.validate_solution(solution, tickers)
        assert is_valid
        assert error == ""

    def test_validate_solution_missing_variables(self, decoder_4bit):
        """Test solution validation with missing variables."""
        tickers = ['AAPL', 'MSFT']
        solution = {
            'AAPL_0': 1, 'AAPL_1': 0, 'AAPL_2': 1  # Missing AAPL_3 and all MSFT
        }

        is_valid, error = decoder_4bit.validate_solution(solution, tickers)
        assert not is_valid
        assert "Missing variables" in error

    def test_validate_solution_extra_variables(self, decoder_4bit):
        """Test solution validation with extra variables."""
        tickers = ['AAPL']
        solution = {
            'AAPL_0': 1, 'AAPL_1': 0, 'AAPL_2': 1, 'AAPL_3': 0,
            'MSFT_0': 1, 'MSFT_1': 1, 'MSFT_2': 1, 'MSFT_3': 1
        }

        is_valid, error = decoder_4bit.validate_solution(solution, tickers)
        assert not is_valid
        assert "Unexpected variables" in error

    def test_validate_solution_non_binary(self, decoder_4bit):
        """Test solution validation with non-binary values."""
        tickers = ['AAPL']
        solution = {
            'AAPL_0': 1, 'AAPL_1': 0, 'AAPL_2': 2, 'AAPL_3': 0  # 2 is invalid
        }

        is_valid, error = decoder_4bit.validate_solution(solution, tickers)
        assert not is_valid
        assert "Non-binary values" in error

    def test_get_weight_range(self, decoder):
        """Test weight range calculation."""
        min_w, max_w = decoder.get_weight_range()
        assert min_w == 0.0
        assert np.isclose(max_w, 1.0)

    def test_get_discretization_step(self, decoder):
        """Test discretization step."""
        step = decoder.get_discretization_step()
        assert np.isclose(step, 1/1023)

    def test_encode_weight(self, decoder_4bit):
        """Test encoding continuous weight to binary."""
        # K = 15
        # weight = 0.0 → integer = 0 → binary = [0,0,0,0]
        # weight = 1.0 → integer = 15 → binary = [1,1,1,1]
        # weight = 0.5 → integer = 7.5 ≈ 8 → binary = [0,0,0,1]

        binary_0 = decoder_4bit.encode_weight(0.0)
        assert binary_0 == [0, 0, 0, 0]

        binary_1 = decoder_4bit.encode_weight(1.0)
        assert binary_1 == [1, 1, 1, 1]

        binary_half = decoder_4bit.encode_weight(0.5)
        # 0.5 * 15 = 7.5 ≈ 8 = 0b1000 = [0,0,0,1]
        assert binary_half == [0, 0, 0, 1]

    def test_encode_weight_invalid(self, decoder):
        """Test encoding with invalid weights."""
        with pytest.raises(ValueError, match="must be in"):
            decoder.encode_weight(-0.1)

        with pytest.raises(ValueError, match="must be in"):
            decoder.encode_weight(1.5)

    def test_decode_weight(self, decoder_4bit):
        """Test decoding binary to continuous weight."""
        # [0,0,0,0] → 0 → 0/15 = 0.0
        # [1,1,1,1] → 15 → 15/15 = 1.0
        # [0,0,0,1] → 8 → 8/15

        weight_0 = decoder_4bit.decode_weight([0, 0, 0, 0])
        assert weight_0 == 0.0

        weight_1 = decoder_4bit.decode_weight([1, 1, 1, 1])
        assert weight_1 == 1.0

        weight_8 = decoder_4bit.decode_weight([0, 0, 0, 1])
        assert np.isclose(weight_8, 8/15)

    def test_decode_weight_invalid_length(self, decoder_4bit):
        """Test decoding with wrong binary length."""
        with pytest.raises(ValueError, match="must have length"):
            decoder_4bit.decode_weight([0, 0, 0])  # Too short

        with pytest.raises(ValueError, match="must have length"):
            decoder_4bit.decode_weight([0, 0, 0, 0, 0])  # Too long

    def test_encode_decode_roundtrip(self, decoder):
        """Test encode/decode roundtrip preserves values (within discretization)."""
        test_weights = [0.0, 0.25, 0.5, 0.75, 1.0]

        for w in test_weights:
            binary = decoder.encode_weight(w)
            decoded = decoder.decode_weight(binary)
            # Should be close within discretization step
            assert abs(decoded - w) <= 2 * decoder.get_discretization_step()

    def test_get_statistics(self, decoder):
        """Test weight statistics calculation."""
        weights = pd.Series({
            'AAPL': 0.3,
            'MSFT': 0.2,
            'GOOGL': 0.5,
            'AMZN': 0.0,
            'TSLA': 0.0
        })

        stats = decoder.get_statistics(weights)

        assert stats['n_assets'] == 5
        assert stats['n_nonzero'] == 3
        assert np.isclose(stats['sum'], 1.0)
        assert stats['min'] == 0.0
        assert stats['max'] == 0.5
        assert np.isclose(stats['mean'], 0.2)
        assert np.isclose(stats['sparsity'], 2/5)

    def test_empty_solution(self, decoder):
        """Test handling of empty solution."""
        tickers = []
        solution = {}

        weights = decoder.decode_cluster_solution(solution, tickers)
        assert len(weights) == 0

    def test_large_cluster(self, decoder):
        """Test decoding with maximum cluster size (18 assets)."""
        tickers = [f"TICKER_{i}" for i in range(18)]
        solution = {}

        # Set alternating patterns
        for i, ticker in enumerate(tickers):
            for q in range(10):
                solution[f"{ticker}_{q}"] = i % 2

        weights = decoder.decode_cluster_solution(solution, tickers)

        assert len(weights) == 18
        assert all(weights >= 0)
        assert all(weights <= 1)
