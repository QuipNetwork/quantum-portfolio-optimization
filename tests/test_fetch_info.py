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

"""Tests for fetch_info command."""

import pytest
import csv
from pathlib import Path
from click.testing import CliRunner
from qpo.cli import cli


class TestFetchInfo:
    """Test fetch-info CLI command."""

    def test_fetch_info_basic(self, tmp_path):
        """Test basic fetch-info functionality."""
        runner = CliRunner()
        output_file = tmp_path / "test_info.csv"

        result = runner.invoke(cli, [
            'fetch-info',
            'AAPL',
            '--output', str(output_file)
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Check CSV content
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 1
            assert rows[0]['symbol'] == 'AAPL'
            assert rows[0]['sector'] == 'Technology'
            assert rows[0]['industry'] != ''
            assert 'trailingPE' in rows[0]

    def test_fetch_info_multiple_tickers(self, tmp_path):
        """Test fetching multiple tickers."""
        runner = CliRunner()
        output_file = tmp_path / "test_info.csv"

        result = runner.invoke(cli, [
            'fetch-info',
            'AAPL', 'MSFT', 'GOOGL',
            '--output', str(output_file),
            '--delay', '0.3'
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Check we got all 3 tickers
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3

            symbols = {row['symbol'] for row in rows}
            assert symbols == {'AAPL', 'MSFT', 'GOOGL'}

    def test_fetch_info_from_file(self, tmp_path):
        """Test fetching from input file."""
        runner = CliRunner()
        input_file = tmp_path / "tickers.txt"
        output_file = tmp_path / "test_info.csv"

        # Create input file
        input_file.write_text("AAPL\nMSFT\n# Comment\n\nGOOGL\n")

        result = runner.invoke(cli, [
            'fetch-info',
            '--input', str(input_file),
            '--output', str(output_file)
        ])

        assert result.exit_code == 0

        # Should have 3 tickers (comment and blank line ignored)
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3

    def test_fetch_info_append(self, tmp_path):
        """Test append mode."""
        runner = CliRunner()
        output_file = tmp_path / "test_info.csv"

        # First fetch
        result1 = runner.invoke(cli, [
            'fetch-info',
            'AAPL',
            '--output', str(output_file)
        ])
        assert result1.exit_code == 0

        # Append new ticker
        result2 = runner.invoke(cli, [
            'fetch-info',
            'MSFT',
            '--output', str(output_file),
            '--append'
        ])
        assert result2.exit_code == 0

        # Should have both tickers
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            symbols = {row['symbol'] for row in rows}
            assert symbols == {'AAPL', 'MSFT'}

    def test_fetch_info_append_skip_existing(self, tmp_path):
        """Test that append mode skips already-fetched tickers."""
        runner = CliRunner()
        output_file = tmp_path / "test_info.csv"

        # First fetch
        result1 = runner.invoke(cli, [
            'fetch-info',
            'AAPL', 'MSFT',
            '--output', str(output_file)
        ])
        assert result1.exit_code == 0

        # Try to append existing ticker
        result2 = runner.invoke(cli, [
            'fetch-info',
            'AAPL',
            '--output', str(output_file),
            '--append'
        ])

        # Should skip AAPL
        assert "All tickers already fetched" in result2.output

        # Should still have only 2 rows
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2

    def test_fetch_info_no_tickers(self):
        """Test error when no tickers provided."""
        runner = CliRunner()

        result = runner.invoke(cli, ['fetch-info'])

        assert result.exit_code != 0
        assert "No tickers provided" in result.output

    def test_fetch_info_csv_columns(self, tmp_path):
        """Test that CSV has all expected columns."""
        runner = CliRunner()
        output_file = tmp_path / "test_info.csv"

        result = runner.invoke(cli, [
            'fetch-info',
            'AAPL',
            '--output', str(output_file)
        ])

        assert result.exit_code == 0

        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            # Check for key columns
            expected_columns = [
                'symbol', 'shortName', 'longName',
                'sector', 'industry',
                'marketCap', 'currentPrice',
                'trailingPE', 'forwardPE',
                'beta', 'exchange'
            ]

            for col in expected_columns:
                assert col in fieldnames, f"Missing column: {col}"

    def test_fetch_info_with_sector_clusterer(self, tmp_path):
        """Test integration with SectorClusterer."""
        import pandas as pd
        from clustering import SectorClusterer

        runner = CliRunner()
        output_file = tmp_path / "test_info.csv"

        # Fetch some stock info
        result = runner.invoke(cli, [
            'fetch-info',
            'AAPL', 'MSFT', 'JPM', 'XOM',
            '--output', str(output_file)
        ])

        assert result.exit_code == 0

        # Create fake returns data
        returns = pd.DataFrame({
            'AAPL': [0.01, -0.02, 0.03],
            'MSFT': [0.02, -0.01, 0.02],
            'JPM': [-0.01, 0.01, -0.02],
            'XOM': [0.03, -0.03, 0.01],
        })

        # Use SectorClusterer with CSV file
        clusterer = SectorClusterer(sector_map=str(output_file))
        clusters = clusterer.cluster(returns)

        # Should create clusters by sector
        assert len(clusters) > 0

        # Technology stocks should be together
        tech_clusters = [
            tickers for name, tickers in clusters.items()
            if 'Technology' in name
        ]
        assert len(tech_clusters) > 0

        # Check that AAPL and MSFT are in same sector
        tech_tickers = set()
        for tickers in tech_clusters:
            tech_tickers.update(tickers)

        assert 'AAPL' in tech_tickers
        assert 'MSFT' in tech_tickers

    @pytest.mark.slow
    def test_fetch_info_invalid_ticker(self, tmp_path):
        """Test handling of invalid ticker."""
        runner = CliRunner()
        output_file = tmp_path / "test_info.csv"

        # Use a definitely invalid ticker pattern
        result = runner.invoke(cli, [
            'fetch-info',
            'AAPL', 'ZZZZZ999999INVALID',
            '--output', str(output_file)
        ])

        # Should still succeed with partial results
        assert result.exit_code == 0

        # Should have at least AAPL (might or might not show failed message depending on yfinance behavior)
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) >= 1

            # Verify AAPL is there
            symbols = {row['symbol'] for row in rows}
            assert 'AAPL' in symbols
