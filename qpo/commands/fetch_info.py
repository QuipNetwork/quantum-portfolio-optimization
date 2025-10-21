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

"""Fetch info subcommand - download stock information from Yahoo Finance."""

import click
import csv
import time
from pathlib import Path
from qpo.utils.yahoo_api import get_stock_info


# Yahoo Finance rate limiting - be respectful
RATE_LIMIT_DELAY = 0.5  # seconds between requests


def extract_stock_info(info_dict):
    """
    Extract relevant fields from Yahoo Finance info dict.

    Args:
        info_dict: Raw info dictionary from yfinance

    Returns:
        dict: Cleaned stock information
    """
    if not info_dict:
        return None

    # Extract common fields that exist for most stocks
    return {
        'symbol': info_dict.get('symbol', ''),
        'shortName': info_dict.get('shortName', ''),
        'longName': info_dict.get('longName', ''),
        'sector': info_dict.get('sector', ''),
        'industry': info_dict.get('industry', ''),
        'marketCap': info_dict.get('marketCap', ''),
        'currentPrice': info_dict.get('currentPrice', ''),
        'previousClose': info_dict.get('previousClose', ''),
        'trailingPE': info_dict.get('trailingPE', ''),
        'forwardPE': info_dict.get('forwardPE', ''),
        'dividendYield': info_dict.get('dividendYield', ''),
        'beta': info_dict.get('beta', ''),
        'fiftyTwoWeekHigh': info_dict.get('fiftyTwoWeekHigh', ''),
        'fiftyTwoWeekLow': info_dict.get('fiftyTwoWeekLow', ''),
        'volume': info_dict.get('volume', ''),
        'averageVolume': info_dict.get('averageVolume', ''),
        'exchange': info_dict.get('exchange', ''),
        'currency': info_dict.get('currency', ''),
        'country': info_dict.get('country', ''),
        'website': info_dict.get('website', ''),
    }


@click.command()
@click.argument('tickers', nargs=-1)
@click.option('--input', '-i', 'input_file', type=click.Path(exists=True),
              help='Input file with ticker symbols (one per line)')
@click.option('--output', '-o', default='stock_info.csv',
              type=click.Path(), help='Output CSV file (default: stock_info.csv)')
@click.option('--append', '-a', is_flag=True,
              help='Append to existing file instead of overwriting')
@click.option('--delay', '-d', default=0.5, type=float,
              help='Delay between API calls in seconds (default: 0.5)')
def fetch_info(tickers, input_file, output, append, delay):
    """Download stock information from Yahoo Finance and save to CSV.

    Fetches fundamental data including sector, industry, market cap, P/E ratio,
    and other common stock metrics. The CSV file serves as the cache - use the
    same output file to avoid re-downloading data.

    Examples:

        # Fetch info for specific tickers
        qpo fetch-info AAPL MSFT GOOGL

        # Fetch from a file of tickers
        qpo fetch-info --input tickers.txt --output stocks.csv

        # Append to existing file (skip already fetched tickers)
        qpo fetch-info TSLA NVDA --output stocks.csv --append

        # Use custom delay for rate limiting
        qpo fetch-info --input large_list.txt --delay 1.0
    """
    # Collect tickers from arguments and/or input file
    ticker_list = list(tickers)

    if input_file:
        with open(input_file, 'r') as f:
            file_tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            ticker_list.extend(file_tickers)

    if not ticker_list:
        raise click.UsageError("No tickers provided. Use arguments or --input file")

    # Remove duplicates while preserving order
    seen = set()
    ticker_list = [t.upper() for t in ticker_list if not (t.upper() in seen or seen.add(t.upper()))]

    # Check if output file exists and load already-fetched tickers
    output_path = Path(output)
    existing_tickers = set()

    if append and output_path.exists():
        click.echo(f"Loading existing data from {output}...")
        try:
            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                existing_tickers = {row['symbol'] for row in reader if 'symbol' in row}
            click.echo(f"Found {len(existing_tickers)} existing tickers")

            # Filter out already-fetched tickers
            ticker_list = [t for t in ticker_list if t not in existing_tickers]

            if not ticker_list:
                click.echo("All tickers already fetched. Nothing to do.")
                return
        except Exception as e:
            click.echo(f"Warning: Could not read existing file: {e}")
            append = False  # Fall back to overwrite mode

    click.echo(f"Fetching info for {len(ticker_list)} ticker{'s' if len(ticker_list) > 1 else ''}...")

    # Fetch stock info
    results = []
    failed_tickers = []

    with click.progressbar(ticker_list, label='Downloading stock info') as bar:
        for ticker in bar:
            try:
                # Respect rate limits
                if len(results) > 0:
                    time.sleep(delay)

                # Fetch info
                info = get_stock_info(ticker)

                if info:
                    stock_data = extract_stock_info(info)
                    if stock_data:
                        results.append(stock_data)
                    else:
                        failed_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)

            except KeyboardInterrupt:
                click.echo("\n\nInterrupted by user. Saving partial results...")
                break
            except Exception as e:
                failed_tickers.append(ticker)

    if not results:
        click.echo("\nNo data was successfully fetched.")
        return

    # Write results to CSV
    fieldnames = list(results[0].keys())

    # Determine write mode
    mode = 'a' if append and output_path.exists() else 'w'
    write_header = mode == 'w' or not output_path.exists()

    try:
        with open(output_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerows(results)

        # Report results
        click.echo(f"\nSuccessfully fetched: {len(results)}/{len(ticker_list)}")

        if failed_tickers:
            click.echo(f"\nFailed tickers ({len(failed_tickers)}):")
            for ticker in failed_tickers[:10]:  # Show max 10
                click.echo(f"  - {ticker}")
            if len(failed_tickers) > 10:
                click.echo(f"  ... and {len(failed_tickers) - 10} more")

        click.echo(f"\nData saved to: {output_path.absolute()}")

        # Show sample of what was fetched
        if results:
            click.echo("\nSample data (first ticker):")
            sample = results[0]
            click.echo(f"  Symbol: {sample.get('symbol', 'N/A')}")
            click.echo(f"  Name: {sample.get('longName', 'N/A')}")
            click.echo(f"  Sector: {sample.get('sector', 'N/A')}")
            click.echo(f"  Industry: {sample.get('industry', 'N/A')}")
            click.echo(f"  P/E Ratio: {sample.get('trailingPE', 'N/A')}")

    except Exception as e:
        click.echo(f"\nError writing to file: {e}")
        raise click.ClickException(f"Failed to write output file: {e}")
