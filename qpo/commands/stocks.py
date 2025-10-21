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

"""Stocks subcommand - list active stocks with sufficient history."""

import click
import requests
from qpo.utils.yahoo_api import has_sufficient_history


@click.command()
@click.option('--min-years', default=2, help='Minimum years of trading history required')
@click.option('--output', '-o', type=click.Path(), help='Output file (default: stdout)')
@click.option('--limit', default=None, type=int, help='Maximum number of stocks to return (default: no limit)')
def stocks(min_years, output, limit):
    """Query Yahoo Finance for active stocks with sufficient trading history.

    This command fetches a list of popular/active stocks and filters them
    to only include those with at least MIN_YEARS of trading history.
    """
    click.echo(f"Fetching active stocks with at least {min_years} years of history...")

    # Use a predefined list of popular stocks from major indices
    # In a production system, you might scrape this from Yahoo Finance screeners
    # or use a more comprehensive data source
    candidate_tickers = get_popular_tickers()

    valid_stocks = []

    with click.progressbar(candidate_tickers,
                          label='Checking stock history') as bar:
        for ticker in bar:
            if limit and len(valid_stocks) >= limit:
                break

            if has_sufficient_history(ticker, min_years):
                valid_stocks.append(ticker)

    # Output results
    if output:
        with open(output, 'w') as f:
            f.write('\n'.join(valid_stocks))
        click.echo(f"\nWrote {len(valid_stocks)} stocks to {output}")
    else:
        for ticker in valid_stocks:
            click.echo(ticker)

    click.echo(f"\nFound {len(valid_stocks)} stocks with {min_years}+ years of history")


def get_popular_tickers():
    """Get a list of popular stock tickers.

    Returns a curated list of tickers from major indices (S&P 500, NASDAQ, etc.).
    In production, this could fetch from a live API or screener.
    """
    # S&P 500 sample tickers (popular large-cap stocks)
    sp500_sample = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
        'V', 'WMT', 'XOM', 'JPM', 'PG', 'MA', 'LLY', 'AVGO', 'HD', 'CVX',
        'MRK', 'ABBV', 'KO', 'COST', 'PEP', 'ADBE', 'ORCL', 'TMO', 'MCD', 'CSCO',
        'ACN', 'ABT', 'CRM', 'WFC', 'NFLX', 'DHR', 'DIS', 'VZ', 'AMD', 'TXN',
        'INTC', 'CMCSA', 'QCOM', 'NEE', 'PM', 'UPS', 'RTX', 'INTU', 'COP', 'HON',
        'IBM', 'BA', 'SBUX', 'AMGN', 'CAT', 'GE', 'LOW', 'ELV', 'SPGI', 'DE',
        'GS', 'BLK', 'AXP', 'BKNG', 'LMT', 'MDLZ', 'ADI', 'SYK', 'GILD', 'ADP',
        'MMC', 'TJX', 'VRTX', 'C', 'REGN', 'PLD', 'AMT', 'CVS', 'ISRG', 'ZTS',
        'MO', 'CI', 'SO', 'SCHW', 'CB', 'DUK', 'PGR', 'BDX', 'EOG', 'BSX',
        'USB', 'BMY', 'NOC', 'ITW', 'HCA', 'ETN', 'SLB', 'MMM', 'APD', 'PNC'
    ]

    # Additional tech/growth stocks
    tech_stocks = [
        'CRM', 'SNOW', 'SHOP', 'SQ', 'PYPL', 'UBER', 'LYFT', 'ABNB', 'COIN',
        'RBLX', 'ZM', 'DOCU', 'TWLO', 'DDOG', 'NET', 'CRWD', 'PANW', 'ZS'
    ]

    return sp500_sample + tech_stocks
