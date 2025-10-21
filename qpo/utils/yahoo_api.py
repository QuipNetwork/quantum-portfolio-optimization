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

"""Yahoo Finance API utilities."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_stock_info(ticker):
    """Get basic info for a stock ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        dict: Stock information or None if ticker is invalid
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception:
        return None


def has_sufficient_history(ticker, min_years=2):
    """Check if a stock has at least min_years of trading history.

    Args:
        ticker: Stock ticker symbol
        min_years: Minimum years of history required

    Returns:
        bool: True if stock has sufficient history
    """
    try:
        stock = yf.Ticker(ticker)
        # Get max available history
        hist = stock.history(period="max")

        if hist.empty:
            return False

        # Check if earliest date is at least min_years ago
        earliest_date = hist.index[0]
        cutoff_date = datetime.now() - timedelta(days=min_years * 365)

        return earliest_date.to_pydatetime().replace(tzinfo=None) <= cutoff_date
    except Exception:
        return False


def download_stock_data(ticker, period=None, start=None, end=None):
    """Download historical stock data.

    Args:
        ticker: Stock ticker symbol
        period: Period string (e.g., "1y", "2y", "5y", "max")
        start: Start date (YYYY-MM-DD string or datetime)
        end: End date (YYYY-MM-DD string or datetime)

    Returns:
        pd.DataFrame: Historical stock data or None on error
    """
    try:
        stock = yf.Ticker(ticker)

        if period:
            hist = stock.history(period=period)
        elif start:
            hist = stock.history(start=start, end=end)
        else:
            raise ValueError("Must provide either period or start date")

        return hist if not hist.empty else None
    except Exception as e:
        return None
