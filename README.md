# QPO - QUBO Portfolio Optimization CLI

A Python command-line tool for portfolio optimization using quantum computing (QUBO) techniques.

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the package in editable mode:
```bash
pip install -e .
```

## Usage

### 1. Get Active Stocks

List active stocks with at least 2 years of trading history:

```bash
qpo stocks --min-years 2 --limit 50
```

Save to file:
```bash
qpo stocks --output tickers.txt
```

### 2. Fetch Historical Data

Download historical price data for specific stocks:

```bash
# Using period (convenient)
qpo fetch AAPL MSFT GOOGL --period 2y --output-dir data/stocks

# Using date range (precise)
qpo fetch TSLA --start 2020-01-01 --end 2023-12-31

# From file
qpo fetch --input tickers.txt --period 1y
```

Period options: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `max`

### 3. Create Portfolio Matrix

Merge individual stock CSVs into a single portfolio matrix:

```bash
qpo portfolio data/stocks/ --output portfolio.csv
```

Advanced options:
```bash
# Use adjusted close prices
qpo portfolio data/stocks/ --column "Adj Close"

# Handle missing data with interpolation
qpo portfolio data/stocks/ --fill-method interpolate
```

## Example Workflow

```bash
# 1. Get list of stocks
qpo stocks --limit 20 --output tickers.txt

# 2. Fetch 2 years of data
qpo fetch --input tickers.txt --period 2y --output-dir data/stocks

# 3. Create portfolio matrix
qpo portfolio data/stocks/ --output portfolio.csv
```

## Development

The package structure:
```
qpo/
├── cli.py              # Main CLI entry point
├── commands/           # Subcommands
│   ├── stocks.py      # List active stocks
│   ├── fetch.py       # Download stock data
│   └── portfolio.py   # Create portfolio matrix
└── utils/
    └── yahoo_api.py   # Yahoo Finance utilities
```

## License

Copyright (C) 2025 Postquant Labs Incorporated

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU Affero General Public License** as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but **WITHOUT ANY WARRANTY**; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

### AGPL Network Use Notice

**Important:** If you run a modified version of this software as a network service (e.g., as a web API, cloud service, or any other networked application), you **must** make the complete source code of your modified version available to users of that service under the terms of the AGPLv3 license. This is a key requirement of the AGPL license that distinguishes it from the standard GPL.

### License Files

- **[COPYING](COPYING)** - Full text of the GNU Affero General Public License v3.0
- **[NOTICE](NOTICE)** - Third-party software attributions and copyright notices

### Third-Party Dependencies

This project uses several open-source libraries. See the [NOTICE](NOTICE) file for detailed attribution and license information for all dependencies.

All third-party dependencies are compatible with AGPLv3. Key dependencies include:
- Apache-2.0 licensed: requests, yfinance, cvxpy, D-Wave Ocean SDK components
- BSD-3-Clause licensed: pandas, NumPy, SciPy, scikit-learn, seaborn, Click
- MIT licensed: PyYAML, setuptools
- GPL-3.0 licensed: ECOS (compatible with AGPL-3.0)
