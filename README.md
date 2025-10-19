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
