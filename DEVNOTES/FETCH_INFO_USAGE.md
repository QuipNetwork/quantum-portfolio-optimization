# Fetch Stock Info - User Guide

The `qpo fetch-info` command downloads fundamental stock information from Yahoo Finance and saves it to a CSV file. The CSV file serves as a cache and can be used directly with the `SectorClusterer` for portfolio clustering.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Command Options](#command-options)
3. [Usage Examples](#usage-examples)
4. [CSV Output Format](#csv-output-format)
5. [Integration with SectorClusterer](#integration-with-sectorclustering)
6. [Rate Limiting](#rate-limiting)
7. [Error Handling](#error-handling)

## Quick Start

```bash
# Fetch info for specific tickers
qpo fetch-info AAPL MSFT GOOGL

# Fetch from a file
qpo fetch-info --input tickers.txt --output stock_info.csv

# Append to existing file (acts as cache)
qpo fetch-info TSLA NVDA --output stock_info.csv --append
```

## Command Options

```
Usage: qpo fetch-info [OPTIONS] [TICKERS]...

Options:
  -i, --input PATH    Input file with ticker symbols (one per line)
  -o, --output PATH   Output CSV file (default: stock_info.csv)
  -a, --append        Append to existing file instead of overwriting
  -d, --delay FLOAT   Delay between API calls in seconds (default: 0.5)
  --help              Show this message and exit
```

### Ticker Input

You can provide tickers in two ways:

1. **Command-line arguments**: `qpo fetch-info AAPL MSFT GOOGL`
2. **Input file**: `qpo fetch-info --input tickers.txt`
3. **Both**: Arguments and input file tickers are combined

The input file should have one ticker per line. Lines starting with `#` and empty lines are ignored.

### Output File

The output CSV file contains fundamental stock data. By default, it's saved as `stock_info.csv`.

**Important**: The CSV file serves as the cache. Use `--append` to add new tickers without re-downloading existing ones.

## Usage Examples

### Example 1: Basic Usage

```bash
qpo fetch-info AAPL MSFT
```

Output:
```
Fetching info for 2 tickers...
Downloading stock info

Successfully fetched: 2/2

Data saved to: stock_info.csv

Sample data (first ticker):
  Symbol: AAPL
  Name: Apple Inc.
  Sector: Technology
  Industry: Consumer Electronics
  P/E Ratio: 38.34
```

### Example 2: Fetch from File

Create a file `tickers.txt`:
```
AAPL
MSFT
GOOGL
# This is a comment
TSLA

JPM
```

Run the command:
```bash
qpo fetch-info --input tickers.txt --output stocks.csv
```

### Example 3: Append Mode (Caching)

First fetch:
```bash
qpo fetch-info AAPL MSFT --output stocks.csv
```

Later, add more tickers:
```bash
qpo fetch-info TSLA NVDA --output stocks.csv --append
```

The file now contains all 4 tickers. If you run:
```bash
qpo fetch-info AAPL --output stocks.csv --append
```

It will skip AAPL since it's already in the file:
```
Loading existing data from stocks.csv...
Found 4 existing tickers
All tickers already fetched. Nothing to do.
```

### Example 4: Complete Portfolio Workflow

Extract tickers from `portfolio.csv`:
```bash
head -1 portfolio.csv | tr ',' '\n' | tail -n +2 > tickers.txt
```

Fetch stock info:
```bash
qpo fetch-info --input tickers.txt --output portfolio_info.csv
```

This fetches info for all 115 tickers in the portfolio.

### Example 5: Custom Rate Limiting

For large lists, increase the delay to avoid rate limits:
```bash
qpo fetch-info --input large_list.txt --delay 1.0
```

This adds a 1-second delay between each API call.

## CSV Output Format

The CSV file contains 20 columns with fundamental stock data:

| Column | Description | Example |
|--------|-------------|---------|
| `symbol` | Ticker symbol | AAPL |
| `shortName` | Short company name | Apple Inc. |
| `longName` | Full company name | Apple Inc. |
| `sector` | GICS sector | Technology |
| `industry` | GICS industry | Consumer Electronics |
| `marketCap` | Market capitalization | 3744081903616 |
| `currentPrice` | Current stock price | 252.29 |
| `previousClose` | Previous closing price | 247.45 |
| `trailingPE` | Trailing P/E ratio | 38.34 |
| `forwardPE` | Forward P/E ratio | 30.36 |
| `dividendYield` | Dividend yield | 0.0041 |
| `beta` | Stock beta | 1.094 |
| `fiftyTwoWeekHigh` | 52-week high | 260.10 |
| `fiftyTwoWeekLow` | 52-week low | 169.21 |
| `volume` | Current volume | 49146961 |
| `averageVolume` | Average volume | 54379123 |
| `exchange` | Exchange code | NMS (NASDAQ) |
| `currency` | Currency | USD |
| `country` | Country | United States |
| `website` | Company website | https://www.apple.com |

### Sample CSV Output

```csv
symbol,shortName,longName,sector,industry,marketCap,currentPrice,...
AAPL,Apple Inc.,Apple Inc.,Technology,Consumer Electronics,3744081903616,252.29,...
MSFT,Microsoft Corporation,Microsoft Corporation,Technology,Software - Infrastructure,3817525739520,513.58,...
GOOGL,Alphabet Inc.,Alphabet Inc.,Communication Services,Internet Content & Information,3066071089152,253.3,...
```

## Integration with SectorClusterer

The CSV file can be used directly with `SectorClusterer` for sector-based portfolio clustering:

### Python Usage

```python
import pandas as pd
from clustering import SectorClusterer

# Load portfolio returns
portfolio = pd.read_csv('portfolio.csv', index_col=0, parse_dates=True)
returns = portfolio.pct_change().dropna()

# Method 1: Cluster by sector (11 GICS sectors)
clusterer = SectorClusterer(
    sector_map='portfolio_info.csv',  # Path to CSV from fetch-info
    max_cluster_size=18
)
clusters = clusterer.cluster(returns)

print(f"Created {len(clusters)} sector-based clusters")
# Output: Created 13 sector-based clusters
#   Technology_1: 18 assets
#   Technology_2: 11 assets
#   Healthcare_1: 18 assets
#   Financial Services: 17 assets
#   ...

# Method 2: Cluster by industry (more granular - ~50 industries)
clusterer = SectorClusterer(
    sector_map='portfolio_info.csv',
    max_cluster_size=18,
    use_industry=True  # Use industry instead of sector
)
clusters = clusterer.cluster(returns)

print(f"Created {len(clusters)} industry-based clusters")
# Output: Created 49 industry-based clusters
#   Semiconductors: 8 assets
#   Software - Infrastructure: 12 assets
#   Banks - Diversified: 3 assets
#   ...
```

### Advantages Over On-Demand Fetching

1. **Speed**: No API calls during clustering (instant)
2. **Reliability**: Works offline, no network issues
3. **Consistency**: Same sector data across runs
4. **Reusability**: One CSV for multiple analyses

### Complete Example

```bash
# Step 1: Extract tickers from portfolio
head -1 portfolio.csv | tr ',' '\n' | tail -n +2 > tickers.txt

# Step 2: Fetch stock info once
qpo fetch-info --input tickers.txt --output stocks.csv

# Step 3: Use in clustering (Python)
python << EOF
import pandas as pd
from clustering import SectorClusterer

portfolio = pd.read_csv('portfolio.csv', index_col=0, parse_dates=True)
returns = portfolio.pct_change().dropna()

# Sector-based clustering
clusterer = SectorClusterer(sector_map='stocks.csv')
clusters = clusterer.cluster(returns)

# Show results
for cluster_id, tickers in clusters.items():
    print(f"{cluster_id}: {len(tickers)} assets")
EOF
```

## Rate Limiting

Yahoo Finance has rate limits to prevent abuse. The command respects these limits:

- **Default delay**: 0.5 seconds between requests
- **Recommended for large lists**: 1.0 second (`--delay 1.0`)
- **Progress bar**: Shows real-time progress during fetching

### Interrupt and Resume

You can interrupt the fetch with `Ctrl+C`. Partial results are automatically saved:

```
Downloading stock info  [################----]  75%

Interrupted by user. Saving partial results...

Successfully fetched: 85/115

Data saved to: stock_info.csv
```

Then resume with append mode:
```bash
qpo fetch-info --input tickers.txt --output stock_info.csv --append
```

This will fetch only the remaining 30 tickers.

## Error Handling

### Invalid Tickers

Invalid tickers are skipped and reported:

```
Successfully fetched: 3/4

Failed tickers (1):
  - INVALIDTICKER

Data saved to: stock_info.csv
```

The CSV will contain only the successful tickers.

### Network Errors

If a network error occurs, the command will:
1. Save any successfully fetched data
2. Report which tickers failed
3. Exit with an error message

Use `--append` to retry failed tickers later.

### Missing Data

Some tickers may have incomplete data (e.g., no P/E ratio for non-profitable companies). Missing fields are stored as empty strings in the CSV.

## Best Practices

1. **Cache Strategy**: Use one CSV file per portfolio and update with `--append`
2. **Rate Limiting**: Use `--delay 1.0` for lists >50 tickers
3. **Validation**: Check `Failed tickers` in output for data quality
4. **Updates**: Re-fetch periodically (quarterly) as sectors/industries can change
5. **Backup**: Keep the CSV in version control for reproducibility

## Troubleshooting

### "No tickers provided"

You must provide tickers via arguments or `--input` file.

### "All tickers already fetched"

When using `--append`, all tickers are already in the file. This is normal and means your cache is up-to-date.

### Rate limit errors

Increase `--delay` to 1.0 or higher:
```bash
qpo fetch-info --input large_list.txt --delay 1.5
```

### Empty sector/industry

Some tickers (e.g., ETFs, new IPOs) may not have sector data. These will show `Unknown` in the clustering output.

## See Also

- [SectorClusterer Documentation](../clustering/README.md)
- [Clustering Comparison Tool](../tools/clustering_comparison.py)
- [Portfolio Optimization Guide](SPEC.md)
