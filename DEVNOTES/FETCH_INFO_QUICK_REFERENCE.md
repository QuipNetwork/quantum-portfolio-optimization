# Fetch-Info Quick Reference

## Command Syntax

```bash
qpo fetch-info [OPTIONS] [TICKERS]...
```

## Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--input` | `-i` | PATH | - | Input file with tickers (one per line) |
| `--output` | `-o` | PATH | `stock_info.csv` | Output CSV file |
| `--append` | `-a` | flag | false | Append to existing file |
| `--delay` | `-d` | float | 0.5 | Delay between API calls (seconds) |

## Common Usage Patterns

### Basic Usage
```bash
# Single ticker
qpo fetch-info AAPL

# Multiple tickers
qpo fetch-info AAPL MSFT GOOGL TSLA

# Custom output file
qpo fetch-info AAPL MSFT --output tech_stocks.csv
```

### Input from File
```bash
# From file
qpo fetch-info --input tickers.txt

# With custom output
qpo fetch-info --input tickers.txt --output stocks.csv
```

### Caching with Append Mode
```bash
# Initial fetch
qpo fetch-info AAPL MSFT --output cache.csv

# Add more tickers (skips AAPL, MSFT)
qpo fetch-info AAPL GOOGL TSLA --output cache.csv --append

# Result: cache.csv now has AAPL, MSFT, GOOGL, TSLA
```

### Portfolio Workflow
```bash
# Extract tickers from portfolio.csv
head -1 portfolio.csv | tr ',' '\n' | tail -n +2 > tickers.txt

# Fetch all stock info
qpo fetch-info --input tickers.txt --output portfolio_info.csv

# Use with SectorClusterer (Python)
python -c "
from clustering import SectorClusterer
import pandas as pd

returns = pd.read_csv('portfolio.csv', index_col=0, parse_dates=True).pct_change().dropna()
clusterer = SectorClusterer(sector_map='portfolio_info.csv')
clusters = clusterer.cluster(returns)
print(f'Created {len(clusters)} clusters')
"
```

### Rate Limiting
```bash
# Default (0.5s delay = 2 req/sec)
qpo fetch-info --input tickers.txt

# Slower for large lists (1s delay = 1 req/sec)
qpo fetch-info --input large_list.txt --delay 1.0

# Faster for small lists (0.2s delay = 5 req/sec) - use with caution
qpo fetch-info AAPL MSFT GOOGL --delay 0.2
```

## CSV Output Columns

| Column | Description | Example |
|--------|-------------|---------|
| `symbol` | Ticker | AAPL |
| `shortName` | Short name | Apple Inc. |
| `longName` | Full name | Apple Inc. |
| `sector` | GICS sector | Technology |
| `industry` | GICS industry | Consumer Electronics |
| `marketCap` | Market cap | 3744081903616 |
| `currentPrice` | Current price | 252.29 |
| `previousClose` | Prev close | 247.45 |
| `trailingPE` | P/E ratio | 38.34 |
| `forwardPE` | Forward P/E | 30.36 |
| `dividendYield` | Div yield | 0.0041 |
| `beta` | Beta | 1.094 |
| `fiftyTwoWeekHigh` | 52w high | 260.10 |
| `fiftyTwoWeekLow` | 52w low | 169.21 |
| `volume` | Volume | 49146961 |
| `averageVolume` | Avg volume | 54379123 |
| `exchange` | Exchange | NMS |
| `currency` | Currency | USD |
| `country` | Country | United States |
| `website` | Website | https://www.apple.com |

## Integration with SectorClusterer

### Sector Clustering (11 GICS Sectors)
```python
from clustering import SectorClusterer

clusterer = SectorClusterer(sector_map='stocks.csv')
clusters = clusterer.cluster(returns)
```

### Industry Clustering (~50 Industries)
```python
clusterer = SectorClusterer(
    sector_map='stocks.csv',
    use_industry=True
)
clusters = clusterer.cluster(returns)
```

## Typical Workflows

### 1. One-Time Setup
```bash
qpo fetch-info --input all_tickers.txt --output stocks.csv
```

### 2. Incremental Updates
```bash
# Add new tickers
qpo fetch-info NEW1 NEW2 --output stocks.csv --append

# Or from file
qpo fetch-info --input new_tickers.txt --output stocks.csv --append
```

### 3. Periodic Refresh
```bash
# Quarterly: re-fetch all to update fundamentals
qpo fetch-info --input all_tickers.txt --output stocks.csv
```

### 4. Resume After Interrupt
```bash
# Interrupted during fetch (Ctrl+C)
# Resume with append mode
qpo fetch-info --input tickers.txt --output stocks.csv --append
```

## Error Handling

### No Tickers Provided
```bash
$ qpo fetch-info
Error: No tickers provided. Use arguments or --input file
```

### Invalid Ticker
```bash
$ qpo fetch-info AAPL INVALID123
# Output:
Successfully fetched: 1/2
Failed tickers (1):
  - INVALID123
```

### Already Fetched
```bash
$ qpo fetch-info AAPL --output cache.csv --append
# If AAPL already in cache.csv:
Loading existing data from cache.csv...
Found 1 existing tickers
All tickers already fetched. Nothing to do.
```

## Performance Tips

1. **Use append mode** for incremental updates
2. **Increase delay** for large lists (`--delay 1.0`)
3. **Interrupt-safe**: Use Ctrl+C, then resume with `--append`
4. **Cache strategy**: One CSV per portfolio, update quarterly
5. **Batch processing**: Group tickers in files for organization

## Example: Complete Portfolio Setup

```bash
#!/bin/bash
# Complete portfolio setup script

# 1. Extract tickers
head -1 portfolio.csv | tr ',' '\n' | tail -n +2 > tickers.txt

# 2. Fetch stock info
qpo fetch-info --input tickers.txt \
               --output portfolio_info.csv \
               --delay 0.5

# 3. Verify results
echo "Fetched $(tail -n +2 portfolio_info.csv | wc -l) stocks"
echo "Sectors: $(tail -n +2 portfolio_info.csv | cut -d',' -f4 | sort -u | wc -l)"

# 4. Use in Python
python << 'EOF'
from clustering import SectorClusterer
import pandas as pd

portfolio = pd.read_csv('portfolio.csv', index_col=0, parse_dates=True)
returns = portfolio.pct_change().dropna()

clusterer = SectorClusterer(sector_map='portfolio_info.csv')
clusters = clusterer.cluster(returns)

for cluster_id, tickers in sorted(clusters.items()):
    print(f"{cluster_id}: {len(tickers)} assets")
EOF
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Rate limit errors | Increase `--delay` to 1.0 or higher |
| Empty sector/industry | Normal for ETFs/new IPOs (will show "Unknown") |
| Network timeout | Check internet connection, retry with `--append` |
| Partial results | Use `--append` to fetch remaining tickers |
| Permission denied | Check output directory write permissions |

## See Also

- [Full Documentation](FETCH_INFO_USAGE.md)
- [SectorClusterer Guide](../clustering/README.md)
- [Examples](../examples/fetch_info_example.sh)
