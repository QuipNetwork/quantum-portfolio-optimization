# Clustering Comparison Tool - Refactored CLI

## Overview

The `clustering_comparison.py` CLI has been refactored to be simpler and more intuitive. The tool now automatically infers data dimensions from input files and uses a straightforward parameter structure.

## What Changed

### Removed Parameters

❌ **Removed**: `--is-prices` / `--no-is-prices`
- **Why**: The tool now always expects price data and automatically computes returns
- **Migration**: Just remove these flags - the default behavior is what you want

❌ **Removed**: `--n-assets`
- **Why**: Automatically determined from CSV columns
- **Migration**: No action needed - dimensions are inferred

❌ **Removed**: `--n-days`
- **Why**: Automatically determined from CSV rows
- **Migration**: No action needed - dimensions are inferred

❌ **Removed**: `--fetch-sectors` and `--sector-cache`
- **Why**: Replaced with simpler `--portfolio-info-csv` parameter
- **Migration**: Use `qpo fetch-info` to create the CSV first, then pass via `--portfolio-info-csv`

### New/Changed Parameters

✅ **New**: `--portfolio-info-csv`
- Path to CSV file with portfolio metadata (from `qpo fetch-info`)
- Replaces the fetch/cache mechanism with a simple file reference
- Optional - if not provided, sector clustering is skipped

✅ **Unchanged**: `--portfolio-csv`
- Path to portfolio price data (Date, TICKER1, TICKER2, ...)
- Now always expects prices (not returns)

✅ **Unchanged**: `--max-cluster-size` (default: 18)
✅ **Unchanged**: `--seed` (default: 42)
✅ **Unchanged**: `--output-dir` (default: output)

## New CLI Syntax

```bash
python tools/clustering_comparison.py [OPTIONS]

Options:
  --portfolio-csv PORTFOLIO_CSV
                        Path to CSV file with portfolio price data
  --portfolio-info-csv PORTFOLIO_INFO_CSV
                        Path to CSV file with portfolio metadata
  --max-cluster-size MAX_CLUSTER_SIZE
                        Maximum cluster size constraint (default: 18)
  --seed SEED           Random seed for synthetic data (default: 42)
  --output-dir OUTPUT_DIR
                        Output directory for reports and plots (default: output)
```

## Migration Guide

### Before (Old CLI)

```bash
# Old way - complex and verbose
python tools/clustering_comparison.py \
  --portfolio-csv portfolio.csv \
  --is-prices \
  --n-assets 115 \
  --n-days 502 \
  --fetch-sectors \
  --sector-cache output/sector_cache.json \
  --output-dir output/results
```

### After (New CLI)

```bash
# Step 1: Fetch portfolio info once (if needed)
qpo fetch-info --input tickers.txt --output stock_info.csv

# Step 2: Run comparison - simple and intuitive
python tools/clustering_comparison.py \
  --portfolio-csv portfolio.csv \
  --portfolio-info-csv stock_info.csv \
  --output-dir output/results
```

**Key improvements**:
- No need to specify `--is-prices` (always prices)
- No need to specify `--n-assets` and `--n-days` (auto-detected)
- No need for `--fetch-sectors` flag (just provide the CSV)
- Cleaner separation: fetch data once, use many times

## Usage Examples

### Example 1: Basic Usage (No Sector Data)

```bash
# Just run comparison on portfolio
python tools/clustering_comparison.py --portfolio-csv portfolio.csv
```

**Output**: Runs 6 clustering methods (skips sector clustering)

### Example 2: With Sector Data

```bash
# First time: fetch sector info
qpo fetch-info --input tickers.txt --output stock_info.csv

# Then: run comparison with sector clustering
python tools/clustering_comparison.py \
  --portfolio-csv portfolio.csv \
  --portfolio-info-csv stock_info.csv
```

**Output**: Runs 7 clustering methods (includes sector clustering)

### Example 3: Custom Constraints

```bash
python tools/clustering_comparison.py \
  --portfolio-csv portfolio.csv \
  --max-cluster-size 12 \
  --output-dir results/
```

### Example 4: Synthetic Data (Testing)

```bash
# Generate synthetic data for testing
python tools/clustering_comparison.py --seed 42
```

**Output**: Creates 50 assets, 252 days of synthetic returns

### Example 5: Complete Workflow

```bash
# Extract tickers from portfolio
head -1 portfolio.csv | tr ',' '\n' | tail -n +2 > tickers.txt

# Fetch metadata once
qpo fetch-info --input tickers.txt --output portfolio_info.csv

# Run comparison (can run multiple times with different settings)
python tools/clustering_comparison.py \
  --portfolio-csv portfolio.csv \
  --portfolio-info-csv portfolio_info.csv \
  --max-cluster-size 18 \
  --output-dir output/clusters_18

python tools/clustering_comparison.py \
  --portfolio-csv portfolio.csv \
  --portfolio-info-csv portfolio_info.csv \
  --max-cluster-size 12 \
  --output-dir output/clusters_12
```

## Benefits of Refactored CLI

1. **Simpler**: Fewer parameters to remember
2. **Intuitive**: File always contains prices, dimensions auto-detected
3. **Cleaner**: Separation between data fetching and analysis
4. **Reusable**: Fetch portfolio info once, use for multiple analyses
5. **Consistent**: Aligns with `qpo fetch-info` workflow

## Technical Details

### Automatic Dimension Detection

```python
# Old way - manual specification
returns = generate_synthetic_returns(n_assets=115, n_days=502)

# New way - automatic from CSV
data = pd.read_csv(portfolio_csv, index_col=0, parse_dates=True)
n_assets = len(data.columns)  # Auto-detected
n_days = len(data)             # Auto-detected
returns = data.pct_change().dropna()
```

### Portfolio Info CSV Format

The `--portfolio-info-csv` file should have at minimum:
- `symbol` column: Ticker symbols
- `sector` column: GICS sector classification

Example (from `qpo fetch-info`):
```csv
symbol,shortName,longName,sector,industry,...
AAPL,Apple Inc.,Apple Inc.,Technology,Consumer Electronics,...
MSFT,Microsoft,Microsoft Corporation,Technology,Software - Infrastructure,...
```

Only `symbol` and `sector` columns are required for clustering. Other columns are ignored.

### Graceful Degradation

If `--portfolio-info-csv` is:
- Not provided: Sector clustering is skipped (6 methods run)
- Provided but missing: Warning shown, continues without sector clustering
- Provided but invalid: Warning shown, continues without sector clustering

## Backward Compatibility

⚠️ **Breaking Changes**: The old flags are no longer supported

If you have scripts using the old CLI, update them:

```bash
# Old (will not work)
python tools/clustering_comparison.py --portfolio-csv data.csv --is-prices

# New (correct)
python tools/clustering_comparison.py --portfolio-csv data.csv
```

## Testing

All functionality has been tested:

✅ With portfolio CSV and portfolio info CSV (7 methods)
✅ With portfolio CSV only (6 methods)
✅ With synthetic data (6 methods)
✅ With missing portfolio info file (graceful degradation)
✅ With custom max-cluster-size
✅ With custom output directory

## See Also

- [Fetch Info Command Guide](FETCH_INFO_USAGE.md)
- [Clustering Methods README](../clustering/README.md)
- [Original Tool Documentation](../tools/clustering_comparison.py)
