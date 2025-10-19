"""Portfolio subcommand - merge stock data into portfolio matrix."""

import click
import pandas as pd
from pathlib import Path


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', default='portfolio.csv',
              type=click.Path(), help='Output portfolio CSV file')
@click.option('--column', '-c', default='Close',
              type=click.Choice(['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']),
              help='Which price column to use')
@click.option('--fill-method', default='ffill',
              type=click.Choice(['ffill', 'bfill', 'interpolate', 'drop']),
              help='Method for handling missing data')
def portfolio(input_dir, output, column, fill_method):
    """Merge individual stock CSV files into a single portfolio matrix.

    Reads all CSV files from INPUT_DIR and creates a portfolio.csv where:
    - Each row is a date
    - Each column is a ticker symbol
    - Values are the stock prices (default: Close price)

    Examples:

        qpo portfolio data/stocks/ --output portfolio.csv

        qpo portfolio data/ --column "Adj Close" --fill-method interpolate
    """
    input_path = Path(input_dir)

    # Find all CSV files
    csv_files = list(input_path.glob('*.csv'))

    if not csv_files:
        raise click.UsageError(f"No CSV files found in {input_dir}")

    click.echo(f"Found {len(csv_files)} stock data files")
    click.echo(f"Building portfolio matrix using '{column}' prices...")

    # Dictionary to store each stock's data
    stock_data = {}

    for csv_file in csv_files:
        ticker = csv_file.stem  # Filename without extension

        try:
            # Read CSV, parse dates
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

            if column not in df.columns:
                click.echo(f"Warning: {ticker} missing '{column}' column, skipping")
                continue

            # Extract the desired price column
            stock_data[ticker] = df[column]

        except Exception as e:
            click.echo(f"Warning: Failed to read {csv_file.name}: {e}")
            continue

    if not stock_data:
        raise click.ClickException("No valid stock data could be loaded")

    # Merge all stocks into a single DataFrame
    portfolio_df = pd.DataFrame(stock_data)

    # Handle missing data
    original_shape = portfolio_df.shape
    if fill_method == 'ffill':
        portfolio_df = portfolio_df.fillna(method='ffill')
    elif fill_method == 'bfill':
        portfolio_df = portfolio_df.fillna(method='bfill')
    elif fill_method == 'interpolate':
        portfolio_df = portfolio_df.interpolate(method='linear')
    elif fill_method == 'drop':
        portfolio_df = portfolio_df.dropna()

    # Sort by date
    portfolio_df = portfolio_df.sort_index()

    # Save to CSV
    portfolio_df.to_csv(output)

    # Report results
    click.echo(f"\nPortfolio matrix created:")
    click.echo(f"  - Tickers: {len(portfolio_df.columns)}")
    click.echo(f"  - Date range: {portfolio_df.index[0].date()} to {portfolio_df.index[-1].date()}")
    click.echo(f"  - Trading days: {len(portfolio_df)}")
    click.echo(f"  - Missing data handling: {fill_method}")

    if fill_method == 'drop':
        rows_dropped = original_shape[0] - portfolio_df.shape[0]
        if rows_dropped > 0:
            click.echo(f"  - Rows dropped: {rows_dropped}")

    click.echo(f"\nSaved to: {output}")
