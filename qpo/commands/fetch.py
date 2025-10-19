"""Fetch subcommand - download historical stock data."""

import click
import os
from pathlib import Path
from qpo.utils.yahoo_api import download_stock_data


@click.command()
@click.argument('tickers', nargs=-1)
@click.option('--input', '-i', 'input_file', type=click.Path(exists=True),
              help='Input file with ticker symbols (one per line)')
@click.option('--output-dir', '-o', default='data/stocks',
              type=click.Path(), help='Output directory for CSV files')
@click.option('--period', '-p', help='Time period (e.g., 1y, 2y, 5y, max)')
@click.option('--start', help='Start date (YYYY-MM-DD)')
@click.option('--end', help='End date (YYYY-MM-DD)')
def fetch(tickers, input_file, output_dir, period, start, end):
    """Download historical stock price data from Yahoo Finance.

    Accepts tickers as arguments or from an input file. Data is saved as
    individual CSV files named {TICKER}.csv in the output directory.

    Examples:

        qpo fetch AAPL MSFT GOOGL --period 1y

        qpo fetch --input stocks.txt --period 2y --output-dir data/

        qpo fetch TSLA --start 2020-01-01 --end 2023-12-31
    """
    # Validate date range options
    if not period and not start:
        raise click.UsageError("Must provide either --period or --start date")

    if period and start:
        raise click.UsageError("Cannot use both --period and --start/--end")

    # Collect tickers from arguments and/or input file
    ticker_list = list(tickers)

    if input_file:
        with open(input_file, 'r') as f:
            file_tickers = [line.strip() for line in f if line.strip()]
            ticker_list.extend(file_tickers)

    if not ticker_list:
        raise click.UsageError("No tickers provided. Use arguments or --input file")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Downloading data for {len(ticker_list)} tickers...")

    success_count = 0
    failed_tickers = []

    with click.progressbar(ticker_list, label='Fetching stock data') as bar:
        for ticker in bar:
            # Download data
            if period:
                data = download_stock_data(ticker, period=period)
            else:
                data = download_stock_data(ticker, start=start, end=end)

            if data is not None and not data.empty:
                # Save to CSV
                output_file = output_path / f"{ticker}.csv"
                data.to_csv(output_file)
                success_count += 1
            else:
                failed_tickers.append(ticker)

    # Report results
    click.echo(f"\nSuccessfully downloaded: {success_count}/{len(ticker_list)}")

    if failed_tickers:
        click.echo(f"\nFailed tickers ({len(failed_tickers)}):")
        for ticker in failed_tickers:
            click.echo(f"  - {ticker}")

    click.echo(f"\nData saved to: {output_path.absolute()}")
