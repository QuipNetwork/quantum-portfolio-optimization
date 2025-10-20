"""Main CLI entry point for qpo."""

import click
from qpo.commands import stocks, fetch, portfolio, fetch_info


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """QUBO Portfolio Optimization CLI Tool."""
    pass


# Register subcommands
cli.add_command(stocks.stocks)
cli.add_command(fetch.fetch)
cli.add_command(portfolio.portfolio)
cli.add_command(fetch_info.fetch_info, name='fetch-info')


if __name__ == '__main__':
    cli()
