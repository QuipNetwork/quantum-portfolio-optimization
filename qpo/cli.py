"""Main CLI entry point for qpo."""

import click
from qpo.commands import stocks, fetch, portfolio


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """QUBO Portfolio Optimization CLI Tool."""
    pass


# Register subcommands
cli.add_command(stocks.stocks)
cli.add_command(fetch.fetch)
cli.add_command(portfolio.portfolio)


if __name__ == '__main__':
    cli()
