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
