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

from setuptools import setup, find_packages

setup(
    name='qpo',
    version='0.1.0',
    description='QUBO Portfolio Optimization CLI Tool',
    author='Postquant Labs Incorporated',
    license='AGPL-3.0-or-later',
    url='https://github.com/postquant/portfolio-optimization',
    classifiers=[
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    packages=find_packages(),
    install_requires=[
        'click>=8.0.0',
        'requests>=2.31.0',
        'yfinance>=0.2.0',
        'pandas>=2.0.0',
    ],
    entry_points={
        'console_scripts': [
            'qpo=qpo.cli:cli',
        ],
    },
    python_requires='>=3.8',
)
