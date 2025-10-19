from setuptools import setup, find_packages

setup(
    name='qpo',
    version='0.1.0',
    description='QUBO Portfolio Optimization CLI Tool',
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
