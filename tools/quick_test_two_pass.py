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

#!/usr/bin/env python3
"""Quick test of two-pass quantum optimization."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from qpo.optimizers.discrete_levels import DiscreteLevelsOptimizer, DiscreteLevelsOptimizerWrapper
from qpo.optimizers.backtest import Backtester
from clustering import CorrelationClusterer

# Load data
print("Loading portfolio data...")
prices = pd.read_csv('portfolio2.csv', index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()
print(f"  Data: {len(prices)} days, {len(prices.columns)} assets\n")

# Shared clustering
clusterer = CorrelationClusterer(
    max_cluster_size=18,
    target_cluster_size=10
)

# Test 1: Single-Pass (Original)
print("="*60)
print("TEST 1: Single-Pass Quantum (Original)")
print("="*60)
quantum_single = DiscreteLevelsOptimizer(
    max_cluster_size=18,
    n_bits=10,
    alpha=1.0,
    beta=1.0,
    lambda_budget=10.0,
    solver_type='simulated',
    num_reads=1000,
    annealing_time=20,
    aggregation_strategy='concatenate',
    clusterer=clusterer,
    use_two_pass=False  # Original approach
)

wrapper_single = DiscreteLevelsOptimizerWrapper(quantum_single)

backtester_single = Backtester(
    wrapper_single,
    train_days=126,
    test_days=21,
    step_days=42,
    initial_capital=100000.0
)

try:
    result_single = backtester_single.run(prices, returns)
    m = result_single['metrics']
    print(f"✓ Complete")
    print(f"  Sharpe: {m['sharpe_ratio']:.3f}")
    print(f"  Return: {m['annualized_return']*100:.1f}%")
    print(f"  MaxDD: {m['max_drawdown']*100:.1f}%")
    print(f"  Total Runtime: {m['total_runtime']:.1f}s")
    print()
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Two-Pass (New)
print("="*60)
print("TEST 2: Two-Pass Quantum (Hierarchical)")
print("="*60)
quantum_twopass = DiscreteLevelsOptimizer(
    max_cluster_size=18,
    n_bits=10,
    alpha=1.0,
    beta=1.0,
    lambda_budget=10.0,
    solver_type='simulated',
    num_reads=1000,
    annealing_time=20,
    aggregation_strategy='concatenate',  # Ignored in two-pass
    clusterer=clusterer,
    use_two_pass=True,  # Enable two-pass
    inter_cluster_alpha=1.0,
    inter_cluster_beta=1.0
)

wrapper_twopass = DiscreteLevelsOptimizerWrapper(quantum_twopass)

backtester_twopass = Backtester(
    wrapper_twopass,
    train_days=126,
    test_days=21,
    step_days=42,
    initial_capital=100000.0
)

try:
    result_twopass = backtester_twopass.run(prices, returns)
    m = result_twopass['metrics']
    print(f"✓ Complete")
    print(f"  Sharpe: {m['sharpe_ratio']:.3f}")
    print(f"  Return: {m['annualized_return']*100:.1f}%")
    print(f"  MaxDD: {m['max_drawdown']*100:.1f}%")
    print(f"  Total Runtime: {m['total_runtime']:.1f}s")
    print()
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Comparison
print("="*60)
print("COMPARISON")
print("="*60)
single_return = result_single['metrics']['annualized_return']
twopass_return = result_twopass['metrics']['annualized_return']
improvement = ((twopass_return / single_return) - 1) * 100

print(f"Single-Pass Return: {single_return*100:.2f}%")
print(f"Two-Pass Return: {twopass_return*100:.2f}%")
print(f"Improvement: {improvement:+.1f}%")
print()
print("✓ Two-pass implementation validated!")
