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
"""Unit test for two-pass hierarchical optimization."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from qpo.optimizers.discrete_levels import DiscreteLevelsOptimizer
from clustering import CorrelationClusterer

print("="*60)
print("TWO-PASS HIERARCHICAL OPTIMIZATION UNIT TEST")
print("="*60)
print()

# Load data
print("Loading portfolio data...")
prices = pd.read_csv('portfolio.csv', index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()

# Use last 126 days for training
train_returns = returns.iloc[-126:]
print(f"  Training data: {len(train_returns)} days, {len(train_returns.columns)} assets")
print()

# Shared clustering
clusterer = CorrelationClusterer(
    max_cluster_size=18,
    target_cluster_size=10
)

# Test 1: Single-Pass Optimization
print("-"*60)
print("Test 1: Single-Pass Optimization (Original)")
print("-"*60)

optimizer_single = DiscreteLevelsOptimizer(
    max_cluster_size=18,
    n_bits=10,
    alpha=1.0,
    beta=1.0,
    lambda_budget=10.0,
    solver_type='simulated',
    num_reads=100,  # Reduced for speed
    annealing_time=20,
    aggregation_strategy='concatenate',
    clusterer=clusterer,
    use_two_pass=False
)

try:
    print("Running single-pass optimization...")
    result_single = optimizer_single.optimize(train_returns)

    weights_single = result_single.weights

    print(f"✓ Success")
    print(f"  Total weight: {weights_single.sum():.4f}")
    print(f"  Non-zero assets: {(weights_single > 1e-6).sum()}/{len(weights_single)}")
    print(f"  Expected return: {result_single.expected_return*100:.2f}%")
    print(f"  Expected volatility: {result_single.volatility*100:.2f}%")
    print(f"  Sharpe: {result_single.sharpe_ratio:.3f}")
    print(f"  Mode: {result_single.solver_info['optimization_mode']}")
    print()
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Two-Pass Optimization
print("-"*60)
print("Test 2: Two-Pass Optimization (Hierarchical)")
print("-"*60)

optimizer_twopass = DiscreteLevelsOptimizer(
    max_cluster_size=18,
    n_bits=10,
    alpha=1.0,
    beta=1.0,
    lambda_budget=10.0,
    solver_type='simulated',
    num_reads=100,  # Reduced for speed
    annealing_time=20,
    aggregation_strategy='concatenate',  # Ignored
    clusterer=clusterer,
    use_two_pass=True,  # Enable hierarchical
    inter_cluster_alpha=1.0,
    inter_cluster_beta=1.0
)

try:
    print("Running two-pass hierarchical optimization...")
    result_twopass = optimizer_twopass.optimize(train_returns)

    weights_twopass = result_twopass.weights

    print(f"✓ Success")
    print(f"  Total weight: {weights_twopass.sum():.4f}")
    print(f"  Non-zero assets: {(weights_twopass > 1e-6).sum()}/{len(weights_twopass)}")
    print(f"  Expected return: {result_twopass.expected_return*100:.2f}%")
    print(f"  Expected volatility: {result_twopass.volatility*100:.2f}%")
    print(f"  Sharpe: {result_twopass.sharpe_ratio:.3f}")
    print(f"  Mode: {result_twopass.solver_info['optimization_mode']}")

    # Check for Pass 2 metadata
    if 'pass2_metadata' in result_twopass.solver_info:
        pass2 = result_twopass.solver_info['pass2_metadata']
        print(f"\n  Pass 2 Cluster Allocation:")
        for cid, allocation in pass2['cluster_allocation'].items():
            print(f"    {cid}: {allocation:.3f}")
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
print(f"Single-Pass Sharpe: {result_single.sharpe_ratio:.4f}")
print(f"Two-Pass Sharpe: {result_twopass.sharpe_ratio:.4f}")

# Verify weights are different
weight_diff = np.abs(weights_single - weights_twopass).sum()
print(f"\nTotal weight difference: {weight_diff:.4f}")

if weight_diff > 1e-6:
    print("✓ Two-pass produces different allocation (as expected)")
else:
    print("⚠ Warning: Two-pass produced same allocation as single-pass")

print()
print("="*60)
print("UNIT TEST COMPLETE")
print("="*60)
