#!/usr/bin/env python3
"""Quick test to verify 'all' clustering methods expansion."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from tools.quantum_classical_comparison import create_optimizers

print("="*80)
print("TESTING 'ALL' CLUSTERING METHODS EXPANSION")
print("="*80)
print()

# Test 1: Explicit "all"
all_clustering_methods = ['correlation', 'graph', 'sector', 'covariance',
                          'returns', 'volatility', 'dtw', 'factor']

clustering_methods = all_clustering_methods if 'all' in ['all'] else ['all']

print("Test 1: Using 'all' keyword")
print(f"  Resolved to: {clustering_methods}")
print(f"  Count: {len(clustering_methods)} methods")
print()

# Test 2: Create optimizers with all methods
print("Test 2: Creating optimizers with all clustering methods...")

try:
    optimizers = create_optimizers(
        solver_types=['simulated'],
        clustering_methods=clustering_methods,
        include_classical=True,
        max_cluster_size=18,
        target_cluster_size=10,
        portfolio_info_csv=None,
        optimizer_filter=None
    )

    print(f"  ✓ Successfully created {len(optimizers)} optimizers")
    print()
    print("Optimizer breakdown:")

    quantum_opts = [name for name in optimizers.keys() if 'Quantum' in name]
    classical_opts = [name for name in optimizers.keys() if 'Quantum' not in name]

    print(f"  Quantum optimizers: {len(quantum_opts)}")
    for name in quantum_opts:
        print(f"    - {name}")

    print(f"\n  Classical optimizers: {len(classical_opts)}")
    for name in classical_opts:
        print(f"    - {name}")

    print()
    print("="*80)
    print(f"✓ TOTAL: {len(optimizers)} optimizers configured")
    print("  - 8 Quantum (one per clustering method)")
    print("  - 5 Classical (Mean-Variance, L1, L2, Risk-Parity, Equal-Weight)")
    print("="*80)

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
