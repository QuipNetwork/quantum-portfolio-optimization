"""Quick test of quantum optimizer wrapper."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qpo.optimizers.quantum import IndependentClustersOptimizer, QuantumOptimizerWrapper
from clustering import CorrelationClusterer

print("="*80)
print("QUANTUM OPTIMIZER WRAPPER TEST")
print("="*80)

# Generate simple synthetic data
print("\n1. Generating synthetic data...")
np.random.seed(42)
n_assets = 20
n_days = 100

dates = pd.date_range('2024-01-01', periods=n_days)
tickers = [f'ASSET_{i:02d}' for i in range(n_assets)]

returns_data = {}
for ticker in tickers:
    returns_data[ticker] = np.random.normal(0.001, 0.02, n_days)

returns = pd.DataFrame(returns_data, index=dates)
print(f"   ✓ Created {n_assets} assets, {n_days} days")

# Create optimizer
print("\n2. Creating quantum optimizer...")
clusterer = CorrelationClusterer(max_cluster_size=10, target_cluster_size=5)
quantum_opt = IndependentClustersOptimizer(
    max_cluster_size=10,
    n_bits=10,
    solver_type='simulated',
    num_reads=100,  # Fewer reads for faster testing
    clusterer=clusterer
)
print("   ✓ Optimizer created")

# Test without wrapper (dataclass return)
print("\n3. Testing direct call (dataclass format)...")
result1 = quantum_opt.optimize(returns, return_dict=False)
print(f"   ✓ Type: {type(result1).__name__}")
print(f"   ✓ Success: {result1.success}")
print(f"   ✓ N Clusters: {result1.n_clusters}")
print(f"   ✓ Runtime: {result1.runtime:.4f}s")
print(f"   ✓ Solver-only: {result1.solver_info.get('solver_only_runtime', 'N/A'):.4f}s")

# Test with dict return
print("\n4. Testing direct call (dict format)...")
result2 = quantum_opt.optimize(returns, return_dict=True)
print(f"   ✓ Type: {type(result2).__name__}")
print(f"   ✓ Keys: {list(result2.keys())}")
print(f"   ✓ Runtime: {result2['runtime']:.4f}s")
print(f"   ✓ Solver-only: {result2.get('solver_only_runtime', 'N/A'):.4f}s")

# Test wrapper
print("\n5. Testing wrapper (auto dict format)...")
wrapper = QuantumOptimizerWrapper(quantum_opt)
result3 = wrapper.optimize(returns)
print(f"   ✓ Type: {type(result3).__name__}")
print(f"   ✓ Keys: {list(result3.keys())}")
print(f"   ✓ Weights shape: {result3['weights'].shape}")
print(f"   ✓ Sharpe: {result3['metrics']['sharpe_ratio']:.3f}")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED")
print("="*80)
