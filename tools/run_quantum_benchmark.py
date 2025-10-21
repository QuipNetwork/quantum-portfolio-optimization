#!/usr/bin/env python3
"""
Simplified CLI for running quantum portfolio optimizer benchmarks.

Quick start examples:

1. Run complete benchmark (DEFAULT: all 8 clustering methods + all 5 classical optimizers):
   python tools/run_quantum_benchmark.py --portfolio-csv portfolio.csv

2. Test specific clustering algorithms only:
   python tools/run_quantum_benchmark.py --portfolio-csv portfolio.csv --clustering-methods correlation graph factor

3. Quantum only (no classical comparison):
   python tools/run_quantum_benchmark.py --portfolio-csv portfolio.csv --no-classical

4. Classical optimizers only (no quantum):
   python tools/run_quantum_benchmark.py --portfolio-csv portfolio.csv --optimizers mean-variance l1 l2 risk-parity equal-weight

5. Compare QPU vs simulated annealing (all clustering methods on both):
   python tools/run_quantum_benchmark.py --portfolio-csv portfolio.csv --solver-types qpu simulated

6. Fast test (shorter windows, fewer methods):
   python tools/run_quantum_benchmark.py --portfolio-csv portfolio.csv --clustering-methods correlation --train-days 126 --test-days 10

Available clustering methods: all, correlation, graph, sector, covariance, returns, volatility, dtw, factor
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import and run the main comparison script
from tools.quantum_classical_comparison import main

if __name__ == '__main__':
    # Print header
    print(__doc__)
    print("="*80)
    print()

    # Run main function
    main()
