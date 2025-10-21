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
