# Testing on a Linux box with NVIDIA hardware

Instructions for running the QHDOPT, NVIDIA cuOpt, OpenPhiSolve, and
Pasqal solver integrations on a Linux machine. NVIDIA hardware is
needed for cuOpt only — the other three (QHDOPT, OpenPhiSolve, Pasqal)
run on CPU and on macOS as well.

The Pasqal stack (sections 6–7 below) was added on branch
`feat/add-pasqal-solvers`; QHDOPT / cuOpt / OpenPhiSolve sections
match what landed on `feat/add-qhd-cuopt-phisolve-solvers`.

## Prerequisites

- Linux (Ubuntu 22.04+) — or macOS for everything except cuOpt
- NVIDIA GPU with compute capability 7.0+ (Volta or newer; Turing,
  Ampere, and Hopper all fine) — cuOpt only
- NVIDIA driver supporting CUDA 12.x (`nvidia-smi` should report
  Driver Version >= 525) — cuOpt only
- **Python 3.12 is the recommended target for this setup.** Each
  optional solver has its own version constraints, and 3.12 is the
  only version inside all of them simultaneously:
    - cuOpt: 3.10–3.12 (no 3.13+ wheels published)
    - QHDOPT: 3.10–3.12 stock; 3.13 needs the `--no-deps` workaround
      in §3 because qhdopt 0.0.1 hard-pins `numpy<1.28` and
      `scipy==1.11.4` which don't build on 3.13
    - Pasqal qubo-solver: most versions cap at `<3.13`; the 0.7.x
      versions advertise `<=3.14` but pip interprets that strictly
      (3.14.0 only, not 3.14.x patches) — so 3.14 is effectively
      unsupported and 3.12 stays the safe choice
    - PhiSolve, NL, CQM, dimod: 3.10+
  If you only need Pasqal (skipping cuOpt), Python 3.13 works with
  the QHDOPT workaround. Python 3.14 does not work for any of the
  optional solver wheels currently published.

## 1. Clone and check out the branch

```bash
git clone git@gitlab.com:quip.network/quantum-portfolio-optimization.git
cd quantum-portfolio-optimization
git switch feat/add-pasqal-solvers   # most recent, includes everything
```

## 2. Set up a Python environment for the experiments

If you already have an `experiments/simple/.venv` on a different
Python version (commonly happens because `python3` on Ubuntu 24.04+
and recent macOS defaults to 3.13 or 3.14), delete and recreate it
with the version that satisfies every solver:

```bash
cd experiments/simple
rm -rf .venv                          # or use `trash .venv` on macOS
python3.12 -m venv .venv              # NOT `python3 -m venv .venv`
source .venv/bin/activate
python --version                      # confirm: Python 3.12.x
pip install --upgrade pip

# Base dependencies (numpy, scipy, dimod, dwave-*, pytest, python-dotenv)
pip install -r requirements.txt
# Extra packages used by some optional solvers on GPU boxes:
pip install sympy jax "jax[cuda12]"
```

If `python3.12` is not found, install it first:

```bash
# Ubuntu 22.04+ / Debian
sudo apt install python3.12 python3.12-venv

# macOS (Homebrew)
brew install python@3.12
```

## 3. Install QHDOPT

On Python 3.12 (recommended):

```bash
pip install qhdopt
```

**On Python 3.13** (only if you skipped cuOpt and chose 3.13 over
3.12) you have to bypass two broken pins (`numpy<1.28` and
`scipy==1.11.4`, neither of which builds on 3.13). The qhdopt
runtime works fine with newer numpy / scipy:

```bash
pip install --no-deps qhdopt
pip install --no-deps cyipopt simuq sympy "jax[cpu]" jaxlib qiskit
```

Quick smoke check (works for both Python versions):

```bash
python -c "from qhdopt import QHD; print('qhdopt:', QHD.__module__)"
```

## 4. Install NVIDIA cuOpt

cuOpt ships pre-built wheels on NVIDIA's PyPI. Match the CUDA major
version to your driver:

```bash
# For CUDA 12.x systems (most common)
pip install --extra-index-url=https://pypi.nvidia.com \
    "cuopt-cu12" "nvidia-nvjitlink-cu12" "rapids-logger"

# For CUDA 13.x systems
# pip install --extra-index-url=https://pypi.nvidia.com \
#     "cuopt-cu13" "nvidia-nvjitlink-cu13" "rapids-logger"
```

Quick smoke check:

```bash
python -c "from cuopt.linear_programming.problem import Problem, VType, MINIMIZE; print('cuopt OK')"
nvidia-smi  # confirm GPU is visible
```

## 5. Install OpenPhiSolve

PhiSolve isn't on PyPI — install from source:

```bash
# From somewhere outside the portfolio-optimization repo
git clone https://github.com/Artephi-Computing/OpenPhiSolve.git
cd OpenPhiSolve
pip install -e .
cd -  # back to experiments/simple
```

Quick smoke check:

```bash
python -c "from phisolve import PhiMIQP, QUBO, MIQP, QIHD, PDQP; print('phisolve OK')"
```

## 6. Install Pasqal solver stack

Pasqal's neutral-atom libraries — used by `pasqal_portfolio.py` for the
qubosolver / MIS / raw-Pulser benchmark rows. All three paths run on
local CPU emulators (`LocalEmulator` for qubosolver, `qutip` for MIS
and raw Pulser), so no Pasqal cloud account is required.

```bash
pip install qubo-solver maximum-independent-set pulser pulser-simulation qutip
```

This pulls in ~40 transitive dependencies including `torch`, `cplex`,
`pasqal-cloud`, and `qoolqit`. Total install is ~1–2 GB.

Quick smoke check:

```bash
python -c "
from qubosolver import QUBOInstance
from mis import MISSolver
from pulser.backends import QutipBackendV2
print('Pasqal stack OK')
"
```

## 7. Run the individual demos

```bash
cd experiments/simple  # if not already there
source .venv/bin/activate

python qhd_portfolio.py       # QHD demo (QP + SymPy paths)
python cuopt_portfolio.py     # cuOpt demo (MILP + QP paths)
python phi_portfolio.py       # PhiSolve demo (QUBO + MIQP paths)
python pasqal_portfolio.py    # Pasqal demo (qubo + MIS + Pulser paths)
```

Each should print its selected assets, total score, and a feasibility
check. Expected score on the example problem is **17** (assets A+C+E).

## 8. Run the test suites

```bash
python -m pytest test_qhd_portfolio.py -v
python -m pytest test_cuopt_portfolio.py -v
python -m pytest test_phi_portfolio.py -v
python -m pytest test_pasqal_portfolio.py -v

# Or all at once
python -m pytest . -v
```

On a properly configured box you should see all four suites fully
execute (not skip). Expected counts: ~34 QHD, ~34 cuOpt, ~33 PhiSolve,
20 Pasqal, plus the 92 existing QUBO/CQM/Slack tests.

## 9. Run the full benchmark

```bash
# Quick sanity run on the PDF example
python benchmark_simple_qubo.py --problem-set simple --num-trials 1

# Random problem with more assets — good stress test for GPU solvers.
# Note: Pasqal-Pulser is hard-capped at n<=12 by Qutip emulation cost,
# so it shows as failed (or returns trivial output) beyond that.
python benchmark_simple_qubo.py --problem-set random \
    --num-assets 20 --num-trials 5 --seed 42
```

You should see all 21 solver rows report real scores and runtimes
(or graceful FAIL for solvers whose dependencies are missing). Things
to look for:

- **cuOpt-MILP** should match `ILP (scipy)` in score but be meaningfully
  faster on larger problems
- **cuOpt-QP**, **QHD-QP**, **Pasqal-QUBO**, and **Phi-QUBO** all solve
  the same QUBO penalty-matrix energy landscape with different
  quantum-inspired backends. They tend to converge to the same local
  optima — a property of the penalty-matrix formulation, not the
  backends
- **Phi-MIQP** and **QHD-SymPy** use native linear constraints (MIQP)
  rather than penalty matrices and consistently find the optimum
- **Pasqal-MIS** uses a pairwise conflict-graph relaxation plus
  classical score-aware post-selection — it can match the global
  optimum when the optimal portfolio is pairwise-feasible
- **Pasqal-Pulser** is sensitive to pulse parameters (Rabi amplitude,
  detuning sweep, total duration). The defaults in `pasqal_portfolio.py`
  are tuned for the PDF example; expect degradation on larger / more
  constrained problems
- Watch for out-of-memory errors on very large problems — cuOpt and
  PhiSolve both use JAX/CUDA and may need smaller batch sizes

## 10. Optional: switch PhiSolve to GPU

`phi_portfolio.py` defaults to `device="cpu"`. To use the GPU, edit
`solve_qubo` / `solve_miqp` calls in `benchmark_simple_qubo.py` (or
pass through the benchmark) to use `device="gpu"`. A quick way to
test manually:

```bash
python -c "
from phi_portfolio import PhiPortfolioOptimizer
from simple_portfolio_qubo import get_example_assets, get_example_constraints
a, c = get_example_assets(), get_example_constraints()
opt = PhiPortfolioOptimizer(assets=a, budget=c['budget'],
    max_duration=c['max_duration'], max_cardinality=c['max_cardinality'])
print(opt.solve_qubo(device='gpu'))
"
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cuopt-MILP` shows FAIL but package installed | `nvidia-smi` — check driver version and GPU visibility |
| JAX/PhiSolve complains about CUDA version | `pip install "jax[cuda12]"` matching your CUDA runtime |
| `cuopt-cu12` wheel not found | Check Python version — stick to 3.10/3.11/3.12 |
| OOM on large random problems | Lower `--num-assets` or `n_shots` in phi/qhd solve calls |
| Test suite hangs on QHD/Phi | First run JIT-compiles — give it a minute, subsequent runs are fast |
| `qhdopt` install fails on Python 3.13 with `scipy` build error | Use the `--no-deps` workaround in §3 |
| `Pasqal-Pulser` row returns score 0 / empty selection | Adiabatic sweep params not tuned for the problem — `pasqal_portfolio._build_adiabatic_sequence` uses defaults; bump Rabi amplitude or duration if the device caps allow |
| `pulser.exceptions.sequence.RadiusError: All atoms must be at most 38 µm…` | Register layout overflowed `AnalogDevice.max_radial_distance` — already handled in `_build_pulser_register` via `0.95*max_step` cap; this should not fire under normal use |
