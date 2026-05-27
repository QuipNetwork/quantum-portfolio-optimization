# Testing on a Linux box with NVIDIA hardware

Instructions for running the QHDOPT, NVIDIA cuOpt, and OpenPhiSolve
solver integrations (branch `feat/add-qhd-cuopt-phisolve-solvers`)
on a Linux machine with an NVIDIA GPU.

## Prerequisites

- Linux (Ubuntu 22.04+ or similar)
- NVIDIA GPU with compute capability 7.0+ (Volta or newer; Turing,
  Ampere, and Hopper all fine)
- NVIDIA driver supporting CUDA 12.x (`nvidia-smi` should report
  Driver Version >= 525)
- Python 3.10–3.12 (cuOpt packages don't ship for 3.14 yet)

## 1. Clone and check out the branch

```bash
git clone git@gitlab.com:quip.network/quantum-portfolio-optimization.git
cd quantum-portfolio-optimization
git switch feat/add-qhd-cuopt-phisolve-solvers
```

## 2. Set up a Python environment for the experiments

```bash
cd experiments/simple
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Base dependencies for existing solvers
pip install numpy scipy dimod dwave-neal dwave-system dwave-optimization \
            pytest sympy jax "jax[cuda12]"
```

## 3. Install QHDOPT

```bash
pip install qhdopt
```

Quick smoke check:

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

## 6. Run the individual demos

```bash
cd experiments/simple  # if not already there
source .venv/bin/activate

python qhd_portfolio.py       # QHD demo (QP + SymPy paths)
python cuopt_portfolio.py     # cuOpt demo (MILP + QP paths)
python phi_portfolio.py       # PhiSolve demo (QUBO + MIQP paths)
```

Each should print its selected assets, total score, and a feasibility
check. Expected score on the example problem is **17** (assets A+C+E).

## 7. Run the test suites

```bash
python -m pytest test_qhd_portfolio.py -v
python -m pytest test_cuopt_portfolio.py -v
python -m pytest test_phi_portfolio.py -v

# Or all at once
python -m pytest . -v
```

On a properly configured GPU box you should see all three suites fully
execute (not skip). Expected counts: ~34 QHD, ~34 cuOpt, ~33 PhiSolve,
plus the 92 existing QUBO/CQM/Slack tests.

## 8. Run the full benchmark

```bash
# Quick sanity run on the PDF example
python benchmark_simple_qubo.py --problem-set simple --num-trials 1

# Random problem with more assets — good stress test for GPU solvers
python benchmark_simple_qubo.py --problem-set random \
    --num-assets 20 --num-trials 5 --seed 42
```

You should see all 18 solvers report real scores and runtimes. Things
to look for:

- **cuOpt-MILP** should match `ILP (scipy)` in score but be meaningfully
  faster on larger problems
- **cuOpt-QP** and **QHD-QP** solve the same energy landscape — compare
  runtime and solution quality
- **Phi-QUBO** and **Phi-MIQP** should find feasible solutions; QIHD
  quality depends on `n_shots` / `n_steps` (tune in `phi_portfolio.py`
  if needed)
- Watch for out-of-memory errors on very large problems — cuOpt and
  PhiSolve both use JAX/CUDA and may need smaller batch sizes

## 9. Optional: switch PhiSolve to GPU

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
