# Pasqal Solver Notes

This experiments/simple/ directory benchmarks three Pasqal neutral-atom
solver paths against the same portfolio asset-selection problem the
other solvers (D-Wave SA, D-Wave CQM, NL, QHDOPT, cuOpt, OpenPhiSolve)
target. Two of the three Pasqal paths are NOT directly comparable to
the other rows. This document explains what each row measures.

## Pasqal-QUBO (comparable)

Backend: `qubo-solver` 0.5.x (PyPI package name; imported as
`qubosolver`) with `LocalEmulator` (neutral-atom emulator running
locally).

Input: the same QUBO penalty matrix used by D-Wave SA, cuOpt-QP, and
phi-QUBO — soft penalties for budget, duration, and cardinality.

This row is comparable. Differences in score or runtime against the
other QUBO rows are attributable to the backend, not the problem
formulation.

## Pasqal-Slack (comparable, but interesting tradeoff)

Backend: `qubo-solver` 0.5.x with `LocalEmulator`, on the
`SlackPortfolioQUBO` matrix instead of `SimplePortfolioQUBO`.

The slack formulation converts each `≤` constraint into an `=`
constraint via auxiliary binary slack variables — eliminating the
"penalty pushes toward equality with target" bias of the simple
penalty matrix. The resulting QUBO is mathematically more correct:
its global minimum aligns with the true portfolio optimum (D-Wave
SA confirms this on the same matrix).

Caveat: the slack formulation produces a much larger QUBO matrix
(asset bits + ~30 slack bits per constraint at default precision).
Pasqal's local emulator scales poorly with problem size, so the
default precision is set coarse (budget=1.0, duration=5.0) to keep
runtimes around 3 s rather than the ~4 min that finer precision
takes. With coarse precision, the matrix is correct but the
emulator under-samples the larger state space — empirically this
produces solutions that are sometimes WORSE than Pasqal-QUBO on
small problems, despite the better formulation. A clean
demonstration that solver quality matters as much as formulation
quality, especially at scale.

## Pasqal-MIS (NOT directly comparable)

Backend: `maximum-independent-set` 0.3.x with the local Qutip emulator.

The portfolio problem is re-encoded as a pairwise asset-conflict
graph: nodes are assets, edges connect pairs whose combined price
exceeds the budget OR whose combined duration exceeds the
max-duration constraint. The quantum step finds a Maximum
Independent Set of that graph — the largest set of mutually
non-conflicting assets.

Caveats:
- The MIS encoding only captures PAIRWISE constraints. Multi-asset
  budget interactions (e.g., three cheap assets that together exceed
  the budget) are not represented.
- The cardinality constraint is not encoded in the quantum step.
- Asset scores are not encoded in the quantum step at all — MIS is
  unweighted.

After the quantum step, we apply a classical score-aware greedy
post-selection: from the returned independent set, take assets in
decreasing score order until the next would violate the full budget,
duration, or cardinality constraint. This recovers some score signal,
but the optimization itself was unweighted.

## Pasqal-Pulser (NOT directly comparable; n ≤ 12)

Backend: raw `pulser` 1.6.x with the local `QutipBackendV2` simulator.

A 1D atom register is built that mirrors the same pairwise conflict
graph as Pasqal-MIS — conflicting pairs placed within Rydberg
blockade radius, non-conflicting pairs outside it. A linear-detuning
adiabatic sweep at constant Rabi amplitude is then run on the
register, and the most-common bitstring is sampled and repaired to
feasibility.

Additional caveats on top of the Pasqal-MIS caveats:
- The 1D layout cannot in general realize an arbitrary conflict
  graph (only "interval-graph"-like structures are exact).
- Rydberg interactions are 1/r^6 repulsive only — the encoded
  "constraint" strength varies smoothly across pairs, not as a
  hard pairwise penalty.
- Pulse parameters (Rabi amplitude, total duration, detuning
  endpoints) are heuristic and were not tuned per problem instance.
- Local Qutip emulation cost scales as 2^n; the solver hard-caps at
  n ≤ 12. This rules it out for any realistic portfolio size.

This row is included as a demonstration that the problem can be
expressed at the hardware-control level on neutral atoms, not as a
benchmark of solver quality.
