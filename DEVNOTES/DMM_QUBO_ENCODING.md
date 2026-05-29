# DMM-Based QUBO Encoding on Neutral Atoms

How to attach a **Detuning Map Modulator (DMM)** channel to a Pulser `Sequence` so the **diagonal of a QUBO** is encoded as **per-atom detuning**, complementing the **off-diagonal** that is encoded by **atom positions**. Sections 1–9 are the general mechanics; section 10 is how `experiments/simple/pasqal_portfolio.py` actually uses them.

**Assumes:** a symmetric QUBO `Q` (minimizing `xᵀ Q x`, negative-leaning diagonal), a 2D register from a classical pre-optimizer that matches `C₆/r_ij⁶` to `|off-diagonal(Q)|`, and a device that exposes a DMM channel.

## 1. What the DMM Encodes

The Rydberg Hamiltonian under analog control:

```
H(t) = (Ω(t)/2) Σ_i σ_i^x  −  Σ_i δ_i(t) n_i  +  Σ_{i<j} (C₆/r_ij^6) n_i n_j
       global drive          detuning            pairwise interaction
```

The DMM overlays a **static per-atom mask** on the otherwise-global detuning:

```
δ_i(t) = δ_global(t)  +  ε_i · δ_DMM(t)
         (all atoms)      (atom-i-specific; encodes diag(Q))
```

- `ε_i ∈ [0, 1]` — dimensionless per-atom weight, fixed for the run.
- `δ_DMM(t)` — the DMM waveform (`ConstantWaveform` is typical for QAA).

**Sign convention.** With `ε ∈ [0,1]` and a *negative* `δ_DMM`, higher `ε` → more negative total detuning → `|1⟩` more expensive → excitation **suppressed**:

```
ε_i ↑  ⟺  diag(Q)_i ↑  ⟺  asset i less desirable
ε_i ↓  ⟺  diag(Q)_i ↓  ⟺  asset i more desirable
```

## 2. Device Prerequisite

The DMM is not on every device — use one with a populated `dmm_channels`.

| Device | DMM | Notes |
|---|---|---|
| `DigitalAnalogDevice` | **yes** (`dmm_0`) | Recommended target. |
| `MockDevice` | yes | Permissive limits, for exploration. |
| `AnalogDevice` | **no** | Raises on `config_detuning_map`. |

```python
assert device.dmm_channels, f"{device.name} has no DMM channel"
```

## 3. Computing the ε Weights

Three recipes, increasing in principled-ness, plus a range guarantee.

### 3a. Score-only (used by both repo paths)

Uses only the per-asset objective. Correct when constraints are handled elsewhere (e.g. by the register geometry) or when `diag(Q)` is dominated by `−score`.

```python
rng = score.max() - score.min()
epsilon = (score.max() - score) / rng if rng > 0 else np.zeros(n)
```

### 3b. Direct: full QUBO diagonal

Reads `ε` straight from `diag(Q)` — canonical PASQAL advice.

```python
diag = np.diag(Q)
rng = diag.max() - diag.min()
epsilon = (diag - diag.min()) / rng if rng > 0 else np.zeros(n)
```

**Failure mode:** when `diag(Q)` is dominated by the constraint linear-bias term (`−2λb·g`), `ε` reflects "how many tight constraints touch this atom" more than asset quality, which can invert the ordering on ill-conditioned QUBOs.

### 3c. α-decoupled diagonal

For QUBOs where constraints overlap on the same atoms: build a *separate* DMM-only diagonal with constraint penalties scaled by `α ∈ [0,1]` (`α=0` → score-only, `α=1` → full `diag(Q)`). Atom positions still use the full `λ`; only `ε` sees the reduced version. Sweep `α ∈ {0, 0.05, 0.1, 0.25, 0.5, 1.0}` and keep the value with the highest target-bitstring frequency on a noiseless emulator.

### 3d. Range guarantee

The DMM requires `ε ∈ [0,1]`. After any recipe:

```python
assert 0.0 <= epsilon.min() <= epsilon.max() <= 1.0
```

## 4. Attaching the DMM — Two-Step Construction

```
Stage 1 (spatial)                 Stage 2 (temporal)
register.define_detuning_map  →   seq.add_dmm_detuning(waveform, "dmm_0")
  {qubit_id: ε}                    applies waveform scaled per-atom by ε
```

with a `seq.config_detuning_map(det_map, "dmm_0")` between them.

```python
# 1. Spatial mask — keys must match the Register's qubit_ids, values in [0,1]
det_map = register.define_detuning_map({f"q{i}": float(epsilon[i]) for i in range(n)})

# 2. Bind it to a hardware DMM channel (must precede add_dmm_detuning)
seq = Sequence(register, device)
seq.declare_channel("global", "rydberg_global")
seq.config_detuning_map(det_map, "dmm_0")

# 3. Temporal waveform — duration must match the global pulse (they run in parallel)
seq.add_dmm_detuning(ConstantWaveform(T, dmm_amplitude), "dmm_0")
```

## 5. Choosing the DMM Amplitude

The DMM amplitude is set relative to the global sweep's final detuning `δ_f`:

```python
dmm_amplitude = -delta_f * alpha_dmm   # alpha_dmm ∈ (0,1), typically 0.5–0.8
```

At the end of the sweep the global channel pushes every atom toward `|1⟩` (positive `δ_f`); the DMM subtracts `ε_i · δ_f · α_dmm` from high-ε atoms, partially undoing that push for "bad" assets. `α_dmm = 1` would exactly cancel it for `ε=1`, so keep `α_dmm < 1` for residual cancellation.

## 6. Hardware Limit Checks

```python
dmm_ch = device.dmm_channels["dmm_0"]

# Per-atom floor
if dmm_ch.bottom_detuning is not None:
    dmm_amplitude = max(dmm_amplitude, dmm_ch.bottom_detuning)

# Total-array floor (sum over atoms, related to laser power)
if dmm_ch.total_bottom_detuning is not None:
    total = float(np.sum(epsilon)) * dmm_amplitude
    if total < dmm_ch.total_bottom_detuning:
        dmm_amplitude *= dmm_ch.total_bottom_detuning / total
```

Scaling preserves the relative per-atom ordering; only the magnitude shrinks.

## 7. Full Reference Pattern

```python
from pulser import Pulse, Register, Sequence
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform, ConstantWaveform
import numpy as np

# Prereqs: register, epsilon ∈ [0,1], pulse endpoints Omega / delta_0 / delta_f
device = DigitalAnalogDevice
assert device.dmm_channels, "device has no DMM channel"

det_map = register.define_detuning_map(
    {f"q{i}": float(epsilon[i]) for i in range(len(epsilon))}
)
seq = Sequence(register, device)
seq.declare_channel("global", "rydberg_global")
seq.config_detuning_map(det_map, "dmm_0")

T = 5500
omega_wf = InterpolatedWaveform(T, [1e-9, Omega*0.4, Omega*0.9, Omega,
                                    Omega*0.7, Omega*0.3, 1e-9])
delta_wf = InterpolatedWaveform(T, [delta_0, delta_0*0.7, delta_0*0.2,
                                    delta_f*0.2, delta_f*0.4, delta_f*0.7, delta_f])
seq.add(Pulse(omega_wf, delta_wf, phase=0.0), "global")

dmm_ch = device.dmm_channels["dmm_0"]
dmm_amplitude = -delta_f * 0.6
if dmm_ch.bottom_detuning is not None:
    dmm_amplitude = max(dmm_amplitude, dmm_ch.bottom_detuning)
if dmm_ch.total_bottom_detuning is not None:
    total = float(np.sum(epsilon)) * dmm_amplitude
    if total < dmm_ch.total_bottom_detuning:
        dmm_amplitude *= dmm_ch.total_bottom_detuning / total

seq.add_dmm_detuning(ConstantWaveform(T, dmm_amplitude), "dmm_0")
```

## 8. Verification

Log the effective per-atom detuning at the end of the sweep:

```python
for i in range(n):
    eff_det = delta_f + epsilon[i] * dmm_amplitude
    print(f"q{i}: δ_eff = {eff_det:.4f}")
```

**Expected:** `ε ≈ 0` (best assets) → `δ_eff ≈ δ_f` (positive, drives `|1⟩`); `ε ≈ 1` (worst) → `δ_eff ≈ δ_f + dmm_amplitude` (near zero/negative, suppresses `|1⟩`). If the ordering contradicts the QUBO's expected solution, the bug is upstream in `ε` (§3) or the QUBO formulation, not the DMM attachment.

## 9. Common Pitfalls

| Symptom | Likely cause |
|---|---|
| `RuntimeError: device has no DMM channel` | Using `AnalogDevice`; switch to `DigitalAnalogDevice`. |
| `KeyError: 'dmm_0'` | Channel name mismatch; check `device.dmm_channels`. |
| All atoms collapse to `\|0…0⟩` | `dmm_amplitude` too negative; reduce it or check `bottom_detuning`. |
| All atoms collapse to `\|1…1⟩` | `dmm_amplitude` too small / wrong sign; DMM suppresses nothing. |
| Top bitstring is the *opposite* of expected | `ε` ordering inverted (§1 convention needs negative `dmm_amplitude`). |
| `ValueError: weight out of [0,1]` | `ε` not normalized; assert range (§3d) before the map. |
| Duration mismatch | DMM waveform `T` must equal the global pulse duration. |

## 10. How This Repo Uses the DMM

`experiments/simple/pasqal_portfolio.py` exposes two DMM-equipped Pulser paths that take opposite approaches:

- **`Pasqal-Pulser-DMM`** — reduces the problem to a **conflict graph** (constraints in the geometry, objective in the DMM): a weighted MIS.
- **`Pasqal-Pulser-QUBO`** — embeds the **full QUBO** by hand (off-diagonal → positions, diagonal → detuning).

### 10.1 The pitfall a full-QUBO embedding must avoid

> The Rydberg interaction `C₆/r⁶` is **positive-only** and, at the magnitudes a real QUBO needs, behaves as a **hard blockade** (you cannot excite both atoms), not the **finite penalty** the QUBO intends.

So a large positive `Q[i,j]` — which the QUBO minimum may happily pay — can become a hard "these two can't both be selected." If the optimum *requires* that pair and the pulse's blockade radius covers their distance, the embedding has forbidden the optimum.

Measured on the example (assets A–E, optimum **A,E = 17**) with a **naive fixed Rabi** `Ω = 0.5·max_amp`:

| Check | Result |
|---|---|
| QUBO global min over all 2⁵ states | A,E, score 17 — the formulation is correct |
| `Q[A,E]` | 1094 — dominated by the duration cross-term `2·λ_d·d_A·d_E` |
| A,E distance in the embedding | ~4.1 µm, inside the ~9.4 µm blockade → **A,E blockaded** |

**The fix:** don't hold `Ω` fixed — **tie it to the register's own strongest interaction**, `Ω = 0.5·U_max` with `U_max = C₆/min_pairwise_distance⁶`, so the blockade radius scales *with* the embedded geometry instead of against it.

A methodology rule for **both** paths: **don't let post-processing hide the truth.** Score-aware repair could reconstruct a feasible optimum from an infeasible sample and inflate the reported score — but that is the classical repair talking, not the quantum encoding. Selection is therefore **best already-feasible sample**, with repair as a last resort only when nothing feasible was sampled.

### 10.2 `Pasqal-Pulser-DMM` — weighted MIS

- **Register = conflict graph.** Edge `(i,j)` iff the pair is infeasible (combined price or duration over limit); conflict edges inside the blockade, others outside. Feasible pairs like A,E are *not* edges, so they stay free to be co-excited. A binary target embeds in 2D far more reliably than an arbitrary QUBO matrix.
- **DMM = score-only ε (§3a).** Constraints already live in the geometry, so there is no constraint term left to dominate the diagonal — `ε = normalize(−score)` is exactly right. This is a *weighted* MIS: it prefers the higher-scoring **A,E (17)** over the larger-but-lower-scoring A,C,D (14) the unweighted MIS path settles on, and reaches A,E through genuine sampling.

### 10.3 `Pasqal-Pulser-QUBO` — manual full-QUBO embedding

- **Off-diagonal → positions:** match `C₆/r_ij⁶` to the symmetric off-diagonal directly (Nelder-Mead + compactness penalty + ring start), then rescale the register up to honour the minimum atom spacing.
- **Diagonal → detuning:** the QUBO's negative linear bias (`−2λb·g`) avoids a trivial all-zeros minimum; the global sweep gives the bulk detuning and the DMM adds a per-atom score tilt (`ε = normalize(−score)`).
- **Energy scale `Ω = 0.5·U_max`** (§10.1) — the key to keeping the optimum reachable.

On the example this reaches **14, not 17** — and the reason is the formulation, not the code: `Q[A,E] = 1094` forces A,E to ~4.1 µm, and even at the device's maximum Rabi (Ω ≈ 15.7 rad/µs, blockade ≈ 8.4 µm) the pair stays blockaded. Tying `Ω` to `U_max` shrinks the blockade (9.4 → 8.4 µm) but cannot uncover a pair whose required interaction exceeds the device's Rabi ceiling. Treat the score as a property of *formulation × embedding × device*; a friendlier instance with smaller, well-spread off-diagonals reaches the optimum exactly.

### 10.4 Practical guidance

- Two working recipes: **(a)** hard pairwise feasibility in the **geometry** + objective in the **DMM** (weighted MIS); or **(b)** embed the **full QUBO** off-diagonal but **tie `Ω` to `U_max`** (manual QUBO).
- When embedding a full QUBO, run the §8 verification *and* check the optimum: compare pairwise distances against `device.rydberg_blockade_radius(Ω)` using the *same* `Ω` the pulse uses. If a pair the optimum needs co-excited sits inside the blockade, raise `Ω` toward the device max or fix the formulation — large near-uniform off-diagonals (e.g. an over-weighted `λ_d` producing big `d_i·d_j` cross terms) are the usual culprit.
