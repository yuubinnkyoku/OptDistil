# Minimum optimizer computation frontier

Status: **main run complete** on `experiment/stochastic-distillation`.

Question: can the 2-parameter anisotropic NormGrad explanation be reduced
further, and what is the gain actually made of?

Artifacts:

- `artifacts/frontier-main.json`
- `artifacts/frontier-quick.json`

## Protocol

| piece | design |
|---|---|
| Source family | `two_layer` + `residual` planted tanh-MLP |
| Conditions | train `30/300`; IID test `10/100/1000/3000` |
| Width / samples / steps / batch | 8 / 48 / 18 / 8 |
| Val / test tasks | 3 / 4 per architecture |
| Tuning | coordinate descent, `--cd-rounds 3`, 8-point LR grid |
| Adversarial families | `all_matrix`, `all_vector`, `iso_shape` |
| Formulas | absolute LRs, **no validation search** (`base=0.1`) |

Commit at run time: recorded in artifact JSON.

## 1. Is one global scale enough?

Paired audit of `min-compute-main.json` (same protocol as prior ladder):

| comparison | mean diff (role − uniform) | bootstrap 95% | wins | sign p | Cohen d_z |
|---|---:|---|---:|---:|---:|
| NormGrad base A | −0.0142 | [−0.0204, −0.0088] | 26/32 | 0.0005 | −0.83 |
| NormGrad reparam B | −0.0132 | [−0.0197, −0.0068] | 23/32 | 0.020 | −0.69 |
| AdamW base A | −0.0097 | [−0.0160, −0.0042] | 23/32 | 0.020 | −0.55 |
| AdamW reparam B | +0.0009 | [−0.0205, +0.0293] | 20/32 | 0.22 | +0.01 |

**Verdict:** for NormGrad, **1-param uniform is statistically distinguishable
from 2-param roles** on both families. Cannot reduce the free scalar count to 1
without a real loss. AdamW’s role gain does **not** survive reparameterization.

## 2. Group-count / partition frontier (base family)

Coordinate descent, same rounds and candidate grid for every partition.

| method | free scalars | test mean | Δ vs k=1 | bootstrap 95% | sign p |
|---|---:|---:|---:|---|---:|
| **per-tensor oracle** | 5 | **0.0679** | −0.0212 | [−0.0290, −0.0139] | 1e-4 |
| **k=2 numel** | 2 | **0.0717** | −0.0175 | [−0.0248, −0.0115] | 2e-5 |
| k=3 numel | 3 | 0.0737 | −0.0155 | [−0.0230, −0.0086] | 0.034 |
| k=2 true roles / ndim / index_mod | 2 | 0.0749 | −0.0142 | [−0.0209, −0.0087] | 4e-4 |
| k=2 random (best) | 2 | 0.0793 | −0.0099 | [−0.0149, −0.0050] | 2e-5 |
| **k=1 uniform** | 1 | **0.0891** | 0 | — | — |

Paired best-k2 − k1: mean **−0.0175**, CI **[−0.0248, −0.0115]**, 28/32 wins.

**Complexity/performance frontier on this family:**

```text
free scalars:  1        2           3           5
test mean:   0.089 → 0.072      0.074       0.068
             uniform  numel-2    numel-3     per-tensor
```

Going 2→3 does **not** help; 2→5 buys only ~0.004 further. **k=2 by numel is
the knee.**

## 3. Property formulas without validation search

Absolute LRs from geometry only (`base=0.1`, no val):

| formula | test mean | vs val-tuned k=2 (0.0717) |
|---|---:|---|
| **numel_rank** `c=0.1*(0.25+0.75*rank(numel))` | **0.0748** | +0.0031 |
| ndim_scaled (matrix=0.1, vector=0.033) | 0.0759 | +0.0042 |
| uniform 0.1 | 0.0891 | +0.0174 |
| LARS-like `c∝‖θ‖_init` | 0.0891 | +0.0174 |
| inv_sqrt_numel_normalized | 0.1405 | worse |
| inv_sqrt_fan_in | 0.1680 | worse |
| inv_sqrt_numel / inv_numel (raw) | 0.41 / 0.81 | broken scale |

**A zero-search numel-rank formula nearly matches validation-tuned 2-group
NormGrad.** Theory prior: optimal unit-direction step is
`c* = ‖g‖ / λ_g` (Rayleigh quotient); shape-only formulas cannot recover
curvature, but **size-rank** is a strong proxy on this family. LARS ‖θ‖ and
μP-style 1/√fan-in do **not** help here.

## 4. What the gain is made of

| hypothesis | evidence |
|---|---|
| matrix/vector semantics | **rejected** — numel split ≥ true roles; random 2-way often works |
| tensor size / numel class | **supported** — best k=2 is numel; formula numel_rank ≈ val-tuned |
| ndim | weak — same as roles on this family; formula close |
| pure HP freedom | **partial** — random k=2 sometimes collapses to k=1; not all randoms help |
| deep curvature structure | **not required** — iso_shape oracle ≈ k=1 |
| Jacobian / LARS / fan-in formulas | **not supported** on this planting |

## 5. Adversarial families (break current partition)

### all_matrix (3 matrices, no biases)

- true roles collapse to 1 group (all `"matrix"`)
- **k=3 numel / oracle: 0.093 vs k=1: 0.117** — size heterogeneity still helps

### all_vector (3 vectors)

- true roles collapse to 1 group
- **best k=2: 0.020 vs k=1 ≈ 0.064** — large gain without any matrix

### iso_shape (4 equal-shape matrices, κ staircase 1…condition)

- all methods ~0.092–0.095; oracle ≈ k=2 ≈ k=1
- **Equal-shape least-squares blocks do not need heterogeneous static LRs**
  under NormGrad + this budget

**Conclusion:** the 2-group gain on the base family is **size-class
heterogeneity**, not matrix/vector structure and not fine curvature.

## 6. Minimum optimizer computation (supported)

On the planted synthetic family used by OptDistil:

1. **Direction:** per-tensor normalized gradient (NormGrad). SGD collapses
   under reparameterization; AdamW is worse.
2. **Scale:** either
   - **1 free scalar is not enough** (paired, significant), and
   - **2 free scalars on a numel split**, or
   - **zero free scalars beyond a fixed base** via the `numel_rank` formula
     (within ~0.003 of val-tuned k=2).
3. **Do not need:** learned students, matrix/vector semantics, k≥3 groups on
   this family, LARS/init-norm formulas, or per-tensor oracles for the last
   ~0.004.

Stopping rule applied: further complexity (k>2, learned adaptation) does not
yield convincing out-of-sample benefit beyond the numel knee.

## Reproduce

```bash
uv run --locked --extra cpu pytest -q tests/test_computation_frontier.py

uv run --locked --extra cpu python scripts/probe_computation_frontier.py \
  --output artifacts/frontier-main.json

uv run --locked --extra cpu python scripts/probe_computation_frontier.py \
  --quick --output artifacts/frontier-quick.json
```

## Engineering notes

- New: `multitensor/frontier.py`, `all_matrix` / `all_vector` / `iso_shape`
  tasks, `scripts/probe_computation_frontier.py`,
  `tests/test_computation_frontier.py`.
- Single-tensor and prior multitensor/reparam paths preserved.
- Per-tensor oracle uses one scale vector keyed by tensor index; mixed
  architectures union group labels.
