---
feature: minimum-optimizer-computation
status: delivered
updated: 2026-02-16
branch: experiment/stochastic-distillation
commits: c117667..3103dab
---

# Minimum Optimizer Computation

## Report

**What was built** — A falsification-first probe of the minimum optimizer
computation on the planted multitensor family and whether the 2-parameter
matrix/vector role NormGrad rule is a transferable principle. Added role
permutation helpers, role-scaled SGD/AdamW, a three-layer MLP task,
role-aligned reparameterization, the `probe_minimum_computation.py` ladder
(A–E), a report script, unit tests, and `docs/experiments/minimum_computation.md`.

**What was learned** —

1. NormGrad + 2 validation-tuned role LRs is the strongest cheap method
   (base 0.0749, reparam 0.0823). Uniform NormGrad is close; SGD collapses
   under reparameterization; AdamW is worse than NormGrad. No learned
   optimizer is required.
2. Random balanced two-way partitions match or slightly beat true
   matrix/vector roles (bootstrap CI of best random − true excludes 0 in
   favor of random). The *labels* are not special.
3. Frozen source ratio transfers to three_layer and width OOD within the
   same planting geometry.
4. Under role-aligned inversion (`s_matrix≫s_vector`), retuned LRs flip to
   matrix/vector = 0.05 and beat the frozen source ratio (CI [0.020, 0.046]).
   The *ratio* is a family artifact, not a universal principle.

**Verification** — `uv run --locked --extra cpu pytest -q` → 140 passed;
`uv run --locked --extra cpu ruff check .` clean on new files;
main artifact `artifacts/min-compute-main.json` (runtime ≈ 186s).

**Journey log** —

1. Worktree creation was blocked by a shared-registry hook; continued on
   `experiment/stochastic-distillation` after user consent for isolation
   (not available).
2. First inversion scale direction was theoretically inverted
   (`c ∝ 1/s`, not `c ∝ s`); corrected before the main run.
3. Quick smoke showed uniform NormGrad beating roles (tiny n); main run
   restored the expected role>uniform ordering — always use main budgets
   for ranking claims.
4. Matrix/vector naming is post-hoc; the mechanism is 2-way heterogeneous
   NormGrad LRs under this planting geometry.

## [S1] Problem

The 153-parameter Student no longer outperforms a validation-tuned 2-parameter
static matrix/vector role NormGrad rule under exact reparameterization stress.
The remaining research questions are:

1. What is the **minimum optimizer computation** actually required on this
   benchmark family?
2. Is the static **matrix/vector role rule a transferable principle**, or a
   benchmark artifact of planted tanh-MLP regression with specific role
   geometry?

Prior work (Muon/Moonlight, μP, Fixup, Path-SGD) already uses role-split
parameter updates, but does **not** publish the claim that a 2-parameter static
NormGrad rule beats a learned student under diagonal reparameterization.

## [S2] Design

Four falsification-first experiments share one validation-only protocol and
fair hyperparameter budgets. Student capacity is **not** increased. Existing
single-tensor and multitensor paths stay intact.

### S2.1 Minimum-computation ladder

Compare analytic optimizers that differ only in update geometry and the number
of free LRs (tuned on validation only):

| method | free scalars | geometry |
|---|---:|---|
| SGD uniform | 1 | raw gradient |
| SGD role LRs | 2 | raw gradient, matrix/vector split |
| NormGrad uniform | 1 | per-tensor unit direction |
| NormGrad role LRs | 2 | current 2-param winner |
| AdamW uniform | 1 | per-element second moment |
| AdamW role LRs | 2 | same + role LR |

Evaluated on (a) the existing multitensor family and (b) the existing
reparameterization stress family.

### S2.2 Role-label permutation (is matrix/vector special?)

Keep the 2-way static NormGrad protocol. Compare:

1. true matrix/vector roles
2. swapped labels (vector as matrix, matrix as vector)
3. several random balanced 2-way partitions of tensors (validation-retuned)

If random partitions match true roles, “matrix vs vector” is post-hoc naming,
not a special structural principle.

### S2.3 Frozen-ratio transfer

Tune role LRs once on the source family. On each target domain, allow **only a
one-dimensional global scalar** (ratio frozen). Targets:

- three-layer MLP (new architecture, same regression protocol)
- wider nets
- noisier batches
- condition-number OOD

Compare against fully retuned 2-role NormGrad and uniform NormGrad.

### S2.4 Inverted-role reparameterization family

Construct tasks where the theoretically required θ-space role ratio **flips**:

- Privileged match: ordinary NormGrad scale `c_i = lr / s_i`.
- Set matrices `s_matrix ≫ s_vector` (e.g. 10 vs 0.1) so vectors need larger
  θ-space steps than matrices.

If the source-family ratio (matrix ≫ vector) fails here while re-tuning
recovers, the static rule is a **task-family artifact**, not a universal
principle. If the frozen source ratio still wins, the result is surprising and
would re-open an adaptive-claim investigation.

### S2.5 Protocol rules

- Tune only on validation splits (never IID/OOD test).
- Paired tasks and batch sequences across methods.
- Fair LR candidate grids (same cardinality where comparable).
- Record mean/std/median, bootstrap CI, win fraction, finite fraction.
- No increase of Teacher/Student capacity to rescue prior claims.
- Preserve existing single-tensor paths.

## [S3] Out of Scope

- Larger learned teachers or students.
- Real-image CNN/Transformer training (deferred until ladder + inversion settle).
- Claiming optimizer distillation success if the 2-param static rule remains
  sufficient.
- Merging to `main`.

## Tasks

- [x] T1: Role-permutation helpers + role-scaled SGD/AdamW teachers — acceptance: unit tests pass (covers: S2.1, S2.2)
- [x] T2: Three-layer MLP multitensor task — acceptance: analytic grads match autograd (covers: S2.3)
- [x] T3: Role-aligned reparameterization helpers — acceptance: scales applied consistently; privileged rule still uses `s` (covers: S2.4)
- [x] T4: `probe_minimum_computation.py` running Experiments A–E — acceptance: writes reproducible JSON artifact (covers: S2.1–S2.4)
- [x] T5: Unit + integration tests — acceptance: `pytest -q tests/test_minimum_computation.py` (covers: S2.1–S2.4)
- [x] T6: Run main probe + write experiment doc with falsification conclusions — acceptance: negative results documented (covers: S1)
