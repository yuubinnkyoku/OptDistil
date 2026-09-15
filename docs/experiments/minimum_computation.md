# Minimum optimizer computation and role-rule falsification

Status: **main run complete** on `experiment/stochastic-distillation`.

Questions:

1. What is the **minimum optimizer computation** required on the planted
   multitensor family and its reparameterization stress variant?
2. Is the validation-tuned **matrix/vector static role NormGrad rule** a
   transferable principle, or a benchmark artifact?

Artifacts:

- `artifacts/min-compute-main.json` (canonical payload)
- `artifacts/min-compute-quick.json` (smoke)

Literature context (not a substitute for these tests): matrix/vector role-split
updates are known practice (Muon + AdamW; μP; Fixup). Path-SGD is the classical
rescaling-invariant counterpart to function-space NormGrad. **No prior result
found** that a 2-parameter static role NormGrad rule beats a 153-parameter
learned student under diagonal reparameterization.

## Protocol

| piece | design |
|---|---|
| Source family | `two_layer` + `residual` planted tanh-MLP regression |
| Conditions | train/tune `30/300`; IID test `10/100/1000/3000` |
| Width / samples / steps / batch | 8 / 48 / 18 / 8 |
| Val tasks / test tasks | 3 / 4 per architecture |
| LR candidates | same 8-point grid for uniform and role methods |
| Role partitions | true matrix/vector, swapped, 3 random balanced 2-way |
| Transfer targets | `three_layer` (new), width OOD (`width=16`) |
| Inversion | role-aligned scales `s_matrix=10`, `s_vector=0.1` (jittered) |
| Privileged match | ordinary NormGrad scale `c_i = lr / s_i` |
| Tuning | validation only; never IID/OOD test |

Commit at run time: `c117667b301992a4852de8ad4d28914e17826405`.

## A/B. Computation ladder

Lower test mean loss ratio is better.

### A. Base multitensor family

| method | free scalars | test mean |
|---|---:|---:|
| **NormGrad role LRs** | 2 | **0.0749** |
| NormGrad uniform | 1 | 0.0891 |
| AdamW role LRs | 2 | 0.1036 |
| AdamW uniform | 1 | 0.1133 |
| SGD uniform / role | 1 / 2 | 0.2723 |
| identity no-op | 0 | 1.0000 |

### B. IID reparameterization (train-range scales)

| method | free scalars | test mean |
|---|---:|---:|
| **NormGrad role LRs** | 2 | **0.0823** |
| NormGrad uniform | 1 | 0.0955 |
| AdamW uniform | 1 | 0.1541 |
| AdamW role LRs | 2 | 0.1550 |
| identity no-op | 0 | 1.0000 |
| SGD uniform / role | 1 / 2 | ~1e18 (collapsed) |

**Minimum sufficient computation on this family:** per-tensor unit gradient
direction (NormGrad) plus **two** validation-tuned role LRs. A single uniform
NormGrad LR is competitive (gap ≈ 0.014 on A, 0.013 on B). Raw SGD is not
sufficient under reparameterization; AdamW is strictly worse than NormGrad.

No learned optimizer is required.

## C. Role-label permutation — is matrix/vector special?

| partition | kind | test mean | vs true mean diff | wins better than true |
|---|---|---:|---:|---:|
| true matrix/vector | true | 0.0749 | 0.0000 | 0/32 |
| swapped labels | swap | 0.0749 | 0.0000 | 0/32 |
| random partition 0 | random | 0.0746 | −0.0003 | 17/32 |
| random partition 1 | random | 0.0720 | −0.0030 | 12/32 |
| random partition 2 | random | 0.0747 | −0.0002 | 8/32 |

Bootstrap 95% CI for best random − true: **[−0.0057, −0.0004]** (random slightly
better). Caveat: “best of 3 randoms” is mildly optimistic as a point estimate;
the qualitative claim (true labels are not required) still holds for every
random partition shown.

**Conclusion:** the *name* “matrix vs vector” is not special. Any balanced
two-way partition of tensors, re-tuned on validation with the same budget,
matches or slightly beats the structural matrix/vector split. The useful
ingredient is **two-way heterogeneous NormGrad LRs**, not tensor rank.

Note: the swapped-label control is a pure rename when both LRs are re-tuned,
so zero difference is expected by construction; the informative controls are
the random partitions.

## D. Frozen-ratio transfer (within planting geometry)

Source shared-role LRs: `{matrix: 0.1, vector: 0.03}` (ratio ≈ 3.33).

| target | global scale | frozen ratio | fully retuned | uniform | frozen − retuned |
|---|---:|---:|---:|---:|---:|
| three_layer | 1.0 | 0.1408 | 0.1408 | 0.2278 | 0.0000 |
| width_ood (16) | 1.5 | 0.0984 | 0.1046 | 0.1157 | −0.0062 |

**Within the same planted-MLP geometry**, freezing the relative role ratio and
retuning only a global scalar transfers to a deeper architecture and wider
widths. This does **not** save the universal-principle claim once function-space
scales change (Experiment E).

## E. Inversion — falsification of a universal matrix≫vector ratio

Role-aligned reparameterization with `s_matrix = 10`, `s_vector = 0.1` (plus
mild log-uniform jitter). Privileged theory: θ-space NormGrad scale should be
`c_i ∝ 1/s_i`, so **vectors need larger θ-steps**.

| quantity | value |
|---|---:|
| source ratio matrix/vector | 3.333 |
| **retuned ratio matrix/vector** | **0.050** |
| retuned LRs | `matrix=0.01`, `vector=0.2` |
| frozen source-ratio mean | 0.1069 |
| retuned role mean | **0.0746** |
| uniform NormGrad mean | 0.1008 |
| frozen − retuned | +0.0323 (frozen better on only 5/32) |

Bootstrap 95% CI for frozen − retuned: **[0.0198, 0.0463]** (retuned clearly
better).

**Conclusion:** the empirical matrix≫vector ratio is a **family artifact of the
base planting / scale geometry**, not a transferable optimizer principle. When
function-space reparameterization inverts the required θ-step sizes, the
source-family ratio fails and re-tuning flips the ratio.

## Interpretation

| claim | status |
|---|---|
| Minimum compute on this family = NormGrad + 2 role LRs | **supported** |
| Uniform NormGrad is almost as good | supported (gap ~0.014) |
| Matrix/vector role *labels* are special | **falsified** (random partitions match) |
| Matrix≫vector *ratio* transfers universally | **falsified** (inversion flips it) |
| Ratio transfers within same planting geometry | supported (three_layer, width) |
| Learned Student / larger optimizer needed | **not supported** |

Revised working statement:

> On these planted synthetic benchmarks, the useful optimizer computation is
> per-tensor normalized-gradient direction plus a **small number of
> validation-tuned heterogeneous LRs**. Calling that rule “matrix vs vector”
> overfits the partition to the current architecture names; the specific
> ratio overfits the current function-space scale geometry.

This is narrower than “fixed tensor-role learning rates are a transferable
principle” and narrower than “optimizer distillation found adaptive structure.”

## What this does not settle

1. Whether a real-data architecture family (CNN/Transformer) needs more than
   two LR groups or more than NormGrad geometry.
2. Whether LARS/LAMB-style dynamic trust ratios beat static two-role LRs when
   norms drift over long horizons.
3. Whether the 153p Student could win on a family where **no** static partition
   is sufficient — current evidence never required that.

## Recommendation

1. Report the **minimum-computation ladder** and **permutation/inversion
   falsifications** as the primary contribution of this branch, not Student
   capacity scaling.
2. Do **not** claim a universal matrix/vector optimizer principle.
3. Do **not** introduce larger teachers to “rescue” adaptive claims on this
   family.
4. Next decisive experiment (if any): a **non-planted** or real-data family
   where tensor roles and function-scale geometry differ enough that static
   two-way LRs fail, with LARS/AdamW role baselines already in the ladder.

## Reproduce

```bash
# unit tests
uv run --locked --extra cpu pytest -q tests/test_minimum_computation.py

# quick smoke
uv run --locked --extra cpu python scripts/probe_minimum_computation.py \
  --quick --output artifacts/min-compute-quick.json

# main
uv run --locked --extra cpu python scripts/probe_minimum_computation.py \
  --output artifacts/min-compute-main.json

# report
uv run --locked --extra cpu python scripts/report_minimum_computation.py \
  --input artifacts/min-compute-main.json
```

## Engineering notes

- New package pieces: `multitensor/roles.py`, `multitensor/ladder.py`,
  `RoleScaledSGD` / `RoleScaledAdamW` in `teachers.py`,
  `ThreeLayerMLPRegressionTask` + `make_three_layer_mlp` in `tasks.py`,
  `role_aligned_scales` in `reparam.py`.
- Existing single-tensor and prior multitensor/reparam paths are preserved.
- Full suite at commit time: `140 passed`.
