# Multi-tensor NormGrad allocation mechanism

This experiment asks why the 153-parameter direction-only Student can slightly outperform its tensor-wise NormGrad teacher on the stochastic multi-tensor benchmark, and whether that advantage requires a learned dynamic optimizer at all.

## Verified mechanism run

- Workflow run: `34750297519`
- Job: `103705517493`
- Artifact: `multitensor-allocation-mechanism`
- Artifact ID: `10314879300`
- Artifact SHA256: `99c2c8ecf5c1df4abc75398e68b5fe6abd33279edb633b0eaedeb597d74ac76d`
- Artifact commit: `8a07650bd95254f12cf541a966d15fe5f2f39054`
- Ruff: passed
- Tests: 18 passed
- Student seeds: 5
- Teacher: tensor-wise local NormGrad, validation-selected LR `0.06`
- Batch size: 8
- Architectures: two-layer MLP and residual MLP
- Student: shared `8 -> 8 -> 8 -> 1` MLP, 153 trainable parameters

All calibration/tuning splits are independent from IID/OOD test tasks.

## Causal Student interventions

- `full_student`: original Student update.
- `tensor_projection`: project each tensor's Student update onto that tensor's unit negative-gradient direction. This removes within-tensor directional residuals while preserving the Student's dynamic signed scale for every tensor.
- `global_projection`: preserve only one shared dynamic coefficient across all tensor-local negative-gradient directions. This removes tensor-wise allocation.
- `frozen_role`: use calibration-split median Student projection coefficient for each tensor name.
- `equalized_frozen`: replace all frozen role scales by one median scale.
- `swapped_frozen`: swap `W1/W2` and `b1/b2` frozen role scales.
- `teacher_norm`: original local NormGrad teacher, LR `0.06` for every tensor.

Lower loss ratio is better.

| update rule | IID mean | IID std | OOD mean | OOD std |
|---|---:|---:|---:|---:|
| full Student | 0.06456 | 0.00244 | **0.06363** | 0.00146 |
| tensor projection | **0.06389** | 0.00261 | 0.06377 | 0.00219 |
| global projection | 0.07997 | 0.00914 | 0.08123 | 0.00835 |
| frozen Student role scales | 0.06802 | 0.00402 | 0.07049 | 0.00412 |
| equalized frozen scales | 0.07195 | 0.00506 | 0.07650 | 0.00490 |
| swapped frozen scales | 0.06725 | 0.00383 | 0.07101 | 0.00418 |
| uniform NormGrad teacher | 0.06710 | 0.00000 | 0.06979 | 0.00000 |

`tensor_projection` is slightly better than the full Student on IID in all 5 Student seeds (mean paired difference `-0.000678`) and effectively tied on OOD (`+0.000147`). Therefore within-tensor directional rotation is not needed for the Student's advantage.

Removing tensor-wise scale allocation with `global_projection` is much worse than tensor projection in every seed:

- IID: `+0.01608`
- OOD: `+0.01745`

Student-derived frozen role scales do not preserve the dynamic projected Student:

- IID frozen-role minus tensor-projection: `+0.00413`
- OOD: `+0.00672`

The simple `W1/W2`, `b1/b2` swap is not a clean causal hit, so exact first/second-layer identity should not be over-interpreted.

## Student scale pattern

The uniform teacher scale is `0.06`. Across the 5 Student seeds, the seed-mean calibration coefficients relative to the teacher LR are:

| tensor role | Student scale / teacher LR |
|---|---:|
| W1 | 0.967 |
| W2 | 0.915 |
| W_skip | 1.571 |
| b1 | 0.298 |
| b2 | 0.553 |

The Student approximately keeps the teacher scale on the main weight matrices, enlarges the residual skip update, and strongly suppresses bias updates.

## Architecture-specific caution

The global-projection failure is mainly a two-layer effect:

| rule | residual IID | two-layer IID |
|---|---:|---:|
| full Student | 0.04346 | 0.08567 |
| tensor projection | 0.03955 | 0.08822 |
| global projection | **0.03886** | 0.12108 |
| uniform NormGrad | 0.04066 | 0.09354 |

A shared dynamic coefficient is already sufficient for the residual architecture in this run but fails badly for the two-layer architecture.

## Strong analytic control: validation-tuned static role scales

The mechanism result above does **not** establish that a learned dynamic policy is necessary. To test the strongest simple alternative, a second experiment tuned static NormGrad scales directly on an independent validation split using deterministic coordinate descent over positive candidate scales.

Verified run:

- Workflow run: `34750732339`
- Artifact: `multitensor-static-role-control`
- Artifact ID: `10315244487`
- Artifact SHA256: `fcb610c4ee84cedd5cb9da87611ed222376f05191ac5b65fa5a326242533a53b`
- Artifact commit: `720b96ff4e20d5b1290a6598a16107c99059dda8`
- Ruff: passed (focused workflow suppresses B023 for a synchronous validation closure)
- Mechanism tests: 19 passed

Two controls were tuned:

1. `tuned_static_shared`: one role-scale dictionary shared across both architectures.
2. `tuned_static_architecture`: a separate static role-scale dictionary per architecture; this is a stronger analytic upper bound because it is explicitly architecture-aware.

### Selected scales

Uniform NormGrad uses `0.06` for every role.

Shared static validation optimum:

| role | scale | relative to uniform teacher |
|---|---:|---:|
| W1 | 0.150 | 2.50x |
| W2 | 0.039 | 0.65x |
| W_skip | 0.048 | 0.80x |
| b1 | 0.012 | 0.20x |
| b2 | 0.012 | 0.20x |

Architecture-aware validation optima:

- two-layer: `W1=0.150`, `W2=0.048`, `b1=0.012`, `b2=0.012`
- residual: `W1=0.120`, `W2=0.012`, `W_skip=0.060`, `b1=0.018`, `b2=0.018`

### Static analytic result

| optimizer | IID mean | OOD mean |
|---|---:|---:|
| uniform NormGrad | 0.06710 | 0.06979 |
| 153p full Student | 0.06456 | 0.06363 |
| Student tensor projection | 0.06389 | 0.06377 |
| **validation-tuned static shared role NormGrad** | **0.05061** | **0.04844** |
| **validation-tuned static architecture-specific NormGrad** | **0.04906** | **0.04740** |

The static analytic controls beat both the original teacher and the distilled Student by a large margin. The stronger architecture-specific control is only modestly better than the shared-role control, so the result is not merely architecture-label memorization.

Architecture-specific IID means for the strongest static control:

- residual: `0.03567`
- two-layer: `0.06245`

For comparison, uniform NormGrad gives `0.04066` and `0.09354`, respectively.

## Revised interpretation

The previous tentative explanation — that the Student beats its teacher because it learned an essential state-dependent tensor-wise allocation policy — is **not supported** by the stronger analytic control.

What is supported is narrower:

1. Direction-only distillation does not simply copy the uniform NormGrad update norm.
2. The 153p Student spontaneously develops non-uniform tensor-wise update magnitudes while learning the teacher's directions.
3. This incidental reallocation is enough to slightly beat the uniform NormGrad teacher.
4. However, a very small validation-tuned static per-role NormGrad rule is substantially better than the Student.

The correct interpretation of the teacher-beating result is therefore:

> Direction-only distillation acts as an imperfect amortizer of layer/tensor learning-rate allocation on this benchmark, rather than discovering a superior dynamic optimizer policy.

This also means the current benchmark is still too easy to justify a larger learned Teacher. The strongest optimizer in this family is now a handful of static role scales plus normalized-gradient directions.

## Next required benchmark refinement

The next task family should make fixed per-role scales insufficient by varying the relative curvature/sensitivity of tensor roles across tasks. A useful stress test is task-level reparameterization or layer-scale randomization while keeping the same role names. The goal is to force the optimal `W1/b1/W2/b2/W_skip` scales to change across tasks.

Then compare:

- uniform NormGrad
- validation-tuned static shared role scales
- architecture-specific static role scales
- 153p distilled Student
- projected Student

If the Student adapts across per-task reparameterizations while static role scales fail, feature-conditioned dynamic allocation becomes a meaningful capability. If static scales remain strongest, the current NormGrad compression result should be presented primarily as compact amortized layer-wise LR tuning, not generic learned-optimizer distillation.
