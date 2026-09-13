# Multi-tensor NormGrad allocation mechanism

This experiment asks why the 153-parameter direction-only Student can slightly outperform its tensor-wise NormGrad teacher on the stochastic multi-tensor benchmark.

## Verified run

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

All allocation calibration uses an independent split. IID and OOD test tasks are not used to choose scales.

## Causal update interventions

For each Student update, the following interventions are evaluated in closed loop.

- `full_student`: original Student update.
- `tensor_projection`: project each tensor's Student update onto that tensor's unit negative-gradient direction. This removes within-tensor directional residuals while preserving the Student's dynamic signed scale for every tensor.
- `global_projection`: project the entire multi-tensor Student update onto one shared scale multiplying all tensor-local negative-gradient directions. This removes tensor-wise allocation as well as within-tensor directional residuals.
- `frozen_role`: replace the dynamic Student coefficients by calibration-split median coefficients for each tensor name.
- `equalized_frozen`: replace all frozen role scales by one median scale.
- `swapped_frozen`: swap `W1/W2` and `b1/b2` frozen role scales.
- `teacher_norm`: restore the original local NormGrad teacher, using LR `0.06` for every tensor.

Lower loss ratio is better.

## Main result

| update rule | IID mean | IID std | OOD mean | OOD std |
|---|---:|---:|---:|---:|
| full Student | 0.06456 | 0.00244 | **0.06363** | 0.00146 |
| tensor projection | **0.06389** | 0.00261 | 0.06377 | 0.00219 |
| global projection | 0.07997 | 0.00914 | 0.08123 | 0.00835 |
| frozen role scales | 0.06802 | 0.00402 | 0.07049 | 0.00412 |
| equalized frozen scales | 0.07195 | 0.00506 | 0.07650 | 0.00490 |
| swapped frozen scales | 0.06725 | 0.00383 | 0.07101 | 0.00418 |
| NormGrad teacher | 0.06710 | 0.00000 | 0.06979 | 0.00000 |

The tensor projection is better than the full Student on IID in all 5 Student seeds, with a mean paired difference of `-0.000678`. On OOD, the two are effectively tied: mean paired difference `+0.000147`.

Removing tensor-wise allocation with global projection is substantially worse than tensor projection in every seed:

- IID: `+0.01608` mean loss-ratio difference, 0/5 wins for global projection.
- OOD: `+0.01745`, 0/5 wins.

Freezing the Student's per-role scales also loses performance in every seed:

- IID frozen-role minus dynamic tensor projection: `+0.00413`, 0/5 wins.
- OOD: `+0.00672`, 0/5 wins.

Equalizing the frozen role scales loses another `+0.00393` IID and `+0.00601` OOD on average.

The simple `W1/W2`, `b1/b2` swap is not a clean causal hit: it slightly improves over the original frozen-role assignment on IID (`-0.00077`) and is only slightly worse on OOD (`+0.00052`). Therefore the evidence supports scale heterogeneity, but not a strong claim that those exact first/second-layer role identities are individually causal.

## Learned scale pattern

The teacher uses scale `0.06` for every tensor. Across the 5 Student seeds, the median calibration coefficients relative to teacher LR have the following seed-mean pattern:

| tensor role | Student scale / teacher LR |
|---|---:|
| W1 | 0.967 |
| W2 | 0.915 |
| W_skip | 1.571 |
| b1 | 0.298 |
| b2 | 0.553 |

The pattern is highly non-uniform. The Student approximately preserves the teacher scale on the main weight matrices, increases the residual skip-matrix update, and strongly suppresses bias updates, especially `b1`.

Representative per-seed relative scales are stable in ordering even though their absolute magnitude changes with Student seed. For example, `b1` is only about `0.24--0.43x` the teacher scale, while `W_skip` is about `1.28--2.13x`.

## Architecture-specific caution

The overall global-projection failure is driven primarily by the two-layer architecture. Averaged over Student seeds:

| rule | residual IID | two-layer IID |
|---|---:|---:|
| full Student | 0.04346 | 0.08567 |
| tensor projection | 0.03955 | 0.08822 |
| global projection | **0.03886** | 0.12108 |
| NormGrad teacher | 0.04066 | 0.09354 |

Thus a shared global coefficient is already sufficient for the residual architecture in this run, but fails badly for the two-layer architecture. The mechanism should be described as task/architecture-dependent tensor-wise allocation rather than a universal requirement for every architecture.

## Interpretation

The strongest conclusion supported by this experiment is:

> The Student's small advantage over local NormGrad does not require within-tensor directional rotation. It is explained primarily by allowing state-dependent, tensor-specific update magnitudes on otherwise ordinary negative-gradient directions.

The experiment does **not** yet prove that a learned dynamic policy is necessary. `frozen_role` uses Student-derived median coefficients rather than validation-optimized analytic per-role learning rates. A stronger analytic control must therefore tune per-role NormGrad scales directly on an independent validation split before claiming that dynamic allocation itself is essential.

The next required control is validation-only static per-role NormGrad scale tuning. If it reaches the projected Student, the correct interpretation is that distillation amortizes layer-wise learning-rate allocation. If it remains behind the dynamic projection, the next step is to fit cheap analytic state-dependent scale rules from gradient/parameter/global statistics before increasing Student or Teacher capacity.
