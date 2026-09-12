# Stochastic optimizer distillation (FrozenReadoutMLP)

Status: preliminary multi-seed evidence. Student size is fixed at **153 parameters**
(`8 -> 8 -> 8 -> 1` TinyMLPOptimizer, matrix-aware 8 features, no secant/L-BFGS inputs).

Artifacts:

- `artifacts/stochastic-distillation-merged.json` (canonical merged payload)
- `artifacts/stoch-adamw-b32-main.json` (5-seed main path, AdamW@32)
- `artifacts/stoch-normgrad-b8-main.json` (5-seed main path, NormGrad@8)
- `artifacts/stoch-*-controls-ablations.json` (2-seed negative controls / feature ablations)

Commit at measurement time: `a5305f93b1c0ccd940af02116de41a04f56fd8df`.

## Protocol

Task: `FrozenReadoutMLPTask` with mini-batch gradients; full-data loss is the metric.
Train conditions `30 / 300`; test-only OOD conditions `10 / 100 / 1000 / 3000`.

Split seed bases are disjoint:

| split | seed base |
|---|---|
| teacher LR validation | 411000 |
| distillation train | 421000 |
| scale validation | 431000 |
| meta train | 441000 |
| meta validation | 451000 |
| IID test | 461000 |
| OOD test | 471000 |

Teacher LR and student deployment scale are selected **only** on their validation splits.
Closed-loop meta-finetuning observes mini-batch gradients on the student's own trajectory
(stop-gradient observations), backpropagates student update → parameter → full-data loss,
and does **not** differentiate the observed gradient.

Regimes:

1. AdamW teacher, batch 32
2. Normalized-gradient teacher, batch 8

Compared methods per regime: Teacher, distilled joint student, distilled direction-only
student, distilled+closed-loop-meta student, direct-meta student, AdamW, NormGrad,
raw L-BFGS two-scale, and negative controls.

## Main 5-seed results (IID test mean loss ratio L_T/L_0)

### AdamW @ batch 32

| method | test mean | OOD mean | notes |
|---|---:|---:|---|
| AdamW teacher | 0.0297 | 0.0309 | tuned lr=0.06 |
| NormGrad analytic | **0.0254** | 0.0244 | strongest analytic |
| Muon analytic | 0.0868 | 0.0846 | weak |
| raw L-BFGS two-scale | 0.0568 | 0.0117 | median test 0.012, rare blow-ups |
| distill joint | 0.0394 | 0.0456 | cos(teacher)=0.77 |
| distill direction-only | 0.0934 | 0.1097 | worse than joint |
| distill + closed-loop meta | 0.0347 | 0.0393 | beats teacher on 1/5 seeds |
| direct meta (no distill) | 0.8634 | 0.7459 | fails |

Paired meta − distill test delta: mean −0.0047 (meta better on 3/5 seeds).
Paired distill − direct-meta: mean −0.824 (distill better on 5/5 seeds).

### NormGrad @ batch 8

| method | test mean | OOD mean | notes |
|---|---:|---:|---|
| NormGrad teacher | 0.0342 | 0.0309 | tuned lr=0.1 |
| AdamW analytic | 0.0422 | 0.0350 | weaker here |
| Muon analytic | 0.0967 | 0.0873 | weak |
| raw L-BFGS two-scale | 1.5475 | 0.0120 | collapses on IID test |
| distill joint | 0.0345 | 0.0361 | ≈ teacher |
| distill direction-only | 0.0383 | 0.0338 | similar |
| distill + closed-loop meta | 0.0340 | 0.0359 | beats teacher on 3/5 seeds |
| direct meta (no distill) | 0.8785 | 0.7576 | fails |

Paired meta − distill test delta: mean −0.0005 (meta better on 3/5 seeds).
Paired distill − direct-meta: mean −0.844 (distill better on 5/5 seeds).

## What is supported

1. **Distillation is a strong prior.** In both regimes, distilled students massively
   outperform direct-meta students with the same 153-parameter architecture and the same
   closed-loop objective. This is the cleanest positive result.
2. **NormGrad policy is compressible.** At batch 8, the distilled student matches the
   NormGrad teacher, retains OOD performance, and closed-loop meta can edge past the
   teacher. This is partial success for “tiny student compresses a strong stochastic
   update policy.”
3. **Teacher-imitation geometry is real for NormGrad.** Student–teacher cosine ≈ 0.98,
   and the student is also nearly aligned with −grad (as expected for NormGrad).
4. **Raw L-BFGS is not a competitive stochastic baseline** under these mini-batch noise
   levels, especially at batch 8.
5. **Muon remains weaker** than AdamW/NormGrad here; large Muon-scale compute is not
   justified yet.

## What is rejected or weakened

1. **“Student distills AdamW-specific knowledge.”** For AdamW@32, the `norm_only`
   negative control (replace teacher direction with −grad, keep teacher norm) reaches
   test mean **0.0290**, essentially matching the distilled joint student (0.0394) and
   the AdamW teacher (0.0297). Distillation is therefore not uniquely encoding AdamW
   direction structure beyond a well-scaled normalized descent.
2. **Strong success criterion is not met for AdamW@32.** The strongest analytic baseline
   (NormGrad 0.0254) beats every learned student in that regime. Distilled+meta reaches
   0.0347, still worse than NormGrad.
3. **Teacher-shuffled / coordinate-permuted / random-direction labels** degrade students
   (as expected). Those controls do **not** match the distilled student for NormGrad,
   so the NormGrad result is not explained by arbitrary label noise.
4. **Feature capacity is not the first bottleneck.** Matrix-aware / elementwise /
   no-progress students are close. `no_ema` is competitive or better in these short
   horizons. Gram cubic / hybrid Gram are not clearly better. Increasing student size is
   not the immediate next step.

## Batch-noise transfer matrix

AdamW@32-trained joint student (seed 401000), IID test mean:

| eval batch | 4 | 8 | 16 | 32 (train) | 64 |
|---|---:|---:|---:|---:|---:|
| loss ratio | 0.123 | 0.058 | 0.046 | 0.037 | 0.036 |

NormGrad@8-trained joint student:

| eval batch | 4 | 8 (train) | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| loss ratio | 0.050 | 0.035 | 0.029 | 0.030 | 0.031 |

Interpretation: students are not a single brittle fixed step that only works at the
training batch. Larger batches (less noise) remain usable; smaller/noisier batches than
training degrade performance. Some stochastic-robustness policy is present, but it does
not fully transfer to substantially noisier gradients.

## Negative controls (test mean; 2 seeds, reduced task counts)

AdamW@32:

| control | test | vs distill joint 0.0394 |
|---|---:|---|
| shuffle_tasks | 0.749 | much worse |
| permute_coords | 0.169 | worse |
| norm_only | **0.0290** | matches/beats |
| random_unit | 2.322 | much worse |
| analytic_baseline_trajectory (NormGrad labels) | 0.862 | much worse |

NormGrad@8:

| control | test | vs distill joint 0.0345 |
|---|---:|---|
| shuffle_tasks | 0.176 | worse |
| permute_coords | 0.591 | worse |
| norm_only | 0.0401 | slightly worse |
| random_unit | 0.284 | worse |
| analytic_baseline_trajectory (AdamW labels) | 0.0600 | worse |

## Feature ablations (test mean; 2 seeds, reduced task counts)

AdamW@32: matrix_aware 0.040, elementwise 0.042, no_progress 0.043, hybrid_gram 0.049,
gram_cubic 0.061, no_ema 0.029.

NormGrad@8: no_progress 0.037, elementwise 0.041, gram_cubic 0.048, matrix_aware 0.061,
no_ema 0.071, hybrid_gram 0.095.

These ablations were run with fewer tasks/seeds than the main path, so they are diagnostic
rather than definitive rankings.

## Claim revision

The evidence does **not** support the strongest form of “optimizer knowledge distillation”
for AdamW on this stochastic benchmark. A more accurate working claim is:

> A 153-parameter student can be trained to a competitive tiny stochastic update policy
> when supervised on a strong analytic teacher (especially NormGrad under mini-batch
> noise). Distillation is a far better prior than from-scratch closed-loop meta-training.
> For AdamW@32, much of the recoverable signal is well-scaled normalized descent rather
> than AdamW-specific direction knowledge.

## Is a large learned teacher worth it next?

**Not yet as the primary path.**

Evidence against immediately scaling the teacher:

- No analytic teacher in this suite is clearly stronger than NormGrad after noise.
- Muon and raw L-BFGS are weaker, so “more teacher capacity” has no demonstrated ceiling
  to chase on this task family.
- The AdamW student gap is explained at least partly by a simple `norm_only` control,
  which a larger teacher would not automatically fix.
- Direct-meta failure shows the bottleneck is the learning signal / prior, not only
  teacher expressivity.

Evidence for revisiting later:

- Distillation clearly beats direct-meta, so a stronger *and* actually better stochastic
  teacher could transfer.
- NormGrad distillation already reaches teacher parity, so the compression pipeline works
  when the teacher policy is the right target.

Recommended next steps before large learned teachers:

1. Treat NormGrad@small-batch as the primary compression target; report that result.
2. Decide whether AdamW is worth pursuing on this task family, or move to a family where
   AdamW/Muon genuinely dominate NormGrad.
3. Increase horizon / harder noise schedules to separate fixed-scale policies from true
   stochastic robustness.
4. Only then introduce a large learned teacher that is empirically stronger than the
   analytic baselines under the same noise protocol.
