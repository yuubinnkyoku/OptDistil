# Contextual learned-teacher experiments

This note records the first OptDistil experiments that replace a shared per-element MLP teacher with a teacher that can exchange information across the full optimized matrix.

## Setup

The deployment student remains fixed at 153 trainable parameters (`8 -> 8 -> 8 -> 1`) and only receives the existing NPU-friendly matrix-aware features. The teacher is allowed to be heavier.

The learned teachers all start from the same bias-corrected Adam-like base update. Their learned policy heads are zero-initialized, so architecture changes do not change the initial optimizer policy.

The main matched-capacity comparison is:

| teacher | trainable parameters | extra structure |
| --- | ---: | --- |
| `MetaMLPTeacher(hidden_dim=128)` | 18,435 | shared MLP with row/column summary features |
| `MetaAttentionTeacher(d_model=32, depth=2)` | 17,699 | full self-attention across matrix elements |

The existing MLP teacher is not strictly local: its inputs already include row/column gradient RMS and row/column momentum RMS. Therefore this ablation asks whether full token-to-token interaction adds value beyond those cheap structural summaries.

## Frozen-readout nonlinear task: quick probe

Eight-step rollouts, 12 outer iterations, two meta-train and two meta-validation tasks per condition, four held-out test tasks per condition.

### Teacher results

| teacher | test loss ratio |
| --- | ---: |
| MLP-128 | **0.03814** |
| Attention-32x2 | 0.03899 |
| Attention-64x2 | 0.03973 |

Full attention did not improve the teacher on this probe.

### Fixed 153-parameter distilled student

| teacher source | student test loss ratio, mean over 2 seeds |
| --- | ---: |
| MLP-128 | **0.10151** |
| Attention-32x2 | 0.13154 |
| Attention-64x2 | 0.17362 |

The more contextual teachers were harder to compress in this short probe even though the final student architecture and student-visible features were unchanged.

## Dense coupled quadratics: quick probe

The objective is

`0.5 * ||left @ (W - target) @ right||_F^2`

with random dense SPD row/column factors and Hessian conditions 30 and 300. This produces a dense Hessian over `vec(W)`.

With 20 outer iterations and three learned-teacher seeds:

| optimizer | test loss ratio |
| --- | ---: |
| tuned Muon | 0.18078 |
| tuned norm-gradient control | 0.04440 |
| MLP-128 | **0.01589 +/- 0.00057** |
| Attention-32x2 | 0.01597 +/- 0.00017 |

Both learned teachers substantially beat the tuned analytic controls on this task distribution, but static full attention was effectively tied with the cheaper structural MLP.

## Dense coupled quadratics: extended matched-capacity run

The matched-capacity comparison was extended to 60 outer iterations, four train and four validation tasks per condition, eight test tasks per condition, and three teacher seeds.

| teacher | parameters | validation loss ratio | test loss ratio |
| --- | ---: | ---: | ---: |
| MLP-128 | 18,435 | **0.00913 +/- 0.00049** | 0.01325 +/- 0.00055 |
| Attention-32x2 | 17,699 | 0.00960 +/- 0.00048 | **0.01297 +/- 0.00116** |

The attention teacher is about 2.1% better in mean held-out test loss ratio, but the three-seed distributions overlap and validation slightly favors the MLP. This is not enough evidence to claim a real attention advantage.

For reference, the same extended test split gives 0.03879 for the tuned norm-gradient control and 0.19781 for tuned Muon.

## Interpretation

The first contextual experiment changes the working hypothesis:

1. Increasing teacher parameter count alone is not the main missing ingredient.
2. Static full-matrix attention provides little benefit beyond the row/column structural summaries already available to `MetaMLPTeacher` on the current tasks.
3. The next useful capability to add is temporal full-matrix state rather than more static width. Heavy analytical optimizers such as covariance/Kronecker preconditioners derive much of their extra information from statistics accumulated across steps.
4. Distillability remains a separate axis from teacher quality. A teacher can improve its own rollout without producing an update policy that a fixed 153-parameter student can imitate well.

The next experiment therefore adds persistent per-token hidden state before/through full-matrix attention and compares local, static-context, and recurrent-context teachers under a matched outer-training budget.
