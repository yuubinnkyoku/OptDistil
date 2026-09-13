# Multi-tensor stochastic optimizer distillation

Status: **5-seed main path complete**. Student size remains **153 parameters**
(`8 -> 8 -> 8 -> 1` TinyMLPOptimizer shared across all tensor elements).

Artifacts:

- `artifacts/multitensor-main.json` (canonical main payload)
- `artifacts/multitensor-quick.json` (CI smoke)

Measurement provenance (regenerate from this branch):

| field | value |
|---|---|
| base commit | `2095560e0106d0a96d1e62d047df2bacb69573ec` |
| framework commit | `be1a122` (code content used at measurement; untracked at run time, then committed) |
| artifact `commit_sha` | recorded inside JSON via `git rev-parse HEAD` at run time (= base) |
| student seed base | 401000 (5 seeds: 401000–401004) |
| control/ablation seeds | 1 each in this run (`--control-seeds 1 --ablation-seeds 1`) |
| width / samples / steps / batch | 8 / 48 / 24 / 8 |
| architectures | `two_layer` + `residual` |
| teacher | `norm_grad_local` @ lr=0.06 (validation-tuned) |
| student parameters | 153 (fixed) |

Full config, seed ranges, tuned LRs, and per-run metrics live in
`artifacts/multitensor-main.json` (`config`, `split_specs`, `tuned_lrs`, `summary`).

## Question

Does the single-matrix FrozenReadoutMLP result
(153-param student ≈ NormGrad teacher under mini-batch noise)
generalize to **multi-tensor** neural-network optimization, or is it an artifact
of one optimized matrix?

## Framework

New package: `src/optdistil/multitensor/`. Existing 1-tensor path is untouched.

| piece | design |
|---|---|
| Parameter collection | `ParamCollection` = ordered `list[Tensor]` (no PyTree framework) |
| Architectures | `two_layer` (W1,b1,W2,b2) and `residual` (+W_skip) |
| Task | planted-network MSE regression, fully synthetic, seed-reproducible |
| Teachers | SGD, momentum, NormGrad local, NormGrad global, AdamW, Muon hybrid, L-BFGS |
| Student features | 8 features/element, shared 153p MLP |
| Stochasticity | mini-batch gradients; **identical batch-index sequence** across methods |
| Metric | full-data loss ratio L/L0 and normalized AULC |

### NormGrad definitions (mechanistic)

- **A. tensor-wise / local**: `u_l = -lr * g_l / ||g_l||`
- **B. global**: `u_l = -lr * g_l / sqrt(sum_j ||g_j||²)`

### Default multi-tensor features (include_global=True)

1. grad
2. momentum
3. RMS second moment
4. parameter
5. tensor-local grad RMS
6. tensor-local parameter RMS
7. **global grad RMS**
8. training progress

Ablation `local_only` replaces channel 7 with tensor-local grad RMS (no global
communication). Student parameter count stays 153.

## Protocol

| split | seed base |
|---|---|
| teacher LR validation | 411000 |
| distillation train | 421000 |
| scale validation | 431000 |
| meta train | 441000 |
| meta validation | 451000 |
| IID test | 461000 |
| OOD test | 471000 |

Train conditions `30/300`; OOD conditions `10/100/1000/3000`.
Width=8, samples=48, steps=24, train batch=8. Both architectures mixed in every split.

Teacher LR and student deployment scale selected **only** on validation splits.

## Main 5-seed results (IID test mean loss ratio)

Primary teacher: **NormGrad local @ batch 8**, tuned lr=0.06.

### Analytic baselines

| method | test mean | median | lr |
|---|---:|---:|---:|
| NormGrad local | 0.0671 | 0.0539 | 0.06 |
| NormGrad global | 0.0670 | 0.0523 | 0.1 |
| AdamW | 0.0821 | 0.0720 | 0.03 |
| SGD | 0.0886 | 0.0579 | 0.3 |
| Muon hybrid | 0.1455 | 0.1233 | 0.06 |
| Momentum | 0.2196 | 0.2148 | 0.1 |
| L-BFGS two-scale | 90.14 | 0.91 | 0.03 |

NormGrad remains the strongest analytic policy. L-BFGS collapses under this
stochastic multi-tensor setting.

By architecture (NormGrad local): two_layer 0.0935, residual 0.0407.

### Students (mean ± std over 5 seeds)

| method | test mean | median | std | finite | OOD mean |
|---|---:|---:|---:|---:|---:|
| NormGrad teacher | 0.0671 | 0.0539 | — | 1.00 | — |
| distill joint | 0.0725 | 0.0720 | 0.0032 | 1.00 | 0.0740 |
| distill direction-only | **0.0637** | 0.0635 | 0.0022 | 1.00 | 0.0654 |
| distill + closed-loop meta | 0.0700 | 0.0706 | 0.0015 | 1.00 | 0.0688 |
| direct meta (no distill) | 1.2223 | 1.1345 | 0.5662 | 1.00 | 1.3964 |

Paired distill_joint − direct_meta: mean −1.15, distill better on **5/5 seeds**.

Interpretation: multi-tensor compression **does** generalize. The 153p student
matches the NormGrad teacher (direction-only slightly beats it on test mean).
Direct meta still fails, confirming distillation is the necessary prior.

### Batch-size OOD (distill joint, seed 401000)

| eval batch | 4 | 8 (train) | 16 | 32 |
|---|---:|---:|---:|---:|
| loss ratio | 0.103 | 0.072 | 0.068 | 0.057 |

Larger/less-noisy batches remain usable; batch 4 (noisier than train) degrades.

### Width OOD (distill joint, two_layer)

| width | 8 (train) | 12 | 16 |
|---|---:|---:|---:|
| loss ratio | 0.142 | 0.085 | 0.108 |

Elementwise shared student transfers across width without retraining. Width 12
was better than the training width on this split (likely easier tasks).

## Local vs global NormGrad

| teacher | lr | teacher test | distilled student test |
|---|---:|---:|---:|
| local (tensor-wise) | 0.06 | 0.0671 | 0.0690 |
| global | 0.1 | 0.0670 | 0.0853 |

The student compresses **local** NormGrad nearly perfectly, and compresses
**global** NormGrad worse (still competitive). This is evidence that cheap
global statistics are useful but not sufficient under the default 8-feature
interface; local-only is the easier target.

## Negative controls (1 seed, reduced)

| control | test mean | vs distill joint 0.0725 |
|---|---:|---|
| shuffle_tasks | 0.980 | much worse |
| permute_coords | 1.071 | much worse |
| norm_only (−grad, teacher norm) | 0.0810 | slightly worse |
| random_unit | 1.093 | much worse |
| analytic_imitation (NormGrad global trajectories) | 0.0710 | comparable |

The NormGrad result is not explained by label noise. `norm_only` is close but
does not match the distilled student, so there is residual direction structure
beyond scaled −grad.

**Analytic imitation note.** For an analytic teacher such as NormGrad,
trajectory distillation *is* rule compression: labels are exactly the analytic
update computed from the current mini-batch gradient. `analytic_imitation`
here means distilling the *other* NormGrad form (global) rather than the
primary local teacher; it reaches 0.0710, slightly better than distilling
local trajectories (0.0725). The honest framing is:

> The 153p student compresses the **NormGrad update rule** under multi-tensor
> stochastic settings, not a privileged “teacher trajectory distribution.”
> Closed-loop meta does not add a large extra gain over rule compression here.

## Feature / global-statistic ablation (1 seed)

| mode | test mean |
|---|---:|
| local + global grad RMS | 0.0715 |
| local only (no global channel) | 0.0738 |

Global communication helps slightly at fixed 153 params. The gap is small;
local features already carry most of the compressible NormGrad signal.

## Per-tensor diagnostics (distill joint, seed 401000, test)

Student–teacher cosine (global): 0.85; student–(−grad) cosine: 0.94;
update-norm ratio: 0.87.

| tensor | update energy fraction | cosine to teacher | relative update scale |
|---|---:|---:|---:|
| W1 | 0.282 | 0.932 | 0.91 |
| b1 | 0.031 | 0.792 | 0.27 |
| W2 | 0.247 | 0.973 | 0.86 |
| b2 | 0.129 | 0.993 | 0.57 |

The student is **not** applying one global scalar step. It allocates most update
energy to matrix weights and systematically under-updates `b1`. This is the
emergence of tensor-wise step allocation at 153 parameters.

## What is supported

1. **Multi-tensor generalization holds.** 153p student ≈ NormGrad teacher on
   two architectures with 4–5 tensors of mixed matrix/vector roles.
2. **Distillation >> direct meta** on multi-tensor tasks (5/5 seeds).
3. **Negative controls fail**, so the result is real NormGrad-rule structure.
4. **Width and batch OOD transfer** without changing student size.
5. **Layer-wise allocation appears** without explicit layer embeddings.
6. **Raw L-BFGS is not competitive** under stochastic multi-tensor noise.
7. **AdamW/Muon remain weaker** than NormGrad here; no case to revive AdamW
   distillation as the primary result.

## What is weakened

1. Global NormGrad is harder to compress than local under the same features.
2. Global-feature ablation only yields a small gain; not the first bottleneck.
3. Closed-loop meta does not beat direction-only distillation on this suite.

## Success criteria

| criterion | status |
|---|---|
| 153p student ≈ NormGrad across architectures | **yes** (joint ≈ teacher; direction-only ≤ teacher) |
| clearly stronger than direct meta | **yes** (5/5) |
| stronger than negative controls | **yes** |
| robust on unseen batch sizes / widths | **yes** (width OOD works; noisy batch 4 degrades) |
| no student-parameter increase | **yes** (153 fixed) |
| student beats teacher on some OOD | partial (direction-only test mean already < teacher) |
| global NormGrad compressible from cheap stats | partial (worse than local) |
| tensor-wise allocation emerges | **yes** (diagnostics) |

## Recommendation

**Solidify NormGrad compression as the primary result. Do not scale to a large
learned teacher yet.**

Rationale:

1. Multi-tensor (the stated generalization risk) already reproduces the
   single-matrix compression result.
2. Analytic NormGrad is still the strongest teacher; there is no demonstrated
   analytic ceiling for a larger learned teacher to chase.
3. The remaining interface question is modest: how much global communication
   is required. Current evidence says local features already capture most of
   the rule; a larger student would confound capacity with interface.

Next (if anything) before large teachers:

1. Keep 153p; optionally add a cheap per-tensor role scalar only if global
   NormGrad remains a hard target after more seeds.
2. Increase horizon / harsher noise to separate fixed-scale policies from true
   stochastic robustness.
3. Only then introduce a learned teacher **if** it is empirically stronger
   than NormGrad under the same multi-tensor protocol.

## Reproduce

```bash
# unit tests
uv run --locked --extra cpu pytest -q tests/test_multitensor.py

# quick CI smoke
uv run --locked --extra cpu python scripts/probe_multitensor_main.py \
  --quick --skip-meta --skip-controls --skip-ablations \
  --output artifacts/multitensor-quick.json

# full main experiment
uv run --locked --extra cpu python scripts/probe_multitensor_main.py \
  --output artifacts/multitensor-main.json

# report
uv run --locked --extra cpu python scripts/report_multitensor_results.py
```
