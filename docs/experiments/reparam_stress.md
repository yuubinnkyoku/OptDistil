# Exact multi-tensor reparameterization stress benchmark

Status: **main run complete** on `experiment/stochastic-distillation`.

Question: is the 153-param Student explained by **fixed tensor-role learning rates**, or does it
truly infer **state-dependent adaptive scales** under an unseen positive diagonal reparameterization
`p_i = s_i * θ_i`?

Student cannot observe `s_i`. Only privileged controls may use `s_i`. Test/OOD information is never
used for hyperparameter tuning.

Artifacts:

- `artifacts/reparam-stress-main.json` (canonical payload)
- `artifacts/reparam-quick-smoke.json` (CI smoke; may be absent locally)

## Protocol

| piece | design |
|---|---|
| Reparameterization | per-tensor positive scalar `s_i`, `p_i = s_i θ_i` |
| Objective | identical function-space loss `L(p)` |
| Gradients | `g_θ = s ⊙ g_p` |
| Train range | `s_i ~ log-uniform[0.5, 2.0]` |
| IID test | same range, unseen seeds |
| OOD | `log-uniform[0.25, 4.0]` |
| Strong OOD | `log-uniform[0.1, 10.0]` |
| Architectures | `two_layer` + `residual` |
| Conditions (train/tune) | 30 / 300 |
| Batch / width / steps (main) | 8 / 8 / 18 |
| Student seeds | 5 (`401000–401004`) |
| Teacher labels for distillation | privileged function-space local NormGrad |
| Student size | 153 parameters (unchanged) |
| Structured controllers | 27 parameters (global / per-tensor) |

Privileged function-space NormGrad on θ:

```
u_θ_priv = u_θ_ordinary / s = -lr * g_θ / (s ||g_θ||)
```

which equals function-space local NormGrad on `p`. Ordinary local NormGrad keeps
`u_θ = -lr * g_θ/||g_θ||`, so function-space step size scales with `s`.

## Methods compared (same batch sequences / tasks)

1. ordinary local NormGrad
2. privileged function-space NormGrad (uses `s`)
3. validation-tuned static shared-role NormGrad (matrix/vector LRs)
4. validation-tuned architecture-specific static-role NormGrad
5. full 153p Student (distilled on privileged trajectories)
6. Student projected onto per-tensor NormGrad direction
7. frozen Student-derived role scales
8. structured tiny global controller (`u_l = -a_l c_t g_l/||g_l||`)
9. structured tiny per-tensor controller (`c_{t,l}` from cheap stats)

## Seed bases

| split | seed base |
|---|---|
| scale validation | 511000 |
| distill train | 521000 |
| role validation / tuning | 531000 |
| IID test | 561000 |
| OOD mild | 571000 |
| OOD strong | 581000 |
| reparam scale seed base | 601000 |
| structured fit seed | 611000 |

## Main results (mean loss ratio; lower is better)

Provenance: commit `7074915` at measurement start; artifact records `git rev-parse HEAD`.
Tuned on validation only: ordinary/privileged lr=0.1; shared roles `{matrix: 0.1, vector: 0.01}`.

| method | IID mean | OOD mild | strong OOD |
|---|---:|---:|---:|
| ordinary local NormGrad | 0.0999 | 0.1503 | 0.2212 |
| privileged function-space NormGrad (uses s) | 0.0872 | 0.0827 | 0.0757 |
| **static shared-role NormGrad (2p)** | **0.0750** | **0.1063** | **0.1672** |
| static architecture-role NormGrad | 0.0682 | 0.0841 | 0.1585 |
| frozen Student-derived role scales | 0.0975 | 0.1528 | 0.1446 |
| structured global (27p) | 0.1287 | 0.1660 | 0.4262 |
| structured per-tensor (27p) | 0.1287 | 0.1660 | 0.4262 |
| 153p Student (mean over 5 seeds) | 0.3224 | 3.745 | 111.85 |
| Student projected (mean over 5 seeds) | 0.2981 | — | — |

Student seed spread on IID is large (0.099–0.980); one seed largely fails to learn. Even the best
student seeds do not beat static shared-role on average, and OOD means collapse.

### Experiment B — static-role generalization ceiling

Static shared-role LRs tuned once on the train-range reparameterization distribution were transferred
without retuning:

| transfer | mean loss ratio |
|---|---:|
| batch 4 | 0.1104 |
| batch 16 | 0.0723 |
| batch 32 | 0.0680 |
| width 12 | 0.0967 |
| width 16 | 0.1110 |
| width 24 | 0.1436 |
| arch residual | 0.0687 |
| arch two_layer | 0.0884 |
| residual-tuned → two_layer | 0.1323 |
| condition OOD 10/100/1000/3000 | 0.0723 |

The fixed rule degrades mildly with noisier batches (4) and larger widths, but remains usable.

### Experiment C — effective Student scale

`effective_scale(t,l)` is the projection of the Student update onto the local NormGrad direction
`d_l = -g_l/||g_l||`.

| split | n | hidden ΔR² | multi-var R² (obs) | hidden Spearman | grad_rms Spearman |
|---|---:|---:|---:|---:|---:|
| IID | 1296 | 0.048 | 0.210 | 0.156 | 0.576 |
| OOD | 972 | 0.009 | 0.372 | 0.223 | 0.247 |

Hidden-scale incremental R² is small and Spearman with `s_i` is weak: the Student is **not** cleanly
recovering `s_i` from the 8-feature interface.

### Experiment D — structured tiny controls

| controller | params | IID mean | OOD mean |
|---|---:|---:|---:|
| static shared-role baseline | 2 | 0.0750 | — |
| structured global | 27 | 0.1287 | 0.1660 |
| structured per-tensor | 27 | 0.1287 | 0.1660 |
| 153p Student | 153 | 0.3224 | 3.745 |

Both structured controllers beat the 153p Student on IID but lose to the 2-parameter static role
rule. Deployment candidate on this family is the static role rule, not a learned controller.

Runtime primitives: NormGrad direction + tiny scalar MLP over cheap stats. State size: role
coefficients + controller weights (no Adam-style per-element second moments beyond existing student
EMA features used only for the learned controllers).

## Interpretation (pre-registered rules)

| claim | status |
|---|---|
| **A. fixed tensor-role LR is sufficient** | **supported** |
| B. dynamic tensor-wise scale adaptation is necessary | not supported |
| C. within-tensor direction adaptation is necessary | not supported |
| D. a <50-param structured optimizer captures the useful behavior | secondary: matches/beats Student, but static-role is stronger |

Primary conclusion:

> Under this exact reparameterization stress benchmark, the 153p Student does **not** outperform a
> validation-tuned static shared-role NormGrad baseline. Therefore we **cannot claim** that the
> Student is an adaptive learned optimizer that infers tensor-wise step allocation from observable
> statistics. The useful behavior is already captured by a fixed matrix/vector role learning-rate
> rule (2 parameters).

This is consistent with the earlier multi-tensor result (153p ≈ NormGrad) **without** requiring
hidden-scale inference: compression of the NormGrad rule is real, but adaptive scale inference is
not the active ingredient under reparameterization OOD.

## What is supported

1. Multi-tensor NormGrad compression remains valid as a rule-compression result.
2. Static shared-role NormGrad is a strong, cheap, and more robust baseline under scale OOD.
3. Privileged function-space NormGrad beats ordinary NormGrad, so reparameterization does distort
   coordinate-wise steps as intended.
4. The 153p Student trained on privileged labels **fails to transfer** to strong reparameterization
   OOD (mean loss ratio can exceed 1).
5. Hidden `s_i` is not strongly recoverable from the current 8-feature student interface.

## What is weakened

1. Any claim that the 153p Student is a state-dependent adaptive optimizer.
2. Deployment priority of the 153p Student over a 2-parameter static role rule on this family.
3. The need for a larger Student to chase adaptive scale inference before introducing larger teachers.

## Recommendation

**Treat fixed tensor-role learning rates as the sufficient explanation. Do not claim adaptive scale
inference. Do not scale Student capacity or introduce large teachers for this question.**

If a future adaptive claim is desired, the interface must expose statistics that actually identify
`s_i` (Experiment C says the current 8 features do not), and the adaptive method must beat static
shared-role on reparameterization OOD.

## Reproduce

```bash
# unit tests
uv run --locked --extra cpu pytest -q tests/test_reparam_stress.py

# quick smoke
uv run --locked --extra cpu python scripts/probe_reparam_stress.py \
  --quick --skip-structured --skip-experiment-b \
  --output artifacts/reparam-quick.json

# main experiment
uv run --locked --extra cpu python scripts/probe_reparam_stress.py \
  --student-seeds 5 --distill-epochs 12 --steps 18 \
  --scale-validation-tasks 3 --distill-train-tasks 4 --role-validation-tasks 3 \
  --iid-test-tasks 4 --ood-test-tasks 3 \
  --output artifacts/reparam-stress-main.json

# report
uv run --locked --extra cpu python scripts/report_reparam_results.py \
  --input artifacts/reparam-stress-main.json
```

## Engineering notes

- Existing single-tensor and non-reparameterized multi-tensor paths are untouched.
- New package pieces: `multitensor/reparam.py`, `static_role.py`, `structured.py`,
  `effective_scale.py`, `methods.py`, `stats_utils.py`.
- Statistics: 5+ seeds, paired task comparison, mean/std/median, bootstrap 95% CI, win fraction,
  IID/OOD separated, finite rollout fraction.
- Commits stay on `experiment/stochastic-distillation` only.
