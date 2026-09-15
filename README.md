# OptDistil

**Distilling powerful optimizers into tiny learned optimizers.**

OptDistil is an experimental research project for studying whether expensive, high-capacity optimizers can teach much smaller learned optimizers that retain most of their optimization performance while being cheap enough for deployment on constrained hardware such as NPUs.

## Core question

> If the student optimizer is kept tiny and fixed, how much stronger can it become as the teacher optimizer becomes more capable?

The project is designed around three stages:

1. **Teacher generation** — run a strong optimizer or learned optimizer and record optimization trajectories.
2. **Optimizer distillation** — train a tiny student optimizer to imitate useful aspects of the teacher update rule.
3. **Student evaluation / meta-finetuning** — evaluate the distilled optimizer independently and optionally optimize it further against downstream training loss.

## Current milestone

Deterministic quadratic / coupled-quadratic work showed that raw L-BFGS is very strong
when gradients are exact, which made optimizer distillation a weak story on full-batch
tasks. The active milestone therefore moves to **stochastic mini-batch optimization**
on `FrozenReadoutMLPTask`:

- mini-batch gradients with full-data evaluation loss;
- two teacher regimes: AdamW @ batch 32 and normalized gradient @ batch 8;
- fixed 153-parameter matrix-aware student;
- supervised distillation, closed-loop meta-finetuning, and direct-meta baseline;
- condition-number OOD (`10 / 100 / 1000 / 3000`), batch-noise transfer, negative
  controls, and feature ablations.

Preliminary multi-seed findings (details in `docs/experiments/stochastic_distillation.md`):

- Distillation is a much stronger prior than from-scratch closed-loop meta-training
  in both regimes.
- The NormGrad@8 teacher policy is approximately compressible into the 153-parameter
  student; closed-loop meta can slightly improve on the teacher.
- For AdamW@32, a `norm_only` negative control matches the distilled student, so the
  result is weaker than “AdamW-specific knowledge distillation.”
- Raw L-BFGS collapses under mini-batch noise; Muon remains weaker than AdamW/NormGrad
  on this benchmark.

The working claim is therefore closer to **tiny stochastic update-policy learning with
distillation as a strong prior** than to a blanket optimizer-knowledge-distillation
success. Large learned teachers are deferred until a teacher is actually stronger than
these analytic baselines under the same noise protocol.

A follow-up **reparameterization stress benchmark** (`docs/experiments/reparam_stress.md`)
asks whether the 153-param multi-tensor Student is an adaptive scale inferer or whether
fixed tensor-role learning rates already explain the result:

- positive diagonal `p_i = s_i θ_i` with unseen IID/OOD scale ranges;
- Student cannot observe `s_i`; privileged controls can;
- 5 seeds, paired tasks, bootstrap CIs;
- static shared-role NormGrad (2 parameters) **beats** the 153p Student and 27p
  structured controllers under this stress test.

A further **minimum-computation / role-falsification** study
(`docs/experiments/minimum_computation.md`) shows:

- NormGrad + two validation-tuned role LRs is the strongest cheap method on this
  family; uniform NormGrad is close; SGD collapses under reparameterization;
- random balanced two-way tensor partitions match the matrix/vector split, so
  the *labels* are not special;
- under role-aligned reparameterization that inverts required θ-steps, the
  source matrix≫vector ratio fails and re-tuning flips the ratio — the *ratio*
  is a family artifact, not a universal principle.

Primary supported claim: **per-tensor normalized gradient plus a small number of
validation-tuned heterogeneous LRs is sufficient on this synthetic family**.
Do not describe the Student as a state-dependent adaptive learned optimizer, and
do not claim a universal matrix/vector optimizer principle.

## Development with uv

OptDistil uses [uv](https://docs.astral.sh/uv/) for dependency management and reproducible development environments. Python 3.11 is pinned for local development in `.python-version`, while the package itself supports Python 3.10 and newer.

PyTorch is selected explicitly as either a CPU-only or CUDA 13.0 build. The two extras are mutually exclusive.

### CPU

```bash
uv sync --extra cpu
uv run --extra cpu pytest -q
uv run --extra cpu ruff check .
uv run --extra cpu python scripts/smoke_distill.py
```

The CPU extra uses PyTorch's CPU-only wheel index, so CUDA runtime packages are not downloaded.

### NVIDIA GPU / CUDA 13.0

```bash
uv sync --extra cu130
uv run --extra cu130 pytest -q
uv run --extra cu130 python scripts/smoke_distill.py
```

Use `cu130` only on systems with a sufficiently recent NVIDIA driver. The CUDA runtime used by PyTorch comes from the wheel environment; a separately installed CUDA toolkit is not required for ordinary PyTorch execution.

After changing dependencies, refresh the lockfile with:

```bash
uv lock
```

Commit both `pyproject.toml` and `uv.lock` when dependency resolution changes.

## Initial scope

- Teachers: AdamW, Muon, heavier matrix optimizers, and eventually large learned optimizers.
- Students: Celo2-base-like tiny MLP optimizers and NPU-friendly variants.
- Distillation targets: update direction, update magnitude, rollout behavior, and downstream loss.
- Evaluation: loss vs. steps, wall-clock time, optimizer state size, and eventually loss vs. energy.

## Repository layout

```text
OptDistil/
├── configs/              # experiment configuration examples and conventions
├── docs/                 # design notes and research decisions
├── scripts/              # runnable experiments and data-generation entry points
├── src/optdistil/
│   ├── teachers/         # teacher optimizer adapters
│   ├── students/         # tiny learned optimizer architectures
│   ├── distill/          # features, trajectories, objectives, training
│   ├── tasks/            # cheap inner-loop tasks
│   └── metrics/          # optimizer/evaluation metrics
└── tests/                # lightweight correctness tests
```

## Next experiments

1. Treat the minimum-computation ladder + role-permutation/inversion results as
   the current baseline story; do not scale Student capacity on this family.
2. If the research continues, move to a non-planted or real-data family where
   static two-way LRs may fail, with LARS/AdamW role baselines already available.
3. Only introduce a learned teacher if it is empirically stronger than
   NormGrad + role LRs under the same fair-budget protocol.
