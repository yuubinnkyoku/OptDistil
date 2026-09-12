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

1. Treat NormGrad under mini-batch noise as the primary compression target and harden
   that claim with longer horizons.
2. Decide whether AdamW is still a useful teacher on this task family, or move to a
   family where AdamW/Muon genuinely dominate normalized gradient.
3. Separate fixed step-size policies from true stochastic robustness with stronger
   batch-noise transfer tests.
4. Only then introduce large learned teachers that are empirically stronger than
   AdamW/NormGrad/Muon under the same protocol.
5. Continue NPU-oriented student observations and cost measurements.
