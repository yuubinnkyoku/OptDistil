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

The first end-to-end path is implemented:

- functional AdamW teacher;
- reference Muon teacher with Newton-Schulz orthogonalization;
- teacher-independent trajectory features;
- serializable trajectory records/datasets;
- a Celo2-base-like `8 -> 8 -> 8 -> 1` student (153 parameters);
- direction + magnitude distillation losses;
- a deterministic quadratic smoke task;
- CI covering lint, unit tests, and end-to-end smoke distillation.

The current student observation is deliberately small and teacher-independent:

`gradient, momentum, RMS, parameter, gradient sign, log|gradient|, parameter RMS, training progress`.

This makes teacher-size scaling experiments meaningful: a larger teacher is not allowed to secretly give the student more information.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/smoke_distill.py
```

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

1. Compare direct training of the 153-parameter student against teacher-distilled initialization.
2. Sweep teacher capacity while keeping the student architecture and observations fixed.
3. Add Muon-generated trajectories and compare single-teacher vs. mixed-teacher distillation.
4. Add rollout loss and downstream meta-finetuning.
5. Introduce NPU-native student observations and measure loss per wall-clock time / joule.
