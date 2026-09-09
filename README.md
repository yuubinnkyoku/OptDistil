# OptDistil

**Distilling powerful optimizers into tiny learned optimizers.**

OptDistil is an experimental research project for studying whether expensive, high-capacity optimizers can teach much smaller learned optimizers that retain most of their optimization performance while being cheap enough for deployment on constrained hardware such as NPUs.

## Core question

> If the student optimizer is kept tiny and fixed, how much stronger can it become as the teacher optimizer becomes more capable?

The project is designed around three stages:

1. **Teacher generation** — run a strong optimizer or learned optimizer and record optimization trajectories.
2. **Optimizer distillation** — train a tiny student optimizer to imitate useful aspects of the teacher update rule.
3. **Student evaluation / meta-finetuning** — evaluate the distilled optimizer independently and optionally optimize it further against downstream training loss.

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
├── scripts/              # entry points for data generation, distillation, evaluation
├── src/optdistil/
│   ├── teachers/         # teacher optimizer adapters
│   ├── students/         # tiny learned optimizer architectures
│   ├── distill/          # trajectory datasets and distillation objectives
│   ├── tasks/            # inner-loop training tasks
│   └── metrics/          # optimizer/evaluation metrics
└── tests/                # lightweight correctness tests
```

## Status

Early research scaffold. The first milestone is a minimal teacher → trajectory → student pipeline on a tiny model before adding expensive teachers or device-specific backends.
