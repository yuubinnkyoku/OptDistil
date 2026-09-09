# Architecture

OptDistil separates **teacher behavior**, **student capacity**, and **evaluation tasks** so each can be changed independently.

## Data flow

```text
training task
    │
    ▼
teacher optimizer ──► trajectory records
                         │
                         ▼
                  distillation objective
                         │
                         ▼
                  tiny student optimizer
                         │
                         ▼
                 independent benchmark
```

## Design rules

### Teachers

A teacher may be a conventional optimizer, a matrix optimizer, or a learned optimizer. Teacher code should expose the update information needed for trajectory collection without leaking teacher-specific assumptions into the student implementation.

### Students

Students are deployment targets. Their feature set, state size, parameter count, and allowed operations are explicit parts of the experiment. A particularly important setting is a Celo2-base-like tiny MLP whose size remains fixed while teacher capacity grows.

### Distillation

Distillation should support more than raw update MSE. Initial objectives to compare are:

- update direction similarity;
- update magnitude similarity;
- short rollout agreement;
- downstream task loss after imitation.

### Tasks

Start with tasks small enough to run many times. Scale only after the distillation effect is visible on controlled problems.

### Metrics

Always separate optimization quality from execution cost. At minimum record:

- loss versus optimization step;
- loss versus wall-clock time;
- optimizer state bytes per model parameter;
- student optimizer parameter count.

For NPU experiments, additionally record energy or a device-level proxy when measurement is available.

## First research milestone

Hold the student architecture fixed and vary only teacher capacity. The key experiment is whether a stronger teacher produces a stronger distilled student even though student inference cost remains unchanged.
