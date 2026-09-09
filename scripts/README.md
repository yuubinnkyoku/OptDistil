# Scripts

Thin command-line entry points belong here. Keep reusable implementation under `src/optdistil/`.

Planned entry points:

```text
scripts/
├── generate_trajectories.py   # run teacher optimizers and record updates
├── distill.py                 # train a student from saved teacher trajectories
├── meta_finetune.py           # optional task-loss fine-tuning after imitation
└── benchmark.py               # compare teacher, student, AdamW, Muon, etc.
```

Scripts should remain small wrappers around package APIs so experiments can also be driven from tests or notebooks without duplicating logic.
