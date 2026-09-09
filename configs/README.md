# Configs

Keep experiment configuration separate from implementation code.

Suggested naming:

```text
configs/
├── teacher/      # teacher optimizer and trajectory-generation settings
├── student/      # student architecture and feature settings
├── distill/      # imitation / rollout loss settings
└── benchmark/    # evaluation task settings
```

Do not commit machine-specific paths, credentials, or large generated artifacts here.

The first experiment should stay intentionally small: one tiny task, one teacher, one fixed-size student, and one distillation objective. Add configuration complexity only when an experiment actually needs it.
