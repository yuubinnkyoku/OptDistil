# Teacher strength versus fixed-student distillability

This experiment asks whether a stronger optimizer teacher automatically produces a better fixed-size distilled optimizer.

## Controlled setup

- Dense coupled quadratic tasks with Hessian conditions 30 and 300.
- Eight optimization steps.
- Disjoint splits for learned-teacher meta-training, learned-teacher validation, analytic learning-rate validation, distillation training, student validation, and final testing.
- Every distilled student is the same 153-parameter `8 -> 8 -> 8 -> 1` TinyMLP.
- Every student sees the same matrix-aware eight-feature observation.
- Three independent student seeds per teacher source.
- Joint distillation loss: 0.7 direction + 0.3 log-magnitude.

A task-aware Newton teacher is included as a privileged second-order oracle. It knows the coupled-quadratic row and column factors; the student does not. Damped Newton teachers with learning rates 0.25 and 0.5 are distilled. Exact Newton with learning rate 1 is reported only as a teacher ceiling because it converges in one step and makes the remaining trajectory targets nearly zero.

## Quick results

| teacher source | teacher test loss ratio | 153p student test loss ratio |
| --- | ---: | ---: |
| Muon | 0.16200 | 0.04839 +/- 0.00907 |
| norm-gradient | 0.04256 | **0.03434 +/- 0.00249** |
| learned MLP-128 | 0.02058 | 0.03888 +/- 0.00567 |
| Newton, lr=0.25 | 0.01002 | 0.10802 +/- 0.03382 |
| Newton, lr=0.50 | **1.53e-5** | 0.08446 +/- 0.00638 |
| exact Newton ceiling | 0 | not distilled |

The teacher ranking and student ranking are clearly different. The two privileged Newton teachers are dramatically stronger than the other teachers on their own rollouts, yet their distilled students are much worse.

The mean final imitation losses show the same pattern:

| source | final distillation loss |
| --- | ---: |
| norm-gradient | 0.04282 |
| learned MLP-128 | 0.13873 |
| Muon | 0.23906 |
| Newton, lr=0.25 | 0.30486 |
| Newton, lr=0.50 | 0.61153 |

This makes a simple output-scale explanation unlikely. The fixed student has difficulty representing or inferring the oracle update itself.

## Interpretation

The experiment falsifies the naive form of the teacher-scaling hypothesis on the current student interface:

> stronger teacher != better fixed student.

Teacher capability and teacher distillability are separate axes. In particular, a teacher can exploit information that is not recoverable from the student's observation. The Newton oracle knows dense second-order row/column structure, while the current student receives only per-element state and cheap row/column RMS summaries.

This suggests that simply spending more compute on the teacher is not enough. The next experiments should enlarge the *basis of directions available to the fixed student* while keeping the learned network itself tiny.

A promising NPU-friendly direction is to replace some scalar summary features with one-step matrix polynomial features such as

`(G @ G.T) @ G`

and the analogous momentum transform. These require only a small number of dense matrix multiplies, allow cross-coordinate mixing, and keep the TinyMLP parameter count unchanged at 153. This tests whether a small amount of structured deployment compute unlocks knowledge that is otherwise impossible to distill from strong matrix-aware teachers.
