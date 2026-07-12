# Basic pilot stopped checkpoint

Status: **stopped by user request on 2026-07-12 UTC**

All evaluator and inference services were stopped, and both H200s were released.
This run must not be interpreted as a four-problem score because only two
problems reached the mandatory selector-produced final-proof gate.

## Complete and gradeable

| Problem | Final source | Selector vote | DeepSeek v4 Flash passes |
|---|---|---:|---|
| PB-Basic-001 | `select:P1(5/5)` | 5/5 | 7, 7 |
| PB-Basic-002 | `select:P2(5/5)` | 5/5 | 7, 7 |

These are valid completed results. Their full traces, selected proofs, raw
two-pass grader responses, and interim aggregate are committed in this run.

## Partial and not gradeable

| Problem | GPU | Last event time | Prove events | Verify events | Refine events | Final selector result |
|---|---:|---:|---:|---:|---:|---|
| PB-Basic-003 | 0 | 2,299.4 s | 8 | 37 | 7 | none |
| PB-Basic-004 | 1 | 2,080.8 s | 4 | 6 | 1 | none |

The append-only event streams for these partial problems are preserved for
audit. Neither problem has a stage JSON, selected proof, final source, or
DeepSeek grade. They must be restarted from the beginning if evaluated later;
partial calls are not promoted to fallback answers.

An earlier serial PB-Basic-001 attempt was also interrupted before selection.
Its partial event stream remains under `generation/basic/raw/` and is not part
of the completed PB-Basic-001 result from the clean two-GPU run.
