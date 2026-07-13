# AIME 2026 Q10 Target-Only Diagnostic Run

## Status

This run is incomplete and is not the requested production DFlash evaluation.
It used the BF16 target model with FA3, TP1, DP8, and `dflash: false`. The run
was stopped during the final initial-proof solution continuation after that
configuration mismatch was identified. Verification, refinement, selection,
and external grading did not run.

- Run ID: `aime-2026-q10-full-20260713`
- Code commit: `241b2b775752d22b94ffd5279216689990e4a58d`
- Dataset: `MathArena/aime_2026`, problem 10
- Gold answer: `156`
- Server: BF16 target-only, FA3, TP1 x DP8
- Manifest terminal state: `failed`, `CancelledError()`
- All eight H200s were active during the 32-request generation batch.

## Persisted Results

The client persisted 31 of 32 initial proof calls. The last call was cancelled
during its forced solution continuation and therefore has no call record.

| Metric | Value |
| --- | ---: |
| Persisted proofs | 31 |
| Valid generation XML | 31 |
| Failed calls | 0 |
| Natural completions | 26 |
| Persisted forced continuations | 5 |
| Additional forced continuation interrupted | 1 |
| Exact `156` answers | 16 (51.6%) |
| Mean completion tokens | 47,688.7 |
| Min / max completion tokens | 19,108 / 75,335 |
| Mean call latency | 1,189.8 seconds |

Boxed-answer distribution:

| Answer | Count |
| --- | ---: |
| 156 | 16 |
| 21 | 9 |
| 103 | 2 |
| 134 | 1 |
| 151 | 1 |
| 155 | 1 |
| 49 | 1 |

Among the five persisted forced continuations, all five produced valid XML,
but only one produced the exact gold answer. The five answers were `155`, `21`,
`156`, `21`, and `21`.

## Red Flags

### Corrupted or ambiguous problem notation

The exact prover prompt contains:

```text
so that ${}\overline{AC}$ is perpendicular $\overline{BC}$
```

That condition is impossible for the unchanged 13-14-15 triangle. The answer
split is strongly associated with two repairs invented by the model:

- Gold-consistent proofs interpret it as `A'C'` perpendicular to the original
  `BC`; these commonly obtain `156`.
- Many incorrect proofs interpret it as original `AC` perpendicular to rotated
  `B'C'`; this commonly obtains `21` and also produced `134` and `155` variants.

The source problem notation should be verified before treating model accuracy
or verifier rankings from this prompt as meaningful.

### Excessive reasoning length

Even the shortest persisted call used 19,108 completion tokens for a coordinate
geometry problem. The mean was 47,689 tokens. Several reasoning traces repeat
the notation-interpretation discussion many times before giving a short final
proof. This is a latency and cost concern independent of answer correctness.

### Overconfident self-evaluation

Incorrect proofs frequently emit `<score>1</score>` and claim that no gaps
remain. Representative examples include:

- `r01-p0005`: uses the non-gold `AC` perpendicular `B'C'` interpretation,
  concludes `21`, and asserts that the alternative rotation has the same area.
- `r01-p0022`: uses the same interpretation, concludes `134`, and claims that
  polygon simplicity was verified without showing that verification.
- `r01-p0014`: a forced continuation concludes `155` after inconsistent angle
  and coordinate arithmetic, then rates itself fully correct.

The generated self-score is not a reliable correctness signal.

## Artifact Map

- `generation/problems/10/calls.jsonl`: 31 complete proof records, including
  hidden reasoning, final XML, token usage, continuations, and latency.
- `generation/problems/10/prompts/*.json`: exact prompt that exposed the
  notation issue.
- `run_manifest.json`: authoritative target-only configuration and cancelled
  terminal state.
- `server_validation.json`: validated FA3 TP1 x DP8 target-only server state.
- `server-live.log`: DP0-DP7 activity and continuation transitions.
- `input/dataset_metadata.json`: source dataset revision and hashes.

No round summary, verifier result, refinement, selected proof, or grader result
exists because the run was stopped before those phases.
