# IMO 2025 PR17 search with 128K generation length

This directory records the complete outcome of the six-problem IMO 2025 evaluation
completed on 2026-07-15. The search used the PR17 production settings, except that
`search.max_completion_tokens` was raised from 65,536 to 128,000. Problems were run
sequentially on one 8x H200 node. Problem 4 was completed immediately before the
remaining five problems with the same frozen search settings and is included here as
part of the six-problem result.

The source branch was based on commit
`f3d0be2b5fbeef4e546c3b36b1de23faa9497da1` from
`bogoconic1/aimo-proof-pilot-inference`.

## Configuration

Server and model:

- BF16 target: `/workspace/models/opd-32b-deploy`
- BF16 draft: `/workspace/models/dflash-32b-draft-v2test-phaseL`
- TP2 x DP4 on 8x H200
- FA3, page size 1, deterministic inference
- DFlash enabled
- Context length: 262,144
- Maximum running requests: 64 per DP replica
- Client concurrency: 96

Search:

- 32 proof attempts per round
- 16 verifier calls per valid proof
- Top 8 parents, 4 refinements per parent, 4 analyses per refinement
- Up to 4 rounds, early-stop threshold 0.99999
- Temperature 1.0, top-p 0.95
- Main generation limit: 128,000 tokens
- Solution and verifier continuation limits: 16,384 tokens
- Minimum valid verifications: 4

The exact configuration is preserved in [search-config.yaml](search-config.yaml).

## Search results

| Problem | Selected proof | Rounds | Internal mean | Valid candidates | Disqualified | Verifier calls |
|---|---|---:|---:|---:|---:|---:|
| P1 | `r03-p0023` | 3 | 1.00000 | 90/96 | 6 | 1,440 |
| P2 | `r02-p0003` | 2 | 1.00000 | 64/64 | 0 | 1,024 |
| P3 | `r04-p0003` | 4 | 0.65625 | 109/128 | 19 | 1,744 |
| P4 | `r04-p0010` | 4 | 1.00000 | 125/128 | 3 | 2,000 |
| P5 | `r03-p0028` | 3 | 1.00000 | 95/96 | 1 | 1,520 |
| P6 | `r01-p0012` | 1 | 1.00000 | 32/32 | 0 | 512 |
| **Total** | | **17** | | **515/544** | **29** | **8,240** |

Candidates without valid solution XML were disqualified and were not sent to the
verifier. All 8,240 attempted verifier calls produced valid verifier outputs.

### Per-round progression

| Problem | Round | Valid/attempted | Best proof | Internal mean | Verifiers | Early stop |
|---|---:|---:|---|---:|---:|---|
| P1 | 1 | 27/32 | `r01-p0018` | 0.21875 | 432 | No |
| P1 | 2 | 31/32 | `r02-p0004` | 0.93750 | 496 | No |
| P1 | 3 | 32/32 | `r03-p0023` | 1.00000 | 512 | Yes |
| P2 | 1 | 32/32 | `r01-p0007` | 0.84375 | 512 | No |
| P2 | 2 | 32/32 | `r02-p0003` | 1.00000 | 512 | Yes |
| P3 | 1 | 26/32 | `r01-p0013` | 0.06250 | 416 | No |
| P3 | 2 | 26/32 | `r01-p0013` | 0.06250 | 416 | No |
| P3 | 3 | 29/32 | `r03-p0027` | 0.62500 | 464 | No |
| P3 | 4 | 28/32 | `r04-p0003` | 0.65625 | 448 | No |
| P4 | 1 | 30/32 | `r01-p0002` | 0.84375 | 480 | No |
| P4 | 2 | 31/32 | `r02-p0025` | 0.93750 | 496 | No |
| P4 | 3 | 32/32 | `r02-p0025` | 0.93750 | 512 | No |
| P4 | 4 | 32/32 | `r04-p0010` | 1.00000 | 512 | Yes |
| P5 | 1 | 31/32 | `r01-p0015` | 0.87500 | 496 | No |
| P5 | 2 | 32/32 | `r01-p0015` | 0.87500 | 512 | No |
| P5 | 3 | 32/32 | `r03-p0028` | 1.00000 | 512 | Yes |
| P6 | 1 | 32/32 | `r01-p0012` | 1.00000 | 512 | Yes |

The raw round records are in [search/rounds](search/rounds), the machine-readable
table is [search/round-summary.csv](search/round-summary.csv), and the full selected
proof records are in [search/finals](search/finals). The exact six submitted proofs
are also preserved in [submission.csv](submission.csv).

## Strict GPT-5.6 grading

Each selected proof was graded independently 64 times using `gpt-5.6-sol` with high
reasoning and the problem-specific MathArena rubric as the sole scoring rubric. There
were 384 completed attempts and no API or parsing failures.

| Problem | Distribution | Raw mean | Zero count | Zero-veto score |
|---|---|---:|---:|---:|
| P1 | 1x3, 15x4, 3x5, 2x6, 43x7 | 6.109375 | 0 | 6.109375 |
| P2 | 59x0, 5x2 | 0.156250 | 59 | 0 |
| P3 | 1x0, 54x1, 5x2, 2x5, 2x6 | 1.343750 | 1 | 0 |
| P4 | 64x7 | 7.000000 | 0 | 7.000000 |
| P5 | 26x3, 26x5, 12x6 | 4.375000 | 0 | 4.375000 |
| P6 | 62x1, 2x2 | 1.031250 | 0 | 1.031250 |

Without zero-veto:

- Total: **20.015625/42**
- Mean: **3.3359375/7**

With PR17 zero-veto:

- Total: **18.515625/42**
- Mean: **3.0859375/7**

All individual scores and rationales are in [grading/attempts.jsonl](grading/attempts.jsonl).
The exact prompt used for each problem is in [grading/prompts](grading/prompts), and
[grading/scores.csv](grading/scores.csv) provides a compact 384-row score table.

## P1 grading instability

The P1 arithmetic mean of 6.109375 is reproducible from the current 64 records, but
it should not be treated as a stable literal-rubric score.

The prior 128K P1 run selected `r04-p0006` and received 55 scores of 3 and 9 scores
of 2, for a mean of 2.859375. This run selected a genuinely different proof,
`r03-p0023`, and received scores ranging from 3 through 7. The two runs used exactly
the same system prompt and the same user prompt before the proposed solution.

The current graders disagree about method-specific checkpoints 4-7:

- The 3-point interpretation says the alternative proof does not explicitly establish
  the required boundary, origin, hypotenuse, and one-line checkpoints.
- The 7-point interpretation treats the proof's forced-line reduction and three-boundary
  classification as an equivalent or stronger argument and awards all checkpoints.

Because the grader prompt says to follow the checkpoint descriptions exactly and not
infer missing arguments, a conservative literal reading gives P1 **3/7**, or at most
**4/7** if its boundary-family argument is accepted for checkpoint 4. The reported
6.109375 is therefore retained as the observed 64-call mean, alongside this caveat.
The machine-readable comparison is
[grading/p1-prior-run-comparison.json](grading/p1-prior-run-comparison.json).

## Artifact scope

This result bundle includes every outcome needed to audit the run:

- all six problem statements and selected proofs;
- exact search and grading configuration;
- every final search record and every round summary;
- all 384 grader scores, findings, reasoning, usage, latency, and prompt hashes;
- one exact grading prompt per problem;
- raw and zero-veto aggregates; and
- the prior/current P1 comparison.

The 299 MB request-level search trace is not committed because it repeats full prompts,
model reasoning, and responses for thousands of calls. The complete candidate and
verifier outcome counts from that trace are retained in the round records and summary.
No credentials or API keys are present in this directory.
