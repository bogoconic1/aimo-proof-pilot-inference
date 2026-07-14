# Gemini IMO 2025 strict one-shot grading

This run grades the public Gemini Deep Think IMO 2025 solutions with the
repository's current problem-specific MathArena rubrics.

- Source: <https://storage.googleapis.com/deepmind-media/gemini/IMO_2025.pdf>
- Source SHA-256: `90040afd878059d3a8d20974b7bff8d4ef648a96352f61ed65540ed468eca935`
- Grader: `gpt-5.6-sol`
- Reasoning effort: `high`
- Calls: one independent API call per problem, with no retries
- Prompt format: current production system and user prompts
- Aggregation: none; each score is the single returned grade

The PDF contains solutions for P1-P5 and ends after P5 on page 13. P6 was
submitted to the grader as a missing solution and therefore received zero.

## Scores

| Problem | PDF pages | Score | Main grading outcome |
|---|---:|---:|---|
| P1 | 1-2 | 3/7 | Complete alternative proof; rubric-specific checkpoints 4-7 were denied. |
| P2 | 3-5 | 5/7 | Valid alternative tangency proof; rubric-specific checkpoints 3 and 5 were denied. |
| P3 | 6-8 | 7/7 | All rubric checkpoints satisfied. |
| P4 | 9-10 | 7/7 | All rubric checkpoints satisfied. |
| P5 | 11-13 | 7/7 | All rubric checkpoints satisfied. |
| P6 | absent | 0/7 | No P6 solution is present in the source PDF. |

- Total including missing P6: **29/42**
- Total over the five provided solutions: **29/35**

## Rubric-path mismatch

The grader explicitly described P1 as "mathematically complete" through a
stronger structural reduction and convex-boundary count, but awarded only 3/7
because the proof did not reproduce the prescribed left/bottom boundary,
origin, hypotenuse, and final one-line arguments.

The same issue appears on P2. The grader accepted the final tangency argument as
valid but withheld two points because the solution did not establish the
rubric's specific intersection and incenter characterizations.

This is not evidence that grading should be more lenient. It shows that a strict
rubric can still be incomplete when its checkpoints encode one solution route
rather than the mathematical obligations of the problem. A complete alternative
proof must remain eligible for full credit while unsupported or incorrect claims
still receive no credit.

## Artifacts

- `records.jsonl`: sanitized per-problem findings, scores, usage, hashes, and
  latency. Full copied proof text is deliberately excluded.
- `summary.json`: compact score summary.
- `run_manifest.json`: source, prompt, model, and call protocol metadata.
