# SGLang determinism and proof diversity audit

This audit measures the throughput benefit and proof-diversity cost of disabling
SGLang deterministic inference in the repository's TP2 x DP4 BF16+DFlash
deployment. It was run on eight NVIDIA H200 GPUs with SGLang
`0.5.14.dev20260618+g343aeeef39` and PyTorch `2.11.0+cu130`.

## Matched-output throughput

The decode-only A/B issued 32 concurrent requests with 8,192 generated tokens
each. Greedy decoding forced both arms to generate the same full-output SHA-256,
so output-path and speculative-acceptance luck could not confound the result.

| Mode | Aggregate output tokens/s | Wall time |
|---|---:|---:|
| Determinism on | 8,860.27 | 29.59 s |
| Determinism off | 11,959.71 | 21.92 s |

Disabling determinism improved aggregate decode throughput by 34.98% and reduced
wall time by 25.92%. SGLang changed the sampling backend from PyTorch to
FlashInfer and enabled custom all-reduce v2; deterministic TP2 forced tree NCCL
all-reduce and disabled custom all-reduce.

## Production P5 diversity

The diversity test used human IMO 2026 Problem 5 (dataset `id=4`) and the exact
first-round production generation path:

- the verbatim checked-in prover prompt;
- 32 proof candidates with distinct stable request seeds;
- temperature 1.0 and top-p 0.95;
- concurrency 96;
- a 128,000-token maximum with normal EOS/XML stopping;
- the production 16,384-token continuation policy when required.

No fixed output length or `ignore_eos` override was used. All 32 nondeterministic
requests stopped naturally, produced valid XML, and completed in one physical
request. Their completion lengths ranged from 32,006 to 113,480 tokens.

| Mode | Valid proofs | Unique full outputs | Unique parsed proofs | Duplicate groups |
|---|---:|---:|---:|---:|
| Determinism off | 32 | **16** | **16** | 16 pairs |
| Determinism on, pinned control | 32 | **32** | **32** | 0 |

Every nondeterministic candidate was duplicated exactly once:
`r01-p0000 == r01-p0001`, `r01-p0002 == r01-p0003`, through
`r01-p0030 == r01-p0031`. Each pair carried two different requested seeds but
had identical reasoning, answer content, parsed proof, completion length, and
SHA-256 values.

The installed SGLang build only materializes per-request `sampling_seed` values
when global deterministic inference is enabled. With it disabled, DP replicas
sample from worker RNG state; synchronized streams can therefore collapse
independent proof requests. In this production-shaped test, 2,322,852 generated
tokens yielded only 16 distinct proof attempts.

The hash-only evidence is in
[`p5-off-uniqueness.json`](p5-off-uniqueness.json). It contains no proof text.
The deterministic control summary is in
[`p5-on-control-summary.json`](p5-on-control-summary.json); its source trace is
pinned at commit `7c21511`.

## Reproduce

Start and validate the desired server configuration, then run the audit against
that already-running endpoint. `--config` supplies the production model, search,
and endpoint values; `--server-determinism` records the actual server mode being
tested.

```bash
python evaluation/harness/audit_generation_diversity.py \
  --config config.yaml \
  --input /workspace/test.csv \
  --problem-index 4 \
  --server-determinism off \
  --artifacts-dir /workspace/diversity-audit-p5-off \
  --summary /workspace/diversity-audit-p5-off.json
```

The artifacts directory must not already exist, preventing cached calls from
silently changing the measurement. Raw proof text remains in the local call
trace; only the hash summary should be published unless traces are intentionally
part of the evaluation artifact.

## Recommendation

Keep deterministic inference enabled for multi-replica proof search until the
nondeterministic path gives each DP worker an independent sampling stream or
honors per-request seeds. If nondeterministic inference is enabled for its 35%
decode throughput gain, make exact-output uniqueness a required validation gate.
