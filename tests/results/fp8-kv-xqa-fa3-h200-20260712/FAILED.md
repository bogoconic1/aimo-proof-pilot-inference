# Failed FP8-KV DFlash experiment on two H200 GPUs

## Status

**Failed. Do not deploy FP8 KV for both the target and DFlash draft.**

The experiment doubled target KV capacity, but uncalibrated FP8 draft KV destroyed
DFlash agreement with the BF16 target. The full matched workload completed 2.84x
slower than the committed BF16 FA4 reference.

## Results

| Configuration | Full KV tokens/GPU | Wall time | Aggregate throughput | DFlash acceptance |
|---|---:|---:|---:|---:|
| BF16 FA4 reference | 544,384 | 80.024 s | 3,275.817 tok/s | about 3.1-3.5 tokens |
| FA3, target+draft FP8 KV | 1,088,994 | 227.435 s | 1,152.612 tok/s | about 1.0-1.16 tokens |

The FP8 run generated all 32 requests to exactly 8,192 tokens, for 262,144 total
completion tokens. It was 64.815% slower in aggregate throughput and took 184.207%
more wall time than BF16 FA4.

## Memory result

At `mem_fraction_static=0.82`, FP8 E4M3 increased the target full-attention pool
from 544,384 to 1,088,994 tokens per GPU and the SWA pool from 108,800 to 217,798.
The mandatory page-1 DFlash ring reduced the draft pool from a would-be 1,088,994
tokens to 69,696 tokens.

Four contexts of 262,144 tokens require 1,048,576 full-attention slots, so the
measured FP8 target pool would fit them with 40,418 slots remaining.

## Failure mechanism

The checkpoint contains no calibrated FP8 KV scaling factors. SGLang explicitly
reported that target and draft KV used scale 1.0. The fixed three-equation smoke
test remained correct, but DFlash draft acceptance collapsed from roughly 3.3
tokens to almost exactly the mandatory one target token. Draft work therefore
became overhead instead of acceleration.

This result does not show that target-only FP8 KV is unusable. It shows that the
DFlash draft KV must remain BF16 unless calibrated draft scales are produced and
validated.

## XQA control startup failure

The FA3-prefill/TRT-LLM-XQA-decode BF16 control did not reach serving. During
DFlash CUDA-graph capture, FlashInfer XQA rejected SGLang metadata with:

```
AssertionError: Mask is required for speculative decoding
```

The official FlashInfer implementation requires a packed uint16 lower-triangular
mask whenever XQA receives a query block longer than one token. This SGLang build
did not forward that mask for DFlash target verification. No backend fallback was
used.

## Workload contract

The completed run used the BF16 OPD target and BF16 DFlash weights, FP8 E4M3 KV
for both models, FA3 target/draft attention, page size 1, the compact draft ring,
TP1/DP2, 32 simultaneous requests, the exact IMO 2025 problem 1 production prompt,
the committed 32 IDs and seeds, temperature 1.0, top-p 0.95, no warm-up, a flushed
prefix cache, and 8,192 completion tokens per request.

## Artifacts

- `fa3-fp8-result.json`: aggregate completed benchmark
- `fa3-fp8-requests.json`: all 32 per-request records and output hashes
- `fa3-fp8-server-info.json`: strict live server/config validation
- `fa3-fp8-server.log`: capacity, dtype, ring and runtime evidence
- `fa3-fp8-equation-*`: fixed correctness smoke test
- `xqa-bf16-server.log`: preserved XQA speculative-mask startup failure
- Three exact YAML experiment configurations
