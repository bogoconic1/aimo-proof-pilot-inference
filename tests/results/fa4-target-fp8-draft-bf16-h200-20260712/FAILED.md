# FA4 target-FP8 / draft-BF16 experiment — FAILED

Date: 2026-07-12  
Branch: `experiment/fa4-target-fp8-draft-bf16`  
Base: `b7ccd2f` (latest merged FA4 `main` when the branch was created)

## Requested configuration

- Target weights: BF16 OPD-32B
- Target KV: `fp8_e4m3`
- Draft weights: BF16 DFlash
- Draft KV: `bfloat16`
- Attention: FA4 for target and draft
- Parallelism: TP1 / DP2 on two H200 GPUs
- Page size: 128
- Static memory fraction: 0.82
- DFlash: enabled, block size 8, production sampling unchanged
- Fallbacks: none

## Result

This exact configuration is unsupported by the installed FA4 kernel and is marked failed. No throughput benchmark was run because the server cannot complete CUDA-graph capture or execute its first attention forward.

The decisive error is:

```text
AssertionError: inputs must have the same dtype
```

The BF16 target produces BF16 queries while the FP8 target cache supplies FP8 keys and values. FA4 requires Q, K, and V to have the same dtype. SGLang also explicitly avoids casting FA4 queries and keys to FP8 because this FA4 implementation does not support FP8 Q/K.

## Capacity-planner finding and fix

The first startup exposed an independent SGLang planning bug. The original hybrid-SWA planner sized only the FP8 target pool and selected 1,088,896 full-history tokens. After allocating that target pool, only 20.34 GiB remained; the inherited BF16 draft allocation then failed with 1.63 GiB free while requesting another 2.08 GiB buffer.

The branch now budgets the two explicit pools together before either is allocated. The corrected live calculation on both DP replicas was:

| Quantity | Value |
|---|---:|
| Profiled static KV budget | 57,094,674,841 bytes |
| FP8 hybrid-target cost | 52,428 bytes per full-history token |
| BF16 default-draft cost | 32,768 bytes per token |
| Combined cost | 85,196 bytes per token |
| Page-aligned capacity | 670,080 tokens per GPU |

With that correction, allocation succeeded exactly as configured:

- Target SWA FP8 pool: 134,016 tokens; K 6.14 GiB, V 6.14 GiB.
- Target full FP8 pool: 670,080 tokens; K 10.23 GiB, V 10.23 GiB.
- Draft BF16 pool: 670,080 tokens; K 10.23 GiB, V 10.23 GiB.
- Free memory after both pools: 20.32 GiB per GPU.

This gives more full-history capacity than the all-BF16 FA4 setup (about 544K tokens), but less than all-FP8 (about 1.089M), as expected from the combined byte widths.

## Startup attempts

1. `server-attempt1-oom.log`: strict dtypes were honored, but the target-only capacity estimate caused the BF16 draft allocation to OOM.
2. `server-attempt2-classification.log`: the first planner fix failed closed because SGLang classifies this all-sliding draft under `DefaultPoolConfigurator`, not `HybridSWAPoolConfigurator`.
3. `server.log`: the allocator-matched formula succeeded; both pools allocated, then FA4 rejected mixed BF16-Q / FP8-KV tensors on its first graph forward.

## Conclusion

Do not deploy this FA4 target-FP8 / draft-BF16 configuration with the current kernel. Making it run would require changing the requested semantics—for example using another attention backend, dequantizing the target cache, or adding genuine mixed-dtype/FP8 support to FA4. None was attempted because this experiment forbids fallback.
