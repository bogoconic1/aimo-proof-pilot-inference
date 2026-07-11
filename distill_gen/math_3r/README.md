# math_3r — multi-agent proof-data generator (DeepSeek-Math-V2, simplified)

A batched multi-agent proving pipeline. Per problem it runs:

```
6 Provers (same prompt, parallel) → drop invalid → each valid proof × verify_k Verifiers
→ rank (verifier mean/min, self_score, length) → Merge-Refiners (top-4 only)
→ Selectors (majority vote over refined candidates) → Clean → fallback
```

**Every stage call** (prove/verify/refine/select) — its prompt, `reasoning_content`, `content`, and usage —
is saved as a trainable distillation sample (multi-role distillation aligned to the Nemotron proof/verification
shape). This same structure is the blueprint for the deployed Kaggle agentic loop.

## Layout

| file | purpose |
|------|---------|
| `prompts/{prover,verifier,refiner,selector}.txt` | Lean two-section (`===SYSTEM===`/`===USER===`) templates. Outputs use XML tags (`<solution>`, `<score>0\|0.5\|1</score>`, `<evaluation>`, `<selected_id>`) so parsing is unambiguous and proofs can freely use `\boxed{}`. |
| `prompts/fallback.txt` | Partial-progress opener when no complete proof exists. |
| `prompts.py` | Template loading + render + system/user split. |
| `parser.py` | Dataclasses + XML-tag parsing + validity checks. |
| `rank.py` | Candidate ranking. |
| `bundle.py` | Builds refine/select bundles (with token-budget truncation). |
| `clean.py` | Deterministic cleaning (strips self-eval/boxed/XML/meta) + fallback. |
| `pipeline.py` | `Engine` + `solve_problem` (the 5-stage flow, returns the full trace). |
| `run.py` | Batch CLI: reads input parquet, cross-problem concurrency, writes full-trace JSONL + run_meta, resume. |
| `gen_refsel.py` | Augmentation: re-samples bundles over an existing proof pool to add refine/select samples. |
| `select_hard.py` | Difficulty filter producing `hard2000.parquet`. |
| `*.parquet` | Problem subsets: `hard2000` (harder 2000), `random16`/`random500` (unbiased samples), `proofbench_v2`. |

## Usage

```bash
# smoke on a 16-problem sample
DEEPSEEK_API_KEY=... uv run python distill_gen/math_3r/run.py \
    --input distill_gen/math_3r/random16.parquet --run-id r3_smoke16

# full run on the hard-2000 subset
DEEPSEEK_API_KEY=... uv run python distill_gen/math_3r/run.py \
    --input distill_gen/math_3r/hard2000.parquet --run-id r3
```

## Notes

- Diversity comes purely from backend sampling (`reasoning=high`); provers/verifiers/refiners share one prompt each.
- Effort is `high`, all stages use `max_tokens=180000`. Bad samples are dropped (no retry); a whole-problem
  failure falls back to the best verifier-confirmed original proof.
- Validity: not truncated ∧ has `<solution>` ∧ `<score> ∈ {0,0.5,1}` ∧ `len(solution) > 500`.
