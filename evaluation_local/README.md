# evaluation_local — local generation + offline grading

Same IMO-ProofBench evaluation as [`../evaluation`](../evaluation), split for an air-gapped setup:
GPU nodes host the model but **cannot reach the DeepSeek grading API**, so generation and grading
run on different machines.

1. **Serve** the model locally (SGLang, OpenAI-compatible).
2. **Generate** proofs against that endpoint (`gen_local.py`) — reuses the same templates and writes
   the same `responses.jsonl` schema as `../evaluation`.
3. **Grade** on an API-connected box (`grade_all.sh`) with the *unchanged* calibrated grader from
   `../evaluation/harness/grade_proofs.py`.

## Layout
| path | purpose |
|---|---|
| `servers/serve_sd.sh` | serve the soft-distill 7B (olmo3_sink, fp8, 200k ctx) on 1 GPU |
| `servers/serve.sh` | generic bare-metal SGLang launcher (any model) |
| `harness/gen_local.py` | local prompt-template generation (adapts `run_template_sweep.py`) |
| `harness/gen_full.sh` | full sweep: 8 templates × 60 problems × k=3 |
| `harness/extract_proof_local.py` | meta-strip local runs, reusing the evaluation harness |
| `grade_all.sh` | turnkey grading of all local runs via the DeepSeek API |

## Usage
```bash
# on the GPU box
bash servers/serve_sd.sh                       # override PROOF_PILOT_ROOT / SGLANG_SIF as needed
bash harness/gen_full.sh mymodel default 30000 150000
# on an API-connected box
export DEEPSEEK_API_KEY=...
bash grade_all.sh
```

## Note
`max_tokens` matters: these models are distilled on long proofs (p50 ~89k tokens); 60k truncates
many, 150k lets them finish cleanly.
