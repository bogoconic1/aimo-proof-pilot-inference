# AIMO Proof Pilot Inference

This branch runs the generate-verify-refine proof harness through OpenRouter with
`deepseek/deepseek-v4-flash`. It reads `test.csv`, uses the same model for proof
generation, internal verification, and refinement, then writes `submission.csv`.
There is no external grader.

## Configuration

`config.yaml` is the complete runtime contract. Its provider section is:

```yaml
provider:
  base_url: https://openrouter.ai/api/v1
  model: deepseek/deepseek-v4-flash
  api_key_env: OPENROUTER_API_KEY
  reasoning_effort: high
```

`reasoning_effort` is passed unchanged to OpenRouter; `high` is only the
checked-in default. The checked-in search policy is unchanged from `main`:

| Setting | Value |
|---|---:|
| Proofs per round | 32 |
| Verifications per proof | 16 |
| Top proofs | 8 |
| Refinements per proof | 4 |
| Maximum rounds | 4 |
| Temperature | 1.0 |
| Top-p | 0.95 |
| First output segment | 128,000 tokens |
| Solution continuation | 16,384 tokens |
| Verifier continuation | 16,384 tokens |
| Concurrency | 96 |

The harness sends streaming Chat Completions requests with the configured
sampling values and stable per-call seed. It ignores OpenRouter's SSE keepalive
comments and reconstructs the response content, reasoning, reasoning details,
finish reason, and usage. If a length-limited response lacks complete XML, it
makes the same single configured continuation through OpenRouter assistant
prefill while preserving `reasoning_details`.

Every physical OpenRouter call uses exponential backoff with seven retries.
HTTP responses are retried only for status codes `429`, `500`, `502`, `503`,
`504`, and `529`; transport failures and streams missing their terminal chunk
are also retried. Failed persisted attempts are retried on resume, while a
completed sample ID is always reused.

## Input

Create `test.csv` with exactly two lowercase columns:

```csv
id,problem
0,"First complete problem statement"
1,"Second complete problem statement"
```

IDs must be nonempty and unique. Quote fields containing commas or newlines.

## Local usage

Install the runtime dependencies:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r evaluation/requirements.txt
```

Set `OPENROUTER_API_KEY` in the environment or an env file, then run:

```bash
CONFIG="$PWD/config.yaml" \
ENV_FILE="$PWD/.env" \
PYTHON="$PWD/.venv/bin/python" \
bash run_submission.sh test.csv submission.csv
```

For the checked-in six-problem IMO 2026 run, use:

```bash
./run_imo_2026.sh
```

It prints one start/completion line per OpenRouter request and appends the same
output to `run_imo_2026.log`. The resumable request records remain under
`submission_artifacts/problems/<row>/calls.jsonl`.

The output has exactly these columns:

```csv
id,proof
```

## Docker usage

Prepare a workspace containing `config.yaml`, `test.csv`, and `.env`:

```text
workspace/
  .env
  config.yaml
  test.csv
```

Build and run the API-only image:

```bash
docker build -t aimo-proof-pilot-openrouter .
docker run --rm \
  -v "$PWD/workspace:/workspace" \
  -e CONFIG=/workspace/config.yaml \
  aimo-proof-pilot-openrouter submission
```

The container writes `/workspace/submission.csv` and stores resumable state in
`/workspace/submission_artifacts`.

## Resume behavior

After every completed search round, the current top proof is atomically written
to `submission.csv`. Re-running with the same configuration, `test.csv`, and
artifacts directory reuses completed calls. The runner rejects mismatched pinned
inputs or configuration rather than combining different searches.

Use a different directory for a new run:

```bash
ARTIFACTS_DIR=submission_artifacts_candidate_2 \
CONFIG="$PWD/config.yaml" \
PYTHON="$PWD/.venv/bin/python" \
bash run_submission.sh test.csv submission.csv
```
