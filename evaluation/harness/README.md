# Eval harness

A three-stage ProofBench evaluation pipeline, all using the OpenAI-compatible `/v1/chat/completions` API (the same client for local SGLang and the DeepSeek API). Run it from the **main proof-pilot venv** (needs `requests`/`httpx`/`pandas`/`numpy`/`sympy`).

```
run_eval.py        problems -> model -> runs/<run_id>/responses.jsonl  (k candidates/problem; supports notool / pytool)
grader.py          responses + reference solution/rubric -> grades.jsonl  (0/1/6/7; --limit to grade only the first N)
score.py           grades -> summary.json + tables                  (mean/almost+/correct/best-of-k, by subset/level/category)
client.py          synchronous HTTP client (retries, health, reasoning knobs, chat_raw for the tool loop)
async_client.py    async HTTP client (httpx, pooled, for ~1000-way concurrency)
tool_loop.py       native function-calling proof loop (for the prover, pytool; stores lossless messages, cache-friendly tool budget)
grade_proofs.py    grade a run's candidates with the calibrated flash high_notool grader (async; --passes, grade-all/intersect)
make_review_html.py package a run into a self-contained HTML viewer (prompt/per-turn reasoning/tool code/output/proof, opens offline)
calibrate_grader.py grader calibration (async; gradingbench × multiple configs, computes accuracy/MAE/Pearson)
tools/             python tool sandbox: python_tool.py (vendored), safe_session.py (resource-limited), _exec_driver.py
```

### ⚠️ Data-persistence principle (mandatory when calling an API)

**Every LLM/API call must persist its complete raw record to disk, one row at a time — never store only a summary or discard it afterward.** API calls are expensive, and thinking mode is stochastic = not reproducible; without saving, any later audit (tool success rate, grader rationale, failure cases) means re-running — wasting money and losing data. Store at minimum:

- **Full messages**: system/user/assistant, **the tool_calls' code, and the tool-returned output** (for every turn).
- **The full model output** `content` **and `reasoning_content` (the thinking)**.
- `finish_reason`, `usage` (prompt/completion/**reasoning** tokens), `latency`, seed, and all sampling/reasoning parameters.
- **Append + resume** row by row (the host is unstable; see the memory note `host-memory-instability`).

> Lesson learned: `calibrate_grader.py` originally stored only the `n_tool_calls` count and threw away the transcript, so we couldn't go back and check whether the tool calls succeeded — only re-run. **Better to over-save than to regret it later.** Summaries (scores/metrics) are derived artifacts; the raw messages are the real data.

### reasoning / tool knobs

- `--reasoning {default,no_think,high,max}` (client `_apply_reasoning`): maps to DeepSeek thinking/reasoning_effort; `high`/`max` automatically drop temperature/top_p (ignored in thinking mode). See the memory note `deepseek-v4-api`.
- `--condition pytool`: gives the model a native `execute_python` tool (numpy/sympy sandbox), running a multi-turn tool loop until it produces a final proof. `--max-turns` caps the number of turns.
  - **cache-friendly tool budget (`--max-tool-calls`, default 24)**: the per-problem tool-call limit is **written into the tool description** (a stable cached prefix) plus a `[k remaining]` countdown at the tail of each tool result (suffix); when exhausted it feeds `BUDGET_SPENT_MSG`, and the **tools array never changes** (so the prefix cache is not invalidated). `--max-turns` (default 32) is only a hard backstop. **Do not detach tools** — mutating the tools array causes a prefix-cache miss on the longest call. Measured (k=4×60): zero tool-lock blanks, `n_tool_calls` never exceeded the limit. This replaces the old max_turns-64 + `FINAL_NUDGE` detach scheme.
  - Historical lesson: too small a `--max-turns` (10) lets flash high burn all its turns and return an empty proof (usable dropped falsely to 19/60); with the budget scheme a large max_turns is unnecessary.
  - **Persistence (lossless)**: `candidates_raw.jsonl` stores per row a `messages` field (full conversation: user→assistant with `content`+**per-turn `reasoning_content`**+raw `tool_calls`→tool with `tool_call_id`/`name`/output+countdown→final) plus `turns` (per-turn finish/tokens). **notool uses the same schema** (messages=user+assistant with reasoning_content). `run_meta` separately stores `tools` (the exact schema) + `repro` (git/sha).
  - **A tool-call turn's `reasoning_content` is fed back into the API** (DeepSeek thinking_mode requires the thinking from a tool-call turn to be carried into subsequent turns, otherwise a cold-cache miss returns 400); this is **free** (context cache dedups it, prompt_tokens unchanged, measured usage A=B). Non-tool-call turns don't need it fed back. See the memory note `deepseek-v4-api`.

### Python tool sandbox (safety measures)

**High-concurrency tool use must use `tools/safe_session.py` (`SafePythonSession`)**, not the bare `SecureLightweightPythonSession` in `python_tool.py`: the latter has **no RAM limit, and its timeout doesn't fire off the main thread** (this once coincided with a host/terminal crash at concurrency 400). `SafePythonSession` runs each exec in a subprocess with an **RLIMIT_AS memory cap + wall-clock SIGKILL + in-subprocess SIGALRM** (triple protection); its interface and behavior are identical from the LLM's point of view (`execute(code)->stdout`, variables persist across calls). Both `tool_loop.py` and `calibrate_grader.py` now use it.
(`tools/python_tool.py` is vendored from `agent-factory-3/mcp_tools/python_tool/secure_python_session.py` for its import restrictions; the resource limits are added on top by `safe_session.py`.)

**Math libraries available inside the sandbox** (verified, running on the main venv): `numpy scipy sympy mpmath networkx gmpy2 galois` + the full stdlib (math/cmath/fractions/decimal/itertools/…). Blocked: `os/sys/subprocess/socket/pathlib/pickle/importlib`, `pandas/matplotlib`, builtin `open/eval/exec/getattr/setattr`. `gmpy2` (bignum/number theory) / `galois` (finite fields) are installed via `uv add` (recorded in pyproject). The tool description lists these libs for the model.

### Rules for long / high-concurrency jobs

Under heavy load the host can byte-flip (memory note `host-memory-instability`). For long runs: (1) run `setsid` detached (a terminal restart won't take the job with it); (2) keep concurrency conservative (especially pytool — the subprocess sandbox has a memory cost); (3) append + resume row by row (both `run_eval.py` and `calibrate_grader.py` support this, deduping by the output file's key).

## 1. Start the SGLang server (local OLMo / post-trained model)

A separate venv lives at `../serving/.venv` (sglang 0.5.9 / torch 2.9.1+cu128, **isolated from the main training venv** to avoid disturbing torch 2.12+cu126). Needs `ninja` (already installed in that venv).

```bash
cd ../serving
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  nohup .venv/bin/python -m sglang.launch_server \
    --model-path ${OLMO3_7B:-/models/Olmo-3-7B-Think} \
    --host 127.0.0.1 --port 30000 --tp 1 \
    --mem-fraction-static 0.85 --context-length 32768 --trust-remote-code \
    > logs/server.log 2>&1 &
# ready: curl -s http://127.0.0.1:30000/health
```

Shut down (kill the whole tree, otherwise the scheduler subprocess keeps holding the GPU; see docs/tokenizer/deploy.md):
```bash
pkill -9 -f sglang.launch_server
```

## 2. Run the evaluation

```bash
PY=python
cd harness

# generate
$PY run_eval.py --data ../data/subset_dev.csv \
  --base-url http://127.0.0.1:30000/v1 --served-model default \
  --model-name olmo3-7b-think --condition notool \
  --k 1 --temperature 0.7 --max-tokens 32768

# grade (once you have a DeepSeek key)
$PY grader.py --run-id olmo3-7b-think__notool --data ../data/subset_dev.csv \
  --grader deepseek --base-url https://api.deepseek.com/v1 \
  --served-model <pro-model-id> --api-key-env DEEPSEEK_API_KEY
#   no key yet -> --grader stub (null scores, only validates the pipeline)

# aggregate
$PY score.py --run-id olmo3-7b-think__notool
```

DeepSeek teacher evaluation: point `run_eval.py` at `--base-url https://api.deepseek.com/v1 --served-model <id> --api-key-env DEEPSEEK_API_KEY`; no local server needed.

pytool condition (give the model a python tool): `run_eval.py --condition pytool --reasoning high --max-tokens 131072 --max-tool-calls 24 --max-turns 32` (the budget 24 is written into the tool description; max_turns is only a backstop).

## 3. Grader calibration (calibrate_grader.py)

Validates grader accuracy (vs the gradingbench human scores) — a prerequisite for trusting the scores. Async, high-concurrency, resumable.

```bash
$PY calibrate_grader.py --data ../data/gradingbench.csv --n 200 \
  --base-url https://api.deepseek.com/v1 --served-model deepseek-v4-flash \
  --api-key-env DEEPSEEK_API_KEY --concurrency 60 \
  --configs high_notool,high_pytool,max_notool,max_pytool \
  --max-tokens 65536 --max-turns 30 --run-id grader_calibration
```

Outputs `runs/<run_id>/grades_all.jsonl` + `summary.json` (per-config coverage/accuracy/MAE%/Pearson/token). Calibration conclusions are in `../results/grader_calibration.md` (**use flash high_notool for ranking; consider pro max_pytool only for a per-problem verifier**).

## Known issues (smoke 2026-06-01)

- **`--max-tokens` must be large enough**: Olmo-3-7B-Think reasons very long; at 8192, 7 of 8 problems were truncated (`finish=length`). Production eval uses **32768** (context-length is already set to 32768). Truncated = an incomplete proof, which the grader usually scores 0/1.
- Think models write reasoning directly into content (no separate reasoning channel, `reasoning_tokens=0`); the grader sees the full text including the thinking. To grade only the final proof, extract it before the grader (currently the whole thing is sent for grading).
- The 8-problem subset_dev is Algebra-heavy (a side effect of stratified sampling for difficulty); it's fine for a smoke test — run the full 60 problems for the real evaluation.

## DeepSeek (2026-06-03)

- model ids: `deepseek-v4-flash` / `deepseek-v4-pro`, base_url `https://api.deepseek.com/v1`. `run_eval.py` needs no change — just point at it (default = thinking on + effort high).
- **`--max-tokens` is a combined reasoning+output budget**: flash high on hard problems spends the whole budget thinking and leaves `content` empty. 32768 → 36/60 empty proofs; production DeepSeek thinking uses **131072**. A truncation (finish=length) usually has empty content = no proof, not merely "cut short".
- Thinking mode **ignores temperature/top_p** (including the grader's `--temperature 0`); only no_think (`thinking:{type:disabled}`) honors sampling parameters. The three conditions no_think/high/max map as in `plan.md` §current-status and the memory note `deepseek-v4-api`.
- flash high has ~3% reasoning runaway (>131k with no conclusion); re-rolling recovers most, and the remainder are recorded as incomplete=0.
- flash high baseline, all 60 problems: `runs/dsv4-flash__high_131k/` (`gen_summary.json`).
- flash high **pytool**, all 60 problems: `runs/dsv4-flash__high_pytool/` (max_turns 64, lossless messages). usable 55/60; finish stop 55/tool_calls 5 (5 blanks = tool-locked hitting 64).
- **notool vs pytool coverage**: intersection 53, union 60 (full coverage), both empty 0. The failure modes are complementary (notool = reasoning-runaway, pytool = tool-lock). See `plan.md` §current-status (2026-06-04).
