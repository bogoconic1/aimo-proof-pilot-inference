"""Native function-calling proof loop (DeepSeek V4 + OpenAI-compatible).

One tool: `execute_python`, backed by a subprocess SafePythonSession (numpy/sympy/scipy;
no filesystem/network/subprocess). Loop: call the model with the tool -> if it returns
tool_calls, run each and feed stdout back -> repeat until it returns a final proof
(finish=stop), is truncated (finish=length), or hits max_turns. Reasoning (thinking) runs
alongside tool calls; the final proof is the last assistant `content`.

The model gets a per-problem tool-call BUDGET, communicated cache-friendly: the cap is in
the tool description (cached prefix) and counted down in each tool result (suffix). We hold
the tools array constant for the whole problem (never drop it) so the prefix cache is never
invalidated; when the budget is spent we stop executing and tell the model to write the
proof. Verified against DeepSeek flash high: thinking + tool_calls coexist.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
from safe_session import SafePythonSession  # noqa: E402

# Budget is communicated to the model two cache-friendly ways: (1) the per-problem cap
# is baked into the tool DESCRIPTION (part of the stable, cached prefix — constant for the
# whole problem); (2) each tool result carries a remaining-calls countdown in its content
# (the growing suffix, never part of the cached tools prefix). We NEVER drop/replace the
# tools array mid-problem — that would invalidate the prefix cache on the (usually longest)
# final call. When the budget is spent we stop executing and tell the model, in-band, to
# write the proof now; tools stay attached.
def _py_tool(max_calls: int) -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python in a persistent sandbox to support your proof "
                "(numpy, scipy, sympy, mpmath, networkx, gmpy2, galois available; no "
                "file/network/subprocess). Returns stdout. Use it for numerical checks, "
                "small bounded exhaustive search, symbolic computation, exact big-integer / "
                "number-theory work (gmpy2), finite fields (galois), or graph/combinatorics "
                "(networkx). It cannot prove anything on its own — you must still "
                f"write the full rigorous proof yourself. You may call this tool at most "
                f"{max_calls} times for this problem; spend them sparingly, then write your "
                "complete final proof with no further tool calls. Each result tells you how "
                "many calls remain."),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string",
                             "description": "Python code; use print() to see results."},
                },
                "required": ["code"],
            },
        },
    }]


BUDGET_SPENT_MSG = ("[No tool calls remaining for this problem. Do not call the tool "
                    "again. Write your complete, rigorous final proof now.]")


def run_proof_with_tools(client, user_prompt: str, *, reasoning: str, max_tokens: int,
                         max_turns: int = 32, tool_timeout: float = 15.0,
                         max_tool_calls: int = 24, max_tool_output: int = 4000,
                         tools: list | None = None) -> dict:
    """Drive the tool loop for one problem. Returns a dict compatible with the notool
    path (text / finish_reason / prompt_tokens / completion_tokens / reasoning_tokens /
    latency_s) plus n_turns / n_tool_calls and two audit fields:

      - "messages": the COMPLETE faithful conversation (lossless), OpenAI-message shape.
        user prompt -> each assistant turn (content + reasoning_content thinking + raw
        tool_calls verbatim) -> each tool turn (tool_call_id + name + the exact output
        fed back, incl the remaining-calls countdown) -> final assistant turn. This is the
        teacher signal we archive; the `messages` SENT to the API are the same EXCEPT they
        carry reasoning_content only on tool-call turns (DeepSeek thinking_mode REQUIRES a
        tool-call turn's reasoning_content back in later turns; it's free — context cache
        dedups it, prompt_tokens unchanged — and avoids a cold-cache-miss 400). See
        memory deepseek-v4-api.
      - "turns": per-turn (finish / completion_tokens / reasoning_tokens) usage.

    Tool budget: the model may call execute_python at most `max_tool_calls` times; this cap
    is stated in the tool description (cached prefix) and counted down in each tool result
    (suffix). The tools array is held CONSTANT for the whole problem so the prefix cache is
    never invalidated. Once the budget is spent we stop executing and return BUDGET_SPENT_MSG
    as the tool result; `max_turns` is a hard backstop above the budget.

    finish_reason semantics: "stop" = model produced a final proof; "length" = a turn
    was truncated (proof may be partial/empty); "max_turns" = hit the turn cap still
    wanting tools (empty proof).
    """
    tools = tools or _py_tool(max_tool_calls)
    sess = SafePythonSession(timeout=tool_timeout + 5, mem_mb=4096)
    sent: list[dict] = [{"role": "user", "content": user_prompt}]   # clean, sent to API
    saved: list[dict] = [{"role": "user", "content": user_prompt}]  # faithful, archived
    turns: list[dict] = []
    n_tool_calls = 0
    ptoks = ctoks = rtoks = 0
    t0 = time.monotonic()
    final_text, finish = "", "max_turns"

    for turn in range(max_turns):
        out = client.chat_raw(sent, reasoning=reasoning, max_tokens=max_tokens,
                              tools=tools)
        ptoks += out["prompt_tokens"] or 0
        ctoks += out["completion_tokens"] or 0
        rtoks += out["reasoning_tokens"] or 0
        m = out["message"]
        finish = out["finish_reason"]
        tcs = m.get("tool_calls") or []
        reasoning_text = m.get("reasoning_content") or ""
        turns.append({"turn": turn, "finish": finish,
                      "completion_tokens": out["completion_tokens"],
                      "reasoning_tokens": out["reasoning_tokens"]})

        if finish == "tool_calls" and tcs:
            # SENT: per DeepSeek thinking_mode spec, a tool-call turn's reasoning_content
            # MUST be passed back in all subsequent turns (else 400 / lost thinking thread);
            # the thinking trace is preserved across tool calls within one user turn.
            sent.append({"role": "assistant", "content": m.get("content") or "",
                         "reasoning_content": reasoning_text, "tool_calls": tcs})
            # SAVED: faithful turn incl thinking text + raw tool_calls (id/fn/arguments).
            saved.append({"role": "assistant", "content": m.get("content") or "",
                          "reasoning_content": reasoning_text,
                          "tool_calls": tcs, "finish_reason": finish})
            for c in tcs:
                fn = c.get("function", {})
                try:
                    code = json.loads(fn.get("arguments") or "{}").get("code", "")
                except Exception as e:  # noqa: BLE001 - malformed tool arguments
                    code, content = "", f"[bad tool arguments: {e}]"
                else:
                    if n_tool_calls >= max_tool_calls:
                        content = BUDGET_SPENT_MSG  # tools stay attached; stop executing
                    else:
                        res = sess.execute(code)
                        n_tool_calls += 1
                        remaining = max_tool_calls - n_tool_calls
                        content = res[:max_tool_output] + f"\n[{remaining} tool call(s) remaining]"
                sent.append({"role": "tool", "tool_call_id": c["id"], "content": content})
                saved.append({"role": "tool", "tool_call_id": c["id"],
                              "name": fn.get("name"), "content": content})
            continue

        # finish == "stop" (final proof) or "length" (truncated) -> done
        final_text = m.get("content") or ""
        saved.append({"role": "assistant", "content": final_text,
                      "reasoning_content": reasoning_text, "finish_reason": finish})
        break

    sess.close()
    return {
        "text": final_text,
        "finish_reason": finish,
        "prompt_tokens": ptoks,
        "completion_tokens": ctoks,
        "reasoning_tokens": rtoks,
        "latency_s": round(time.monotonic() - t0, 2),
        "n_turns": len(turns),
        "n_tool_calls": n_tool_calls,
        "messages": saved,
        "turns": turns,
    }
