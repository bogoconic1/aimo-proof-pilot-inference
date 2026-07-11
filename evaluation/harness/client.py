"""Minimal OpenAI-compatible chat client (works for SGLang local server and DeepSeek API).

Both expose POST {base_url}/chat/completions. We deliberately depend only on `requests`
so the harness runs from the main proof-pilot venv without extra installs.

`reasoning` maps our condition labels to DeepSeek V4 reasoning controls:
  - "default" : send nothing (DeepSeek default = thinking on, effort high)
  - "no_think": thinking disabled; temperature/top_p apply
  - "high"/"max": reasoning_effort; thinking on. DeepSeek thinking mode IGNORES
    temperature/top_p, so we drop them to keep payloads honest.
"""
from __future__ import annotations

import time
import requests

REASONING = ("default", "no_think", "high", "max")


def _apply_reasoning(payload: dict, reasoning: str) -> None:
    if reasoning == "default":
        return
    if reasoning == "no_think":
        payload["thinking"] = {"type": "disabled"}
        return
    if reasoning in ("high", "max"):
        payload["reasoning_effort"] = reasoning
        payload.pop("temperature", None)  # thinking mode ignores sampling params
        payload.pop("top_p", None)
        return
    raise ValueError(f"unknown reasoning {reasoning!r}; expected one of {REASONING}")


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 1800.0, max_retries: int = 4):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, payload: dict) -> tuple[dict, float]:
        """POST with retry. Returns (response_json, latency_s)."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_err = None
        for attempt in range(self.max_retries):
            t0 = time.monotonic()
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                return r.json(), round(time.monotonic() - t0, 2)
            except Exception as e:  # noqa: BLE001 - network/JSON errors, retry
                last_err = e
                wait = 2 ** attempt
                print(f"  [client] attempt {attempt + 1}/{self.max_retries} failed: {e} "
                      f"(retry in {wait}s)")
                time.sleep(wait)
        raise RuntimeError(f"request failed after {self.max_retries} retries: {last_err}")

    def _usage(self, data: dict) -> dict:
        usage = data.get("usage", {}) or {}
        details = usage.get("completion_tokens_details") or {}
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
        }

    def chat(self, messages: list[dict], *, temperature: float = 0.7,
             top_p: float = 0.95, max_tokens: int = 8192, seed: int | None = None,
             reasoning: str = "default") -> dict:
        """Single-shot completion. Return {text, finish_reason, prompt_tokens,
        completion_tokens, reasoning_tokens, latency_s}."""
        payload = {
            "model": self.model, "messages": messages,
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        _apply_reasoning(payload, reasoning)
        data, latency = self._post(payload)
        choice = data["choices"][0]
        return {
            "text": choice["message"].get("content") or "",
            "reasoning_content": choice["message"].get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"),
            **self._usage(data),
            "latency_s": latency,
        }

    def chat_raw(self, messages: list[dict], *, max_tokens: int = 8192,
                 reasoning: str = "default", tools: list | None = None,
                 temperature: float = 0.7, top_p: float = 0.95) -> dict:
        """Tool-aware completion. Returns the full assistant `message` (incl any
        `tool_calls`) plus finish_reason and usage. Used by the native function-calling
        loop in tool_loop.py."""
        payload = {
            "model": self.model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p,
        }
        if tools:
            payload["tools"] = tools
        _apply_reasoning(payload, reasoning)
        data, latency = self._post(payload)
        choice = data["choices"][0]
        return {
            "message": choice["message"],
            "finish_reason": choice.get("finish_reason"),
            **self._usage(data),
            "latency_s": latency,
        }

    def health(self) -> bool:
        try:
            # SGLang exposes /health at server root (one level above /v1)
            root = self.base_url.rsplit("/v1", 1)[0]
            return requests.get(f"{root}/health", timeout=10).ok
        except Exception:  # noqa: BLE001
            return False
