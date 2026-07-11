"""Async OpenAI-compatible chat client (httpx) for high-concurrency experiments.

Mirrors client.ChatClient (same `reasoning` mapping, same return dicts) but runs on a
single asyncio event loop with a pooled httpx.AsyncClient, so we can drive ~1000
concurrent requests cheaply instead of spawning 1000 OS threads. The synchronous
client.py stays for the bounded proof-generation path (run_eval.py).
"""
from __future__ import annotations

import asyncio
import time

import httpx

from client import _apply_reasoning  # reuse the no_think/high/max mapping


def _usage(data: dict) -> dict:
    usage = data.get("usage", {}) or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
    }


class AsyncChatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, *,
                 max_connections: int = 1000, timeout: float = 3600.0, max_retries: int = 5):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(max_connections=max_connections,
                                max_keepalive_connections=max_connections),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, payload: dict) -> tuple[dict, float]:
        url = f"{self.base_url}/chat/completions"
        last = None
        for attempt in range(self.max_retries):
            t0 = time.monotonic()
            try:
                r = await self._client.post(url, json=payload)
                r.raise_for_status()
                return r.json(), round(time.monotonic() - t0, 2)
            except Exception as e:  # noqa: BLE001 - network/HTTP errors, retry w/ backoff
                last = e
                await asyncio.sleep(min(2 ** attempt, 30))
        # preserve the underlying cause (httpx errors sometimes have an empty str()) via
        # `from last` + repr, and report attempts (not "retries"): max_retries=1 means 1 try.
        raise RuntimeError(
            f"request failed after {self.max_retries} attempt(s): {last!r}") from last

    async def chat(self, messages: list[dict], *, max_tokens: int = 8192,
                   reasoning: str = "default", temperature: float = 0.7,
                   top_p: float = 0.95) -> dict:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": top_p}
        _apply_reasoning(payload, reasoning)
        data, latency = await self._post(payload)
        ch = data["choices"][0]
        return {"text": ch["message"].get("content") or "",
                "finish_reason": ch.get("finish_reason"), **_usage(data),
                "latency_s": latency}

    async def chat_raw(self, messages: list[dict], *, max_tokens: int = 8192,
                       reasoning: str = "default", tools: list | None = None,
                       temperature: float = 0.7, top_p: float = 0.95) -> dict:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": top_p}
        if tools:
            payload["tools"] = tools
        _apply_reasoning(payload, reasoning)
        data, latency = await self._post(payload)
        ch = data["choices"][0]
        return {"message": ch["message"], "finish_reason": ch.get("finish_reason"),
                **_usage(data), "latency_s": latency}
