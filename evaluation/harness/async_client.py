"""Async client for the OpenRouter DeepSeek V4 Flash chat API."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx


DEFAULT_MAX_RETRIES = 7
BASE_DELAY_S = 2.0
MAX_DELAY_S = 128.0
JITTER_RATIO = 0.25
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504, 529}


class IncompleteOpenRouterStreamError(RuntimeError):
    pass


class OpenRouterStreamError(RuntimeError):
    def __init__(self, error: dict):
        self.error = error
        super().__init__(
            "OpenRouter stream error: "
            + json.dumps(error, ensure_ascii=False)
        )


def _backoff_delay(attempt: int) -> float:
    base = min(BASE_DELAY_S * (2**attempt), MAX_DELAY_S)
    return base * (1 + random.uniform(-JITTER_RATIO, JITTER_RATIO))


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, IncompleteOpenRouterStreamError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_HTTP_STATUS_CODES
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, OpenRouterStreamError):
        return error.error.get("code") in RETRYABLE_HTTP_STATUS_CODES
    return False


async def _retry_with_exponential_backoff(
    func: Callable[[], Awaitable[Any]],
    *,
    request_id: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[Any, int]:
    for attempt in range(max_retries + 1):
        try:
            return await func(), attempt + 1
        except Exception as error:
            if not _is_retryable(error) or attempt == max_retries:
                raise
            delay = _backoff_delay(attempt)
            print(
                f"[request] retry id={request_id} "
                f"attempt={attempt + 1}/{max_retries + 1} "
                f"error={error!r} sleep={delay:.1f}s",
                flush=True,
            )
            await asyncio.sleep(delay)

    raise AssertionError("retry loop exhausted without returning or raising")


def _usage(data: dict) -> dict:
    usage = data["usage"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "cached_prompt_tokens": usage["prompt_tokens_details"]["cached_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "reasoning_tokens": usage["completion_tokens_details"]["reasoning_tokens"],
    }


def _optional_sum(*values: Any) -> int | None:
    integers = [value for value in values if type(value) is int]
    return sum(integers) if integers else None


class AsyncChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        reasoning_effort: str,
        *,
        max_connections: int = 1000,
        timeout: float = 3600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, payload: dict) -> tuple[dict, float]:
        started = time.monotonic()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_details: list[dict] = []
        finish_reason = None
        usage = None
        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    raise RuntimeError(f"invalid OpenRouter SSE line: {line!r}")
                event_data = line.removeprefix("data: ")
                if event_data == "[DONE]":
                    break
                event = json.loads(event_data)
                if "error" in event:
                    raise OpenRouterStreamError(event["error"])
                if event["choices"]:
                    choice = event["choices"][0]
                    delta = choice["delta"]
                    content_parts.append(delta.get("content") or "")
                    reasoning_parts.append(delta.get("reasoning") or "")
                    reasoning_details.extend(delta.get("reasoning_details") or [])
                    if choice["finish_reason"] is not None:
                        finish_reason = choice["finish_reason"]
                if event.get("usage") is not None:
                    usage = event["usage"]
        if usage is None or finish_reason is None:
            raise IncompleteOpenRouterStreamError(
                "OpenRouter stream ended without usage or finish reason"
            )
        data = {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "content": "".join(content_parts),
                        "reasoning": "".join(reasoning_parts),
                        "reasoning_details": reasoning_details,
                    },
                }
            ],
            "usage": usage,
        }
        return data, round(time.monotonic() - started, 3)

    async def chat_raw(
        self,
        messages: list[dict],
        *,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        request_id: str,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": True,
            "reasoning": {
                "effort": self.reasoning_effort,
                "exclude": False,
            },
        }
        print(f"[request] start id={request_id}", flush=True)
        started = time.monotonic()
        try:
            (data, reported_latency), attempts = (
                await _retry_with_exponential_backoff(
                    lambda: self._post(payload),
                    request_id=request_id,
                )
            )
        except Exception as error:
            print(
                f"[request] failed id={request_id} error={error!r}",
                flush=True,
            )
            raise
        latency = round(
            max(reported_latency, time.monotonic() - started),
            3,
        )
        if len(data["choices"]) != 1:
            raise RuntimeError(f"expected one completion, received {len(data['choices'])}")
        choice = data["choices"][0]
        source_message = choice["message"]
        message = {
            "content": source_message["content"],
            "reasoning_content": source_message["reasoning"],
            "reasoning_details": source_message["reasoning_details"],
        }
        usage = _usage(data)
        finish_reason = choice["finish_reason"]
        print(
            f"[request] complete id={request_id} finish={finish_reason} "
            f"tokens={usage['completion_tokens']} attempts={attempts} "
            f"latency={latency:.3f}s",
            flush=True,
        )
        segment = {
            "kind": "chat",
            "request_id": request_id,
            "finish_reason": finish_reason,
            **usage,
            "requested_max_completion_tokens": max_completion_tokens,
            "physical_request_count": attempts,
            "latency_s": latency,
        }
        return {
            "message": message,
            "finish_reason": finish_reason,
            **usage,
            "requested_max_completion_tokens": max_completion_tokens,
            "logical_max_completion_tokens": max_completion_tokens,
            "physical_request_count": attempts,
            "physical_prompt_tokens": usage["prompt_tokens"],
            "segments": [segment],
            "latency_s": latency,
        }

    async def _continue_xml_raw(
        self,
        initial: dict,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        request_id: str,
        role: str,
        opening_tag: str,
        preserve_untagged_content: bool,
    ) -> dict:
        initial_message = initial["message"]
        content = initial_message["content"] or ""
        has_opening_tag = opening_tag in content.lower()
        if has_opening_tag:
            visible_prefix = content
        else:
            untagged_content = content if preserve_untagged_content else ""
            visible_prefix = opening_tag + "\n" + untagged_content

        continuation_id = f"{request_id}/{role}-continuation"
        prefill = {
            "role": "assistant",
            "content": visible_prefix,
            "reasoning_details": initial_message["reasoning_details"],
        }
        continuation = await self.chat_raw(
            [*messages, prefill],
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            request_id=continuation_id,
        )
        continuation_message = continuation["message"]
        trigger = (
            f"length_partial_{role}"
            if has_opening_tag
            else "length_thinking"
            if not content
            else f"length_unstructured_{role}"
        )
        segment = {
            **continuation["segments"][0],
            "kind": f"{role}_continuation",
            "trigger": trigger,
            f"injected_{role}_tag": not has_opening_tag,
        }
        requested_continuation_field = f"requested_{role}_continuation_tokens"
        return {
            **initial,
            "message": {
                "content": visible_prefix + (continuation_message["content"] or ""),
                "reasoning_content": (
                    (initial_message["reasoning_content"] or "")
                    + (continuation_message["reasoning_content"] or "")
                ),
                "reasoning_details": [
                    *initial_message["reasoning_details"],
                    *continuation_message["reasoning_details"],
                ],
            },
            "finish_reason": continuation["finish_reason"],
            "completion_tokens": _optional_sum(
                initial["completion_tokens"], continuation["completion_tokens"]
            ),
            "reasoning_tokens": _optional_sum(
                initial["reasoning_tokens"], continuation["reasoning_tokens"]
            ),
            requested_continuation_field: max_new_tokens,
            "logical_max_completion_tokens": (
                initial["requested_max_completion_tokens"] + max_new_tokens
            ),
            "physical_request_count": _optional_sum(
                initial["physical_request_count"],
                continuation["physical_request_count"],
            ),
            "physical_prompt_tokens": _optional_sum(
                initial["prompt_tokens"], continuation["prompt_tokens"]
            ),
            "segments": [*initial["segments"], segment],
            "latency_s": round(initial["latency_s"] + continuation["latency_s"], 3),
        }

    async def continue_solution_raw(
        self,
        initial: dict,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        request_id: str,
    ) -> dict:
        return await self._continue_xml_raw(
            initial,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            request_id=request_id,
            role="solution",
            opening_tag="<solution>",
            preserve_untagged_content=False,
        )

    async def continue_verification_raw(
        self,
        initial: dict,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        request_id: str,
    ) -> dict:
        return await self._continue_xml_raw(
            initial,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            request_id=request_id,
            role="verifier",
            opening_tag="<evaluation>",
            preserve_untagged_content=True,
        )
