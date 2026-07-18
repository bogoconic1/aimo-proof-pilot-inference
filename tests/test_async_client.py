from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from async_client import (  # noqa: E402
    AsyncChatClient,
    IncompleteOpenRouterStreamError,
)


def completion(
    *,
    content: str,
    reasoning: str,
    reasoning_details: list[dict],
    finish_reason: str = "stop",
    prompt_tokens: int = 200,
    completion_tokens: int = 300,
    reasoning_tokens: int = 100,
) -> dict:
    return {
        "id": "gen-test",
        "model": "deepseek/deepseek-v4-flash",
        "provider": "DeepSeek",
        "choices": [
            {
                "finish_reason": finish_reason,
                "native_finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": reasoning,
                    "reasoning_details": reasoning_details,
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 25},
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens,
            },
        },
    }


def initial_response(*, content: str) -> dict:
    details = [
        {
            "type": "reasoning.text",
            "text": "unfinished reasoning",
            "format": "unknown",
            "index": 0,
        }
    ]
    return {
        "message": {
            "content": content,
            "reasoning_content": "unfinished reasoning",
            "reasoning_details": details,
        },
        "finish_reason": "length",
        "prompt_tokens": 200,
        "cached_prompt_tokens": 25,
        "completion_tokens": 128000,
        "reasoning_tokens": 127000,
        "requested_max_completion_tokens": 128000,
        "logical_max_completion_tokens": 128000,
        "physical_request_count": 1,
        "physical_prompt_tokens": 200,
        "segments": [{"kind": "chat", "finish_reason": "length"}],
        "latency_s": 2.0,
    }


class AsyncClientTests(unittest.TestCase):
    def test_retries_retryable_http_status_with_exponential_wrapper(self):
        async def run():
            client = AsyncChatClient(
                "https://openrouter.ai/api/v1",
                "deepseek/deepseek-v4-flash",
                "test-key",
                "high",
            )
            calls = 0

            async def post(payload: dict) -> tuple[dict, float]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    request = httpx.Request(
                        "POST",
                        "https://openrouter.ai/api/v1/chat/completions",
                    )
                    response = httpx.Response(503, request=request)
                    raise httpx.HTTPStatusError(
                        "service unavailable",
                        request=request,
                        response=response,
                    )
                return (
                    completion(
                        content="answer",
                        reasoning="reasoning",
                        reasoning_details=[],
                    ),
                    0.01,
                )

            client._post = post
            sleep = AsyncMock()
            try:
                with (
                    patch("async_client._backoff_delay", return_value=2.0),
                    patch("async_client.asyncio.sleep", sleep),
                ):
                    result = await client.chat_raw(
                        [{"role": "user", "content": "problem"}],
                        max_completion_tokens=128000,
                        temperature=1.0,
                        top_p=0.95,
                        seed=7,
                        request_id="retry-503",
                    )
            finally:
                await client.aclose()

            self.assertEqual(calls, 2)
            sleep.assert_awaited_once_with(2.0)
            self.assertEqual(result["physical_request_count"], 2)

        asyncio.run(run())

    def test_retries_stream_without_terminal_chunk(self):
        async def run():
            client = AsyncChatClient(
                "https://openrouter.ai/api/v1",
                "deepseek/deepseek-v4-flash",
                "test-key",
                "high",
            )
            calls = 0

            async def post(payload: dict) -> tuple[dict, float]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise IncompleteOpenRouterStreamError(
                        "OpenRouter stream ended without usage or finish reason"
                    )
                return (
                    completion(
                        content="answer",
                        reasoning="reasoning",
                        reasoning_details=[],
                    ),
                    0.01,
                )

            client._post = post
            try:
                with (
                    patch("async_client._backoff_delay", return_value=2.0),
                    patch(
                        "async_client.asyncio.sleep",
                        new=AsyncMock(),
                    ),
                ):
                    result = await client.chat_raw(
                        [{"role": "user", "content": "problem"}],
                        max_completion_tokens=128000,
                        temperature=1.0,
                        top_p=0.95,
                        seed=7,
                        request_id="retry-incomplete-stream",
                    )
            finally:
                await client.aclose()

            self.assertEqual(calls, 2)
            self.assertEqual(result["physical_request_count"], 2)

        asyncio.run(run())

    def test_chat_uses_exact_openrouter_contract(self):
        async def run():
            client = AsyncChatClient(
                "https://openrouter.ai/api/v1",
                "deepseek/deepseek-v4-flash",
                "test-key",
                "xhigh",
            )
            payloads: list[dict] = []

            async def post(payload: dict) -> tuple[dict, float]:
                payloads.append(payload)
                return (
                    completion(
                        content="answer",
                        reasoning="reasoning",
                        reasoning_details=[
                            {
                                "type": "reasoning.text",
                                "text": "reasoning",
                                "format": "unknown",
                                "index": 0,
                            }
                        ],
                    ),
                    0.01,
                )

            client._post = post
            try:
                self.assertEqual(
                    client._client.headers["Authorization"],
                    "Bearer test-key",
                )
                self.assertEqual(
                    client._client.headers["Content-Type"],
                    "application/json",
                )
                result = await client.chat_raw(
                    [{"role": "user", "content": "large prompt"}],
                    max_completion_tokens=128000,
                    temperature=1.0,
                    top_p=0.95,
                    seed=7,
                    request_id="fixed-budget",
                )
            finally:
                await client.aclose()

            self.assertEqual(
                payloads,
                [
                    {
                        "model": "deepseek/deepseek-v4-flash",
                        "messages": [
                            {"role": "user", "content": "large prompt"}
                        ],
                        "max_tokens": 128000,
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "seed": 7,
                        "stream": True,
                        "reasoning": {"effort": "xhigh", "exclude": False},
                    }
                ],
            )
            self.assertEqual(result["message"]["content"], "answer")
            self.assertEqual(result["message"]["reasoning_content"], "reasoning")
            self.assertEqual(result["reasoning_tokens"], 100)
            self.assertEqual(result["requested_max_completion_tokens"], 128000)
            self.assertEqual(result["physical_request_count"], 1)

        asyncio.run(run())

    def test_stream_ignores_keepalives_and_reconstructs_response(self):
        async def run():
            client = AsyncChatClient(
                "https://openrouter.ai/api/v1",
                "deepseek/deepseek-v4-flash",
                "test-key",
                "high",
            )
            await client._client.aclose()

            reasoning_detail = {
                "type": "reasoning.text",
                "text": "reasoning",
                "format": "unknown",
                "index": 0,
            }
            events = [
                ": OPENROUTER PROCESSING",
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "",
                                    "reasoning": "reasoning",
                                    "reasoning_details": [reasoning_detail],
                                },
                                "finish_reason": None,
                            }
                        ],
                        "usage": None,
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": "answer"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": None,
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                            "prompt_tokens_details": {"cached_tokens": 2},
                            "completion_tokens_details": {"reasoning_tokens": 8},
                        },
                    }
                ),
                "data: [DONE]",
            ]

            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    text="\n\n".join(events) + "\n\n",
                )

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                headers={"Authorization": "Bearer test-key"},
            )
            try:
                result = await client.chat_raw(
                    [{"role": "user", "content": "problem"}],
                    max_completion_tokens=128000,
                    temperature=1.0,
                    top_p=0.95,
                    seed=7,
                    request_id="stream-test",
                )
            finally:
                await client.aclose()

            self.assertEqual(result["message"]["content"], "answer")
            self.assertEqual(result["message"]["reasoning_content"], "reasoning")
            self.assertEqual(
                result["message"]["reasoning_details"], [reasoning_detail]
            )
            self.assertEqual(result["finish_reason"], "stop")
            self.assertEqual(result["reasoning_tokens"], 8)

        asyncio.run(run())

    def test_solution_continuation_uses_reasoning_details_and_assistant_prefill(self):
        async def run():
            client = AsyncChatClient(
                "https://openrouter.ai/api/v1",
                "deepseek/deepseek-v4-flash",
                "test-key",
                "high",
            )
            payloads: list[dict] = []

            async def post(payload: dict) -> tuple[dict, float]:
                payloads.append(payload)
                return (
                    completion(
                        content=(
                            "Proof body.</solution>\n"
                            "<self_evaluation>Checked.</self_evaluation>\n"
                            "<score>1</score>"
                        ),
                        reasoning="continued reasoning",
                        reasoning_details=[
                            {
                                "type": "reasoning.text",
                                "text": "continued reasoning",
                                "format": "unknown",
                                "index": 0,
                            }
                        ],
                        prompt_tokens=128200,
                        completion_tokens=200,
                        reasoning_tokens=50,
                    ),
                    3.0,
                )

            client._post = post
            original_messages = [{"role": "user", "content": "problem"}]
            try:
                result = await client.continue_solution_raw(
                    initial_response(content=""),
                    original_messages,
                    max_new_tokens=16384,
                    temperature=1.0,
                    top_p=0.95,
                    seed=7,
                    request_id="round-01/generate/r01-p0000",
                )
            finally:
                await client.aclose()

            prefill = payloads[0]["messages"][-1]
            self.assertEqual(prefill["role"], "assistant")
            self.assertEqual(prefill["content"], "<solution>\n")
            self.assertEqual(
                prefill["reasoning_details"],
                initial_response(content="")["message"]["reasoning_details"],
            )
            self.assertNotIn("reasoning", prefill)
            self.assertEqual(payloads[0]["max_tokens"], 16384)
            self.assertEqual(payloads[0]["seed"], 7)
            self.assertEqual(
                result["message"]["content"],
                "<solution>\nProof body.</solution>\n"
                "<self_evaluation>Checked.</self_evaluation>\n"
                "<score>1</score>",
            )
            self.assertEqual(result["logical_max_completion_tokens"], 144384)
            self.assertEqual(result["physical_request_count"], 2)
            self.assertEqual(result["physical_prompt_tokens"], 128400)
            self.assertTrue(result["segments"][1]["injected_solution_tag"])

        asyncio.run(run())

    def test_verifier_continuation_preserves_partial_visible_content(self):
        async def run():
            client = AsyncChatClient(
                "https://openrouter.ai/api/v1",
                "deepseek/deepseek-v4-flash",
                "test-key",
                "high",
            )
            payloads: list[dict] = []

            async def post(payload: dict) -> tuple[dict, float]:
                payloads.append(payload)
                return (
                    completion(
                        content="continued</evaluation><score>0.75</score>",
                        reasoning="continued reasoning",
                        reasoning_details=[],
                    ),
                    1.0,
                )

            client._post = post
            try:
                result = await client.continue_verification_raw(
                    initial_response(content="partial evaluation "),
                    [{"role": "user", "content": "verify"}],
                    max_new_tokens=16384,
                    temperature=1.0,
                    top_p=0.95,
                    seed=11,
                    request_id="round-01/verify/r01-p0000/v000",
                )
            finally:
                await client.aclose()

            self.assertEqual(
                payloads[0]["messages"][-1]["content"],
                "<evaluation>\npartial evaluation ",
            )
            self.assertEqual(
                result["message"]["content"],
                "<evaluation>\npartial evaluation continued"
                "</evaluation><score>0.75</score>",
            )
            self.assertTrue(result["segments"][1]["injected_verifier_tag"])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
