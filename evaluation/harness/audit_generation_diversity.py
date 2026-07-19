#!/usr/bin/env python3
"""Audit exact first-round proof diversity against a running SGLang server."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from async_client import AsyncChatClient
from eval_config import active_model, load_config
from proof_prompts import parse_generation
from proof_search import ProblemSearch
from run_submission import load_test_csv


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def summarize_records(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["full_sha256"]].append(record)
    duplicate_groups = [
        {
            "count": len(group),
            "proof_ids": [record["proof_id"] for record in group],
            "requested_seeds": [record["requested_seed"] for record in group],
            "completion_tokens": [record["completion_tokens"] for record in group],
            "full_sha256": digest,
        }
        for digest, group in sorted(groups.items())
        if len(group) > 1
    ]
    valid = [record for record in records if record["proof_valid"]]
    return {
        "requests": len(records),
        "valid_proofs": len(valid),
        "unique_full_outputs": len(groups),
        "unique_valid_proofs": len(
            {record["proof_sha256"] for record in valid}
        ),
        "requests_in_duplicate_groups": sum(
            group["count"] for group in duplicate_groups
        ),
        "duplicate_groups": duplicate_groups,
    }


async def run(args: argparse.Namespace) -> None:
    if args.artifacts_dir.exists():
        raise RuntimeError(
            f"artifacts directory already exists: {args.artifacts_dir}"
        )
    config = load_config(args.config)
    inputs = load_test_csv(args.input)
    if not 0 <= args.problem_index < len(inputs):
        raise ValueError(
            f"problem index {args.problem_index} is outside 0..{len(inputs) - 1}"
        )
    row = inputs[args.problem_index]
    model = active_model(config)
    server = config["server"]
    search_config = config["search"]
    client = AsyncChatClient(
        f"http://{server['host']}:{server['port']}/v1",
        str(model.target),
        api_key="EMPTY",
        max_connections=search_config["concurrency"] + 8,
        timeout=float(search_config["request_timeout_seconds"]),
    )
    search = ProblemSearch(
        problem_id=f"row-{args.problem_index:04d}",
        problem=row.problem,
        output_dir=args.artifacts_dir,
        client=client,
        semaphore=asyncio.Semaphore(search_config["concurrency"]),
        config=search_config,
    )
    candidates = search._round_candidates(1)
    started = time.perf_counter()
    try:
        responses = await asyncio.gather(
            *(search._perform(candidate.generation) for candidate in candidates)
        )
    finally:
        await client.aclose()

    records = []
    for candidate, response in zip(candidates, responses, strict=True):
        content = response.get("content") or ""
        reasoning = response.get("reasoning_content") or ""
        try:
            proof, _, self_score = parse_generation(content)
        except ValueError:
            proof = ""
            self_score = None
            proof_valid = False
        else:
            proof_valid = True
        records.append(
            {
                "proof_id": candidate.proof_id,
                "sample_id": response["sample_id"],
                "requested_seed": response["seed"],
                "finish_reason": response.get("finish_reason"),
                "completion_tokens": response.get("completion_tokens"),
                "physical_request_count": response.get("physical_request_count"),
                "xml_valid": response.get("xml_valid"),
                "proof_valid": proof_valid,
                "self_score": self_score,
                "content_sha256": text_sha256(content),
                "reasoning_sha256": text_sha256(reasoning),
                "full_sha256": text_sha256(reasoning + "\0" + content),
                "proof_sha256": text_sha256(proof) if proof_valid else None,
            }
        )

    result = {
        "schema_version": 1,
        "server_determinism": args.server_determinism,
        "problem_index": args.problem_index,
        "problem_id": row.id,
        "problem_sha256": text_sha256(row.problem),
        "config_path": str(args.config.resolve()),
        "search_config": search_config,
        "elapsed_s": time.perf_counter() - started,
        **summarize_records(records),
        "records": records,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"records", "duplicate_groups", "search_config"}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--problem-index", required=True, type=int)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument(
        "--server-determinism", required=True, choices=("on", "off")
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
