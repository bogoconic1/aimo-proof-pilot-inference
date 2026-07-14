"""Math-3R prompts, verifier audit roles, renderers, bundles, and XML parsers.

The prover and refiner templates are copied byte-for-byte from the ycchen-tw
proof-pilot-codes commit bc03a2c71a076990deaad3d712c6889682e12c69. The verifier
retains that XML contract but adds deterministic rubric-blind audit focuses.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "ycchen_math_3r"
PROMPT_SOURCE_COMMIT = "bc03a2c71a076990deaad3d712c6889682e12c69"
SYSTEM_DELIMITER = "===SYSTEM==="
USER_DELIMITER = "===USER==="

VERIFIER_AUDIT_FOCI = {
    "logical_validity": (
        "Trace every deduction and check definitions, implications, algebra, "
        "and dependencies. Look especially for circular reasoning, invalid "
        "converses, and conclusions that do not follow from the stated premises."
    ),
    "proof_completeness": (
        "Inventory the nontrivial claims the proof uses and verify that each is "
        "actually established. Treat words such as clearly, similarly, and one "
        "checks as unsupported unless the required argument is present."
    ),
    "case_coverage": (
        "Check that every case split is exhaustive and that boundary, equality, "
        "degenerate, base, endpoint, and domain cases are handled wherever they "
        "can affect the argument."
    ),
    "adversarial_falsification": (
        "Try to falsify each lemma and coverage claim with minimal, extreme, or "
        "degenerate examples. Accept a claim only after the attempted "
        "counterexamples are ruled out by the written proof."
    ),
}
VERIFIER_AUDIT_ROLES = tuple(VERIFIER_AUDIT_FOCI)

_GENERATION = re.compile(
    r"\s*<solution>(.*?)</solution>\s*"
    r"<self_evaluation>(.*?)</self_evaluation>\s*"
    r"<score>\s*(0(?:\.5)?|1)\s*</score>\s*",
    re.DOTALL,
)
_VERIFICATION = re.compile(
    r"\s*<evaluation>(.*?)</evaluation>\s*"
    r"<suggestions>(.*?)</suggestions>\s*"
    r"<score>\s*(0(?:\.5)?|1)\s*</score>\s*",
    re.DOTALL,
)


@lru_cache(maxsize=None)
def template(name: str) -> str:
    return (PROMPT_ROOT / name).read_text()


def prompt_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((PROMPT_ROOT / name).read_bytes()).hexdigest()
        for name in ("prover.txt", "verifier.txt", "refiner.txt")
    }


def _messages(rendered: str) -> list[dict[str, str]]:
    system, user = rendered.split(USER_DELIMITER, 1)
    if not system.startswith(SYSTEM_DELIMITER):
        raise ValueError("ycchen prompt lacks the system delimiter")
    return [
        {"role": "system", "content": system.removeprefix(SYSTEM_DELIMITER).strip()},
        {"role": "user", "content": user.strip()},
    ]


def generation_messages(problem: str) -> list[dict[str, str]]:
    return _messages(template("prover.txt").replace("{problem}", problem))


def verification_messages(
    problem: str,
    proof: str,
    self_evaluation: str,
    audit_role: str,
) -> list[dict[str, str]]:
    try:
        audit_focus = VERIFIER_AUDIT_FOCI[audit_role]
    except KeyError as error:
        raise ValueError(f"unknown verifier audit role: {audit_role}") from error
    rendered = (
        template("verifier.txt")
        .replace("{problem}", problem)
        .replace("{candidate_solution}", proof)
        .replace("{candidate_self_eval}", self_evaluation)
        .replace("{audit_focus}", audit_focus)
    )
    return _messages(rendered)


def verifier_audit_role(index: int) -> str:
    if type(index) is not int or index < 0:
        raise ValueError("verifier audit index must be a non-negative integer")
    return VERIFIER_AUDIT_ROLES[index % len(VERIFIER_AUDIT_ROLES)]


def refinement_messages(
    problem: str,
    candidate_id: str,
    proof: str,
    self_evaluation: str,
    review_score: float,
    review: str,
) -> list[dict[str, str]]:
    parts = [
        f'<candidate id="{candidate_id}">',
        "<proof>",
        proof,
        "</proof>",
        f'<verifier_review score="{review_score:g}">',
        review,
        "</verifier_review>",
        "<self_evaluation>",
        self_evaluation,
        "</self_evaluation>",
        "</candidate>",
    ]
    rendered = (
        template("refiner.txt")
        .replace("{problem}", problem)
        .replace("{candidate_bundle}", "\n".join(parts))
    )
    return _messages(rendered)


def parse_generation(text: str) -> tuple[str, str, float]:
    match = _GENERATION.fullmatch(text)
    if match is None:
        raise ValueError("generation does not match ycchen's XML contract")
    proof, self_evaluation, score = match.groups()
    proof = proof.strip()
    self_evaluation = self_evaluation.strip()
    if not proof or not self_evaluation:
        raise ValueError("generation contains an empty required XML element")
    return proof, self_evaluation, float(score)


def parse_verification(text: str) -> tuple[str, float]:
    match = _VERIFICATION.fullmatch(text)
    if match is None:
        raise ValueError("verification does not match ycchen's XML contract")
    evaluation, suggestions, score = match.groups()
    if not evaluation.strip() or not suggestions.strip():
        raise ValueError("verification contains an empty required XML element")
    return text.strip(), float(score)
