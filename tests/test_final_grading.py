from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from grade_proofs import aggregate_grades, zero_veto_score  # noqa: E402
from grader import parse_score  # noqa: E402


class FinalGradingTests(unittest.TestCase):
    def test_zero_veto_overrides_all_other_attempts(self):
        scores = [7] * 63 + [0]
        self.assertEqual(zero_veto_score(scores, 64), 0.0)

    def test_no_zero_uses_arithmetic_mean(self):
        scores = [7] * 32 + [6] * 16 + [1] * 16
        self.assertEqual(zero_veto_score(scores, 64), sum(scores) / 64)

    def test_parser_accepts_every_integer_imo_score(self):
        for grade in range(8):
            with self.subTest(grade=grade):
                payload = {
                    "findings": ["Specific finding"],
                    "grade": grade,
                    "reasoning": "Guideline-based justification",
                }
                parsed = parse_score(json.dumps(payload))
                self.assertEqual(parsed["grade"], grade)

    def test_parser_rejects_scores_outside_the_imo_scale(self):
        payload = {
            "findings": ["Specific finding"],
            "grade": 8,
            "reasoning": "Guideline-based justification",
        }
        with self.assertRaisesRegex(ValueError, "off-scale grader grade"):
            parse_score(json.dumps(payload))

    def test_parser_rejects_reordered_fields(self):
        payload = {
            "grade": 7,
            "findings": ["Specific finding"],
            "reasoning": "Guideline-based justification",
        }
        with self.assertRaisesRegex(ValueError, "fields/order differ"):
            parse_score(json.dumps(payload))

    def test_prompt_and_request_require_strict_json_output(self):
        prompt = (REPO / "evaluation/prompts/grader.md").read_text()
        self.assertIn(
            "exactly three fields in this exact order: `\"findings\"`, "
            "`\"grade\"`, `\"reasoning\"`",
            prompt,
        )
        self.assertNotIn("<points>", prompt)
        source = (REPO / "evaluation/harness/grade_proofs.py").read_text()
        self.assertIn(
            'response_format={"type": "json_object"}',
            source,
        )

    def test_aggregate_requires_exact_attempt_sequence(self):
        records = [
            {"problem_id": "1", "attempt": attempt, "score": 7, "error": None}
            for attempt in range(64)
        ]
        summary = aggregate_grades(records, ["1"], 64)
        self.assertEqual(summary["problems"][0]["score_out_of_7"], 7)
        self.assertEqual(summary["overall_score_percent"], 100)
        with self.assertRaisesRegex(RuntimeError, "incomplete grader attempt sequence"):
            aggregate_grades(records[:-1], ["1"], 64)

    def test_aggregate_applies_zero_veto_per_problem_before_overall_mean(self):
        records = []
        for problem_id, scores in (("1", [7] * 64), ("2", [7] * 63 + [0])):
            records.extend(
                {
                    "problem_id": problem_id,
                    "attempt": attempt,
                    "score": score,
                    "error": None,
                }
                for attempt, score in enumerate(scores)
            )
        summary = aggregate_grades(records, ["1", "2"], 64)
        self.assertEqual(summary["overall_score_out_of_7"], 3.5)
        self.assertFalse(summary["problems"][0]["zero_veto_triggered"])
        self.assertTrue(summary["problems"][1]["zero_veto_triggered"])


if __name__ == "__main__":
    unittest.main()
