import sys
import unittest
from pathlib import Path


HARNESS = Path(__file__).resolve().parents[1] / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from audit_generation_diversity import summarize_records  # noqa: E402


class GenerationDiversitySummaryTests(unittest.TestCase):
    def test_reports_duplicate_groups_and_distinct_seeds(self):
        records = [
            {
                "proof_id": "p0",
                "requested_seed": 10,
                "completion_tokens": 100,
                "full_sha256": "same",
                "proof_sha256": "proof-same",
                "proof_valid": True,
            },
            {
                "proof_id": "p1",
                "requested_seed": 11,
                "completion_tokens": 100,
                "full_sha256": "same",
                "proof_sha256": "proof-same",
                "proof_valid": True,
            },
            {
                "proof_id": "p2",
                "requested_seed": 12,
                "completion_tokens": 80,
                "full_sha256": "other",
                "proof_sha256": "proof-other",
                "proof_valid": True,
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["valid_proofs"], 3)
        self.assertEqual(summary["unique_full_outputs"], 2)
        self.assertEqual(summary["unique_valid_proofs"], 2)
        self.assertEqual(summary["requests_in_duplicate_groups"], 2)
        self.assertEqual(
            summary["duplicate_groups"][0]["requested_seeds"], [10, 11]
        )


if __name__ == "__main__":
    unittest.main()
