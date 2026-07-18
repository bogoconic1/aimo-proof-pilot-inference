from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from eval_config import load_config  # noqa: E402


class RuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((REPO / "config.yaml").read_text())

    def load_copy(self, config: dict) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            return load_config(path)

    def test_checked_in_config_uses_exact_openrouter_model(self):
        config = load_config(REPO / "config.yaml")
        self.assertEqual(config["schema_version"], 13)
        self.assertEqual(
            config["provider"],
            {
                "base_url": "https://openrouter.ai/api/v1",
                "model": "deepseek/deepseek-v4-flash",
                "api_key_env": "OPENROUTER_API_KEY",
                "reasoning_effort": "high",
            },
        )

    def test_reasoning_effort_is_passed_through(self):
        efforts = ("max", "xhigh", "high", "medium", "low", "minimal", "none")
        for effort in efforts:
            with self.subTest(effort=effort):
                config = copy.deepcopy(self.config)
                config["provider"]["reasoning_effort"] = effort
                self.assertEqual(
                    self.load_copy(config)["provider"]["reasoning_effort"], effort
                )

    def test_search_policy_is_unchanged(self):
        search = load_config(REPO / "config.yaml")["search"]
        self.assertEqual(search["proofs_per_round"], 32)
        self.assertEqual(search["verifications_per_proof"], 16)
        self.assertEqual(search["top_proofs"], 8)
        self.assertEqual(search["refinements_per_proof"], 4)
        self.assertEqual(search["analyses_per_refinement"], 4)
        self.assertEqual(search["max_rounds"], 4)
        self.assertEqual(search["temperature"], 1.0)
        self.assertEqual(search["top_p"], 0.95)
        self.assertEqual(search["max_completion_tokens"], 128000)
        self.assertEqual(search["solution_continuation_tokens"], 16384)
        self.assertEqual(search["verifier_continuation_tokens"], 16384)
        self.assertEqual(search["concurrency"], 96)
        self.assertEqual(search["seed"], 0)

    def test_rejects_old_local_runtime_sections(self):
        config = copy.deepcopy(self.config)
        config["server"] = {}
        with self.assertRaisesRegex(ValueError, "root keys differ"):
            self.load_copy(config)

    def test_container_submission_requires_config(self):
        env = os.environ.copy()
        env.pop("CONFIG", None)
        result = subprocess.run(
            ["bash", "docker/entrypoint.sh", "submission"],
            cwd=REPO,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CONFIG is required", result.stderr)

    def test_runtime_has_no_gpu_or_sglang_dependency(self):
        dockerfile = (REPO / "Dockerfile").read_text()
        requirements = (REPO / "evaluation/requirements.txt").read_text()
        entrypoint = (REPO / "docker/entrypoint.sh").read_text()
        self.assertNotIn("nvidia", dockerfile.lower())
        self.assertNotIn("cuda", dockerfile.lower())
        self.assertNotIn("sglang", entrypoint.lower())
        self.assertNotIn("flash-attn", requirements)
        self.assertNotIn("cutlass", requirements)


if __name__ == "__main__":
    unittest.main()
