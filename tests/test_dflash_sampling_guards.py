from __future__ import annotations

import unittest
from types import SimpleNamespace

from sglang_patches.patch_dflash_sampling import (
    UNIFORM_MARKER,
    VALIDATION_MARKER,
    patch_dflash_utils_text,
)

try:
    from sglang.srt.speculative.dflash_utils import validate_dflash_request
except ImportError:
    validate_dflash_request = None


class SamplingPatchTransformTests(unittest.TestCase):
    def test_transform_adds_both_guards_and_is_idempotent(self) -> None:
        source = """
def compute_dflash_sampling_correct_drafts_and_bonus():
    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))

def validate_dflash_request(req, enable_overlap):
    return None
"""
        patched = patch_dflash_utils_text(source)
        self.assertIn(UNIFORM_MARKER, patched)
        self.assertIn(VALIDATION_MARKER, patched)
        self.assertEqual(patch_dflash_utils_text(patched), patched)

    def test_transform_fails_when_source_shape_changes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "validate_dflash_request"):
            patch_dflash_utils_text("def unrelated(): pass\n")


def _params(**overrides):
    values = {
        "json_schema": None,
        "regex": None,
        "ebnf": None,
        "structural_tag": None,
        "min_p": 0.0,
        "min_new_tokens": 0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "top_k": 1 << 30,
        "top_p": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _req(**param_overrides):
    return SimpleNamespace(
        return_logprob=False,
        return_hidden_states=False,
        custom_logit_processor=None,
        sampling_params=_params(**param_overrides),
    )


@unittest.skipIf(validate_dflash_request is None, "patched SGLang runtime is not installed")
class DFlashSamplingValidationTests(unittest.TestCase):
    def assert_rejected(self, expected: str, **params) -> None:
        message = validate_dflash_request(_req(**params), enable_overlap=True)
        self.assertIsNotNone(message)
        self.assertIn(expected, message)

    def test_production_top_p_only_is_supported(self) -> None:
        self.assertIsNone(
            validate_dflash_request(
                _req(top_p=0.95),
                enable_overlap=True,
            )
        )

    def test_top_k_only_is_supported(self) -> None:
        self.assertIsNone(
            validate_dflash_request(
                _req(top_k=20),
                enable_overlap=True,
            )
        )

    def test_greedy_with_irrelevant_top_p_is_supported(self) -> None:
        self.assertIsNone(
            validate_dflash_request(
                _req(top_k=1, top_p=0.95),
                enable_overlap=True,
            )
        )

    def test_min_p_is_rejected(self) -> None:
        self.assert_rejected("min_p", min_p=0.1)

    def test_min_new_tokens_is_rejected(self) -> None:
        self.assert_rejected("min_new_tokens", min_new_tokens=3)

    def test_frequency_penalty_is_rejected(self) -> None:
        self.assert_rejected("penalt", frequency_penalty=0.2)

    def test_presence_penalty_is_rejected(self) -> None:
        self.assert_rejected("penalt", presence_penalty=0.2)

    def test_repetition_penalty_is_rejected(self) -> None:
        self.assert_rejected("penalt", repetition_penalty=1.1)

    def test_combined_top_k_top_p_is_rejected(self) -> None:
        self.assert_rejected("combined top_k", top_k=20, top_p=0.95)

    def test_existing_logprob_rejection_remains(self) -> None:
        req = _req()
        req.return_logprob = True
        self.assertIn(
            "return_logprob",
            validate_dflash_request(req, enable_overlap=True),
        )

    def test_custom_logit_processor_is_rejected(self) -> None:
        req = _req()
        req.custom_logit_processor = "serialized"
        self.assertIn(
            "custom logit",
            validate_dflash_request(req, enable_overlap=True),
        )
