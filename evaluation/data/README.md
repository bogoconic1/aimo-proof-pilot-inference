# Pinned MathArena evaluation data

The evaluation runner selects its dataset explicitly from the problem manifest.
No dataset is inferred from a problem ID and there is no fallback source.

| Manifest dataset | Pinned file | SHA-256 | Source |
|---|---|---|---|
| `imo_2025` | `imo_2025.parquet` | `17592c82ae91049ae6215b3cece719fa62d37bcb82f9df16719d436797d03a6f` | `MathArena/imo_2025` |
| `aime_2026` | `aime_2026.parquet` | `d91db799651b4cc1f0734f52792a695c9cc60dac342524b3d8e5b2ff31c3e957` | `MathArena/aime_2026` |

The parquet files are copied without modification from Hugging Face. The AIME
2026 dataset contains 30 records with `problem_idx`, `problem`, and the official
integer `answer`. Its Problem 10 row has answer `156`.

Problem text is fed unchanged to ycchen’s prover, verifier, and refiner prompts.
The selected dataset does not alter the configured search schedule or serving
mode.
