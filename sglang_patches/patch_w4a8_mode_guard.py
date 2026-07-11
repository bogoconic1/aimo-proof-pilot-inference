#!/usr/bin/env python3
"""Gate the Humming import and emit proof when a W4A8 layer is constructed."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


RELATIVE_PATH = Path(
    "layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py"
)
UNGUARDED = "        if _humming_mod().humming_dispatch(layer, x):"
GUARDED = (
    "        if _humming_enabled() and _humming_mod().humming_dispatch(layer, x):"
)
BUILD = "            built = hm.build_humming_w4a8(layer, self.group_size, self.symmetric)"
MARKER = (
    BUILD
    + "\n            if built:\n"
    + '                logger.info("HUMMING_W4A8_LAYER_READY device=%s group_size=%s", '\
    + "layer.weight_packed.device, self.group_size)"
)


def patch_source(source: str) -> str:
    if GUARDED not in source:
        if source.count(UNGUARDED) != 1:
            raise RuntimeError("Expected exactly one unguarded Humming dispatch")
        source = source.replace(UNGUARDED, GUARDED, 1)
    if source.count(GUARDED) != 1:
        raise RuntimeError("Expected exactly one guarded Humming dispatch")
    if MARKER not in source:
        if source.count(BUILD) != 1:
            raise RuntimeError("Expected exactly one Humming build call")
        source = source.replace(BUILD, MARKER, 1)
    if source.count("HUMMING_W4A8_LAYER_READY") != 1:
        raise RuntimeError("Expected exactly one Humming runtime marker")
    return source


def patch_venv(venv: Path) -> None:
    roots = list(venv.glob("lib/python*/site-packages/sglang/srt"))
    if len(roots) != 1:
        raise RuntimeError(f"Expected one sglang/srt under {venv}, found {roots}")
    path = roots[0] / RELATIVE_PATH
    original = path.read_text()
    patched = patch_source(original)
    if patched != original:
        backup = path.with_suffix(path.suffix + ".pre_w4a8_mode_guard")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(patched)
        print(f"  patched: {path.relative_to(roots[0])}")
    else:
        print(f"  verified: {path.relative_to(roots[0])}")
    for pyc in path.parent.glob("compressed_tensors_wNa16*.pyc"):
        pyc.unlink()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <venv_path>")
    patch_venv(Path(sys.argv[1]).resolve())
    print("[patch] W4A8 mode guard and runtime marker verified")


if __name__ == "__main__":
    main()
