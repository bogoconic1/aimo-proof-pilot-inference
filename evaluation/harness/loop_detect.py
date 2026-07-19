"""Degenerate-loop detection for long-CoT proof/verify generations.

Faithful port of Yi-Chia Chen's original Proof-Pilot v2 detectors
(`proof_agent/v2/zlib_runaway_detector.py` + `loopguard.py`), adapted for our
harness's **blocking** completion API: we score the *finished* text post-hoc
instead of streaming. Same signals, same validated thresholds.

Why gzip (not a repetition penalty): a repetition penalty warps the sampling
distribution and corrupts legitimately-repetitive math reasoning (enumerations,
re-derivations). Aborting/rejecting a doomed generation only truncates it — it
does NOT change the distribution — so it is safe on on-policy OPD rollouts.

Two independent detectors; a text is degenerate if EITHER fires:

1. zlib runaway (primary). Signal = zlib_ratio = compressed/raw of a sliding
   12k-char window (lower = more repetitive; genuine reasoning ~0.3, loops ->0).
     - HARD: ratio < 0.05                          -> degenerate (hard token loop).
     - SOFT: ratio < 0.18 for >= 20 consecutive checks -> degenerate.
   The SOFT persistence requirement is what spares legit long math: a real
   enumeration dips below 0.18 but recovers within a few checks; a true loop
   stays sub-0.18 for 60+. Validated on OPD-32B ProofBench: 100% loop catch,
   0/1008 false positives on clean long generations.

2. loopguard local-density (backstop). A verbatim `chunk`=25-char segment
   recurring > `threshold`=8 times within a `span`=1500-char window. Calibrated:
   genuine small-case enumeration tops out ~4; real loops sit at 20+.
"""
from __future__ import annotations

import zlib
from collections import deque

# --- zlib runaway detector (Yi-Chia's validated defaults) ---
WINDOW_CHARS = 12_000
STEP_CHARS = 1_000
HARD_RATIO = 0.05
SOFT_RATIO = 0.18
SOFT_PERSIST = 20

# --- loopguard local-density backstop (Yi-Chia's validated defaults) ---
LG_CHUNK = 25
LG_STEP = 5
LG_THRESHOLD = 8
LG_SPAN = 1500


def zlib_ratio(text: str) -> float:
    """Compressed/raw size ratio. Lower = more repetitive. Empty -> 1.0."""
    b = text.encode("utf-8", "ignore")
    if not b:
        return 1.0
    return len(zlib.compress(b, 6)) / len(b)


def zlib_runaway(
    text: str,
    *,
    window_chars: int = WINDOW_CHARS,
    step_chars: int = STEP_CHARS,
    hard_ratio: float = HARD_RATIO,
    soft_ratio: float = SOFT_RATIO,
    soft_persist: int = SOFT_PERSIST,
) -> bool:
    """True if a sliding zlib window ever trips the hard tier, or the soft tier
    for `soft_persist` consecutive checks. Mirrors the streaming detector run
    offline over the complete text."""
    win: deque[str] = deque(maxlen=window_chars)
    since_check = 0
    soft_run = 0
    for ch in text or "":
        win.append(ch)
        since_check += 1
        if since_check >= step_chars and len(win) >= window_chars:
            since_check = 0
            ratio = zlib_ratio("".join(win))
            if ratio < hard_ratio:
                return True
            if ratio < soft_ratio:
                soft_run += 1
                if soft_run >= soft_persist:
                    return True
            else:
                soft_run = 0
    return False


def loopguard_degenerate(
    text: str,
    *,
    chunk: int = LG_CHUNK,
    step: int = LG_STEP,
    threshold: int = LG_THRESHOLD,
    span: int = LG_SPAN,
) -> bool:
    """True only for a real local loop: one `chunk`-char segment repeated
    > `threshold` times within a `span`-char window. Scattered recurrence and
    small-case checking (spread out, or each instance differs) do NOT trip."""
    t = text or ""
    if len(t) < chunk * 2:
        return False
    pos: dict[str, list[int]] = {}
    for i in range(0, len(t) - chunk, step):
        pos.setdefault(t[i:i + chunk], []).append(i)
    for offs in pos.values():
        if len(offs) <= threshold:
            continue
        j = 0
        for k in range(len(offs)):
            while offs[k] - offs[j] > span:
                j += 1
            if k - j + 1 > threshold:
                return True
    return False


def is_degenerate(text: str) -> bool:
    """A generation is degenerate if the zlib runaway detector OR the loopguard
    local-density backstop fires. Cheap (~ms) and distribution-neutral."""
    return zlib_runaway(text) or loopguard_degenerate(text)
