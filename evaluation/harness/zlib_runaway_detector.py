"""Streaming zlib-based runaway / loop detector for long-CoT generations.

Detects degenerate repetition ("infinite loops") in a streaming generation so
the request can be aborted early instead of running all the way to the token
cap. Aborting only truncates a doomed generation -- it does NOT change the
sampling distribution, so it is safe to use on on-policy OPD rollouts (see
memory opd-rollout-no-distribution-change).

Signal -- zlib compression ratio (compressed / raw) of a sliding window:
LZ77 collapses repeated substrings, so a window stuck in a loop compresses to
~0 while genuine varied reasoning stays ~0.3. The ratio is a cheap,
language-agnostic, model-free proxy for "how repetitive is this window"
(~tens of microseconds on a 12 KB window). It can run on detokenized text or
directly on the byte stream of token ids.

Two-tier decision:
  - HARD: ratio < hard_ratio                         -> abort immediately.
          Catches degenerate token loops (" a?", "+2+2", "1,1,") whose ratio
          crashes far below anything legitimate.
  - SOFT: ratio < soft_ratio sustained for           -> abort.
          >= soft_persist consecutive checks
          Catches paragraph / semantic near-loops. The persistence requirement
          is what spares transient structured output (enumerations, long
          arithmetic), which dip but recover within a few checks.

Empirical basis -- OPD-32B v33/s200 IMO-ProofBench run, full 60-problem agentic
eval (evaluation/results/olmo3_32b_proofbench/loop_analysis.md):
  * 53 cap-hit generations: all loop types caught.
  * 1008 long clean-EOS generations: only 3 tripped the single-threshold form
    (0.30%), all legitimate enumeration/arithmetic that recovered.
  * Their longest sustained-low run was 5/5/9 checks vs >=62 for every real
    loop -> soft_persist=20 gives 0% FP / 100% catch on this dataset.
N.B. that FP sample is tiny (n=3); re-validate on more data before relying on
the soft tier in production.

Defaults below are the validated thresholds.

Streaming use:
    det = RunawayDetector()
    for chunk in stream:                 # chunk = newly decoded text (str)
        v = det.feed(chunk)
        if v.abort:
            engine.abort(); break

Offline use:
    v = scan(full_reasoning_text)
    if v.abort: ...                      # v.position ~ char offset of abort
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# Validated defaults (see module docstring / loop_analysis.md).
WINDOW_CHARS = 12_000
STEP_CHARS = 1_000
HARD_RATIO = 0.05
SOFT_RATIO = 0.18
SOFT_PERSIST = 20


def zlib_ratio(s: str | bytes) -> float:
    """Compressed/raw size ratio. Lower = more repetitive. Empty -> 1.0."""
    b = s.encode("utf-8", "ignore") if isinstance(s, str) else s
    if not b:
        return 1.0
    return len(zlib.compress(b, 6)) / len(b)


@dataclass
class Verdict:
    abort: bool
    reason: str | None = None  # "hard" | "soft" | None
    ratio: float | None = None  # ratio at the deciding/last check
    position: int = 0  # total chars consumed when decided
    soft_run: int = 0  # consecutive sub-soft_ratio checks so far


class RunawayDetector:
    """Sliding-window zlib loop detector. Feed text incrementally via feed()."""

    def __init__(
        self,
        window_chars: int = WINDOW_CHARS,
        step_chars: int = STEP_CHARS,
        hard_ratio: float = HARD_RATIO,
        soft_ratio: float = SOFT_RATIO,
        soft_persist: int = SOFT_PERSIST,
    ) -> None:
        self.window_chars = window_chars
        self.step_chars = step_chars
        self.hard_ratio = hard_ratio
        self.soft_ratio = soft_ratio
        self.soft_persist = soft_persist
        self.reset()

    def reset(self) -> None:
        self._win: deque[str] = deque(maxlen=self.window_chars)
        self._since_check = 0
        self._total = 0
        self._soft_run = 0
        self._aborted = False
        self._last_ratio: float | None = None

    def feed(self, text: str) -> Verdict:
        """Consume newly generated text; evaluate at every step boundary crossed.

        Returns a Verdict each call. Once it aborts, subsequent calls keep
        returning the abort verdict (call reset() to reuse the instance).
        """
        if self._aborted:
            return Verdict(True, "aborted", self._last_ratio, self._total, self._soft_run)
        for ch in text:
            self._win.append(ch)
            self._total += 1
            self._since_check += 1
            if self._since_check >= self.step_chars and len(self._win) >= self.window_chars:
                self._since_check = 0
                v = self._check()
                if v.abort:
                    self._aborted = True
                    return v
        return Verdict(False, None, self._last_ratio, self._total, self._soft_run)

    def _check(self) -> Verdict:
        ratio = zlib_ratio("".join(self._win))
        self._last_ratio = ratio
        if ratio < self.hard_ratio:
            return Verdict(True, "hard", ratio, self._total, self._soft_run)
        if ratio < self.soft_ratio:
            self._soft_run += 1
            if self._soft_run >= self.soft_persist:
                return Verdict(True, "soft", ratio, self._total, self._soft_run)
        else:
            self._soft_run = 0
        return Verdict(False, None, ratio, self._total, self._soft_run)


def scan(text: str, **kwargs) -> Verdict:
    """Run the detector over a complete string; return the first abort verdict
    (or a non-abort verdict if it never trips)."""
    return RunawayDetector(**kwargs).feed(text)


def _iter_stage_candidates(path: Path):
    """Yield (label, reasoning_content, finish_reason) from a stages/*.json."""
    d = json.loads(path.read_text())
    for stage in ("prove", "verify", "refine", "select"):
        for c in d.get("stages", {}).get(stage, []):
            cid = c.get("candidate_id") or c.get("refiner_id") or c.get("selector_id") or "?"
            yield f"{stage}/{cid}", c.get("reasoning_content") or "", c.get("finish_reason")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="raw text file, or a ProofBench stages/*.json (--stage-json)")
    ap.add_argument("--stage-json", action="store_true", help="treat path as a stages/*.json and scan every candidate")
    ap.add_argument("--window", type=int, default=WINDOW_CHARS)
    ap.add_argument("--step", type=int, default=STEP_CHARS)
    ap.add_argument("--hard", type=float, default=HARD_RATIO)
    ap.add_argument("--soft", type=float, default=SOFT_RATIO)
    ap.add_argument("--soft-persist", type=int, default=SOFT_PERSIST)
    args = ap.parse_args()
    kw = dict(window_chars=args.window, step_chars=args.step, hard_ratio=args.hard,
              soft_ratio=args.soft, soft_persist=args.soft_persist)
    p = Path(args.path)

    if args.stage_json:
        n = n_abort = 0
        for label, rc, finish in _iter_stage_candidates(p):
            n += 1
            v = scan(rc, **kw)
            flag = "ABORT" if v.abort else "ok   "
            if v.abort:
                n_abort += 1
            mark = "  [cap-hit]" if finish == "length" else ""
            pos = f"@{v.position}" if v.abort else ""
            rr = f"{v.ratio:.3f}" if v.ratio is not None else "n/a"
            print(f"  {flag} {label:14} len={len(rc):7} reason={v.reason or '-':5} ratio={rr} {pos}{mark}")
        print(f"\n{n_abort}/{n} candidates would abort")
    else:
        v = scan(p.read_text(), **kw)
        print(json.dumps(v.__dict__, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
