"""Strip self-assessment 'meta' sections from template-formatted proof outputs.

The structured prompt templates (t1-t7) make DeepSeek emit, alongside the proof, meta
sections where the model judges its OWN correctness: a Verdict / Claimed score category /
Final status line, a Final self-audit, a Gap check, an Unresolved-issues list, or a
redundant Summary. Those (a) are not proof, (b) leak the model's self-assessment to the
grader, and (c) are unequal across templates (t0/t5 have none). For a fair cross-template
comparison we grade only the mathematical content.

Approach = REMOVE the known meta sections, keep everything else (idea/sketch/lemmas/proof
all stay). For each meta section we locate its header by keyword, then drop from that header
to the next top-level header (or end). If a template's expected meta header isn't found we
keep the full content (fail-safe) and flag it — we never grade an empty body.

Report mode (default): no files changed; prints per-template stats + dumps before/after
samples to runs/_extract_report/.  Apply mode (--apply): adds a `graded_text` field to each
candidate in every run's responses.jsonl (original `text` is left untouched).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent

# Per template: ordered section-title keywords with drop=True for the self-assessment
# 'meta' sections. We locate sections by TITLE keyword (not by numbering/markdown style,
# which varies: '## 1. X', '**1. X**', '## X', '**X:**' all occur). A meta section is
# dropped from its header to the next KNOWN section header (or end).
SECTIONS = {
    "t0": [],
    "t1": [("summary", True), ("detailed solution", False)],
    "t2": [("verdict", True), ("key idea", False), ("final proof", False)],
    "t3": [("main idea", False), ("lemmas", False), ("complete proof", False),
           ("self audit", True)],
    "t4": [("key plan", False), ("complete proof", False), ("gap check", True)],
    "t5": [],
    "t6": [("final status", True), ("main insight", False), ("detailed proof", False),
           ("unresolved", True)],
    "t7": [("claimed", True), ("proof", False)],
}


_LATEX_HDR = re.compile(
    r"^\\(noindent|textbf|textit|paragraph|subparagraph|subsubsection|subsection|section)\b")


def _norm(s: str) -> str:
    """Lowercase, unify unicode hyphens, drop markdown/LaTeX/punct/number -> keyword space."""
    for h in ("‐", "‑", "‒", "–", "—"):
        s = s.replace(h, "-")
    s = s.lower()
    s = re.sub(r"[#*`_:.\-\\{}]", " ", s)
    s = re.sub(r"\d+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_headerish(line: str) -> bool:
    """Markdown/bold/number/LaTeX-marked, or a short plain line. Length is NOT capped for
    bold/markdown lines because a meta header can be a bold-led long paragraph
    ('**Self-audit:** The upper-bound argument ... <long>'). False headers (a meta keyword
    buried in long prose) are instead excluded by _title_at's start-anchor."""
    s = line.strip()
    return bool(s) and (s.startswith(("#", "*")) or bool(re.match(r"^\d+\.", s))
                        or bool(_LATEX_HDR.match(s)) or len(s) <= 60)


def _title_at(line: str, kw: str) -> bool:
    # headerish line whose TITLE contains kw NEAR THE START (after numbering/markdown/a short
    # qualifier like 'Final' or 'noindent textbf') — so a meta keyword buried in a long bolded
    # prose line ('All lemmas are proved...') is not mistaken for a 'lemmas' section header.
    if not _is_headerish(line):
        return False
    n = _norm(line)
    idx = n.find(kw)
    return idx >= 0 and len(n[:idx].split()) <= 3


def _presplit(text: str) -> str:
    """Put onto their own line any section header the model glued mid-line, so section
    detection sees it: '...required format.## 1. Summary', '...is complete.1. Verdict:'."""
    # only a '#'-run glued directly onto non-space text (not the 2nd '#' of a line-start '##')
    text = re.sub(r"(?<=[^\s#\n])(#{1,4}\s+\S)", r"\n\1", text)        # '...format.## 1. Summary'
    # a numbered header glued with NO space after the sentence period ('complete.1. Verdict');
    # legitimate '**Step. 4. ...**' has a space after the period, so it is left intact
    text = re.sub(r"(?<=[.\):])(\d+\.\s+[A-Z][a-z])", r"\n\1", text)
    return text


def strip_meta(sid: str, text: str) -> dict:
    """Return {graded_text, dropped: [kw...], fallback: bool, n_lines_dropped}."""
    secs = SECTIONS.get(sid, [])
    drop_kws = [k for k, drop in secs if drop]
    all_kws = [k for k, _ in secs]
    if not drop_kws or not text.strip():
        return {"graded_text": text, "dropped": [], "fallback": False, "n_lines_dropped": 0}
    lines = _presplit(text).split("\n")
    drop_lines: set[int] = set()
    dropped: list[str] = []
    for kw in drop_kws:
        # strip EVERY occurrence (models sometimes duplicate their whole answer -> 2 meta copies)
        starts = [i for i, l in enumerate(lines) if _title_at(l, kw)]
        for start in starts:
            # end = next line that is the header of ANY OTHER known section
            nxt = next((j for j in range(start + 1, len(lines))
                        if any(_title_at(lines[j], k) for k in all_kws if k != kw)), len(lines))
            drop_lines.update(range(start, nxt))
        if starts:
            dropped.append(kw)
    # t7 sometimes abbreviates "Claimed score category: X" to a bare verdict first line
    if sid == "t7" and not dropped:
        for i, l in enumerate(lines):
            if not l.strip():
                continue
            if re.match(r"^[*#\s]*(complete|almost\s+complete|partial)[*\s.:]*$",
                        l.strip(), re.I):
                drop_lines.add(i)
                dropped.append("bare-verdict")
            break  # only the first non-empty line can be the verdict
    if not dropped:  # no meta section present (or unmatched) -> keep full content, flag
        return {"graded_text": text, "dropped": [], "fallback": True, "n_lines_dropped": 0}
    kept = "\n".join(l for i, l in enumerate(lines) if i not in drop_lines).strip()
    return {"graded_text": kept, "dropped": dropped, "fallback": not kept,
            "n_lines_dropped": len(drop_lines)}


def _runs() -> list[tuple[str, Path]]:
    out = []
    for d in sorted(glob.glob(str(EVAL_ROOT / "runs" / "dsv4-flash__t*__high_notool"))):
        sid = os.path.basename(d).split("__")[1]
        out.append((sid, Path(d)))
    return out


def report(dump_n: int) -> None:
    rep_dir = EVAL_ROOT / "runs" / "_extract_report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'tmpl':>4} {'n':>4} {'stripped':>8} {'fb_clean':>8} {'fb_SUSP':>7} {'empty!':>6} "
          f"{'med_ratio':>9} {'min_ratio':>9}")
    for sid, d in _runs():
        recs = [json.loads(l) for l in (d / "responses.jsonl").open()]
        cands = [(r["problem_id"], j, c) for r in recs for j, c in enumerate(r["candidates"])]
        drop_kws = [k for k, drop in SECTIONS.get(sid, []) if drop]
        ratios, n_strip, n_fb_clean, n_fb_susp, n_empty, worst = [], 0, 0, 0, 0, []
        susp_ex = []
        for pid, j, c in cands:
            full = c.get("text") or ""
            res = strip_meta(sid, full)
            g = res["graded_text"]
            if full.strip():
                ratios.append(len(g) / max(len(full), 1))
            if res["dropped"]:
                n_strip += 1
                worst.append((len(g) / max(len(full), 1), pid, j, full, g, res["dropped"]))
            if res["fallback"] and drop_kws and full.strip():
                # missed? does any meta keyword still appear in the kept text?
                low = _norm(full)
                if any(kw in low for kw in drop_kws):
                    n_fb_susp += 1
                    if len(susp_ex) < dump_n:
                        susp_ex.append((pid, j, full))
                else:
                    n_fb_clean += 1
            if full.strip() and not g.strip():
                n_empty += 1
        worst.sort()
        for ratio, pid, j, full, g, dr in worst[:dump_n]:
            removed = "\n".join(l for l in full.split("\n") if l not in g.split("\n"))
            (rep_dir / f"{sid}__{pid}__c{j}.txt").write_text(
                f"# template {sid}  ratio={ratio:.2f}  dropped={dr}\n"
                f"\n===== REMOVED (meta) =====\n{removed}\n"
                f"\n===== KEPT (graded_text) =====\n{g}\n", encoding="utf-8")
        for pid, j, full in susp_ex:
            (rep_dir / f"SUSP_{sid}__{pid}__c{j}.txt").write_text(
                f"# template {sid}  SUSPECT fallback (meta keyword present but not stripped)\n"
                f"\n===== FULL TEXT =====\n{full}\n", encoding="utf-8")
        mr = round(median(ratios), 2) if ratios else None
        mn = round(min(ratios), 2) if ratios else None
        flag = "  <-- CHECK" if (n_empty or n_fb_susp) else ""
        print(f"{sid:>4} {len(cands):>4} {n_strip:>8} {n_fb_clean:>8} {n_fb_susp:>7} "
              f"{n_empty:>6} {mr:>9} {mn:>9}{flag}")
    print(f"\n[samples] biggest-removal -> {rep_dir}/{{sid}}__*.txt ; "
          f"suspect fallbacks -> {rep_dir}/SUSP_*.txt")


def apply() -> None:
    for sid, d in _runs():
        path = d / "responses.jsonl"
        recs = [json.loads(l) for l in path.open()]
        n = 0
        for r in recs:
            for c in r["candidates"]:
                res = strip_meta(sid, c.get("text") or "")
                c["graded_text"] = res["graded_text"]
                c["extract_dropped"] = res["dropped"]
                c["extract_fallback"] = res["fallback"]
                n += 1
        with path.open("w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[apply] {sid}: graded_text added to {n} candidates")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write graded_text into each responses.jsonl (default: report only)")
    ap.add_argument("--dump", type=int, default=3, help="samples per template to dump")
    args = ap.parse_args()
    if args.apply:
        apply()
    else:
        report(args.dump)


if __name__ == "__main__":
    main()
