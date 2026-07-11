"""Cross-template analysis of the prompt-template sweep grades.

Reads each template run's grades (per-candidate, per-pass) + responses (claimed_status),
computes per-template best-of-3 / mean-of-3 scores by subset/level, paired win-loss vs the
t0 baseline, and overclaiming (claimed completeness vs actual grade). Writes a markdown
report to results/template_sweep.md.

  python analyze_sweep.py --grades grades_flashHigh_2pass.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

# templates that emit a completeness verdict (others have no self-claim)
VERDICT_TEMPLATES = {"t1", "t2", "t3", "t4", "t6", "t7"}


def parse_claim(tid: str, text: str) -> str | None:
    """Re-derive the model's claimed completeness from the raw output (more reliable than the
    generic generation-time heuristic, which missed t7's 'Claimed score category: X')."""
    if tid not in VERDICT_TEMPLATES or not text.strip():
        return None
    h = re.sub(r"[*#`>_]", " ", text)
    # the verdict section can be at the top (t1/t2/t6/t7) or the end (t3 self-audit, t4 gap
    # check) — search the whole text for the marker, then classify the ~150 chars after it
    m = re.search(r"(verdict|final status|claimed score category|gap check|self.?audit)\s*:?\s*",
                  h, re.I)
    seg = h[m.end():m.end() + 150].lower() if m else ""
    if not seg.strip():  # t7 bare verdict first line, e.g. '**Complete**'
        seg = re.sub(r"[*#`]", " ", text.strip().split("\n")[0]).lower()
    if "almost" in seg:
        return "almost"
    if "partial" in seg or "incomplete" in seg or "unresolved gap" in seg or "does not" in seg:
        return "partial"
    if "no material gaps" in seg or "complete" in seg or "correct" in seg or "no gaps" in seg:
        return "complete"
    return None

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent

NAMES = {"t0": "minimal baseline", "t1": "Huang-Yang rigorous", "t2": "HY self-repair",
         "t3": "DeepSeekMath-V2 self-verify", "t4": "STAR-Pólya plan-verify",
         "t5": "Momus dialectic", "t6": "Aletheia gen-verify-revise", "t7": "rubric-aware"}
CLAIM_EXP = {"complete": 7, "almost": 6, "partial": 1}  # claimed -> expected band


def load(grades_name: str):
    """templates -> {pid: {cand: cand_score}}, plus claimed_status + best-of-3 per pid."""
    runs = {}
    for d in sorted(glob.glob(str(EVAL_ROOT / "runs" / "dsv4-flash__t*__high_notool"))):
        tid = os.path.basename(d).split("__")[1]
        # per-candidate score = mean over passes
        passes = defaultdict(list)
        meta = {}
        for l in (Path(d) / grades_name).open():
            r = json.loads(l)
            if r["score"] is not None:
                passes[(r["problem_id"], r["candidate_idx"])].append(r["score"])
            meta[r["problem_id"]] = (r["subset"], r["level"], r["category"])
        cand_score = {k: mean(v) for k, v in passes.items() if v}
        by_pid = defaultdict(list)
        for (pid, _c), s in cand_score.items():
            by_pid[pid].append(s)
        # claimed completeness (re-parsed) + generation token cost per candidate
        claims, ctoks, rtoks = {}, [], []
        for l in (Path(d) / "responses.jsonl").open():
            rr = json.loads(l)
            for ci, c in enumerate(rr["candidates"]):
                claims[(rr["problem_id"], ci)] = parse_claim(tid, c.get("text") or "")
                ctoks.append(c.get("completion_tokens") or 0)
                rtoks.append(c.get("reasoning_tokens") or 0)
        # grader token cost (completion) for this template's run
        gtoks = [json.loads(l).get("completion_tokens") or 0
                 for l in (Path(d) / grades_name).open()]
        runs[tid] = {"best": {p: max(v) for p, v in by_pid.items()},
                     "meanof": {p: mean(v) for p, v in by_pid.items()},
                     "cand_score": cand_score, "claims": claims, "meta": meta,
                     "ctoks": ctoks, "rtoks": rtoks, "gtoks": gtoks}
    return runs


def agg(vals):
    n = len(vals)
    return {"n": n, "mean": round(mean(vals), 3),
            "almost+": round(sum(v >= 6 for v in vals) / n, 3),
            "correct": round(sum(v >= 7 for v in vals) / n, 3)} if n else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", default="grades_flashHigh_2pass.jsonl")
    args = ap.parse_args()
    runs = load(args.grades)
    tids = sorted(runs)
    base = runs["t0"]["best"]

    rows = []
    for t in tids:
        R = runs[t]
        best = R["best"]
        meta = R["meta"]
        sub = defaultdict(list)
        for pid, b in best.items():
            sub[meta[pid][0]].append(b)
        # paired vs t0
        common = sorted(set(best) & set(base))
        deltas = [best[p] - base[p] for p in common]
        win = sum(d > 0.01 for d in deltas)
        loss = sum(d < -0.01 for d in deltas)
        tie = len(deltas) - win - loss
        # overclaiming: candidates claiming 'complete' but graded <6
        oc_n = oc_bad = 0
        for (pid, ci), cl in R["claims"].items():
            if cl == "complete" and (pid, ci) in R["cand_score"]:
                oc_n += 1
                if R["cand_score"][(pid, ci)] < 6:
                    oc_bad += 1
        ct = R["ctoks"]
        rows.append({
            "t": t, "name": NAMES[t],
            "all": agg(list(best.values())),
            "basic": agg(sub.get("basic", [])), "adv": agg(sub.get("advanced", [])),
            "meanof": round(mean(list(R["meanof"].values())), 3),
            "win": win, "loss": loss, "tie": tie,
            "vs0": round(mean(deltas), 3) if deltas else 0.0,
            "oc_n": oc_n, "oc_bad": oc_bad,
            "oc_rate": round(oc_bad / oc_n, 3) if oc_n else None,
            "gen_ctok": round(mean(ct)) if ct else 0,           # mean completion tok / candidate
            "gen_rfrac": round(sum(R["rtoks"]) / max(sum(ct), 1), 2),  # reasoning fraction
            "gen_tot_M": round(sum(ct) / 1e6, 2),               # total completion tok (millions)
            "grade_tot_M": round(sum(R["gtoks"]) / 1e6, 2),     # total grader completion tok
        })

    rows.sort(key=lambda r: -r["all"]["mean"])

    # ---- markdown ----
    L = []
    L.append("# Prompt-template sweep — IMO-ProofBench grades\n")
    L.append("- **Model**: DeepSeek-V4-Flash, reasoning **high**, no-tool, single-round.\n"
             "- **Data**: ProofBench v2, 60 problems (30 basic + 30 advanced), k=3 samples.\n"
             "- **Grader**: flash `high_notool` (paper B.5 verbatim, calibrated), 2 passes/candidate, "
             "graded on the **meta-stripped proof body** (`graded_text`).\n"
             "- **Score** = mean over 2 passes per candidate; **best-of-3** = max over the 3 candidates, "
             "**mean-of-3** = their mean, per problem; table aggregates over 60 problems.\n"
             "- ⚠️ k=3 → per-template differences are noisy; the paired win/loss vs t0 is the more "
             "reliable signal. Grader is flash judging flash (self-grading) and under-scores the "
             "middle band — absolute numbers are conservative.\n")
    L.append("\n## Ranking (by best-of-3 mean)\n")
    L.append("| rank | tmpl | template | best-of-3 | almost+ | correct | mean-of-3 | basic | adv | "
             "vs t0 (Δ) | win/tie/loss | overclaim | gen tok/cand | reason% | gen total | grade total |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        oc = f"{r['oc_bad']}/{r['oc_n']} ({r['oc_rate']})" if r["oc_n"] else "—"
        vs = "—" if r["t"] == "t0" else f"{r['vs0']:+.2f}"
        wl = "—" if r["t"] == "t0" else f"{r['win']}/{r['tie']}/{r['loss']}"
        L.append(f"| {i} | **{r['t']}** | {r['name']} | **{r['all']['mean']}** | "
                 f"{r['all']['almost+']} | {r['all']['correct']} | {r['meanof']} | "
                 f"{r['basic']['mean']} | {r['adv']['mean']} | {vs} | {wl} | {oc} | "
                 f"{r['gen_ctok']:,} | {int(r['gen_rfrac']*100)}% | {r['gen_tot_M']}M | {r['grade_tot_M']}M |")
    gtot = sum(r["gen_tot_M"] for r in rows)
    grtot = sum(r["grade_tot_M"] for r in rows)
    L.append(f"\nTotals: **generation** {gtot:.1f}M completion tokens, "
             f"**grading** {grtot:.1f}M completion tokens (all 8 templates).\n")
    L.append("\n### Notes\n"
             "- **best-of-3** ≈ the agentic best-of-k ceiling; **mean-of-3** ≈ single-sample expectation.\n"
             "- **vs t0 (Δ)** = mean per-problem best-of-3 difference vs the minimal baseline; "
             "**win/tie/loss** counts problems where the template's best-of-3 beats/ties/loses to t0.\n"
             "- **overclaim** = candidates that claimed a *complete* solution but graded <6 "
             "(only templates that emit a completeness verdict: t1/t2/t3/t4/t6/t7).\n"
             "- **gen tok/cand** = mean completion tokens per candidate (generation); **reason%** = "
             "share that is hidden reasoning; **gen total** = completion tokens over all 180 candidates "
             "(60 problems × k=3); **grade total** = grader completion tokens (B.5, 2 passes). "
             "Generation completion is ~97% reasoning — the real cost driver; structured templates that "
             "make the model think/repair more cost more for little or negative quality gain.\n")
    out = EVAL_ROOT / "results" / "template_sweep.md"
    out.write_text("\n".join(L), encoding="utf-8")

    # console
    print(f"{'tmpl':>4} {'name':<28} {'best3':>6} {'corr':>5} {'adv':>5} {'vs0':>6} "
          f"{'w/t/l':>9} {'overclaim':>9} {'gtok/cd':>8} {'genM':>6} {'grdM':>6}")
    for r in rows:
        oc = f"{r['oc_bad']}/{r['oc_n']}" if r["oc_n"] else "—"
        wl = "—" if r["t"] == "t0" else f"{r['win']}/{r['tie']}/{r['loss']}"
        vs = "—" if r["t"] == "t0" else f"{r['vs0']:+.2f}"
        print(f"{r['t']:>4} {r['name']:<28} {r['all']['mean']:>6} {r['all']['correct']:>5} "
              f"{r['adv']['mean']:>5} {vs:>6} {wl:>9} {oc:>9} {r['gen_ctok']:>8,} "
              f"{r['gen_tot_M']:>6} {r['grade_tot_M']:>6}")
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
