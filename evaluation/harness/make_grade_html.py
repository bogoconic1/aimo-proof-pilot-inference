"""Self-contained HTML viewer for grading review.

For each (template, problem) cell it embeds: the problem, the reference Solution + Grading
guidelines (what the grader was given), and every candidate's GRADED proof body
(`graded_text`) with its per-pass grader score + rationale + full grader reasoning. The raw
pre-strip `text` is shown collapsed so a reviewer can see exactly what the meta-stripper
removed. Open it in a browser to sanity-check both the grades and the extraction.

Usage (the 8-problem cross-template smoke):
  python make_grade_html.py \
    --run-ids dsv4-flash__t0__high_notool,...,dsv4-flash__t7__high_notool \
    --grades grades_smoke8.jsonl --data ../data/proofbench_v2.csv \
    --pairs t0:PB-Basic-001,t1:PB-Basic-009,... \
    --out ../runs/grade_smoke8_review.html
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent


def tid_of(run_id: str) -> str:
    return run_id.split("__")[1]


def build_data(run_ids: list[str], grades_name: str, data_csv: str,
               pairs: str | None) -> dict:
    src = pd.read_csv(data_csv).set_index("Problem ID")
    grader_tpl = (EVAL_ROOT / "prompts" / "grader.md").read_text()
    want = None
    if pairs:
        want = {(t.split(":")[0], t.split(":")[1]) for t in pairs.split(",")}

    cells = []
    for run_id in run_ids:
        tid = tid_of(run_id)
        run_dir = EVAL_ROOT / "runs" / run_id
        meta = json.loads((run_dir / "run_meta.json").read_text())
        # candidates (graded_text + raw text) keyed by pid
        resp = {}
        for line in (run_dir / "responses.jsonl").open():
            r = json.loads(line)
            resp[r["problem_id"]] = r
        # grades keyed by (pid, cand) -> list of pass records
        gpath = run_dir / grades_name
        if not gpath.exists():
            continue
        grades = defaultdict(list)
        for line in gpath.open():
            d = json.loads(line)
            grades[(d["problem_id"], d["candidate_idx"])].append(d)

        graded_pids = {pid for (pid, _c) in grades}
        for pid in graded_pids:
            if want is not None and (tid, pid) not in want:
                continue
            r = resp.get(pid)
            if not r:
                continue
            row = src.loc[pid]
            cands = []
            for ci, c in enumerate(r["candidates"]):
                gs = sorted(grades.get((pid, ci), []), key=lambda x: x["pass"])
                if not gs:
                    continue
                scores = [g["score"] for g in gs if g["score"] is not None]
                cands.append({
                    "cand": ci,
                    "graded_text": c.get("graded_text") or c.get("text") or "",
                    "raw_text": c.get("text") or "",
                    "stripped": (c.get("graded_text") is not None
                                 and c.get("graded_text") != c.get("text")),
                    "finish_reason": c.get("finish_reason"),
                    "completion_tokens": c.get("completion_tokens"),
                    "failure_type": c.get("failure_type"),
                    "score_mean": round(mean(scores), 2) if scores else None,
                    "grades": [{"pass": g["pass"], "score": g["score"],
                                "rationale": g.get("rationale") or "",
                                "reasoning": g.get("grader_reasoning") or "",
                                "content": g.get("grader_content") or ""} for g in gs],
                })
            cands.sort(key=lambda x: x["cand"])
            best = max((c["score_mean"] for c in cands if c["score_mean"] is not None),
                       default=None)
            cells.append({
                "template": tid, "template_name": meta.get("template_name"), "run_id": run_id,
                "pid": pid, "subset": r["subset"], "category": r["category"], "level": r["level"],
                "problem": r["problem"], "solution": str(row["Solution"]),
                "guidelines": str(row["Grading guidelines"]),
                "candidates": cands, "best": best,
            })
    cells.sort(key=lambda x: (x["template"], x["pid"]))
    return {"cells": cells, "grader_prompt": grader_tpl}


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>ProofBench grading review</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1e222b;--fg:#e6e6e6;--mut:#9aa4b2;--acc:#6db3f2;
--green:#5fd38a;--red:#f2746b;--yellow:#f2cf6b;--bd:#2a2f3a;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--bd)}
h1{font-size:15px;margin:0}
.layout{display:flex;height:calc(100vh - 46px)}
.side{width:280px;min-width:280px;overflow-y:auto;border-right:1px solid var(--bd);background:var(--panel)}
.main{flex:1;overflow-y:auto;padding:18px}
.cell{padding:8px 12px;border-bottom:1px solid var(--bd);cursor:pointer;font-size:12px}
.cell:hover{background:var(--panel2)}
.cell.on{background:var(--panel2);outline:1px solid var(--acc)}
.cell .t{font-weight:700;color:var(--acc)}
.score{font-weight:700;border-radius:8px;padding:0 7px;font-size:12px}
.s7{background:#16361f;color:var(--green)} .s6{background:#3a3416;color:var(--yellow)}
.s1{background:#3a2616;color:#f2a86b} .s0{background:#3a1d1b;color:var(--red)} .sN{background:#222;color:var(--mut)}
.badges{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 14px}
.badge{background:var(--panel2);border:1px solid var(--bd);border-radius:6px;padding:2px 8px;font-size:12px}
.badge b{color:var(--acc)}
.sec{background:var(--panel);border:1px solid var(--bd);border-radius:8px;margin-bottom:10px;overflow:hidden}
.sec>summary{padding:8px 12px;cursor:pointer;font-weight:600;user-select:none;list-style:none}
.sec>summary::-webkit-details-marker{display:none}
.sec>summary:before{content:"\25B8 ";color:var(--mut)}
.sec[open]>summary:before{content:"\25BE "}
.body{padding:10px 12px;border-top:1px solid var(--bd)}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12.5px/1.55 ui-monospace,Menlo,Consolas,monospace}
.cand{border:1px solid var(--bd);border-left:3px solid var(--acc);border-radius:8px;margin:14px 0;background:var(--panel)}
.cand>summary{padding:8px 12px;cursor:pointer;font-weight:600}
.proof{background:#0d1410;border:1px solid #234;border-left:3px solid var(--green);border-radius:6px;padding:10px;margin:8px 0}
.proof .lbl{color:var(--green);font-size:11px;margin-bottom:4px}
.raw{background:#0b0e13;border:1px solid var(--bd);border-radius:6px;padding:8px;margin:6px 0}
.gr{border:1px solid var(--bd);border-radius:6px;margin:8px 0;background:#12161d}
.gr>summary{padding:6px 10px;cursor:pointer;font-size:13px}
.rationale{border-left:3px solid var(--yellow);padding:8px;margin:4px 0;background:#0b0e13}
.reason{border-left:3px solid var(--acc);padding:8px;margin:4px 0;background:#0b0e13}
.ref{border-left:3px solid var(--mut);padding:8px;background:#0b0e13;border-radius:6px}
.lbl{font-size:11px;color:var(--mut);margin-bottom:4px}
.diff-rm{background:#3a1d1b;color:#f2a89f}
</style></head><body>
<header><h1>ProofBench grading review — graded_text (meta-stripped) + grader scores</h1></header>
<div class="layout">
  <div class="side" id="side"></div>
  <div class="main" id="main"></div>
</div>
<script id="DATA" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('DATA').textContent);
let cur=0;
const esc=s=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const sCls=s=>s==null?'sN':s>=7?'s7':s>=6?'s6':s>=1?'s1':'s0';
const sTxt=s=>s==null?'—':s;

function side(){
  document.getElementById('side').innerHTML=D.cells.map((c,i)=>
    `<div class="cell ${i===cur?'on':''}" onclick="sel(${i})">`+
    `<span class="t">${c.template}</span> ${c.pid} `+
    `<span class="score ${sCls(c.best)}">${sTxt(c.best)}</span><br>`+
    `<span style="color:var(--mut)">${c.subset} · ${c.category} · ${c.level}</span></div>`).join('');
}
// crude word-diff: mark lines present in raw but absent in graded as removed
function removedLines(raw,graded){
  const G=new Set(graded.split('\n').map(x=>x.trim()).filter(Boolean));
  return raw.split('\n').map(l=>{
    const t=l.trim();
    return (t && !G.has(t))?`<span class="diff-rm">${esc(l)}</span>`:esc(l);
  }).join('\n');
}
function detail(){
  const c=D.cells[cur];
  let h=`<div class="badges"><span class="badge"><b>${c.template}</b> ${esc(c.template_name||'')}</span>`+
    `<span class="badge">${c.pid}</span><span class="badge">${c.subset} · ${c.category} · ${c.level}</span>`+
    `<span class="badge">best <b>${sTxt(c.best)}</b>/7</span></div>`;
  h+=`<details class="sec" open><summary>📘 Problem</summary><div class="body"><pre>${esc(c.problem)}</pre></div></details>`;
  h+=`<details class="sec"><summary>📗 Reference solution + grading guidelines (given to grader)</summary>`+
     `<div class="body"><div class="ref"><div class="lbl">SOLUTION</div><pre>${esc(c.solution)}</pre></div>`+
     `<div class="ref" style="margin-top:8px"><div class="lbl">GRADING GUIDELINES</div><pre>${esc(c.guidelines)}</pre></div></div></details>`;
  for(const cd of c.candidates){
    const sm=`<span class="score ${sCls(cd.score_mean)}">${sTxt(cd.score_mean)}</span>`;
    h+=`<details class="cand" open><summary>candidate #${cd.cand} — mean ${sm} `+
       `<span style="color:var(--mut);font-weight:400">· ${cd.finish_reason} · ${cd.completion_tokens} tok`+
       `${cd.stripped?' · meta-stripped':''}</span></summary><div class="body">`;
    h+=`<div class="proof"><div class="lbl">GRADED PROOF (graded_text, ${cd.graded_text.length} chars)</div><pre>${esc(cd.graded_text)}</pre></div>`;
    if(cd.stripped)
      h+=`<details class="gr"><summary>raw pre-strip text (removed meta highlighted)</summary>`+
         `<div class="raw"><pre>${removedLines(cd.raw_text,cd.graded_text)}</pre></div></details>`;
    for(const g of cd.grades){
      h+=`<details class="gr" open><summary>grader pass ${g.pass}: `+
         `<span class="score ${sCls(g.score)}">${sTxt(g.score)}</span>/7</summary>`+
         `<div style="padding:6px 10px">`+
         `<div class="rationale"><div class="lbl">RATIONALE (around &lt;points&gt;)</div><pre>${esc(g.rationale)}</pre></div>`+
         `<details class="gr"><summary>grader reasoning_content (${g.reasoning.length} chars)</summary>`+
         `<div class="reason"><pre>${esc(g.reasoning)}</pre></div></details>`+
         `<details class="gr"><summary>grader full content (${g.content.length} chars)</summary>`+
         `<div class="raw"><pre>${esc(g.content)}</pre></div></details>`+
         `</div></details>`;
    }
    h+=`</div></details>`;
  }
  document.getElementById('main').innerHTML=h;
  document.getElementById('main').scrollTop=0;
}
function sel(i){cur=i;side();detail();}
side();detail();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", required=True)
    ap.add_argument("--grades", default="grades_smoke8.jsonl", help="grade jsonl filename in each run dir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--pairs", default=None, help="restrict to tid:pid,... cells")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]
    data = build_data(run_ids, args.grades, args.data, args.pairs)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    Path(args.out).write_text(HTML.replace("__DATA__", blob), encoding="utf-8")
    nc = sum(len(c["candidates"]) for c in data["cells"])
    print(f"[done] {len(data['cells'])} cells / {nc} candidates -> {args.out} "
          f"({Path(args.out).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
