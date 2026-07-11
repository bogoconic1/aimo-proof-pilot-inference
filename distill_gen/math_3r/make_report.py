"""Render a math_3r run's full traces into a self-contained HTML review page.

Switchable layout: a left sidebar lists the problems; clicking one renders ONLY that problem in
the main pane (built in JS from an embedded JSON blob), and MathJax typesets just that problem —
so the page stays short and responsive regardless of run size. Each API call shows its INPUT
messages (system/user), collapsible reasoning, and RESPONSE content, with badges.

    uv run python distill_gen/math_3r/make_report.py --run-id r3_smoke16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CSS = """
:root{--bg:#0f1115;--card:#1a1d24;--card2:#21252e;--ink:#e6e8ec;--mut:#9aa3b2;--line:#2c313c;
--green:#3fb950;--red:#f85149;--amber:#d29922;--blue:#58a6ff;--violet:#bc8cff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
header{background:#0f1115;border-bottom:1px solid var(--line);padding:12px 20px}
h1{font-size:16px;margin:0 0 8px}
.summary{display:flex;flex-wrap:wrap;gap:8px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
.kpi b{font-size:15px;color:#fff} .kpi span{display:block;color:var(--mut);font-size:10px;text-transform:uppercase;letter-spacing:.04em}
#layout{display:flex;height:calc(100vh - 86px)}
#sidebar{width:300px;flex:none;overflow:auto;border-right:1px solid var(--line);padding:8px}
#sidebar button{display:block;width:100%;text-align:left;background:var(--card);color:var(--ink);
border:1px solid var(--line);border-radius:7px;padding:7px 9px;margin:5px 0;cursor:pointer;font-size:12px;line-height:1.35}
#sidebar button:hover{border-color:var(--blue)} #sidebar button.active{border-color:var(--violet);background:#272233}
#sidebar .n{color:var(--mut);font-family:ui-monospace,monospace}
#content{flex:1;overflow:auto;padding:18px 26px}
h2{font-size:15px;margin:0 0 6px;color:#fff}
.statement{background:#11141a;border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0}
.final{border:1px solid #2ea04366;background:#12251a;border-radius:8px;padding:10px 12px;margin:10px 0}
.stage{margin:16px 0 4px;font-weight:600;color:var(--violet);border-top:1px dashed var(--line);padding-top:10px}
.call{background:var(--card2);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:10px 12px}
.call.win{outline:2px solid #2ea043}
.call-head{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:4px}
.lab{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--mut)}
.badge{font-size:11px;padding:1px 7px;border-radius:20px;border:1px solid var(--line);color:var(--mut)}
.b-valid{color:#fff;background:#1f6f2e;border-color:#2ea043}
.b-invalid{color:#fff;background:#6e2b28;border-color:#f85149}
.b-score{color:#fff;background:#5a4a14;border-color:#d29922}
.b-err,.b-trunc{color:#fff;background:#6e2b28;border-color:#f85149}
.b-tok{color:var(--blue);border-color:#2b4a6b}
details{margin:6px 0}
summary{cursor:pointer;color:var(--mut);font-size:12px;user-select:none}
summary:hover{color:var(--ink)}
.msg{border-left:3px solid var(--line);padding:6px 10px;margin:6px 0;white-space:pre-wrap;background:#11141a;border-radius:0 6px 6px 0}
.msg.system{border-color:var(--amber)} .msg.user{border-color:var(--blue)}
.role{display:block;font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-bottom:3px}
.content{white-space:pre-wrap;background:#11141a;border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin-top:6px}
.reasoning{white-space:pre-wrap;font-size:12px;color:var(--mut);max-height:360px;overflow:auto;background:#0c0e12;border:1px solid var(--line);border-radius:6px;padding:8px;margin:4px 0}
.empty{color:var(--mut);font-style:italic}
"""

MATHJAX = ("<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],"
           "displayMath:[['$$','$$'],['\\\\[','\\\\]']]},"
           "options:{ignoreHtmlClass:'no-math',processHtmlClass:'math'}};</script>"
           "<script async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js'></script>")

JS = r"""
const DATA = __DATA__;
function el(t,c,x){const e=document.createElement(t); if(c)e.className=c; if(x!=null)e.textContent=x; return e;}
function badge(x,c){return el('span','badge '+(c||''),x);}
function vbadge(v){return badge(v?'valid':'invalid', v?'b-valid':'b-invalid');}
function sbadge(s,k){return (s===null||s===undefined)?null:badge(k+' '+s,'b-score');}
function k(n){return Math.round((n||0)/1000)+'k';}
function makeCall(c,opts){
  opts=opts||{};
  const card=el('div','call'+(opts.win?' win':''));
  const h=el('div','call-head'); h.appendChild(el('span','lab',c.label));
  (opts.badges||[]).forEach(b=>{if(b)h.appendChild(b);});
  if(c.error)h.appendChild(badge('error','b-err'));
  if(c.truncated)h.appendChild(badge('truncated','b-trunc'));
  h.appendChild(badge(c.finish_reason));
  h.appendChild(badge('in '+(c.prompt_tokens||0).toLocaleString(),'b-tok'));
  h.appendChild(badge('out '+(c.completion_tokens||0).toLocaleString()+' (reason '+(c.reasoning_tokens||0).toLocaleString()+')','b-tok'));
  card.appendChild(h);
  const dm=el('details'); dm.appendChild(el('summary',null,'input messages ('+(c.messages||[]).length+')'));
  (c.messages||[]).forEach(m=>{const d=el('div','msg '+m.role); d.appendChild(el('span','role',m.role));
    const mc=el('div','math'); mc.textContent=m.content||''; d.appendChild(mc); dm.appendChild(d);});
  card.appendChild(dm);
  const rc=c.reasoning_content||'';
  if(rc){const dr=el('details'); dr.appendChild(el('summary',null,'reasoning ('+rc.length.toLocaleString()+' chars)'));
    const pr=el('pre','reasoning no-math'); pr.textContent=rc; dr.appendChild(pr); card.appendChild(dr);}
  if(c.error){card.appendChild(el('div','content empty','ERROR: '+c.error));}
  else if((c.content||'').trim()){const cc=el('div','content math'); cc.textContent=c.content; card.appendChild(cc);}
  else {card.appendChild(el('div','content empty','(empty content)'));}
  return card;
}
function stageBlock(main,name,calls,optFn){
  main.appendChild(el('div','stage',name+' ('+calls.length+')'));
  calls.forEach(c=>main.appendChild(makeCall(c,optFn(c))));
}
function render(i){
  const r=DATA[i], main=document.getElementById('content'); main.innerHTML='';
  document.querySelectorAll('#sidebar button').forEach((b,j)=>b.classList.toggle('active',j===i));
  main.appendChild(el('h2',null,'#'+(i+1)+' · '+r.origin+' · '+(r.category||'—')));
  const s=el('div','statement math'); s.textContent=r.problem; main.appendChild(s);
  const f=el('div','final'); f.appendChild(el('b',null,'FINAL  '));
  f.appendChild(badge(r.final_source)); f.appendChild(badge('votes '+JSON.stringify(r.selected_ids||[])));
  const fc=el('div','content math'); fc.textContent=r.final_proof; f.appendChild(fc); main.appendChild(f);
  const tt=r.totals;
  main.appendChild(el('div','lab','valid '+r.counts.n_valid_proofs+'/'+r.num_provers+' · refined '+
    r.counts.n_refined_valid+' · calls '+tt.n_calls+' · err '+tt.n_errors+
    ' · in '+tt.prompt_tokens.toLocaleString()+' · out '+tt.completion_tokens.toLocaleString()+
    ' (reason '+tt.reasoning_tokens.toLocaleString()+')'));
  const win=r.selected_id, st=r.stages;
  stageBlock(main,'PROVE',st.prove,c=>({win:c.candidate_id===win,badges:[vbadge(c.valid),sbadge(c.self_score,'self')]}));
  stageBlock(main,'VERIFY',st.verify,c=>({badges:[badge('on '+c.candidate_id),sbadge(c.score,'verdict')]}));
  stageBlock(main,'REFINE',st.refine,c=>({win:c.refiner_id===win,badges:[vbadge(c.valid),sbadge(c.self_score,'self')]}));
  stageBlock(main,'SELECT',st.select,c=>({}));
  if(window.MathJax&&MathJax.typesetPromise)MathJax.typesetPromise([main]);
  main.scrollTop=0;
}
function shortStmt(p){return p.length>72?p.slice(0,72)+'…':p;}
window.addEventListener('DOMContentLoaded',()=>{
  const sb=document.getElementById('sidebar');
  DATA.forEach((r,i)=>{const b=document.createElement('button');
    b.innerHTML='<span class="n">#'+(i+1)+'</span> ['+r.final_source.split(':')[0]+'] '+
      (r.counts.n_valid_proofs)+'/'+r.num_provers+'v';
    b.appendChild(document.createElement('br')); b.appendChild(document.createTextNode(shortStmt(r.problem)));
    b.onclick=()=>render(i); sb.appendChild(b);});
  render(0);
});
"""


def pick_sample(recs: list[dict], n: int) -> list[int]:
    """Curated representative subset (indices): every imperfect problem, some refined-selected,
    some low-margin, then perfect ones balanced across origins, capped at n."""
    imperfect = [i for i, r in enumerate(recs) if r["counts"]["n_valid_proofs"] < r["num_provers"]]
    refined = [i for i, r in enumerate(recs) if (r.get("selected_id") or "").startswith("R")]
    lowmargin = [i for i, r in enumerate(recs)
                 if "(1/" in r["final_source"] or "(2/" in r["final_source"]]
    picked: list[int] = []
    for group, cap in [(imperfect, n), (refined, 3), (lowmargin, 3)]:
        for i in group:
            if i not in picked and len(picked) < n:
                picked.append(i)
    # fill remaining with perfect ones, alternating origin for balance
    by_origin: dict[str, list[int]] = {}
    for i, r in enumerate(recs):
        if i not in picked:
            by_origin.setdefault(r["origin"], []).append(i)
    queues = list(by_origin.values())
    qi = 0
    while len(picked) < n and any(queues):
        q = queues[qi % len(queues)]
        if q:
            picked.append(q.pop(0))
        qi += 1
    return sorted(picked)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="r3_smoke16")
    ap.add_argument("--records", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=0,
                    help="embed only N representative problems (KPIs still over all); 0 = all")
    args = ap.parse_args()

    rec_path = args.records or (HERE / "outputs" / args.run_id / "records.jsonl")
    out_path = args.out or (HERE / "outputs" / args.run_id / "report.html")
    recs = [json.loads(l) for l in open(rec_path) if l.strip()]

    n = len(recs)
    calls = sum(r["totals"]["n_calls"] for r in recs)
    errs = sum(r["totals"]["n_errors"] for r in recs)
    ctok = sum(r["totals"]["completion_tokens"] for r in recs)
    rtok = sum(r["totals"]["reasoning_tokens"] for r in recs)
    ptok = sum(r["totals"]["prompt_tokens"] for r in recs)
    fb = sum(1 for r in recs if not r["final_source"].startswith("select"))
    mean_valid = sum(r["counts"]["n_valid_proofs"] for r in recs) / max(n, 1)
    cost = ctok / 1e6 * 0.28 + ptok / 1e6 * 0.14
    kpis = [("problems", n), ("API calls", calls), ("errors", errs),
            ("mean valid/6", f"{mean_valid:.1f}"), ("fallback", fb),
            ("input tok", f"{ptok/1e6:.2f}M"), ("output tok", f"{ctok/1e6:.2f}M"),
            ("reasoning tok", f"{rtok/1e6:.2f}M"), ("est cost", f"${cost:.2f}")]
    kpi_html = "".join(f'<div class="kpi"><b>{v}</b><span>{kk}</span></div>' for kk, v in kpis)

    sub = recs
    note = ""
    if args.sample and args.sample < n:
        sub = [recs[i] for i in pick_sample(recs, args.sample)]
        note = (f"<div style='color:#d29922;font-size:12px;margin-top:6px'>Showing {len(sub)}/{n} "
                f"representative problems (all valid&lt;6, some refined-selected, and low-consensus ones); the KPIs above are over all {n} problems.</div>")

    data = json.dumps(sub, ensure_ascii=False).replace("</", "<\\/")
    js = JS.replace("__DATA__", data)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>math_3r · {args.run_id}</title>{MATHJAX}<style>{CSS}</style></head><body>"
           f"<header><h1>DSMV2-Simple-3R trace · {args.run_id}</h1>"
           f"<div class='summary'>{kpi_html}</div>{note}</header>"
           f"<div id='layout'><nav id='sidebar'></nav><main id='content'></main></div>"
           f"<script>{js}</script></body></html>")
    out_path.write_text(doc)
    print(f"wrote {out_path}  ({out_path.stat().st_size/1e6:.1f}MB, embedded {len(sub)}/{n} problems, {calls} calls)")


if __name__ == "__main__":
    main()
