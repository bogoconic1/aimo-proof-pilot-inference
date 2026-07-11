"""Build a self-contained HTML viewer for the k=4 smoke runs.

Embeds every candidate's full record (rendered prompt, per-turn reasoning, tool code,
tool output with countdown, final proof, metadata) so the run can be audited offline in a
browser — to verify the prompt and all reasoning/tool data were saved correctly.

Usage:
  python make_review_html.py --configs high_notool,high_pytool,max_notool,max_pytool \
    --suffix _k4smoke --out ../runs/k4smoke_review.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent


def build_data(configs: list[str], suffix: str) -> dict:
    out = {"configs": [], "prompts": {}}
    for name in ("notool", "pytool"):
        p = EVAL_ROOT / "prompts" / f"prover_{name}.md"
        out["prompts"][name] = p.read_text() if p.exists() else "(missing)"
    for cfg in configs:
        run_dir = EVAL_ROOT / "runs" / f"dsv4-flash__{cfg}{suffix}"
        raw = run_dir / "candidates_raw.jsonl"
        meta = run_dir / "run_meta.json"
        cands = [json.loads(l) for l in raw.open()] if raw.exists() else []
        cands.sort(key=lambda r: (r["problem_id"], r["j"]))
        out["configs"].append({
            "id": cfg,
            "meta": json.loads(meta.read_text()) if meta.exists() else {},
            "candidates": cands,
        })
    return out


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>ProofBench k=4 smoke review</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1e222b;--fg:#e6e6e6;--mut:#9aa4b2;--acc:#6db3f2;
--green:#5fd38a;--red:#f2746b;--yellow:#f2cf6b;--bd:#2a2f3a;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:5}
h1{font-size:16px;margin:0 0 6px}
.summary{border-collapse:collapse;font-size:12px;margin-top:4px}
.summary th,.summary td{border:1px solid var(--bd);padding:3px 8px;text-align:right}
.summary th:first-child,.summary td:first-child{text-align:left}
.layout{display:flex;height:calc(100vh - 120px)}
.side{width:300px;min-width:300px;overflow-y:auto;border-right:1px solid var(--bd);background:var(--panel)}
.main{flex:1;overflow-y:auto;padding:16px}
.tabs{display:flex;flex-wrap:wrap;gap:4px;padding:8px}
.tab{padding:4px 8px;border:1px solid var(--bd);border-radius:6px;cursor:pointer;font-size:12px;background:var(--panel2)}
.tab.on{background:var(--acc);color:#06121f;font-weight:600}
.clist{padding:4px 8px}
.crow{padding:5px 8px;border-radius:6px;cursor:pointer;font-size:12px;display:flex;justify-content:space-between;gap:6px}
.crow:hover{background:var(--panel2)}
.crow.on{background:var(--panel2);outline:1px solid var(--acc)}
.pid{font-weight:600}
.dot{font-size:11px;padding:0 5px;border-radius:8px}
.b-stop{color:var(--green)} .b-len{color:var(--yellow)} .b-tc{color:var(--red)} .b-empty{color:var(--red);font-weight:700}
.badges{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.badge{background:var(--panel2);border:1px solid var(--bd);border-radius:6px;padding:2px 8px;font-size:12px}
.badge b{color:var(--acc)}
.sec{background:var(--panel);border:1px solid var(--bd);border-radius:8px;margin-bottom:10px;overflow:hidden}
.sec>summary{padding:8px 12px;cursor:pointer;font-weight:600;user-select:none;list-style:none}
.sec>summary::-webkit-details-marker{display:none}
.sec>summary:before{content:"▸ ";color:var(--mut)}
.sec[open]>summary:before{content:"▾ "}
.body{padding:10px 12px;border-top:1px solid var(--bd)}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace}
.turn{border:1px solid var(--bd);border-radius:8px;margin:8px 0;background:var(--panel)}
.turn>summary{padding:6px 10px;cursor:pointer;font-size:12px;color:var(--mut)}
.code{background:#0b0e13;border:1px solid var(--bd);border-radius:6px;padding:8px;margin:6px 0}
.code .lbl{color:var(--green);font-size:11px;margin-bottom:4px}
.out{background:#0b0e13;border:1px solid var(--bd);border-left:3px solid var(--yellow);border-radius:6px;padding:8px;margin:6px 0}
.out .lbl{color:var(--yellow);font-size:11px;margin-bottom:4px}
.reason{background:#12161d;border-left:3px solid var(--acc);border-radius:6px;padding:8px;margin:6px 0}
.reason .lbl{color:var(--acc);font-size:11px;margin-bottom:4px}
.proof{background:#0d1410;border:1px solid #234;border-left:3px solid var(--green);border-radius:6px;padding:10px}
.proof .lbl{color:var(--green);font-size:11px;margin-bottom:4px}
.prompt{background:#0b0e13;border:1px solid var(--bd);border-radius:6px;padding:10px}
.count{color:var(--yellow);font-weight:600}
.muted{color:var(--mut)}
button.raw{float:right;font-size:11px;background:var(--panel2);color:var(--fg);border:1px solid var(--bd);border-radius:6px;padding:2px 8px;cursor:pointer}
</style></head><body>
<header>
  <h1>ProofBench k=4 smoke — DeepSeek-V4-flash · 4 configs · 5 problems · k=4</h1>
  <div id="sumwrap"></div>
</header>
<div class="layout">
  <div class="side">
    <div class="tabs" id="tabs"></div>
    <div class="clist" id="clist"></div>
  </div>
  <div class="main" id="main"></div>
</div>
<script id="DATA" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('DATA').textContent);
let curCfg = 0, curIdx = 0;
const esc = s => (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const finCls = f => f==='stop'?'b-stop':f==='length'?'b-len':f==='tool_calls'?'b-tc':'b-empty';

function summary(){
  let h='<table class="summary"><tr><th>config</th><th>reasoning</th><th>cond</th><th>max_tok</th>'+
    '<th>budget</th><th>n</th><th>empty</th><th>stop</th><th>med proof</th><th>mean ctok</th><th>mean rtok</th>'+
    '<th>mean tools</th><th>med lat</th></tr>';
  for(const c of DATA.configs){
    const cs=c.candidates, n=cs.length;
    const empty=cs.filter(r=>!(r.text||'').trim()).length;
    const stop=cs.filter(r=>r.finish_reason==='stop').length;
    const med=a=>{const s=[...a].sort((x,y)=>x-y);return s.length?s[Math.floor(s.length/2)]:0;};
    const mean=a=>a.length?Math.round(a.reduce((x,y)=>x+y,0)/a.length):0;
    const tools=cs.map(r=>r.n_tool_calls||0);
    h+=`<tr><td><b>${c.id}</b></td><td>${c.meta.reasoning||''}</td><td>${c.meta.condition||''}</td>`+
       `<td>${c.meta.max_tokens||''}</td><td>${c.meta.max_tool_calls??'-'}</td><td>${n}</td>`+
       `<td class="${empty?'b-empty':''}">${empty}</td><td>${stop}</td>`+
       `<td>${med(cs.map(r=>(r.text||'').length))}c</td><td>${mean(cs.map(r=>r.completion_tokens||0))}</td>`+
       `<td>${mean(cs.map(r=>r.reasoning_tokens||0))}</td>`+
       `<td>${c.meta.condition==='pytool'?(tools.reduce((x,y)=>x+y,0)/n).toFixed(1):'-'}</td>`+
       `<td>${med(cs.map(r=>r.latency_s||0))}s</td></tr>`;
  }
  h+='</table>';
  document.getElementById('sumwrap').innerHTML=h;
}
function tabs(){
  document.getElementById('tabs').innerHTML=DATA.configs.map((c,i)=>
    `<div class="tab ${i===curCfg?'on':''}" onclick="selCfg(${i})">${c.id}</div>`).join('');
}
function clist(){
  const cs=DATA.configs[curCfg].candidates;
  document.getElementById('clist').innerHTML=cs.map((r,i)=>{
    const empty=!(r.text||'').trim();
    const tc=r.n_tool_calls!=null?`<span class="muted">${r.n_tool_calls}🔧</span>`:'';
    return `<div class="crow ${i===curIdx?'on':''}" onclick="selIdx(${i})">`+
      `<span class="pid">${r.problem_id} <span class="muted">#${r.j}</span></span>`+
      `<span><span class="dot ${finCls(r.finish_reason)}">${empty?'EMPTY':r.finish_reason}</span> ${tc}</span></div>`;
  }).join('');
}
function selCfg(i){curCfg=i;curIdx=0;tabs();clist();detail();}
function selIdx(i){curIdx=i;clist();detail();}

function renderMsgs(msgs){
  let h='';
  for(const m of msgs){
    if(m.role==='user'){
      h+=`<details class="sec" open><summary>📨 PROMPT sent (role=user, ${(m.content||'').length} chars)</summary>`+
         `<div class="body"><div class="prompt"><pre>${esc(m.content)}</pre></div></div></details>`;
    } else if(m.role==='assistant'){
      const isTool=m.tool_calls&&m.tool_calls.length;
      const rc=m.reasoning_content||'';
      let inner='';
      if(rc) inner+=`<details class="turn" open><summary>🧠 reasoning_content (${rc.length} chars)</summary>`+
        `<div class="reason"><pre>${esc(rc)}</pre></div></details>`;
      else inner+=`<div class="muted" style="font-size:12px">⚠ no reasoning_content on this turn</div>`;
      if((m.content||'').trim() && !isTool){
        inner+=`<div class="proof"><div class="lbl">FINAL PROOF (content, ${m.content.length} chars) · finish=${esc(m.finish_reason)}</div><pre>${esc(m.content)}</pre></div>`;
      } else if((m.content||'').trim()){
        inner+=`<div class="prompt"><pre>${esc(m.content)}</pre></div>`;
      }
      if(isTool){
        for(const tc of m.tool_calls){
          let code='';try{code=JSON.parse(tc.function.arguments||'{}').code||'';}catch(e){code='<unparseable args>';}
          inner+=`<div class="code"><div class="lbl">🔧 tool_call ${esc(tc.id)} · ${esc(tc.function&&tc.function.name)}</div><pre>${esc(code)}</pre></div>`;
        }
      }
      const tag=isTool?'assistant ▸ tool_calls':'assistant ▸ FINAL';
      h+=`<details class="sec" open><summary>${tag}</summary><div class="body">${inner}</div></details>`;
    } else if(m.role==='tool'){
      const c=m.content||'';
      const cd=c.match(/\[\d+ tool call\(s\) remaining\]|\[No tool calls remaining[^\]]*\]/);
      const body=cd?esc(c.slice(0,cd.index))+'<span class="count">'+esc(c.slice(cd.index))+'</span>':esc(c);
      h+=`<div class="out"><div class="lbl">⬅ tool output (id=${esc(m.tool_call_id)}, name=${esc(m.name)}, ${c.length} chars)</div><pre>${body}</pre></div>`;
    }
  }
  return h;
}

function detail(){
  const r=DATA.configs[curCfg].candidates[curIdx];
  const empty=!(r.text||'').trim();
  let b=`<div class="badges">`+
    `<span class="badge"><b>${r.problem_id}</b> #${r.j}</span>`+
    `<span class="badge">${r.subset} · ${r.category} · ${r.level}</span>`+
    `<span class="badge">finish <b class="${finCls(r.finish_reason)}">${empty?'EMPTY':r.finish_reason}</b></span>`+
    (r.n_tool_calls!=null?`<span class="badge">tool_calls <b>${r.n_tool_calls}</b></span>`:'')+
    (r.n_turns!=null?`<span class="badge">turns <b>${r.n_turns}</b></span>`:'')+
    `<span class="badge">completion <b>${r.completion_tokens}</b></span>`+
    `<span class="badge">reasoning <b>${r.reasoning_tokens}</b></span>`+
    `<span class="badge">prompt_tok <b>${r.prompt_tokens}</b></span>`+
    `<span class="badge">proof <b>${(r.text||'').length}</b>c</span>`+
    `<span class="badge">latency <b>${r.latency_s}</b>s</span>`+
    `<button class="raw" onclick="rawDump()">raw JSON</button></div>`;
  b+=renderMsgs(r.messages||[]);
  document.getElementById('main').innerHTML=b;
  document.getElementById('main').scrollTop=0;
}
function rawDump(){
  const r=DATA.configs[curCfg].candidates[curIdx];
  const w=window.open('','_blank');
  w.document.write('<pre style="white-space:pre-wrap;font:12px monospace;padding:16px">'+
    esc(JSON.stringify(r,null,2))+'</pre>');
}
summary();tabs();clist();detail();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="high_notool,high_pytool,max_notool,max_pytool")
    ap.add_argument("--suffix", default="_k4smoke")
    ap.add_argument("--out", default=str(EVAL_ROOT / "runs" / "k4smoke_review.html"))
    args = ap.parse_args()
    data = build_data(args.configs.split(","), args.suffix)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    Path(args.out).write_text(HTML.replace("__DATA__", blob), encoding="utf-8")
    n = sum(len(c["candidates"]) for c in data["configs"])
    print(f"[done] {n} candidates across {len(data['configs'])} configs -> {args.out} "
          f"({Path(args.out).stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
