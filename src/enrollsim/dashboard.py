"""Static dashboard: reads state/ and audit/ and writes ONE self-contained HTML file (docs/index.html).

No server, no build step, no external libraries. Host the `docs/` folder on Vercel, GitHub Pages or anywhere static.
The dashboard only reads the files the harness already wrote; it never calls the model.
"""
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from .ledger import Ledger

MAX_EVENTS_PER_CASE = 40
MAX_LETTER_CHARS = 1500


def _load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _short(data, n=260):
    s = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return s if len(s) <= n else s[: n - 1] + "…"


def build_data(root):
    root = Path(root)
    reports = root / "audit" / "reports"
    days = []
    for f in sorted(glob.glob(str(reports / "*_summary.json"))):
        s = _load(f)
        if not s:
            continue
        e = s.get("evaluation", {})
        days.append({
            "date": s["sim_date"], "mode": s.get("mode"), "model": s.get("model"),
            "new": s.get("new_applications"), "tokens": s.get("tokens_used_today", 0), "cap": s.get("token_cap"),
            "states": s.get("states", {}), "gate_ok": s.get("gate_ok"), "s1": s.get("audit_s1"), "s2": s.get("audit_s2"),
            "halted": s.get("halted_for_budget"), "accuracy": e.get("outcome_accuracy"), "terminal": e.get("terminal"),
            "correct": e.get("correct"), "open": e.get("open"), "false_accepts": e.get("false_accepts"),
            "false_denials": e.get("false_denials"), "wrong_fields": e.get("wrong_fields_on_accept"),
            "missed_accepts": e.get("missed_accepts"), "by_defect": e.get("by_defect", {}),
            "basket_start": s.get("basket_at_resolve_start"),
        })
    latest = days[-1] if days else None
    audit = compliance = None
    if latest:
        audit = _load(reports / f"{latest['date']}_audit.json", {})
        compliance = _load(reports / f"{latest['date']}_compliance.json", {})
    audit = audit or {}
    compliance = compliance or {}

    ok, n_entries, msg = Ledger.verify(root / "audit" / "ledger.jsonl")
    cases = _load(root / "state" / "cases.json", {})
    gold = _load(root / "audit" / "gold" / "gold.json", {})
    clock = _load(root / "state" / "clock.json", {})
    sim_date = (latest or {}).get("date")

    by_agent = defaultdict(lambda: {"actions": 0, "tokens": 0, "calls": 0})
    actions = Counter()
    token_by_day = defaultdict(int)
    events = defaultdict(list)
    ledger_path = root / "audit" / "ledger.jsonl"
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            a = by_agent[e["agent"]]
            a["actions"] += 1
            a["tokens"] += e.get("tokens") or 0
            if str(e["action"]).startswith("llm:"):
                a["calls"] += 1
            actions[e["action"]] += 1
            token_by_day[e["sim_date"]] += e.get("tokens") or 0
            cid = e.get("case_id")
            if cid and len(events[cid]) < MAX_EVENTS_PER_CASE and e["action"] not in ("sweep",):
                events[cid].append({"seq": e["seq"], "date": e["sim_date"], "agent": e["agent"],
                                    "action": e["action"], "tokens": e.get("tokens") or 0,
                                    "data": _short(e.get("data") or {})})

    rows = []
    from .util import pdate
    today = pdate(sim_date) if sim_date else None
    for cid, c in sorted(cases.items()):
        g = gold.get(cid, {})
        letters = []
        for l in c.get("letters", []):
            try:
                txt = (root / l["file"]).read_text(encoding="utf-8")[:MAX_LETTER_CHARS]
            except OSError:
                txt = "(file missing)"
            letters.append({"kind": l["kind"], "source": l.get("source"), "date": l.get("date"), "text": txt})
        changes = [{"field": x.get("field"), "old": x.get("old"), "new": x.get("new")} if isinstance(x, dict) else {"field": str(x)}
                   for x in c.get("changes", [])]
        age = (today - pdate(c["received_date"])).days if today else None
        rows.append({
            "id": cid, "channel": c.get("channel"), "state": c["state"], "received": c["received_date"],
            "closed": c.get("disposition_date"), "last_code": c.get("last_code"), "reason": c.get("basket_reason"),
            "attempts": c.get("attempts"), "age": age, "flags": c.get("flags", []),
            "defect": g.get("defect"), "expected": g.get("expected_final"),
            "plan": c.get("fields", {}).get("plan_name"), "changes": changes, "letters": letters,
            "events": events.get(cid, []), "missing": c.get("missing_info", []),
        })
    return {
        "generated_from": {"ledger_ok": ok, "ledger_entries": n_entries, "ledger_msg": msg, "sim_date": sim_date,
                           "next_run_date": clock.get("sim_date") if isinstance(clock, dict) else None},
        "days": days, "latest": latest,
        "audit": {"s1": audit.get("s1"), "s2": audit.get("s2"), "findings": audit.get("findings", []),
                  "advisory": audit.get("advisory_llm_review", []), "cases_checked": audit.get("cases_checked")},
        "letter_review": compliance.get("letter_review", []), "compliance": compliance.get("counts", {}),
        "agents": {k: v for k, v in sorted(by_agent.items())}, "actions": dict(actions.most_common()),
        "token_by_day": dict(sorted(token_by_day.items())), "cases": rows,
        "policy": {"tat_days": 10},
    }


def write_dashboard(root, out_dir):
    data = build_data(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", blob)
    (out / "index.html").write_text(html, encoding="utf-8")
    return out / "index.html", len(html)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Enrollment Sim Dashboard</title>
<style>
:root{color-scheme:light;--bg:#f6f5f2;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e3e1db;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#006300;--warn:#8a5a00;--crit:#b02a2a;--chip:#eeede8;--bar:#2a78d6;--bar2:#cde2fb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#121211;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#98978f;--line:#2e2e2b;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#3fc43f;--warn:#fab219;--crit:#ec6a6a;--chip:#262624;--bar:#3987e5;--bar2:#184f95}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#121211;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#98978f;--line:#2e2e2b;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#3fc43f;--warn:#fab219;--crit:#ec6a6a;--chip:#262624;--bar:#3987e5;--bar2:#184f95}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:20px 16px 60px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:16px}
h1{font-size:22px;margin:0}h2{font-size:16px;margin:0 0 4px}.sub{color:var(--ink2);font-size:13px;margin:0 0 12px}
.badge{display:inline-flex;gap:6px;align-items:center;padding:2px 10px;border-radius:99px;background:var(--chip);font-size:12.5px;white-space:nowrap}
.ok{color:var(--good)}.bad{color:var(--crit)}.warn{color:var(--warn)}
button.theme{border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:8px;padding:6px 10px;cursor:pointer}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.kpi,.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px}
.kpi .v{font-size:26px;font-weight:650;line-height:1.1}.kpi .l{color:var(--ink2);font-size:12.5px}.kpi .n{color:var(--muted);font-size:12px;margin-top:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:14px;margin-bottom:14px}
.row{display:grid;grid-template-columns:150px 1fr 74px;gap:8px;align-items:center;padding:3px 0;font-size:13px}
.row .t{height:12px;background:var(--bar2);border-radius:0 4px 4px 0;position:relative}
.row .f{height:100%;background:var(--bar);border-radius:0 4px 4px 0}.row .v{text-align:right;color:var(--ink2);font-variant-numeric:tabular-nums}
.row .k{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600;font-size:12px}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}.legend{display:flex;gap:14px;font-size:12.5px;color:var(--ink2);margin:4px 0 6px;flex-wrap:wrap}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
svg text{fill:var(--ink2);font-size:11px}.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--line);padding:6px 9px;border-radius:8px;font-size:12.5px;box-shadow:0 4px 14px #0003;display:none;z-index:9}
input,select{background:var(--surface);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:7px 10px;font:inherit}
tr.case{cursor:pointer}tr.case:hover{background:var(--chip)}.detail{background:var(--bg);padding:10px 12px;border-radius:8px}
pre{white-space:pre-wrap;word-break:break-word;margin:6px 0;font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace}
.ev{font-size:12.5px;border-left:2px solid var(--line);padding:2px 0 2px 10px;margin:2px 0}.ev b{font-weight:600}.ev span{color:var(--muted)}
.note{font-size:12.5px;color:var(--ink2);background:var(--chip);padding:8px 10px;border-radius:8px;margin-top:8px}
@media(max-width:560px){.row{grid-template-columns:110px 1fr 60px}}
</style></head><body><div class="wrap">
<header><div><h1>Enrollment simulator dashboard</h1><p class="sub" id="sub" style="margin:2px 0 0"></p></div>
<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><span class="badge" id="gate"></span><span class="badge" id="chain"></span><button class="theme" id="th">Theme</button></div></header>
<div class="kpis" id="kpis"></div>
<div class="grid"><div class="card"><h2>Accuracy and tokens by day</h2><p class="sub">Accuracy = correct final outcome among finished cases, scored against the hidden answer key.</p><div id="trend"></div></div>
<div class="card"><h2>Where cases ended up</h2><p class="sub">All cases to date.</p><div id="states"></div></div></div>
<div class="grid"><div class="card"><h2>Accuracy by injected defect</h2><p class="sub">Finished cases only; "open" cases are still in the work basket or waiting on the member.</p><div class="scroll"><table id="defects"></table></div></div>
<div class="card"><h2>Who did the work</h2><p class="sub">Every action and model call is one hash-chained ledger line.</p><div class="scroll"><table id="agents"></table></div><div id="actions" style="margin-top:10px"></div></div></div>
<div class="grid"><div class="card"><h2>Work basket, oldest first</h2><p class="sub">Turnaround target is <span id="tat"></span> days.</p><div class="scroll"><table id="basket"></table></div></div>
<div class="card"><h2>Audit and compliance</h2><p class="sub">Deterministic checks decide pass or fail. The model review is advisory.</p><div id="audit"></div></div></div>
<div class="card" style="margin-bottom:14px"><h2>Case explorer</h2><p class="sub">Click a case to see its ledger timeline, field changes and letters. "Expected" is the hidden answer key, shown here for evaluation only.</p>
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px"><input id="q" placeholder="Search case id, plan, code" style="flex:1;min-width:180px"><select id="fs"></select><select id="fd"></select></div>
<div class="scroll"><table id="cases"></table></div><p class="sub" id="more"></p></div>
<p class="sub">Synthetic data only: fictional people, simplified reply codes, not real CMS rules. Generated from the repository's state/ and audit/ files.</p>
</div><div class="tip" id="tip"></div>
<script id="d" type="application/json">__DATA__</script>
<script>
const D2=JSON.parse(document.getElementById("d").textContent);
const $=id=>document.getElementById(id),esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct=x=>x==null?"–":(x*100).toFixed(1)+"%",num=x=>x==null?"–":Number(x).toLocaleString();
const L=D2.latest||{},G=D2.generated_from;
const tip=$("tip");function showTip(e,h){tip.innerHTML=h;tip.style.display="block";tip.style.left=Math.min(e.clientX+12,innerWidth-230)+"px";tip.style.top=(e.clientY+12)+"px"}
function hideTip(){tip.style.display="none"}
$("th").onclick=()=>{const r=document.documentElement,d=r.getAttribute("data-theme")==="dark"||(!r.getAttribute("data-theme")&&matchMedia("(prefers-color-scheme:dark)").matches);r.setAttribute("data-theme",d?"light":"dark")};
$("sub").textContent=D2.days.length?`Simulated day ${L.date} · ${D2.days.length} run(s) · last run mode: ${L.mode} · model: ${L.model}`:"No runs yet";
$("gate").innerHTML=L.gate_ok?'<span class="ok">✔ Audit gate PASS</span>':'<span class="bad">✖ Audit gate FAIL</span>';
$("chain").innerHTML=G.ledger_ok?`<span class="ok">✔ Ledger intact</span> · ${num(G.ledger_entries)} entries`:'<span class="bad">✖ Ledger broken</span>';
const total=D2.cases.length,cnt=s=>D2.cases.filter(c=>c.state===s).length;
const kp=[[num(total),"Applications","paper + digital"],[pct(L.accuracy),"Outcome accuracy",`${L.correct??0} of ${L.terminal??0} finished`],
[num(L.false_accepts),"False accepts","must stay 0"],[num(L.wrong_fields),"Accepted with a wrong field","not caught by the gate"],
[num(cnt("ACCEPTED")),"Accepted",""],[num(D2.cases.filter(c=>["PENDING_MEMBER","IN_BASKET","ESCALATED","AWAITING_INTAKE"].includes(c.state)).length),"Open","basket, waiting, escalated"],
[num(D2.days.reduce((a,d)=>a+d.tokens,0)),"Tokens used (all days)",`today ${num(L.tokens)} of ${num(L.cap)}`]];
$("kpis").innerHTML=kp.map(k=>`<div class="kpi"><div class="v">${k[0]}</div><div class="l">${k[1]}</div><div class="n">${k[2]}</div></div>`).join("");
function bars(el,items,max){max=max||Math.max(1,...items.map(i=>i[1]));$(el).innerHTML=items.map(i=>`<div class="row" data-t="${esc(i[0])}: ${num(i[1])}"><div class="k" title="${esc(i[0])}">${esc(i[0])}</div><div class="t"><div class="f" style="width:${Math.max(0,i[1]/max*100)}%"></div></div><div class="v">${num(i[1])}</div></div>`).join("")||'<p class="sub">No data yet.</p>';}
const sc={};D2.cases.forEach(c=>sc[c.state]=(sc[c.state]||0)+1);bars("states",Object.entries(sc).sort((a,b)=>b[1]-a[1]));
// trend: line chart, one axis per chart (accuracy, tokens as two small charts)
function line(vals,labels,fmt,color,ymax){const W=460,H=120,p=34,n=vals.length;if(!n)return'<p class="sub">No data yet.</p>';
const mx=ymax??Math.max(1,...vals.filter(v=>v!=null)),x=i=>n===1?W/2:p+(W-p-24)*i/(n-1),y=v=>H-18-(H-30)*(v/mx);
let pts=vals.map((v,i)=>v==null?null:[x(i),y(v)]),path=pts.filter(Boolean).map((q,i)=>(i?"L":"M")+q[0].toFixed(1)+","+q[1].toFixed(1)).join(" ");
let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img"><line x1="${p}" y1="${H-18}" x2="${W-6}" y2="${H-18}" stroke="var(--line)"/><text x="2" y="${y(mx)+4}">${fmt(mx)}</text><text x="2" y="${H-14}">0</text>`;
if(n>1)s+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="2"/>`;
pts.forEach((q,i)=>{if(!q)return;s+=`<circle cx="${q[0]}" cy="${q[1]}" r="4" fill="${color}" stroke="var(--surface)" stroke-width="2"/><circle cx="${q[0]}" cy="${q[1]}" r="14" fill="transparent" data-t="${esc(labels[i])}: ${fmt(vals[i])}"/>`;
if(n<=10)s+=`<text x="${q[0]}" y="${H-3}" text-anchor="middle">${labels[i].slice(5)}</text>`});return s+"</svg>";}
$("trend").innerHTML='<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>Outcome accuracy</span></div>'+line(D2.days.map(d=>d.accuracy),D2.days.map(d=>d.date),v=>pct(v),"var(--s1)",1)+
'<div class="legend"><span><i class="sw" style="background:var(--s2)"></i>Tokens used per day (cap '+num(L.cap)+')</span></div>'+line(D2.days.map(d=>d.tokens),D2.days.map(d=>d.date),v=>num(v),"var(--s2)",L.cap||undefined)+
(D2.days.length<2?'<p class="note">Trend lines fill in as more daily runs are committed.</p>':'');
// defects
const bd=L.by_defect||{};$("defects").innerHTML="<tr><th>Defect</th><th class='n'>Finished</th><th class='n'>Correct</th><th class='n'>Open</th><th>Result</th></tr>"+
Object.entries(bd).sort((a,b)=>b[1].terminal-a[1].terminal).map(([k,v])=>`<tr><td>${esc(k)}</td><td class="n">${v.terminal}</td><td class="n">${v.correct}</td><td class="n">${v.open}</td><td>${v.terminal?(v.correct===v.terminal?'<span class="ok">✔ all correct</span>':'<span class="bad">✖ '+(v.terminal-v.correct)+' wrong</span>'):'<span class="warn">… none finished</span>'}</td></tr>`).join("");
// agents
$("agents").innerHTML="<tr><th>Agent</th><th class='n'>Actions</th><th class='n'>Model calls</th><th class='n'>Tokens</th></tr>"+Object.entries(D2.agents).map(([k,v])=>`<tr><td>${esc(k)}</td><td class="n">${num(v.actions)}</td><td class="n">${num(v.calls)}</td><td class="n">${num(v.tokens)}</td></tr>`).join("");
$("actions").innerHTML='<p class="sub" style="margin:0">Top ledger actions</p><div id="ab"></div>';(function(){const it=Object.entries(D2.actions).slice(0,10),m=Math.max(1,...it.map(i=>i[1]));$("ab").innerHTML=it.map(i=>`<div class="row" data-t="${esc(i[0])}: ${num(i[1])}"><div class="k">${esc(i[0])}</div><div class="t"><div class="f" style="width:${i[1]/m*100}%"></div></div><div class="v">${num(i[1])}</div></div>`).join("")})();
// basket
$("tat").textContent=D2.policy.tat_days;
const open=D2.cases.filter(c=>["IN_BASKET","PENDING_MEMBER","ESCALATED","AWAITING_INTAKE"].includes(c.state)).sort((a,b)=>a.received.localeCompare(b.received)||a.id.localeCompare(b.id));
$("basket").innerHTML="<tr><th>Case</th><th>State</th><th>Received</th><th class='n'>Age (days)</th><th>Reason</th></tr>"+(open.slice(0,25).map(c=>`<tr><td>${esc(c.id)}</td><td>${esc(c.state)}</td><td>${esc(c.received)}</td><td class="n ${c.age>D2.policy.tat_days?"bad":c.age>=8?"warn":""}">${c.age}${c.age>D2.policy.tat_days?" ✖ over":c.age>=8?" ⚠ at risk":""}</td><td>${esc(c.reason||c.last_code||"")}</td></tr>`).join("")||"<tr><td colspan=5>Basket is empty.</td></tr>")+(open.length>25?`<tr><td colspan=5>… and ${open.length-25} more</td></tr>`:"");
// audit
const A=D2.audit,adv=(A.advisory||[]),advBad=adv.filter(a=>a.verdict!=="PASS"),lr=D2.letter_review||[],lrBad=lr.filter(r=>!r.pass);
let h=`<p><b>Deterministic audit:</b> ${A.cases_checked??"–"} cases re-derived from the ledger and stored files. Serious findings S1 <b class="${A.s1?"bad":"ok"}">${A.s1??"–"}</b>, S2 <b class="${A.s2?"bad":"ok"}">${A.s2??"–"}</b>.</p>`;
h+=(A.findings||[]).slice(0,10).map(f=>`<div class="ev"><b>${esc(f.severity)} ${esc(f.code)}</b> ${esc(f.case_id||"")} <span>${esc(f.text)}</span></div>`).join("");
h+=`<p><b>Model review of the audit trail (advisory):</b> ${adv.length-advBad.length} of ${adv.length} sampled cases passed.</p>`+advBad.map(a=>`<div class="ev"><b>${esc(a.case_id)}</b> ${a.findings.map(f=>esc(f.severity)+": "+esc(f.text)).join("<br>")}</div>`).join("");
h+=`<p><b>Letter review (advisory):</b> ${lr.length-lrBad.length} of ${lr.length} sampled letters passed.</p>`+lrBad.map(r=>`<div class="ev"><b>${esc(r.case_id)}</b> ${r.issues.map(esc).join("<br>")}</div>`).join("");
h+=`<p class="sub">Compliance sweep findings: ${Object.keys(D2.compliance).length?esc(JSON.stringify(D2.compliance)):"none"}</p>`;$("audit").innerHTML=h;
// explorer
const st=[...new Set(D2.cases.map(c=>c.state))].sort(),df=[...new Set(D2.cases.map(c=>c.defect).filter(Boolean))].sort();
$("fs").innerHTML='<option value="">All states</option>'+st.map(s=>`<option>${s}</option>`).join("");$("fd").innerHTML='<option value="">All defects</option>'+df.map(s=>`<option>${s}</option>`).join("");
let shown=60;function renderCases(){const q=$("q").value.toLowerCase(),s=$("fs").value,d=$("fd").value;
const rows=D2.cases.filter(c=>(!s||c.state===s)&&(!d||c.defect===d)&&(!q||[c.id,c.plan,c.last_code,c.reason,c.defect].join(" ").toLowerCase().includes(q)));
$("cases").innerHTML="<tr><th>Case</th><th>Channel</th><th>State</th><th>Received</th><th>Last code</th><th>Defect</th><th>Expected</th><th>Match</th></tr>"+rows.slice(0,shown).map(c=>{const fin=["ACCEPTED","DENIED","CLOSED_DUPLICATE","CLOSED_INCOMPLETE"].includes(c.state),m=!c.expected?"":!fin?"…":(c.expected===c.state?'<span class="ok">✔</span>':'<span class="bad">✖</span>');
return `<tr class="case" data-id="${esc(c.id)}"><td>${esc(c.id)}</td><td>${esc(c.channel)}</td><td>${esc(c.state)}</td><td>${esc(c.received)}</td><td>${esc(c.last_code||"")}</td><td>${esc(c.defect||"")}</td><td>${esc(c.expected||"")}</td><td>${m}</td></tr>`}).join("");
$("more").textContent=rows.length>shown?`Showing ${shown} of ${rows.length}. `:`${rows.length} case(s).`;if(rows.length>shown){const b=document.createElement("button");b.textContent="Show more";b.className="theme";b.onclick=()=>{shown+=100;renderCases()};$("more").appendChild(b)}}
["q","fs","fd"].forEach(i=>$(i).oninput=()=>{shown=60;renderCases()});renderCases();
$("cases").onclick=e=>{const tr=e.target.closest("tr.case");if(!tr)return;const nx=tr.nextElementSibling;if(nx&&nx.classList.contains("dt")){nx.remove();return}
const c=D2.cases.find(x=>x.id===tr.dataset.id),r=document.createElement("tr");r.className="dt";
r.innerHTML=`<td colspan="8"><div class="detail"><b>${esc(c.id)}</b> · ${esc(c.plan||"")} · attempts ${c.attempts??0}${c.flags.length?" · flags: "+esc(c.flags.join(", ")):""}${c.missing.length?"<br>Missing from member: "+esc(c.missing.join("; ")):""}
${c.changes.length?"<p><b>Field changes</b></p>"+c.changes.map(x=>`<div class="ev">${esc(x.field)} <span>${esc(x.old??"")} → ${esc(x.new??"")}</span></div>`).join(""):""}
<p><b>Ledger timeline</b></p>${c.events.map(x=>`<div class="ev"><b>${esc(x.agent)}</b> ${esc(x.action)} <span>#${x.seq} · ${esc(x.date)}${x.tokens?" · "+num(x.tokens)+" tokens":""}</span><br><span>${esc(x.data)}</span></div>`).join("")}
${c.letters.map(l=>`<p><b>Letter: ${esc(l.kind)}</b> <span class="badge">${esc(l.source)}</span></p><pre>${esc(l.text)}</pre>`).join("")}</div></td>`;tr.after(r)};
document.addEventListener("mousemove",e=>{const t=e.target.closest("[data-t]");t?showTip(e,t.getAttribute("data-t")):hideTip()});
</script></body></html>
"""
