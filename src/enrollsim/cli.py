"""Command line: python -m enrollsim <run|status|audit|verify-ledger|reset>"""
import argparse
import sys
from pathlib import Path

from .ledger import Ledger
from .llm import LLMError
from .pipeline import Harness


def _root(a):
    return Path(a.root).resolve()


def cmd_run(a):
    over = {}
    if a.cap:
        over["daily_token_cap"] = a.cap
    if a.mock_usage:
        over["mock_usage"] = True
    if a.workers:
        over["llm_concurrency"] = a.workers
    try:
        h = Harness(_root(a), mode=a.mode, overrides=over)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not a.quiet:                      # a live day is minutes of waiting: say what is happening
        h.progress = lambda message: print(message, file=sys.stderr, flush=True)
    worst = 0
    for _ in range(a.days):
        s = h.run_day(a.new)
        e = s["evaluation"]
        t = s.get("timings_seconds", {})
        print(f"[{s['sim_date']}] tokens {s['tokens_used_today']:,}/{s['token_cap']:,}  states {s['states']}  "
              f"audit {'PASS' if s['gate_ok'] else 'FAIL'} (S1 {s['audit_s1']} S2 {s['audit_s2']})  "
              f"accuracy {e['outcome_accuracy']}  false-accepts {e['false_accepts']}  "
              f"{t.get('total', 0)}s/{s.get('llm_calls_today', 0)} calls"
              + ("  ** stopped early: token cap **" if s["halted_for_budget"] else ""))
        if not s["gate_ok"]:
            worst = 2
        if s["halted_for_budget"]:
            break
    return worst


def cmd_status(a):
    h = Harness(_root(a), mode="mock")
    print(f"sim date {h.sim_date}   cases {len(h.cases)}   tokens today {h.budget.used():,}/{h.budget.cap:,}")
    states = {}
    for c in h.cases.values():
        states[c["state"]] = states.get(c["state"], 0) + 1
    print("states:", dict(sorted(states.items())))
    from .agents import OPEN
    rows = sorted((c for c in h.cases.values() if c["state"] in OPEN), key=lambda c: (c["received_date"], c["case_id"]))
    print("oldest open items (resolved oldest-first):")
    for c in rows[:15]:
        print(f"  {c['case_id']}  {c['state']:<16} received {c['received_date']}  {c['basket_reason'] or ''}")
    return 0


def cmd_audit(a):
    h = Harness(_root(a), mode="mock")
    audit, gold = h.audit_only()
    print(f"ledger {'intact' if audit['ledger_ok'] else 'BROKEN'} ({audit['ledger_entries']} entries)  "
          f"S1 {audit['s1']}  S2 {audit['s2']}  accuracy {gold['outcome_accuracy']}")
    for f in audit["findings"][:30]:
        print(f"  {f['severity']} {f['code']} {f['case_id'] or ''}: {f['text']}")
    return 0 if audit["ledger_ok"] and audit["s1"] == 0 else 2


def cmd_verify(a):
    ok, n, msg = Ledger.verify(_root(a) / "audit" / "ledger.jsonl")
    print(f"{'OK' if ok else 'FAILED'}: {msg} ({n} entries)")
    return 0 if ok else 2


def cmd_dashboard(a):
    from .dashboard import write_dashboard
    path, size = write_dashboard(_root(a), Path(a.out) if Path(a.out).is_absolute() else _root(a) / a.out)
    print(f"wrote {path} ({size:,} bytes)")
    return 0


def cmd_reset(a):
    if not a.yes:
        print("this deletes state/, workspace/ and audit/. Re-run with --yes to confirm.")
        return 1
    Harness(_root(a), mode="mock").reset()
    print("reset done")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="enrollsim", description=__doc__)
    ap.add_argument("--root", default=".", help="project folder (default: current directory)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one or more simulated days")
    r.add_argument("--mode", choices=["mock", "live"], default="mock")
    r.add_argument("--days", type=int, default=1)
    r.add_argument("--new", type=int, default=None, help="new applications per day (default: from settings)")
    r.add_argument("--cap", type=int, default=None, help="override the daily token cap")
    r.add_argument("--workers", type=int, default=None,
                   help="how many model calls may be in flight at once (default: settings.llm_concurrency)")
    r.add_argument("--quiet", action="store_true", help="no per-stage progress on stderr")
    r.add_argument("--mock-usage", action="store_true", help="make the mock backend report estimated token use")
    r.set_defaults(fn=cmd_run)
    for name, fn, helptext in (("status", cmd_status, "show queue and token use"),
                               ("audit", cmd_audit, "re-run the audit on the current state"),
                               ("verify-ledger", cmd_verify, "check the hash chain")):
        sub.add_parser(name, help=helptext).set_defaults(fn=fn)
    d = sub.add_parser("dashboard", help="build the static dashboard (docs/index.html)")
    d.add_argument("--out", default="docs")
    d.set_defaults(fn=cmd_dashboard)
    x = sub.add_parser("reset", help="delete all state, workspace and audit data")
    x.add_argument("--yes", action="store_true")
    x.set_defaults(fn=cmd_reset)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
