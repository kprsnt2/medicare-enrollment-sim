"""Explain every case where the run and the hidden answer key disagree.

    PYTHONPATH=src python diagnose.py            # summary + why each miss missed
    PYTHONPATH=src python diagnose.py --all      # also list the cases that agreed
    PYTHONPATH=src python diagnose.py --case APP-20261017-010

It writes `why_cases.txt` (the miss-list, in full) and prints a short version. For each miss it
shows what gold expected, what the run did, which CMS reply code produced the basket reason, and
the ledger timeline of that case - so the reason is readable without re-reading the whole ledger.

An empty miss-list is the good outcome: it means every outcome and every stored field matched.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

CRITICAL = ["first_name", "last_name", "dob", "sex", "mbi", "part_a_eff", "part_b_eff", "zip", "plan_id",
            "pcp_name", "election_type", "sep_reason", "requested_effective_date"]
TERMINAL = {"ACCEPTED", "DENIED", "CLOSED_DUPLICATE", "CLOSED_INCOMPLETE"}
WATCH = ("intake_parsed", "to_basket", "cms_response", "decision", "fix_applied", "fix_rejected",
         "action_rejected", "denial_blocked", "rfi_requested", "escalated", "disposition",
         "llm:paper_intake", "llm:resolver", "llm_failed", "llm_error:paper_intake", "intake_defaulted")


def load(path, default=None):
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def same(a, b):
    return str(a if a is not None else "").strip().lower() == str(b if b is not None else "").strip().lower()


def reject_reason(code, policy):
    """The policy sentence behind a reply code, so the report says why the system rejected it."""
    for section in ("reject_codes", "reply_codes", "codes"):
        entry = (policy.get(section) or {}).get(code)
        if isinstance(entry, dict):
            return entry.get("message") or entry.get("description") or ""
        if isinstance(entry, str):
            return entry
    return ""


def explain(case, gold, policy):
    """Every way this case disagrees with its answer key."""
    problems = []
    expected, actual = gold["expected_final"], case["state"]
    if actual != expected:
        if actual not in TERMINAL:
            problems.append(f"still open as {actual}; gold expects {expected} once it terminates")
        else:
            problems.append(f"ended {actual}, gold expected {expected}")
    if case.get("basket_reason"):
        why = reject_reason(case["basket_reason"], policy)
        problems.append(f"basket reason {case['basket_reason']}"
                        + (f" ({why})" if why else "")
                        + (f", last CMS code {case['last_code']}" if case.get("last_code") else ""))
    if case["state"] == "ACCEPTED" and expected == "ACCEPTED":
        for k in CRITICAL:
            stored, true = (case["fields"] or {}).get(k), gold["true_fields"].get(k)
            if not same(stored, true):
                problems.append(f"field {k}: stored {stored!r}, gold {true!r}"
                                f" (form said {(case.get('original_fields') or {}).get(k)!r})")
    return problems


def timeline(case_id, events, limit=12):
    out = []
    for e in events:
        if e["case_id"] != case_id or e["action"] not in WATCH:
            continue
        out.append(f"  #{e['seq']} {e['agent']} {e['action']} {json.dumps(e['data'], ensure_ascii=False)[:300]}")
    return out[:limit]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="project folder (default: current directory)")
    ap.add_argument("--case", default=None, help="explain one case id and stop")
    ap.add_argument("--all", action="store_true", help="also list the cases that agreed")
    ap.add_argument("--out", default="why_cases.txt")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    cases = load(root / "state" / "cases.json")
    gold = load(root / "audit" / "gold" / "gold.json")
    policy = load(root / "config" / "policy.json")
    events = [json.loads(line) for line in open(root / "audit" / "ledger.jsonl", encoding="utf-8") if line.strip()]

    if a.case:
        c, g = cases.get(a.case), gold.get(a.case)
        if not c or not g:
            print(f"no such case: {a.case}", file=sys.stderr)
            return 1
        print(f"{a.case}  defect {g['defect']}  variant {g.get('variant')}  expected {g['expected_final']}"
              f"  actual {c['state']}  received {c['received_date']}")
        for p in explain(c, g, policy) or ["no disagreement: the run matches the answer key"]:
            print("  - " + p)
        print("  ledger timeline:")
        print("\n".join(timeline(a.case, events)) or "  (nothing logged)")
        return 0

    misses, agreed, by_defect = {}, [], {}
    for cid, c in sorted(cases.items()):
        g = gold.get(cid)
        if not g:
            continue
        problems = explain(c, g, policy)
        d = by_defect.setdefault(g["defect"], Counter())
        d["total"] += 1
        if problems:
            misses[cid] = {"defect": g["defect"], "expected": g["expected_final"], "actual": c["state"],
                           "problems": problems, "timeline": timeline(cid, events)}
            d["miss"] += 1
        else:
            agreed.append(cid)
            d["ok"] += 1

    lines = []
    if not misses:
        lines.append("no disagreements: every outcome and every stored field matches the answer key")
    for cid, m in misses.items():
        lines.append(f"{cid}  defect {m['defect']}  expected {m['expected']}  actual {m['actual']}")
        lines += [f"    - {p}" for p in m["problems"]]
        lines += ["    ledger:"] + m["timeline"]
        lines.append("")
    if a.all:
        lines.append("agreed: " + ", ".join(agreed))
    (root / a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"checked {len(misses) + len(agreed)} cases against the answer key: "
          f"{len(agreed)} agree, {len(misses)} disagree")
    for defect, d in sorted(by_defect.items()):
        flag = f"  <-- {d['miss']} miss" if d["miss"] else ""
        print(f"  {defect:18} {d['ok']:3}/{d['total']:<3}{flag}")
    for cid, m in misses.items():
        print(f"\n{cid}  defect {m['defect']}  expected {m['expected']}  actual {m['actual']}")
        for p in m["problems"]:
            print("  - " + p)
    print(f"\nfull miss-list written to {root / a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
