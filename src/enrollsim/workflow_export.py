"""Workflow export: read state/, audit/ and workspace/ and write ONE JSON file for a web page.

Run it, then point the MEOP site's /workflow page at the file it writes:

    PYTHONPATH=src python -m enrollsim.workflow_export --root . --out workflow_export.json

This is the same idea as dashboard.py - the harness wrote the files, this module only reads them -
with two differences: the output is JSON instead of an HTML page, and it describes the PIPELINE
(the eight stages Harness.run_day executes) rather than a dashboard of numbers.

What it reads: config/policy.json, config/reference.json, config/settings.json,
state/cases.json, state/clock.json, state/tokens.json, audit/ledger.jsonl (hash chain verified
with Ledger.verify), audit/reports/*_summary.json / *_audit.json / *_compliance.json /
*_eval.json, workspace/letters/* and workspace/inbox/*, and audit/gold/gold.json.

About the answer key: audit/gold/gold.json holds, per case, the injected defect and the outcome
the case SHOULD have reached, plus `true_fields` - the fictional correct field values. This module
emits only `defect` and `expected_final` (the published accuracy comparison - docs/index.html shows
it too) and never reads `true_fields` at all. Everything else in the JSON is a count, a ledger
entry, a stored case field or a letter the harness actually issued.

No third-party packages, no network, no model calls, no writes outside the file you name.
"""
import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .ledger import Ledger
from .rules import CODES as REPLY_CODE_TEXT
from .util import pdate

MAX_CASES = 36                 # cases in "cases"; picked from the latest simulated days
MAX_EVENT_CHARS = 120          # one-line summary of a ledger event's data
MAX_LETTER_CHARS = 1200        # a letter can be the longest thing on the page
MAX_EVENTS_PER_CASE = 200      # safety valve only: a day logs a handful of events per case
MAX_RECENT_ACTIONS = 12        # distinct action names in the "all days" top-action table

# The stages below ARE Harness.run_day (src/enrollsim/pipeline.py), in order. Each entry says who
# does the work and where the result lands; the counts and the example are computed from the
# files, so a number is never true only because it is written down here.
STAGES = [
    {
        "step_no": 1, "title": "Applicants", "owner": "applicant_simulator", "engine": "system",
        "artifact": "workspace/inbox/",
        "short": "Synthesise a day of applications",
        "what": "Deterministic generator turns a seed into fictional members: paper forms with OCR noise, "
                "digital JSON payloads, and one injected defect per applicant. Each application also gets a "
                "hidden answer key that only the evaluation stage reads.",
    },
    {
        "step_no": 2, "title": "Intake", "owner": "paper_intake / digital_intake", "engine": "LLM",
        "artifact": "workspace/inbox/ + audit/ledger.jsonl (intake_parsed)",
        "short": "Read the form into 21 fields",
        "what": "Paper: a model reads the scanned text and must return a verbatim evidence quote for every "
                "value; a value it cannot quote is capped at low confidence and flagged. Digital: fields are "
                "copied as submitted, no model call.",
    },
    {
        "step_no": 3, "title": "Processor and mock CMS", "owner": "processor", "engine": "deterministic",
        "artifact": "audit/ledger.jsonl (cms_response)",
        "short": "Completeness check, then submit",
        "what": "Signature, required fields and PCP completeness are checked first. A complete application is "
                "submitted to the mock CMS, whose fixed-order checks answer with ACC, DUP or a REJ-* code.",
    },
    {
        "step_no": 4, "title": "CMS reply", "owner": "processor", "engine": "system",
        "artifact": "audit/ledger.jsonl (cms_response)",
        "short": "Reply code decides the route",
        "what": "ACC and DUP finalise the case; every other code routes it to the work basket with that code "
                "as the reason. Reply codes are simplified stand-ins, not real CMS transaction codes.",
    },
    {
        "step_no": 5, "title": "Work basket", "owner": "processor", "engine": "system",
        "artifact": "state/cases.json + audit/ledger.jsonl (to_basket)",
        "short": "Route the case to a queue",
        "what": "Cases land in one queue with the reason printed on them: missing information, low-confidence "
                "OCR fields, or the CMS reply code. The queue is worked oldest first under a daily capacity.",
    },
    {
        "step_no": 6, "title": "Resolver", "owner": "resolver", "engine": "LLM",
        "artifact": "audit/ledger.jsonl (decision, fix_applied, rfi_requested, escalated)",
        "short": "Fix it, ask for it, or deny it",
        "what": "A model picks FIX, CONFIRM, REQUEST_INFO, DENY or ESCALATE. A fix only lands if its evidence "
                "quote is really in the document; a denial needs a second deterministic gate and is blocked "
                "when the paperwork hints the member may qualify.",
    },
    {
        "step_no": 7, "title": "Notices", "owner": "notice", "engine": "LLM",
        "artifact": "workspace/letters/",
        "short": "Write the member letter",
        "what": "One letter per outcome (ACCEPT, DENY, RFI, CLOSED_INCOMPLETE). The compliance agent checks the "
                "required phrases and rejects the letter for a redo; a fallback template is used if the model "
                "keeps failing.",
    },
    {
        "step_no": 8, "title": "Compliance and audit", "owner": "compliance + auditor", "engine": "deterministic",
        "artifact": "audit/reports/ + audit/gold/",
        "short": "Re-derive the day from the evidence",
        "what": "Deterministic checks escalate the gate (S1 = zero tolerance); an LLM review of a sample is "
                "advisory only. Then every case is scored against the hidden answer key: accuracy, false "
                "accepts, false denials.",
    },
]

# step 5 hands the case back to step 3: a fixed case is resubmitted to the mock CMS.
LOOPS = [{"from_step": 5, "to_step": 3, "label": "resubmission after a fix or a replaced field"}]

DISCLAIMER = (
    "Synthetic data only. Every person, Medicare number and plan is fictional and generated from a seed; "
    "member IDs, plan IDs and case IDs carry the SYN or APP prefix. The CMS reply codes, election windows, "
    "denial reasons and notice wording are simplified stand-ins defined in this repository's config/policy.json "
    "- they are NOT real CMS transaction codes or CMS rules, and nothing here is a compliance determination. "
    "The 'expected' value shown per case is this simulation's own answer key for scoring accuracy, not a legal "
    "outcome. In mock mode the agents are deterministic parsers: the numbers prove the plumbing, guards and "
    "audit trail, not model quality."
)


def _load(path, default=None):
    """Read one JSON file; a missing or broken file falls back to `default` (never raises)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _clip(obj, n):
    """One-line, length-capped rendering of a ledger `data` value."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= n else text[: n - 1] + "\u2026"


def _counts(mapping, order=None):
    """Counter -> plain dict, most frequent first; ties broken by the key so output is stable."""
    c = mapping if isinstance(mapping, Counter) else Counter(mapping)
    keys = order or sorted(c, key=lambda k: (-c[k], str(k)))
    return {k: c[k] for k in keys}


def _summary_line(entry):
    """A one-line human summary of one ledger entry's data, chosen per action."""
    a, d = entry.get("action", ""), entry.get("data") or {}
    if not isinstance(d, dict):
        return _clip(d, MAX_EVENT_CHARS)
    if a.startswith("llm:"):
        out = d.get("output")
        o = out if isinstance(out, dict) else {}
        if a == "llm:resolver":
            return _clip(f"{o.get('action', '?')}: {o.get('diagnosis') or o.get('reason') or ''}", MAX_EVENT_CHARS)
        if a == "llm:paper_intake":
            conf = o.get("confidence") or {}
            low = sorted(k for k, v in conf.items() if isinstance(v, (int, float)) and v < 0.75)
            return _clip(f"parsed {len(o.get('fields') or {})} fields, low confidence: {', '.join(low) or 'none'}",
                         MAX_EVENT_CHARS)
        if a == "llm:notice":
            return _clip(f"'{o.get('subject', '')}'", MAX_EVENT_CHARS)
        if a == "llm:auditor_review":
            return _clip(f"verdict {o.get('verdict', '?')}, {len(o.get('findings') or [])} finding(s)",
                         MAX_EVENT_CHARS)
        if a == "llm:letter_review":
            issues = o.get("issues") or []
            return _clip(f"pass {o.get('pass')}" + (f", issues: {'; '.join(map(str, issues))}" if issues else ""),
                         MAX_EVENT_CHARS)
        if "error" in d:
            return _clip(f"error: {d['error']}", MAX_EVENT_CHARS)
        return _clip(d, MAX_EVENT_CHARS)
    if a == "cms_response":
        return _clip(f"{d.get('code', '?')} - {d.get('message', '')}", MAX_EVENT_CHARS)
    if a == "to_basket":
        extra = {k: v for k, v in d.items() if k != "reason"}
        return _clip(f"reason {d.get('reason', '?')}" + (f", {_clip(extra, 60)}" if extra else ""), MAX_EVENT_CHARS)
    if a == "fix_applied":
        return _clip(f"{d.get('field')}: {d.get('old')!r} -> {d.get('new')!r}, evidence {d.get('evidence')!r}",
                     MAX_EVENT_CHARS)
    if a == "decision":
        return _clip(f"chose {d.get('chosen', '?')} (confidence {d.get('confidence', '?')}): {d.get('diagnosis', '')}",
                     MAX_EVENT_CHARS)
    if a == "letter_issued":
        return _clip(f"{d.get('kind', '?')} letter from {d.get('source', '?')}, {d.get('file', '')}", MAX_EVENT_CHARS)
    if a == "submitted":
        return _clip(f"{d.get('channel', '?')} application arrived", MAX_EVENT_CHARS)
    if a == "intake_parsed":
        flagged, issues = d.get("flagged") or [], d.get("issues") or []
        return _clip(f"fields parsed, flagged {', '.join(flagged) or 'none'}, {len(issues)} evidence issue(s)",
                     MAX_EVENT_CHARS)
    if a == "intake_defaulted":
        return _clip(f"defaulted {d.get('field')} to {d.get('value')!r} ({d.get('basis', '')})", MAX_EVENT_CHARS)
    if a == "disposition":
        return _clip(f"case finalised as {d.get('state', '?')}", MAX_EVENT_CHARS)
    if a == "escalated":
        return _clip(f"sent to a human: {d.get('why', '')}", MAX_EVENT_CHARS)
    if a == "rfi_requested":
        return _clip(f"asked the member for {', '.join(map(str, d.get('missing') or []))}, respond by {d.get('deadline')}",
                     MAX_EVENT_CHARS)
    if a in ("fix_rejected", "action_rejected", "denial_blocked", "letter_rejected"):
        return _clip(f"blocked: {'; '.join(map(str, d.get('errors') or d.get('issues') or [])) or d.get('why', '')}",
                     MAX_EVENT_CHARS)
    return _clip(d, MAX_EVENT_CHARS)


def _read_ledger(path):
    """Parse the ledger the way dashboard.build_data does: a bad line is skipped, not fatal."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _read_letters(root, case):
    """The letter files a case points at, newest last, each clipped to MAX_LETTER_CHARS."""
    letters = []
    for l in case.get("letters", []) or []:
        try:
            text = (Path(root) / l["file"]).read_text(encoding="utf-8")
        except OSError:
            text = "(letter file missing)"
        letters.append({"kind": l.get("kind"), "source": l.get("source"), "date": l.get("date"),
                        "text": text[:MAX_LETTER_CHARS], "truncated": len(text) > MAX_LETTER_CHARS})
    return letters


def _pick_cases(cases, gold):
    """Deterministic spread of cases for the page: newest two simulated days, capped, plus the
    oldest still-open items so a strand that is stuck for days is never invisible."""
    days = sorted({c.get("received_date") for c in cases.values() if c.get("received_date")}, reverse=True)
    recent = [d for d in days[:2]]
    picked = [cid for cid, c in sorted(cases.items()) if c.get("received_date") in recent]
    picked = picked[:MAX_CASES]
    terminal = {"ACCEPTED", "DENIED", "CLOSED_DUPLICATE", "CLOSED_INCOMPLETE"}
    open_items = sorted((cid for cid, c in cases.items() if c.get("state") not in terminal),
                        key=lambda cid: (cases[cid].get("received_date") or "", cid))
    for cid in open_items:
        if len(picked) >= MAX_CASES:
            break
        if cid not in picked:
            picked.append((cid))
    return sorted(set(picked)), recent


def _stage_example(stages_events, case_events):
    """A real one-line example for a stage: earliest ledger event it produced, else the newest case."""
    for e in case_events:
        return f"{e['case_id']} {e['sim_date']}: {e['action']} - {_summary_line(e)}"
    for e in stages_events:
        where = e.get("case_id") or "whole day"
        return f"{where} {e['sim_date']}: {e['action']} - {_summary_line(e)}"
    return ""


def build_export(root):
    """Read the project folder and return the workflow export as a dict (deterministic key order)."""
    root = Path(root)
    cfg, state, reports = root / "config", root / "state", root / "audit" / "reports"
    policy = _load(cfg / "policy.json", {}) or {}
    ref = _load(cfg / "reference.json", {}) or {}
    settings = _load(cfg / "settings.json", {}) or {}
    cases = _load(state / "cases.json", {}) or {}
    clock = _load(state / "clock.json", {}) or {}
    tokens = _load(state / "tokens.json", {}) or {}
    gold = _load(root / "audit" / "gold" / "gold.json", {}) or {}

    # ---- the ledger: verified chain, then indexed by case and by agent --------
    ok, n_entries, msg = Ledger.verify(root / "audit" / "ledger.jsonl")
    raw = _read_ledger(root / "audit" / "ledger.jsonl")
    by_agent = defaultdict(lambda: {"actions": 0, "tokens": 0, "calls": 0})
    actions, tokens_by_action, case_events, raw_by_case = Counter(), Counter(), defaultdict(list), defaultdict(list)
    token_by_day, mode_by_day = defaultdict(int), {}
    first_case_event = {}
    for e in raw:
        a = by_agent[e["agent"]]
        a["actions"] += 1
        a["tokens"] += e.get("tokens") or 0
        if str(e["action"]).startswith("llm:"):
            a["calls"] += 1
            tokens_by_action[e["action"]] += e.get("tokens") or 0
        actions[e["action"]] += 1
        token_by_day[e["sim_date"]] += e.get("tokens") or 0
        if e.get("action") == "day_start":
            mode_by_day[e["sim_date"]] = (e.get("data") or {}).get("mode")
        cid = e.get("case_id")
        if cid:
            raw_by_case[cid].append(e)
            first_case_event.setdefault(e["action"], e)
            if len(case_events[cid]) < MAX_EVENTS_PER_CASE:
                case_events[cid].append({"seq": e["seq"], "date": e["sim_date"], "agent": e["agent"],
                                         "action": e["action"], "summary": _summary_line(e),
                                         "tokens": e.get("tokens") or 0})
    agents = {k: v for k, v in sorted(by_agent.items())}

    # ---- per-day reports ------------------------------------------------------
    days = []
    for f in sorted(glob.glob(str(reports / "*_summary.json"))):
        s = _load(f)
        if not s:
            continue
        ev = s.get("evaluation", {}) or {}
        days.append({
            "date": s.get("sim_date"), "mode": s.get("mode"), "model": s.get("model"),
            "new_applications": s.get("new_applications"),
            "basket_at_resolve_start": s.get("basket_at_resolve_start"),
            "states": s.get("states", {}),
            "accuracy": ev.get("outcome_accuracy"), "terminal": ev.get("terminal"), "correct": ev.get("correct"),
            "open": ev.get("open"), "false_accepts": ev.get("false_accepts"),
            "false_denials": ev.get("false_denials"), "wrong_fields_on_accept": ev.get("wrong_fields_on_accept"),
            "s1": s.get("audit_s1"), "s2": s.get("audit_s2"), "gate_ok": s.get("gate_ok"),
            "ledger_ok": s.get("ledger_ok"), "halted_for_budget": s.get("halted_for_budget"),
            "tokens": s.get("tokens_used_today", 0), "token_cap": s.get("token_cap"),
            "llm_calls": s.get("llm_calls_today"), "workers": s.get("workers"),
            "timings": s.get("timings_seconds", {}),
            "compliance": s.get("compliance", {}), "by_defect": ev.get("by_defect", {}),
            "oldest_open": s.get("oldest_open", []),
        })
    latest = days[-1] if days else None
    sim_date = (latest or {}).get("date") or clock.get("sim_date")

    # ---- meta ---------------------------------------------------------------
    total_case_tokens = sum(e.get("tokens") or 0 for e in raw)
    meta = {
        "sim_date": sim_date,
        "next_run_date": clock.get("sim_date"),
        "day_index": clock.get("day_index"),
        "days_run": len(days),
        "first_day": days[0]["date"] if days else None,
        "modes": sorted({m for m in mode_by_day.values() if m} | {d["mode"] for d in days if d.get("mode")}),
        "latest_mode": (latest or {}).get("mode"),
        "model": settings.get("model"),
        "models_seen": sorted({e["model"] for e in raw if e.get("model")}),
        "token_cap": settings.get("daily_token_cap") or (latest or {}).get("token_cap"),
        "letters_source": _counts(e["data"].get("source") for e in raw
                                  if e.get("action") == "letter_issued" and isinstance(e.get("data"), dict)),
        "tokens_used_total": total_case_tokens,
        "tokens_used_today": token_by_day.get(sim_date, 0),
        "tokens_by_day": dict(sorted(token_by_day.items())),
        "cases_total": len(cases),
        "ledger_ok": ok, "ledger_message": msg, "entries": n_entries,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": {
            "config": ["config/policy.json", "config/reference.json", "config/settings.json"],
            "state": ["state/cases.json", "state/clock.json", "state/tokens.json"],
            "audit": ["audit/ledger.jsonl", "audit/reports/*_summary.json", "audit/reports/*_audit.json",
                      "audit/reports/*_compliance.json", "audit/reports/*_eval.json"],
            "workspace": ["workspace/letters/*", "workspace/inbox/*"],
            "answer_key": "audit/gold/gold.json - only 'defect' and 'expected_final' are read",
        },
    }

    # ---- KPI totals ----------------------------------------------------------
    states_total = Counter(c.get("state") for c in cases.values())
    ticket_kinds = Counter(l.get("kind") for c in cases.values() for l in (c.get("letters") or []))
    kpis = {
        "cases": len(cases),
        "channels": _counts(c.get("channel") for c in cases.values()),
        "states": _counts(states_total),
        "accepted": states_total.get("ACCEPTED", 0),
        "denied": states_total.get("DENIED", 0),
        "closed_incomplete": states_total.get("CLOSED_INCOMPLETE", 0),
        "closed_duplicate": states_total.get("CLOSED_DUPLICATE", 0),
        "open": sum(v for k, v in states_total.items() if k in ("AWAITING_INTAKE", "IN_BASKET", "PENDING_MEMBER", "ESCALATED")),
        "in_basket": states_total.get("IN_BASKET", 0),
        "pending_member": states_total.get("PENDING_MEMBER", 0),
        "escalated": states_total.get("ESCALATED", 0),
        "awaiting_intake": states_total.get("AWAITING_INTAKE", 0),
        "letters": sum(ticket_kinds.values()),
        "letter_kinds": _counts(ticket_kinds),
        "ledger_entries": n_entries,
        "ledger_ok": ok,
        "accuracy": (latest or {}).get("accuracy"),
        "correct": (latest or {}).get("correct"),
        "terminal": (latest or {}).get("terminal"),
        "false_accepts": (latest or {}).get("false_accepts"),
        "false_denials": (latest or {}).get("false_denials"),
        "wrong_fields_on_accept": (latest or {}).get("wrong_fields_on_accept"),
        "s1": (latest or {}).get("s1"), "s2": (latest or {}).get("s2"),
        "gate_ok": (latest or {}).get("gate_ok"),
        "tokens_used_total": total_case_tokens,
        "tokens_used_today": token_by_day.get(sim_date, 0),
        "token_cap": meta["token_cap"],
        "model_calls": sum(1 for e in raw if str(e.get("action")).startswith("llm:")),
        "fixes_applied": sum(len(c.get("changes") or []) for c in cases.values()),
        "fixes_rejected": actions.get("fix_rejected", 0),
        "rfi_sent": actions.get("rfi_requested", 0),
    }

    # ---- stages --------------------------------------------------------------
    letter_events = [(e["sim_date"], e) for e in raw if e.get("action") == "letter_issued"]
    letters_files = glob.glob(str(root / "workspace" / "letters" / "*"))
    inbox_files = glob.glob(str(root / "workspace" / "inbox" / "*"))
    channel_files = Counter("digital" if Path(f).suffix == ".json" else "paper" for f in inbox_files)
    decided = Counter()
    for cid, evs in raw_by_case.items():
        for e in evs:
            if e.get("action") == "decision":
                decided[(e.get("data") or {}).get("chosen")] += 1
    cms_codes = _counts(e["data"].get("code") for e in raw
                        if e.get("action") == "cms_response" and isinstance(e.get("data"), dict))
    basket_reasons = _counts(e["data"].get("reason") for e in raw
                             if e.get("action") == "to_basket" and isinstance(e.get("data"), dict))
    letter_kinds = _counts(e["data"].get("kind") for _, e in letter_events if isinstance(e.get("data"), dict))
    per_stage_actions = {
        1: ["submitted"], 2: ["intake_parsed", "intake_defaulted", "llm:paper_intake"],
        3: ["cms_response", "disposition", "to_basket"], 4: ["cms_response"], 5: ["to_basket"],
        6: ["decision", "fix_applied", "fix_rejected", "rfi_requested", "escalated", "denial_blocked",
            "action_rejected", "llm:resolver"],
        7: ["letter_issued", "letter_rejected", "llm:notice"],
        8: ["sweep", "audit_report", "letter_review", "llm:auditor_review", "llm:letter_review"],
    }
    stage_events = {n: [e for e in raw if e.get("action") in acts] for n, acts in per_stage_actions.items()}

    def case_ev(action):
        return [e for e in raw if e.get("action") == action and e.get("case_id")]

    stage_counts = {
        1: {"applications": len(inbox_files), "channels": _counts(channel_files),
            "by_day": _counts(c.get("received_date") for c in cases.values())},
        2: {"paper_model_calls": actions.get("llm:paper_intake", 0),
            "digital_deterministic": actions.get("intake_parsed", 0) - actions.get("llm:paper_intake", 0),
            "intake_parsed": actions.get("intake_parsed", 0),
            "defaulted_fields": actions.get("intake_defaulted", 0),
            "flags_raised": len({k for c in cases.values() for k in (c.get("flags") or [])}),
            "flags_by_field": _counts(f for c in cases.values() for f in (c.get("flags") or [])),
            "failures": actions.get("llm_failed", 0)},
        3: {"cms_submissions": actions.get("cms_response", 0),
            "completeness_rejections": basket_reasons.get("INCOMPLETE", 0),
            "dispositions": actions.get("disposition", 0),
            "dispositions_by_state": _counts((e.get("data") or {}).get("state") for e in raw
                                             if e.get("action") == "disposition" and isinstance(e.get("data"), dict))},
        4: {"cms_calls": actions.get("cms_response", 0),
            "cases_submitted": len({e["case_id"] for e in raw
                                    if e.get("action") == "cms_response" and e.get("case_id")}),
            "codes": cms_codes,
            "accepted_by_cms": cms_codes.get("ACC", 0),
            "returned_to_basket": sum(v for k, v in cms_codes.items() if k not in ("ACC", "DUP")),
            "closed_duplicate_by_cms": cms_codes.get("DUP", 0)},
        5: {"routed_to_basket": actions.get("to_basket", 0), "reasons": basket_reasons,
            "currently_in_basket": states_total.get("IN_BASKET", 0),
            "resolver_daily_capacity": settings.get("resolver_daily_capacity")},
        6: {"decisions": _counts(decided), "decision_rounds": actions.get("decision", 0),
            "resolver_model_calls": actions.get("llm:resolver", 0),
            "cases_worked": len({e["case_id"] for e in raw if e.get("action") == "decision" and e.get("case_id")}),
            "fixes_applied": actions.get("fix_applied", 0),
            "fixes_rejected_by_guard": actions.get("fix_rejected", 0),
            "rfi_requested": actions.get("rfi_requested", 0),
            "denied_by_resolver": _counts((e.get("data") or {}).get("state") for e in raw
                                          if e.get("action") == "disposition" and isinstance(e.get("data"), dict)
                                          ).get("DENIED", 0),
            "denial_blocked": actions.get("denial_blocked", 0),
            "escalations_logged": actions.get("escalated", 0),
            "action_rejected": actions.get("action_rejected", 0),
            "escalated_states": states_total.get("ESCALATED", 0),
            "pending_member_states": states_total.get("PENDING_MEMBER", 0)},
        7: {"model_calls": actions.get("llm:notice", 0), "letters_issued": len(letter_events),
            "kinds": letter_kinds, "letter_files": len(letters_files),
            "sources": _counts(e["data"].get("source") for _, e in letter_events if isinstance(e.get("data"), dict)),
            "rejected_for_redo": actions.get("letter_rejected", 0)},
        8: {"s1": (latest or {}).get("s1"), "s2": (latest or {}).get("s2"),
            "gate_ok": (latest or {}).get("gate_ok"), "ledger_ok": ok,
            "cases_checked": ((_load(reports / f"{sim_date}_audit.json", {}) or {}).get("cases_checked")
                              if sim_date else None),
            "compliance_findings": actions.get("sweep", 0),
            "advisory_reviews": actions.get("llm:auditor_review", 0),
            "letter_reviews": actions.get("llm:letter_review", 0),
            "accuracy": (latest or {}).get("accuracy"), "terminal": (latest or {}).get("terminal"),
            "correct": (latest or {}).get("correct"), "false_accepts": (latest or {}).get("false_accepts"),
            "by_defect": (latest or {}).get("by_defect", {})},
    }
    stages = []
    for s in STAGES:
        n = s["step_no"]
        entry = dict(s)
        entry["counts"] = stage_counts[n]
        entry["example"] = _stage_example(stage_events[n], case_ev(per_stage_actions[n][0]))
        entry["ledger_actions"] = [a for a in per_stage_actions[n] if a in actions]
        entry["ledger_events"] = sum(actions.get(a, 0) for a in per_stage_actions[n])
        entry["runs_per_day"] = 1
        stages.append(entry)

    # ---- reply-code catalogue (only what this project defines and uses) ------
    code_catalog = []
    wanted = {"ACC", "DUP", "OCR_LOW_CONFIDENCE", "INCOMPLETE"}
    counts = Counter(cms_codes)
    counts.update(basket_reasons)          # update() sums; `+` would keep the larger side instead
    wanted |= set(policy.get("deniable_codes") or [])
    wanted |= {k for k in counts if k}
    used = set(cms_codes) | set(basket_reasons)
    for code in sorted(wanted):
        if code in ("OCR_LOW_CONFIDENCE", "INCOMPLETE"):
            meaning = {"OCR_LOW_CONFIDENCE": "Intake flagged one or more fields as low-confidence OCR; the work "
                                            "basket re-checks them against the attachments.",
                       "INCOMPLETE": "A required item (or the signature) is missing; the member is asked for it."
                       }[code]
            source = "basket reason (routing only, no CMS round trip)"
        else:
            meaning = REPLY_CODE_TEXT.get(code, code)
            source = "mock CMS reply code"
        code_catalog.append({
            "code": code, "meaning": meaning, "source": source,
            "deniable": code in set(policy.get("deniable_codes") or []),
            "used": code in used, "seen": counts.get(code, 0),
        })

    # ---- cases: a deterministic spread for the page -------------------------
    picked, recent_days = _pick_cases(cases, gold)
    case_rows = []
    for cid in picked:
        c = cases[cid]
        g = gold.get(cid) or {}
        changes = [{"field": x.get("field"), "old": x.get("old"), "new": x.get("new"),
                    "evidence": x.get("evidence"), "event_seq": x.get("event_seq")}
                   for x in (c.get("changes") or []) if isinstance(x, dict)]
        if not changes:
            for e in raw_by_case.get(cid, []):
                if e.get("action") == "fix_applied" and isinstance(e.get("data"), dict):
                    d = e["data"]
                    changes.append({"field": d.get("field"), "old": d.get("old"), "new": d.get("new"),
                                    "evidence": d.get("evidence"), "event_seq": e.get("seq")})
        row = {
            "id": cid, "channel": c.get("channel"), "state": c.get("state"),
            "received": c.get("received_date"), "closed": c.get("disposition_date"),
            "defect": g.get("defect"), "expected": g.get("expected_final"),
            "match": (c.get("state") == g.get("expected_final")) if g.get("expected_final") else None,
            "last_code": c.get("last_code"), "last_message": c.get("last_message"),
            "reason": c.get("basket_reason"), "attempts": c.get("attempts"),
            "flags": c.get("flags") or [], "missing_info": c.get("missing_info") or [],
            "rfi_deadline": c.get("rfi_deadline"),
            "plan": (c.get("fields") or {}).get("plan_name"),
            "plan_id": (c.get("fields") or {}).get("plan_id"),
            "county": (c.get("fields") or {}).get("county"),
            "election_type": (c.get("fields") or {}).get("election_type"),
            "effective_date": (c.get("fields") or {}).get("requested_effective_date"),
            "cms_history": [{"date": h.get("date"), "code": h.get("code"), "message": h.get("message")}
                            for h in (c.get("cms_history") or []) if isinstance(h, dict)],
            "changes": changes,
            "letters": _read_letters(root, c),
            "events": case_events.get(cid, []),
            "event_count": len(raw_by_case.get(cid, [])),
        }
        case_rows.append(row)

    # ---- before/after: what the work basket put right ------------------------
    fixed = [r for r in case_rows if r["changes"]]
    reworked = next((r for r in fixed if len(r["cms_history"]) > 1), fixed[0] if fixed else None)
    before_after = None
    if reworked:
        before_after = {
            "case_id": reworked["id"], "channel": reworked["channel"], "defect": reworked["defect"],
            "state": reworked["state"], "expected": reworked["expected"],
            "changes": reworked["changes"], "cms_history": reworked["cms_history"],
            "letters": [{"kind": l["kind"], "text": l["text"][:MAX_LETTER_CHARS]} for l in reworked["letters"]],
            "events": reworked["events"],
        }

    # ---- compliance and audit surface from the latest reports ---------------
    audit_report = _load(reports / f"{sim_date}_audit.json", {}) if sim_date else {}
    compliance_report = _load(reports / f"{sim_date}_compliance.json", {}) if sim_date else {}
    eval_report = _load(reports / f"{sim_date}_eval.json", {}) if sim_date else {}
    cloud = {"audit": {k: (audit_report or {}).get(k) for k in
                       ("sim_date", "ledger_ok", "ledger_entries", "cases_checked", "s1", "s2")},
             "findings": (audit_report or {}).get("findings", []),
             "advisory_reviews": (audit_report or {}).get("advisory_llm_review", []),
             "compliance": {"counts": (compliance_report or {}).get("counts", {}),
                            "findings": (compliance_report or {}).get("findings", []),
                            "letter_review": (compliance_report or {}).get("letter_review", [])},
             "evaluation": {k: (eval_report or {}).get(k) for k in
                            ("terminal", "open", "correct", "outcome_accuracy", "false_accepts",
                             "false_denials", "missed_accepts", "wrong_fields_on_accept")}}

    return {
        "meta": meta,
        "stages": stages,
        "loops": LOOPS,
        "codes": code_catalog,
        "agents": agents,
        "actions": _counts(actions),
        "tokens_by_action": _counts(tokens_by_action),
        "days": days,
        "cases": case_rows,
        "case_selection": {"rule": "every case received on the two most recent simulated days, capped, plus the "
                                   "oldest still-open items until the cap is reached",
                           "days": recent_days, "limit": MAX_CASES, "total_cases": len(cases),
                           "returned": len(case_rows)},
        "kpis": kpis,
        "before_after": before_after,
        "audit_evidence": cloud,
        "config": {"tat_days": policy.get("tat_days"), "tat_warning_days": policy.get("tat_warning_days"),
                   "rfi_wait_days": policy.get("rfi_wait_days"),
                   "resolver_max_attempts": policy.get("resolver_max_attempts"),
                   "ocr_confidence_threshold": policy.get("ocr_confidence_threshold"),
                   "new_applications_per_day": settings.get("new_applications_per_day"),
                   "paper_share": settings.get("paper_share"),
                   "plans": {k: {"name": v.get("name"), "type": v.get("type"), "counties": v.get("counties")}
                             for k, v in sorted((ref.get("plans") or {}).items())},
                   "counties": sorted((ref.get("county_info") or {}))},
        "disclaimer": DISCLAIMER,
    }


def write_export(root, out):
    """Build the export and write it; returns (path, bytes_on_disk)."""
    data = build_export(root)
    path = Path(out)
    if not path.is_absolute():
        path = Path(root) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    # indent=1 keeps a diff of the file readable and the key order is the dict order above
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1).replace("</", "<\\/") + "\n", encoding="utf-8")
    return path, path.stat().st_size


def main(argv=None):
    ap = argparse.ArgumentParser(prog="enrollsim.workflow_export", description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="project folder (default: current directory)")
    ap.add_argument("--out", default="workflow_export.json", help="output JSON path (default: workflow_export.json)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    path, size = write_export(root, a.out)
    data = _load(path, {}) or {}
    meta, kpis = data.get("meta", {}), data.get("kpis", {})
    print(f"wrote {path} ({size:,} bytes)")
    print(f"  {len(data.get('stages', []))} stages, {len(data.get('cases', []))} cases, "
          f"{kpis.get('letters', 0)} letters, {meta.get('entries', 0)} ledger entries "
          f"({'intact' if meta.get('ledger_ok') else 'BROKEN'}), sim date {meta.get('sim_date')}")
    return 0 if meta.get("ledger_ok") else 2


if __name__ == "__main__":
    sys.exit(main())
