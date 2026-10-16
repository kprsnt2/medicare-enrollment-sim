"""The agent roles. Each agent decides; the harness (pipeline.Harness) records and applies.

Agents get data in and return data out. They hold no file handles, no ledger writer and no
API client: everything goes through `h.gw` (budget + schema + ledger) and `h.log`.
"""
from datetime import datetime, timedelta

from . import mock_brain
from .budget import BudgetExceeded
from .ledger import Ledger
from .llm import LLMError
from .rules import completeness_issues, denial_check, evaluate, validate_change
from .synth import FIELDS
from .util import contains_value, derive_effective_date, norm, pdate, sha

TERMINAL = {"ACCEPTED", "DENIED", "CLOSED_DUPLICATE", "CLOSED_INCOMPLETE"}
OPEN = {"AWAITING_INTAKE", "IN_BASKET", "PENDING_MEMBER", "ESCALATED"}
DENY_TEXT = {
    "REJ-AREA": "the county where you live is outside the service area of the plan you chose",
    "REJ-ELECTION": "the date of your request is not within an enrollment period that applies to you",
}


def member_name(f):
    return f"{f.get('first_name', '').title()} {f.get('last_name', '').title()}".strip()


class PaperIntakeAgent:
    name = "paper_intake"

    def __init__(self, h):
        self.h = h

    def parse(self, case):
        h = self.h
        out = h.gw.call(self.name, "paper_intake", {"raw_text": case["source_text"], "field_list": FIELDS},
                        case_id=case["case_id"], keep=h.reserve)
        src, thr = norm(case["source_text"]), h.policy["ocr_confidence_threshold"]
        exempt = set(h.policy["low_confidence_exempt"])
        fields, flags, issues = {}, [], []
        for k in FIELDS:
            v = str(out["fields"].get(k) or "").strip()
            conf = float(out["confidence"].get(k, 0.0))
            quote = out["evidence"].get(k)
            if v:
                if not quote or norm(quote) not in src:       # anti-fabrication: must be quotable
                    conf = min(conf, 0.3)
                    issues.append(f"{k}: evidence quote not found in the scan text")
                elif not contains_value(quote, v):             # corrected an OCR confusion
                    conf = min(conf, 0.6)
            fields[k] = v
            if v and conf < thr and k not in exempt:
                flags.append(k)
        return fields, flags, issues


class DigitalIntakeAgent:
    name = "digital_intake"

    def __init__(self, h):
        self.h = h

    def parse(self, case):
        form = case["payload"]["form"]
        return {k: str(form.get(k, "")).strip() for k in FIELDS}, [], []


class EnrollmentProcessor:
    """The 'system': completeness checks, submission to the mock CMS, routing of the reply."""
    name = "processor"

    def __init__(self, h):
        self.h = h

    def route(self, case):
        h = self.h
        missing = completeness_issues(case["fields"], h.ref, h.policy)
        if missing:
            h.to_basket(case, "INCOMPLETE", missing_items=missing)
        elif case["flags"]:
            h.to_basket(case, "OCR_LOW_CONFIDENCE")
        else:
            self.submit(case)

    def submit(self, case):
        h, f = self.h, case["fields"]
        res = evaluate(f, h.master.get(f.get("mbi")), h.ref, h.policy)
        case["cms_history"].append({"date": h.sim_date, **res})
        case["last_code"], case["last_message"] = res["code"], res["message"]
        h.log(self.name, "cms_response", case["case_id"], code=res["code"], message=res["message"],
              fields_sha=sha(f))
        if res["code"] == "ACC":
            h.finalize(case, "ACCEPTED")
        elif res["code"] == "DUP":
            h.finalize(case, "CLOSED_DUPLICATE")
        else:
            h.to_basket(case, res["code"])
        return res


class ResolverAgent:
    """The work-basket employee: oldest item first, evidence-backed fixes only."""
    name = "resolver"

    def __init__(self, h):
        self.h = h

    def _payload(self, case):
        h, f = self.h, case["fields"]
        age = (pdate(h.sim_date) - pdate(case["received_date"])).days
        return {
            "case_id": case["case_id"], "age_days": age, "tat_days": h.policy["tat_days"],
            "basket_reason": case["basket_reason"], "last_code": case.get("last_code"),
            "last_message": case.get("last_message"), "cms_history": case["cms_history"][-3:],
            "flagged_fields": case["flags"], "missing_items": case.get("missing_items", []),
            "fields": f, "source_text": case["source_text"],
            "reference": {"plans": h.ref["plans"], "zip_to_county": h.ref["zip_to_county"]},
            "policy": {k: h.policy[k] for k in ("election_windows", "iep_window_months", "sep_reasons",
                                                "effective_date_rules", "deniable_codes", "fixable_fields")},
            "derived_effective_dates": {t: derive_effective_date(t, f.get("election_date", ""), h.policy)
                                        for t in ("AEP", "OEP", "IEP", "SEP")},
        }

    def work(self, case):
        h, cid = self.h, case["case_id"]
        limit = h.policy["resolver_max_attempts"]
        while case["state"] == "IN_BASKET" and case["attempts"] < limit and not h.halted:
            case["attempts"] += 1
            try:
                out = h.gw.call(self.name, "resolver", self._payload(case), case_id=cid, keep=h.reserve)
            except BudgetExceeded:
                h.halted = True
                case["attempts"] -= 1
                return
            except LLMError as e:
                h.log(self.name, "llm_failed", cid, error=str(e)[:200])
                continue
            self._apply(case, out)
        if case["state"] == "IN_BASKET" and case["attempts"] >= limit:
            h.escalate(case, "attempt limit reached without a valid resolution")

    def _apply(self, case, out):
        h, cid, act = self.h, case["case_id"], out["action"]
        h.log(self.name, "decision", cid, chosen=act, diagnosis=out["diagnosis"][:300],
              confidence=out["confidence"], n_changes=len(out["changes"]))
        if act in ("FIX", "CONFIRM"):
            if act == "CONFIRM" and (case["basket_reason"] != "OCR_LOW_CONFIDENCE" or out["changes"]):
                return h.log(self.name, "action_rejected", cid, why="CONFIRM is only for OCR checks with no changes")
            if act == "FIX" and not out["changes"]:
                return h.log(self.name, "action_rejected", cid, why="FIX with no changes")
            after = dict(case["fields"])
            for c in out["changes"]:
                after[c["field"]] = c["new_value"]
            errs = []
            for c in out["changes"]:
                ok, why = validate_change(c["field"], c["new_value"], c["evidence_quote"], after,
                                          case["source_text"], h.ref, h.policy)
                if not ok:
                    errs.append(f"{c['field']}: {why}")
            if errs:
                return h.log(self.name, "fix_rejected", cid, errors=errs, proposed=out["changes"])
            for c in out["changes"]:
                old = case["fields"].get(c["field"], "")
                if old == c["new_value"]:
                    continue
                e = h.log(self.name, "fix_applied", cid, field=c["field"], old=old, new=c["new_value"],
                          evidence=c["evidence_quote"])
                case["fields"][c["field"]] = c["new_value"]
                case["changes"].append({"field": c["field"], "old": old, "new": c["new_value"],
                                        "evidence": c["evidence_quote"], "event_seq": e["seq"]})
            case["flags"] = []
            h.processor.submit(case)
        elif act == "REQUEST_INFO":
            case["state"] = "PENDING_MEMBER"
            case["missing_info"] = out["missing_info"] or case.get("missing_items") or ["more information"]
            case["rfi_deadline"] = (pdate(h.sim_date) + timedelta(days=h.policy["rfi_wait_days"])).isoformat()
            h.log(self.name, "rfi_requested", cid, missing=case["missing_info"], deadline=case["rfi_deadline"])
        elif act == "DENY":
            if case.get("last_code") in h.policy["deniable_codes"] and case["basket_reason"] == case["last_code"]:
                ok, why = denial_check(case["last_code"], case["fields"], case["source_text"], h.ref, h.policy)
                if ok:
                    h.finalize(case, "DENIED", deny_code=case["last_code"])
                else:
                    h.log(self.name, "denial_blocked", cid, why=why)
                    h.escalate(case, "denial blocked, needs human review: " + why)
            else:
                h.log(self.name, "action_rejected", cid, why="DENY is not allowed for this reject code")
        else:
            h.escalate(case, out["reason"][:200])


class NoticeAgent:
    name = "notice"

    def __init__(self, h):
        self.h = h

    def payload(self, case, kind):
        h, f = self.h, case["fields"]
        plan = h.ref["plans"].get(f.get("plan_id", ""), {})
        return {"kind": kind, "member_name": member_name(f), "received_date": case["received_date"],
                "plan_name": plan.get("name", f.get("plan_name", "")),
                "effective_date": f.get("requested_effective_date", ""),
                "reason_text": DENY_TEXT.get(case.get("deny_code"), "we could not meet the enrollment requirements"),
                "missing_items": case.get("missing_info", []), "respond_by": case.get("rfi_deadline", ""),
                "contact_placeholder": h.policy["contact_placeholder"],
                "required_phrases": h.policy["letter_required_phrases"][kind]}

    def issue(self, case, kind):
        h, cid = self.h, case["case_id"]
        payload, result, source, feedback = self.payload(case, kind), None, "template", None
        if not h.halted:
            for _ in range(2):
                try:
                    out = h.gw.call(self.name, "notice", dict(payload, compliance_feedback=feedback) if feedback else payload,
                                    case_id=cid, keep=h.reserve)
                except BudgetExceeded:
                    h.halted = True
                    break
                except LLMError as e:
                    h.log(self.name, "llm_failed", cid, error=str(e)[:200])
                    break
                issues = h.compliance.check_letter(kind, out["subject"], out["body"], payload["member_name"])
                if not issues:
                    result, source = out, h.gw.backend.name
                    break
                feedback = issues
                h.log("compliance", "letter_rejected", cid, kind=kind, issues=issues)
        if result is None:
            result = mock_brain.notice(payload)
        path = h.p_letters / f"{cid}_{kind}.txt"
        text = f"Subject: {result['subject']}\n\n{result['body']}\n"
        path.write_text(text, encoding="utf-8")
        case["letters"].append({"kind": kind, "file": str(path.relative_to(h.root)), "sha": sha(text),
                                "source": source, "date": h.sim_date})
        h.log(self.name, "letter_issued", cid, kind=kind, source=source, file=case["letters"][-1]["file"])


class ComplianceAgent:
    """Regulatory checks. Deterministic where the rule is exact; one small advisory LLM review."""
    name = "compliance"
    KIND_FOR_STATE = {"ACCEPTED": "ACCEPT", "DENIED": "DENY", "PENDING_MEMBER": "RFI",
                      "CLOSED_INCOMPLETE": "CLOSED_INCOMPLETE"}

    def __init__(self, h):
        self.h = h

    def check_letter(self, kind, subject, body, name):
        text = norm(subject + " " + body)
        issues = [f"missing required phrase: '{p}'" for p in self.h.policy["letter_required_phrases"][kind]
                  if norm(p) not in text]
        if norm(name) and norm(name) not in text:
            issues.append("member name is missing")
        if "{{" in body or "}}" in body:
            issues.append("unfilled template placeholder")
        return issues

    def sweep(self):
        h = self.h
        today, tat, warn = pdate(h.sim_date), h.policy["tat_days"], h.policy["tat_warning_days"]
        findings = []
        for cid, c in sorted(h.cases.items()):
            age = (today - pdate(c["received_date"])).days
            if c["state"] in TERMINAL:
                took = (pdate(c["disposition_date"]) - pdate(c["received_date"])).days
                if took > tat:
                    findings.append({"case_id": cid, "code": "TAT_BREACH", "detail": f"{took} days"})
                kind = self.KIND_FOR_STATE.get(c["state"])
                if kind and kind not in {l["kind"] for l in c["letters"]}:
                    findings.append({"case_id": cid, "code": "LETTER_MISSING", "detail": kind})
            elif c["state"] in OPEN:
                if age > tat:
                    findings.append({"case_id": cid, "code": "OPEN_OVER_TAT", "detail": f"{age} days"})
                elif age >= warn:
                    findings.append({"case_id": cid, "code": "TAT_AT_RISK", "detail": f"{age} days"})
            for l in c["letters"]:
                p = h.root / l["file"]
                if not p.exists() or sha(p.read_text(encoding="utf-8")) != l["sha"]:
                    findings.append({"case_id": cid, "code": "LETTER_ALTERED_OR_MISSING", "detail": l["file"]})
        review = self._letter_review()
        counts = {}
        for f in findings:
            counts[f["code"]] = counts.get(f["code"], 0) + 1
        h.log(self.name, "sweep", None, counts=counts, findings=findings[:60], letter_review=review)
        return {"counts": counts, "findings": findings, "letter_review": review}

    def _letter_review(self):
        h, out = self.h, []
        todays = [(c, l) for c in h.cases.values() for l in c["letters"] if l["date"] == h.sim_date]
        for c, l in todays[: int(h.settings.get("letter_review_sample", 0))]:
            if h.halted:
                break
            text = (h.root / l["file"]).read_text(encoding="utf-8")
            try:
                r = h.gw.call(self.name, "letter_review", {"kind": l["kind"], "letter": text,
                              "facts": {"state": c["state"], "received": c["received_date"]}}, case_id=c["case_id"])
            except BudgetExceeded:
                h.halted = True
                break
            except LLMError as e:
                h.log(self.name, "llm_failed", c["case_id"], error=str(e)[:200])
                continue
            out.append({"case_id": c["case_id"], "pass": r["pass"], "issues": r["issues"]})
        return out


class AuditorAgent:
    """Independent audit. Reads the ledger and the stored case files, never the agents' own summaries.

    Deterministic invariants decide the gate (S1 = zero tolerance). An LLM reviews a sample of
    cases but is advisory only: an LLM grading an LLM tends to rubber-stamp.
    """
    name = "auditor"

    def __init__(self, h):
        self.h = h

    def audit(self):
        h = self.h
        findings = []

        def add(sev, code, cid, text):
            findings.append({"severity": sev, "code": code, "case_id": cid, "text": text})

        ok, n, msg = Ledger.verify(h.p_ledger)
        if not ok:
            add("S1", "LEDGER_CHAIN_BROKEN", None, msg)
        events = h.ledger.read_all()
        by_seq = {e["seq"]: e for e in events}
        by_case = {}
        for e in events:
            by_case.setdefault(e["case_id"], []).append(e)
            if e["agent"] not in h.policy["allowed_agents"]:
                add("S2", "UNKNOWN_AGENT", e["case_id"], f"event {e['seq']} by '{e['agent']}'")

        for cid, c in sorted(h.cases.items()):
            evs = by_case.get(cid, [])
            replay = dict(c.get("original_fields") or {})
            for ch in c["changes"]:
                replay[ch["field"]] = ch["new"]
                if ch["field"] != "requested_effective_date" and norm(ch["evidence"]) not in norm(c["source_text"]):
                    add("S1", "FIX_EVIDENCE_MISSING", cid, f"{ch['field']}: quote not in the source document")
                e = by_seq.get(ch["event_seq"])
                if (not e or e["action"] != "fix_applied" or e["case_id"] != cid
                        or e["data"].get("field") != ch["field"] or e["data"].get("new") != ch["new"]):
                    add("S1", "FIX_NOT_LOGGED", cid, f"change to {ch['field']} has no matching ledger event")
            if c.get("fields") is not None and replay != c["fields"]:
                add("S1", "FIELDS_CHANGED_WITHOUT_LOG", cid, "stored fields differ from original + logged fixes")
            st = c["state"]
            if st in TERMINAL and c.get("fields"):
                member = h.master.get(c["fields"].get("mbi"))
                if st == "ACCEPTED":
                    r = evaluate(c["fields"], member, h.ref, h.policy, skip_duplicate=True)
                    if r["code"] != "ACC":
                        add("S1", "ACCEPTED_BUT_NOT_ELIGIBLE", cid, f"re-check returns {r['code']}: {r['message']}")
                elif st == "DENIED":
                    r = evaluate(c["fields"], member, h.ref, h.policy)
                    if c.get("deny_code") not in h.policy["deniable_codes"] or r["code"] != c.get("deny_code"):
                        add("S1", "DENIAL_NOT_SUPPORTED", cid, f"re-check returns {r['code']}, denial was {c.get('deny_code')}")
                    else:
                        ok_d, why_d = denial_check(c["deny_code"], c["fields"], c["source_text"], h.ref, h.policy)
                        if not ok_d:
                            add("S1", "DENIAL_GATE_FAILED", cid, why_d)
                elif st == "CLOSED_DUPLICATE":
                    if evaluate(c["fields"], member, h.ref, h.policy)["code"] != "DUP":
                        add("S1", "DUPLICATE_NOT_SUPPORTED", cid, "re-check does not return DUP")
                elif st == "CLOSED_INCOMPLETE" and not any(e["action"] == "rfi_requested" for e in evs):
                    add("S2", "CLOSED_WITHOUT_RFI", cid, "closed as incomplete but no information request was logged")
                if not any(e["action"] == "disposition" for e in evs):
                    add("S2", "NO_DISPOSITION_EVENT", cid, "terminal state without a logged disposition")

        day = h.budget.day()
        tokens_today = 0
        for e in events:
            ts = datetime.fromisoformat(e["ts"])
            if (ts + h.budget.offset).date().isoformat() == day:
                tokens_today += e["tokens"]
        used = h.budget.used()
        if used > h.budget.cap:
            add("S1", "TOKEN_CAP_EXCEEDED", None, f"used {used:,} > cap {h.budget.cap:,}")
        if tokens_today != used:
            add("S2", "TOKEN_ACCOUNTING_MISMATCH", None, f"ledger {tokens_today:,} vs budget file {used:,}")

        advisory = self._llm_sample(by_case, by_seq)
        s1 = sum(1 for f in findings if f["severity"] == "S1")
        report = {"sim_date": h.sim_date, "ledger_ok": ok, "ledger_entries": n, "cases_checked": len(h.cases),
                  "s1": s1, "s2": sum(1 for f in findings if f["severity"] == "S2"),
                  "findings": findings, "advisory_llm_review": advisory}
        h.log(self.name, "audit_report", None, s1=s1, s2=report["s2"], cases=len(h.cases), ledger_ok=ok)
        return report

    def _llm_sample(self, by_case, by_seq):
        h, out = self.h, []
        touched = [c for c in h.cases.values()
                   if any(e["sim_date"] == h.sim_date for e in by_case.get(c["case_id"], []))
                   and (c["changes"] or c["state"] == "DENIED")]
        touched.sort(key=lambda c: (-len(c["changes"]), c["case_id"]))
        for c in touched[: int(h.settings.get("audit_llm_sample", 0))]:
            if h.halted:
                break
            tl = [{"seq": e["seq"], "agent": e["agent"], "action": e["action"], "data": e["data"]}
                  for e in by_case.get(c["case_id"], []) if not e["action"].startswith("llm")][:40]
            payload = {"case_id": c["case_id"], "final_state": c["state"], "original_fields": c.get("original_fields"),
                       "final_fields": c.get("fields"), "timeline": tl, "source_text": c["source_text"][:2500]}
            try:
                r = h.gw.call(self.name, "auditor_review", payload, case_id=c["case_id"])
            except BudgetExceeded:
                h.halted = True
                break
            except LLMError as e:
                h.log(self.name, "llm_failed", c["case_id"], error=str(e)[:200])
                continue
            valid = [f for f in r["findings"] if f["event_seqs"] and all(int(s) in by_seq for s in f["event_seqs"])]
            out.append({"case_id": c["case_id"], "verdict": r["verdict"], "findings": valid,
                        "dropped_unreferenced": len(r["findings"]) - len(valid)})
        return out
