"""The harness: owns state, the audit ledger, the token budget, and runs one simulated day.

Wall-clock is dominated by model calls waiting on the network, and those calls are independent
per case. So the stages below fan the per-case work out over a small thread pool
(`settings.llm_concurrency`, default 8) and apply the results in the original case order. The
ledger, the token budget and the audit gate are all concurrency-safe; the day is still
reproducible case by case, only faster.
"""
import shutil
import time
from datetime import timedelta
from pathlib import Path

from . import synth
from .agents import (OPEN, TERMINAL, AuditorAgent, ComplianceAgent, DigitalIntakeAgent, EnrollmentProcessor,
                     NoticeAgent, PaperIntakeAgent, ResolverAgent)
from .budget import BudgetExceeded, TokenBudget
from .ledger import Ledger
from .llm import LLMError, LLMGateway, MockBackend, OpenAIBackend
from .util import load_json, norm, parallel_map, pdate, save_json

CRITICAL = ["first_name", "last_name", "dob", "sex", "mbi", "part_a_eff", "part_b_eff", "zip", "plan_id",
            "pcp_name", "election_type", "sep_reason", "requested_effective_date"]
DEFAULT_CONCURRENCY = 8


class Harness:
    def __init__(self, root, mode="mock", backend=None, overrides=None):
        self.root = Path(root)
        cfg = self.root / "config"
        self.settings = load_json(cfg / "settings.json")
        self.settings.update(overrides or {})
        self.policy, self.ref = load_json(cfg / "policy.json"), load_json(cfg / "reference.json")
        self.p_state, self.p_ledger = self.root / "state", self.root / "audit" / "ledger.jsonl"
        self.p_inbox, self.p_letters = self.root / "workspace" / "inbox", self.root / "workspace" / "letters"
        self.p_reports, self.p_gold = self.root / "audit" / "reports", self.root / "audit" / "gold" / "gold.json"
        for d in (self.p_state, self.p_inbox, self.p_letters, self.p_reports, self.p_gold.parent):
            d.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.progress = lambda message: None            # the CLI replaces this; tests stay quiet
        # the reserve for compliance/audit can never swallow more than 10% of the cap
        self.reserve = min(int(self.settings["audit_reserve_tokens"]), int(self.settings["daily_token_cap"]) // 10)
        self.ledger = Ledger(self.p_ledger)
        self.clock = load_json(self.p_state / "clock.json", {"sim_date": self.settings["sim_start_date"], "day_index": 0})
        self.cases = load_json(self.p_state / "cases.json", {})
        self.master = load_json(self.p_state / "cms_master.json", {})
        self.gold = load_json(self.p_gold, {})
        self.budget = TokenBudget(self.p_state / "tokens.json", self.settings["daily_token_cap"],
                                  self.settings["budget_day_offset_minutes"])
        self.budget.release_all()                       # drop holds left by a run that was killed
        if backend is None:
            backend = MockBackend(self.settings) if mode == "mock" else OpenAIBackend(self.settings)
        # Concurrency only buys anything when a call waits on the network. A backend that answers
        # instantly (mock mode) is left sequential unless the caller asked for workers explicitly.
        self.workers = max(1, int(self.settings.get("llm_concurrency", DEFAULT_CONCURRENCY)))
        if not (overrides or {}).get("llm_concurrency") and not getattr(backend, "io_bound", True):
            self.workers = 1
        self.gw = LLMGateway(backend, self.ledger, self.budget, self.settings, lambda: self.sim_date)
        self.halted = False
        self.paper, self.digital = PaperIntakeAgent(self), DigitalIntakeAgent(self)
        self.processor, self.resolver = EnrollmentProcessor(self), ResolverAgent(self)
        self.notice, self.compliance, self.auditor = NoticeAgent(self), ComplianceAgent(self), AuditorAgent(self)

    # ---- small helpers -------------------------------------------------
    @property
    def sim_date(self):
        return self.clock["sim_date"]

    def log(self, agent, action, case_id=None, **data):
        return self.ledger.append(sim_date=self.sim_date, agent=agent, action=action, case_id=case_id, data=data)

    def save(self):
        save_json(self.p_state / "clock.json", self.clock)
        save_json(self.p_state / "cases.json", self.cases)
        save_json(self.p_state / "cms_master.json", self.master)
        save_json(self.p_gold, self.gold)

    def to_basket(self, case, reason, **extra):
        case["state"], case["basket_reason"] = "IN_BASKET", reason
        case.update(extra)
        self.log("processor", "to_basket", case["case_id"], reason=reason, **extra)

    def finalize(self, case, state, **extra):
        case["state"], case["disposition_date"], case["basket_reason"] = state, self.sim_date, None
        case.update(extra)
        self.log("processor", "disposition", case["case_id"], state=state, **extra)

    def escalate(self, case, why):
        case["state"] = "ESCALATED"
        self.log("resolver", "escalated", case["case_id"], why=why)

    # ---- stages ----------------------------------------------------------
    def stage_applicants(self, n):
        for it in synth.generate_day(self.sim_date, n, self.settings, self.ref, self.policy):
            cid = it["case_id"]
            self.gold[cid] = it["gold"]                      # hidden answer key, evaluation only
            self.master[it["member"]["mbi"]] = it["member"]
            case = {"case_id": cid, "channel": it["channel"], "received_date": self.sim_date,
                    "source_text": it["source_text"], "state": "AWAITING_INTAKE", "fields": None,
                    "original_fields": None, "flags": [], "basket_reason": None, "attempts": 0,
                    "intake_failures": 0, "cms_history": [], "letters": [], "changes": [],
                    "last_code": None, "disposition_date": None, "rfi_deadline": None}
            if it["channel"] == "digital":
                case["payload"] = it["payload"]
            self.cases[cid] = case
            ext = "txt" if it["channel"] == "paper" else "json"
            (self.p_inbox / f"{cid}.{ext}").write_text(
                it["raw"] if it["channel"] == "paper" else __import__("json").dumps(it["payload"], indent=1),
                encoding="utf-8")
            self.log("applicant_simulator", "submitted", cid, channel=it["channel"])

    def stage_intake(self):
        queue = sorted((c for c in self.cases.values() if c["state"] == "AWAITING_INTAKE"),
                       key=lambda c: (c["received_date"], c["case_id"]))
        for case, ok, value in parallel_map(queue, self._intake_one, self.workers):
            if self.halted:
                break
            if not ok:
                if isinstance(value, BudgetExceeded):
                    self.halted = True
                    break
                raise value
            if value is not None:                     # None = this case failed intake and was logged
                self.processor.route(value)

    def _intake_one(self, case):
        """Parse one application. Runs on a worker thread; only this case is touched."""
        agent = self.paper if case["channel"] == "paper" else self.digital
        try:
            fields, flags, issues = agent.parse(case)
        except BudgetExceeded:
            raise                                     # the caller halts the day
        except LLMError as e:
            case["intake_failures"] += 1
            self.log(agent.name, "llm_failed", case["case_id"], error=str(e)[:200])
            if case["intake_failures"] >= 3:
                self.escalate(case, "intake failed three times")
            return None
        case["fields"], case["original_fields"], case["flags"] = fields, dict(fields), flags
        self.log(agent.name, "intake_parsed", case["case_id"], flagged=flags, issues=issues)
        return case

    def stage_resolve(self):
        queue = sorted((c for c in self.cases.values() if c["state"] == "IN_BASKET"),
                       key=lambda c: (c["received_date"], c["case_id"]))      # oldest first
        selected = queue[: int(self.settings["resolver_daily_capacity"])]
        for _, ok, value in parallel_map(selected, self.resolver.work, self.workers):
            if not ok:
                raise value
            if self.halted:
                break
        return len(queue)

    def stage_rfi_expiry(self):
        today = pdate(self.sim_date)
        for c in sorted(self.cases.values(), key=lambda c: c["case_id"]):
            if c["state"] == "PENDING_MEMBER" and pdate(c["rfi_deadline"]) <= today:
                self.finalize(c, "CLOSED_INCOMPLETE")

    def stage_notices(self):
        need = {"ACCEPTED": "ACCEPT", "DENIED": "DENY", "PENDING_MEMBER": "RFI", "CLOSED_INCOMPLETE": "CLOSED_INCOMPLETE"}
        jobs = []
        for c in sorted(self.cases.values(), key=lambda c: (c["received_date"], c["case_id"])):
            kind = need.get(c["state"])
            if kind and kind not in {l["kind"] for l in c["letters"]}:
                jobs.append((c, kind))
        for _, ok, value in parallel_map(jobs, lambda job: self.notice.issue(job[0], job[1]), self.workers):
            if not ok:
                raise value
            if self.halted:
                break

    # ---- evaluation against the hidden answer key ------------------------
    def evaluate_gold(self):
        s = {"terminal": 0, "open": 0, "correct": 0, "false_accepts": 0, "false_denials": 0,
             "missed_accepts": 0, "wrong_fields_on_accept": 0, "by_defect": {}}
        for cid, c in sorted(self.cases.items()):
            g = self.gold.get(cid)
            if not g:
                continue
            b = s["by_defect"].setdefault(g["defect"], {"terminal": 0, "correct": 0, "open": 0})
            if c["state"] not in TERMINAL:
                s["open"] += 1
                b["open"] += 1
                continue
            s["terminal"] += 1
            b["terminal"] += 1
            exp, ok = g["expected_final"], c["state"] == g["expected_final"]
            if c["state"] == "ACCEPTED" and exp != "ACCEPTED":
                s["false_accepts"] += 1
            if c["state"] == "DENIED" and exp == "ACCEPTED":
                s["false_denials"] += 1
            if exp == "ACCEPTED" and c["state"] != "ACCEPTED":
                s["missed_accepts"] += 1
            if ok and exp == "ACCEPTED":
                if any(norm(c["fields"].get(k)) != norm(g["true_fields"].get(k)) for k in CRITICAL):
                    ok = False
                    s["wrong_fields_on_accept"] += 1
            if ok:
                s["correct"] += 1
                b["correct"] += 1
        s["outcome_accuracy"] = round(s["correct"] / s["terminal"], 4) if s["terminal"] else None
        return s

    # ---- one simulated day -----------------------------------------------
    def _stage(self, label, fn, timings):
        """Run one stage, time it, and report it. A long live run is otherwise a silent terminal."""
        started = time.perf_counter()
        self.progress(f"  {label} ...")
        out = fn()
        timings[label] = round(time.perf_counter() - started, 2)
        self.progress(f"  {label}: {timings[label]:.1f}s")
        return out

    def run_day(self, new_n=None):
        n = int(self.settings["new_applications_per_day"] if new_n is None else new_n)
        self.halted = False
        day = self.sim_date
        timings, started = {}, time.perf_counter()
        self.progress(f"[{day}] {self.mode} mode, {n} new applications, {self.workers} concurrent model calls")
        self.log("harness", "day_start", None, new_applications=n, mode=self.mode, model=self.settings.get("model"),
                 workers=self.workers)
        self._stage("applicants", lambda: self.stage_applicants(n), timings)
        self._stage("intake", self.stage_intake, timings)
        basket_before = self._stage("resolve", self.stage_resolve, timings)
        self._stage("rfi_expiry", self.stage_rfi_expiry, timings)
        self._stage("notices", self.stage_notices, timings)
        comp = self._stage("compliance", self.compliance.sweep, timings)
        audit = self._stage("audit", self.auditor.audit, timings)
        gold = self._stage("evaluate", self.evaluate_gold, timings)
        timings["total"] = round(time.perf_counter() - started, 2)
        states = {}
        for c in self.cases.values():
            states[c["state"]] = states.get(c["state"], 0) + 1
        open_items = sorted((c for c in self.cases.values() if c["state"] in OPEN),
                            key=lambda c: (c["received_date"], c["case_id"]))
        summary = {"sim_date": day, "mode": self.mode, "model": self.settings.get("model"),
                   "new_applications": n, "basket_at_resolve_start": basket_before, "states": states,
                   "tokens_used_today": self.budget.used(), "token_cap": self.budget.cap,
                   "halted_for_budget": self.halted, "compliance": comp["counts"], "audit_s1": audit["s1"],
                   "audit_s2": audit["s2"], "ledger_ok": audit["ledger_ok"], "evaluation": gold,
                   "llm_calls_today": self.budget.calls(), "workers": self.workers, "timings_seconds": timings,
                   "oldest_open": [{"case_id": c["case_id"], "state": c["state"], "received": c["received_date"],
                                    "reason": c["basket_reason"]} for c in open_items[:8]],
                   "gate_ok": bool(audit["ledger_ok"] and audit["s1"] == 0)}
        self.progress(f"[{day}] done in {timings['total']:.1f}s  "
                      f"({self.budget.calls()} model calls today, {self.budget.used():,} tokens)")
        self.log("harness", "day_end", None, gate_ok=summary["gate_ok"], tokens_used_today=summary["tokens_used_today"],
                 seconds=timings["total"], timings=timings)
        save_json(self.p_reports / f"{day}_audit.json", audit)
        save_json(self.p_reports / f"{day}_compliance.json", comp)
        save_json(self.p_reports / f"{day}_eval.json", gold)
        save_json(self.p_reports / f"{day}_summary.json", summary)
        (self.p_reports / f"{day}_summary.md").write_text(render_summary(summary, audit, comp), encoding="utf-8")
        self.clock["sim_date"] = (pdate(day) + timedelta(days=1)).isoformat()
        self.clock["day_index"] += 1
        self.save()
        return summary

    def audit_only(self):
        audit, gold = self.auditor.audit(), self.evaluate_gold()
        return audit, gold

    def reset(self):
        for d in ("state", "workspace", "audit"):
            shutil.rmtree(self.root / d, ignore_errors=True)


def render_summary(s, audit, comp):
    e = s["evaluation"]
    t = s.get("timings_seconds") or {}
    lines = [f"# Day {s['sim_date']} - {s['mode']} mode", "",
             f"- Tokens used today: **{s['tokens_used_today']:,} / {s['token_cap']:,}**"
             + ("  (STOPPED EARLY: cap reached; remaining work carries over)" if s["halted_for_budget"] else ""),
             f"- New applications: {s['new_applications']}   Work basket at start of resolving: {s['basket_at_resolve_start']}",
             f"- Case states: " + ", ".join(f"{k} {v}" for k, v in sorted(s["states"].items())),
             f"- **Audit gate: {'PASS' if s['gate_ok'] else 'FAIL'}**  (S1 {s['audit_s1']}, S2 {s['audit_s2']}, "
             f"ledger {'intact' if s['ledger_ok'] else 'BROKEN'})",
             f"- Compliance: " + (", ".join(f"{k} {v}" for k, v in sorted(s["compliance"].items())) or "no findings"), "",
             "## Where the time went",
             f"- Wall clock for the day: **{t.get('total', '?')}s** across {s.get('llm_calls_today', '?')} model calls "
             f"({s.get('workers', '?')} concurrent)",
             "- " + "   ".join(f"{k} {v}s" for k, v in t.items() if k != "total"), "",
             "## Outcome accuracy against the hidden answer key",
             f"- Terminal cases: {e['terminal']}   Still open: {e['open']}   Correct: {e['correct']}"
             f"   Accuracy: {e['outcome_accuracy']}",
             f"- False accepts: **{e['false_accepts']}**   False denials: {e['false_denials']}   "
             f"Missed accepts: {e['missed_accepts']}   Wrong fields on accepted: {e['wrong_fields_on_accept']}", "",
             "| Defect | Terminal | Correct | Open |", "|---|---|---|---|"]
    for k, v in sorted(e["by_defect"].items()):
        lines.append(f"| {k} | {v['terminal']} | {v['correct']} | {v['open']} |")
    if s["mode"] == "mock":
        lines += ["", "_Mock mode: the 'agents' are deterministic parsers. This proves the plumbing, guards and audit,"
                      " not model quality._"]
    if s["oldest_open"]:
        lines += ["", "## Oldest open items", "| Case | State | Received | Reason |", "|---|---|---|---|"]
        lines += [f"| {o['case_id']} | {o['state']} | {o['received']} | {o['reason'] or ''} |" for o in s["oldest_open"]]
    if audit["findings"]:
        lines += ["", "## Audit findings"] + [f"- **{f['severity']} {f['code']}** {f['case_id'] or ''}: {f['text']}"
                                               for f in audit["findings"][:30]]
    return "\n".join(lines) + "\n"
