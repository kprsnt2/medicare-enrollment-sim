"""The harness: owns state, the audit ledger, the token budget, and runs one simulated day."""
import shutil
from datetime import timedelta
from pathlib import Path

from . import synth
from .agents import (OPEN, TERMINAL, AuditorAgent, ComplianceAgent, DigitalIntakeAgent, EnrollmentProcessor,
                     NoticeAgent, PaperIntakeAgent, ResolverAgent)
from .budget import BudgetExceeded, TokenBudget
from .ledger import Ledger
from .llm import LLMError, LLMGateway, MockBackend, OpenAIBackend
from .util import load_json, norm, pdate, save_json

CRITICAL = ["first_name", "last_name", "dob", "sex", "mbi", "part_a_eff", "part_b_eff", "zip", "plan_id",
            "pcp_name", "election_type", "sep_reason", "requested_effective_date"]


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
        # the reserve for compliance/audit can never swallow more than 10% of the cap
        self.reserve = min(int(self.settings["audit_reserve_tokens"]), int(self.settings["daily_token_cap"]) // 10)
        self.ledger = Ledger(self.p_ledger)
        self.clock = load_json(self.p_state / "clock.json", {"sim_date": self.settings["sim_start_date"], "day_index": 0})
        self.cases = load_json(self.p_state / "cases.json", {})
        self.master = load_json(self.p_state / "cms_master.json", {})
        self.gold = load_json(self.p_gold, {})
        self.budget = TokenBudget(self.p_state / "tokens.json", self.settings["daily_token_cap"],
                                  self.settings["budget_day_offset_minutes"])
        if backend is None:
            backend = MockBackend(self.settings) if mode == "mock" else OpenAIBackend(self.settings)
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
        for c in queue:
            if self.halted:
                break
            agent = self.paper if c["channel"] == "paper" else self.digital
            try:
                fields, flags, issues = agent.parse(c)
            except BudgetExceeded:
                self.halted = True
                break
            except LLMError as e:
                c["intake_failures"] += 1
                self.log(agent.name, "llm_failed", c["case_id"], error=str(e)[:200])
                if c["intake_failures"] >= 3:
                    self.escalate(c, "intake failed three times")
                continue
            c["fields"], c["original_fields"], c["flags"] = fields, dict(fields), flags
            self.log(agent.name, "intake_parsed", c["case_id"], flagged=flags, issues=issues)
            self.processor.route(c)

    def stage_resolve(self):
        queue = sorted((c for c in self.cases.values() if c["state"] == "IN_BASKET"),
                       key=lambda c: (c["received_date"], c["case_id"]))      # oldest first
        for c in queue[: int(self.settings["resolver_daily_capacity"])]:
            if self.halted:
                break
            self.resolver.work(c)
        return len(queue)

    def stage_rfi_expiry(self):
        today = pdate(self.sim_date)
        for c in sorted(self.cases.values(), key=lambda c: c["case_id"]):
            if c["state"] == "PENDING_MEMBER" and pdate(c["rfi_deadline"]) <= today:
                self.finalize(c, "CLOSED_INCOMPLETE")

    def stage_notices(self):
        need = {"ACCEPTED": "ACCEPT", "DENIED": "DENY", "PENDING_MEMBER": "RFI", "CLOSED_INCOMPLETE": "CLOSED_INCOMPLETE"}
        for c in sorted(self.cases.values(), key=lambda c: (c["received_date"], c["case_id"])):
            kind = need.get(c["state"])
            if kind and kind not in {l["kind"] for l in c["letters"]}:
                self.notice.issue(c, kind)

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
    def run_day(self, new_n=None):
        n = int(self.settings["new_applications_per_day"] if new_n is None else new_n)
        self.halted = False
        day = self.sim_date
        self.log("harness", "day_start", None, new_applications=n, mode=self.mode, model=self.settings.get("model"))
        self.stage_applicants(n)
        self.stage_intake()
        basket_before = self.stage_resolve()
        self.stage_rfi_expiry()
        self.stage_notices()
        comp = self.compliance.sweep()
        audit = self.auditor.audit()
        gold = self.evaluate_gold()
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
                   "oldest_open": [{"case_id": c["case_id"], "state": c["state"], "received": c["received_date"],
                                    "reason": c["basket_reason"]} for c in open_items[:8]],
                   "gate_ok": bool(audit["ledger_ok"] and audit["s1"] == 0)}
        self.log("harness", "day_end", None, gate_ok=summary["gate_ok"], tokens_used_today=summary["tokens_used_today"])
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
    lines = [f"# Day {s['sim_date']} - {s['mode']} mode", "",
             f"- Tokens used today: **{s['tokens_used_today']:,} / {s['token_cap']:,}**"
             + ("  (STOPPED EARLY: cap reached; remaining work carries over)" if s["halted_for_budget"] else ""),
             f"- New applications: {s['new_applications']}   Work basket at start of resolving: {s['basket_at_resolve_start']}",
             f"- Case states: " + ", ".join(f"{k} {v}" for k, v in sorted(s["states"].items())),
             f"- **Audit gate: {'PASS' if s['gate_ok'] else 'FAIL'}**  (S1 {s['audit_s1']}, S2 {s['audit_s2']}, "
             f"ledger {'intact' if s['ledger_ok'] else 'BROKEN'})",
             f"- Compliance: " + (", ".join(f"{k} {v}" for k, v in sorted(s["compliance"].items())) or "no findings"), "",
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
