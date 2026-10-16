"""Offline tests. No network, no API key, no cost.

The important ones are adversarial: they plant a lying agent, tamper with stored state and the
ledger, exhaust the token cap, and run the real OpenAI client against a local fake server.
"""
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["ENROLLSIM_FAST_RETRY"] = "1"

from enrollsim.budget import BudgetExceeded, TokenBudget          # noqa: E402
from enrollsim.ledger import Ledger                               # noqa: E402
from enrollsim.llm import LLMError, MockBackend                   # noqa: E402
from enrollsim.pipeline import Harness                            # noqa: E402
from enrollsim.rules import validate_change                       # noqa: E402
from enrollsim.util import load_json                              # noqa: E402


def make_root():
    d = Path(tempfile.mkdtemp(prefix="enrollsim_test_"))
    shutil.copytree(ROOT / "config", d / "config")
    return d


class Base(unittest.TestCase):
    def setUp(self):
        self.root = make_root()
        self.addCleanup(shutil.rmtree, self.root, True)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.pol = load_json(ROOT / "config" / "policy.json")
        self.ref = load_json(ROOT / "config" / "reference.json")
        self.src = ("Name: SMITH, JOHN\nPlan selected: Sunrise Care HMO (SYN-H0001-0X1)\n"
                    "Notes: My regular doctor is Dr. Alan Rao at the clinic.\n"
                    "Attachment A (Medicare card copy): SMITH JOHN / DOB 1948-03-12 / SEX M")
        self.after = {"plan_id": "SYN-H0001-001", "election_type": "AEP", "election_date": "2026-10-15"}

    def v(self, field, new, quote, after=None):
        return validate_change(field, new, quote, after or self.after, self.src, self.ref, self.pol)[0]

    def test_accepts_quoted_value(self):
        self.assertTrue(self.v("dob", "1948-03-12", "DOB 1948-03-12"))

    def test_rejects_quote_not_in_source(self):
        self.assertFalse(self.v("dob", "1950-01-01", "DOB 1950-01-01"))

    def test_rejects_value_not_inside_quote(self):
        self.assertFalse(self.v("dob", "1950-01-01", "DOB 1948-03-12"))

    def test_rejects_field_agents_may_not_change(self):
        self.assertFalse(self.v("signature", "Y", "Name: SMITH, JOHN"))
        self.assertFalse(self.v("election_date", "2026-10-01", "Name: SMITH, JOHN"))

    def test_effective_date_must_match_policy(self):
        self.assertTrue(self.v("requested_effective_date", "2027-01-01", ""))
        self.assertFalse(self.v("requested_effective_date", "2026-11-01", ""))

    def test_plan_must_exist_and_be_named_in_source(self):
        self.assertTrue(self.v("plan_id", "SYN-H0001-001", "Sunrise Care HMO"))
        self.assertFalse(self.v("plan_id", "SYN-H0003-001", "Sunrise Care HMO"))   # name not in source
        self.assertFalse(self.v("plan_id", "SYN-NOPE", "Sunrise Care HMO"))

    def test_pcp_must_be_in_network(self):
        self.assertTrue(self.v("pcp_name", "Dr. Alan Rao", "Dr. Alan Rao"))
        self.assertFalse(self.v("pcp_name", "Dr. Nobody", "Dr. Alan Rao"))

    def test_sep_reason_must_be_on_list(self):
        self.assertTrue(self.v("sep_reason", "LOST_EMPLOYER_COVERAGE", "Name: SMITH, JOHN"))
        self.assertFalse(self.v("sep_reason", "I WANT IT", "Name: SMITH, JOHN"))


class LedgerTests(Base):
    def run_days(self, n=1, **kw):
        h = Harness(self.root, mode="mock", overrides=kw.get("overrides"))
        for _ in range(n):
            h.run_day(kw.get("new", 12))
        return h

    def test_detects_modified_entry(self):
        self.run_days()
        path = self.root / "audit" / "ledger.jsonl"
        self.assertTrue(Ledger.verify(path)[0])
        lines = path.read_text().splitlines()
        lines[5] = lines[5].replace('"agent":', '"agent ":', 1) if '"agent":' in lines[5] else lines[5] + " "
        rec = json.loads(lines[5].strip()) if lines[5].strip().startswith("{") else {}
        rec["tokens"] = 999999                       # change a value without fixing the hash
        lines[5] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n")
        ok, _, msg = Ledger.verify(path)
        self.assertFalse(ok, msg)

    def test_detects_deleted_entry(self):
        self.run_days()
        path = self.root / "audit" / "ledger.jsonl"
        lines = path.read_text().splitlines()
        del lines[10]
        path.write_text("\n".join(lines) + "\n")
        self.assertFalse(Ledger.verify(path)[0])

    def test_chain_continues_across_runs(self):
        self.run_days(1)
        self.run_days(1)                              # a new Harness continues the same chain
        ok, n, msg = Ledger.verify(self.root / "audit" / "ledger.jsonl")
        self.assertTrue(ok, msg)
        self.assertGreater(n, 100)


class PipelineTests(Base):
    def test_mock_end_to_end_is_clean(self):
        h = Harness(self.root, mode="mock")
        for _ in range(3):
            s = h.run_day(25)
        self.assertTrue(s["gate_ok"])
        self.assertEqual(s["audit_s1"], 0)
        self.assertEqual(s["evaluation"]["false_accepts"], 0)
        # every terminal case has the letter its outcome requires
        need = {"ACCEPTED": "ACCEPT", "DENIED": "DENY", "CLOSED_INCOMPLETE": "CLOSED_INCOMPLETE"}
        for c in h.cases.values():
            if c["state"] in need:
                self.assertIn(need[c["state"]], {l["kind"] for l in c["letters"]}, c["case_id"])

    def test_oldest_first_and_capacity(self):
        h = Harness(self.root, mode="mock", overrides={"resolver_daily_capacity": 3})
        for _ in range(3):
            h.run_day(30)
        basket = sorted((c for c in h.cases.values() if c["state"] == "IN_BASKET"),
                        key=lambda c: (c["received_date"], c["case_id"]))
        self.assertGreater(len(basket), 3, "low capacity should leave a backlog")
        worked = [c for c in h.cases.values() if c["attempts"] > 0]
        oldest_open = basket[0]["received_date"]
        # nothing older than the oldest remaining item is still sitting untouched in the basket
        self.assertTrue(all(c["received_date"] >= oldest_open for c in basket))
        self.assertTrue(worked)

    def test_election_windows_and_rollover(self):
        for start in ("2026-12-05", "2026-12-30", "2027-03-29"):
            root = make_root()
            self.addCleanup(shutil.rmtree, root, True)
            h = Harness(root, mode="mock", overrides={"sim_start_date": start})
            for _ in range(5):
                s = h.run_day(40)
            self.assertTrue(s["gate_ok"], start)
            self.assertEqual(s["evaluation"]["outcome_accuracy"], 1.0, start)

    def test_gold_never_reaches_agents_or_ledger(self):
        h = Harness(self.root, mode="mock")
        h.run_day(15)
        blob = (self.root / "audit" / "ledger.jsonl").read_text()
        for secret in ("expected_final", "true_fields", '"defect"'):
            self.assertNotIn(secret, blob)
        c = next(c for c in h.cases.values() if c["state"] in ("ACCEPTED", "DENIED"))
        payload = json.dumps(h.resolver._payload(c))
        for secret in ("expected_final", "true_fields", "demo_mismatch", "gold"):
            self.assertNotIn(secret, payload)


class AdversarialTests(Base):
    def test_lying_resolver_cannot_force_an_accept(self):
        class Fabricator(MockBackend):
            def complete(self, task, system, payload, max_tokens):
                if task == "resolver":
                    out = {"diagnosis": "trust me", "action": "FIX", "reason": "x", "confidence": 0.99,
                           "missing_info": [], "changes": [{"field": "last_name", "new_value": "Fabricated",
                                                            "evidence_quote": "Fabricated"}]}
                    return json.dumps(out), 0, "fabricator"
                return super().complete(task, system, payload, max_tokens)

        h = Harness(self.root, mode="mock", backend=Fabricator({}))
        s = h.run_day(30)
        events = h.ledger.read_all()
        self.assertGreater(sum(1 for e in events if e["action"] == "fix_rejected"), 0)
        self.assertEqual(sum(1 for e in events if e["action"] == "fix_applied"), 0)
        self.assertEqual(s["evaluation"]["false_accepts"], 0)
        self.assertEqual(s["audit_s1"], 0)
        self.assertTrue(any(c["state"] == "ESCALATED" for c in h.cases.values()))

    def test_lying_resolver_cannot_deny_a_fixable_case(self):
        class Denier(MockBackend):
            def complete(self, task, system, payload, max_tokens):
                if task == "resolver":
                    out = {"diagnosis": "deny all", "action": "DENY", "reason": "x", "confidence": 0.9,
                           "missing_info": [], "changes": []}
                    return json.dumps(out), 0, "denier"
                return super().complete(task, system, payload, max_tokens)

        h = Harness(self.root, mode="mock", backend=Denier({}))
        for _ in range(2):
            s = h.run_day(60)
        events = h.ledger.read_all()
        # DENY is only honoured for the deniable codes AND when the second gate finds no special-election cue
        self.assertGreaterEqual(sum(1 for e in events if e["action"] == "denial_blocked"), 1)
        self.assertEqual(s["evaluation"]["false_denials"], 0)
        self.assertEqual(s["audit_s1"], 0)

    def test_auditor_catches_tampered_fields(self):
        h = Harness(self.root, mode="mock")
        h.run_day(20)
        target = next(c for c in h.cases.values() if c["state"] == "ACCEPTED")
        target["fields"]["last_name"] = "Tampered"
        h.save()
        audit, _ = Harness(self.root, mode="mock").audit_only()
        codes = {f["code"] for f in audit["findings"] if f["severity"] == "S1"}
        self.assertIn("FIELDS_CHANGED_WITHOUT_LOG", codes)
        self.assertIn("ACCEPTED_BUT_NOT_ELIGIBLE", codes)

    def test_auditor_catches_forged_evidence(self):
        h = Harness(self.root, mode="mock")
        h.run_day(30)
        target = next(c for c in h.cases.values() if c["changes"])
        target["changes"][0]["evidence"] = "text that is nowhere in the document"
        h.save()
        audit, _ = Harness(self.root, mode="mock").audit_only()
        self.assertIn("FIX_EVIDENCE_MISSING", {f["code"] for f in audit["findings"]})

    def test_auditor_catches_unlogged_change(self):
        h = Harness(self.root, mode="mock")
        h.run_day(30)
        target = next(c for c in h.cases.values() if c["changes"])
        target["changes"][0]["event_seq"] = 1
        h.save()
        audit, _ = Harness(self.root, mode="mock").audit_only()
        self.assertIn("FIX_NOT_LOGGED", {f["code"] for f in audit["findings"]})

    def test_auditor_reports_broken_ledger(self):
        h = Harness(self.root, mode="mock")
        h.run_day(10)
        path = self.root / "audit" / "ledger.jsonl"
        lines = path.read_text().splitlines()
        del lines[3]
        path.write_text("\n".join(lines) + "\n")
        audit, _ = Harness(self.root, mode="mock").audit_only()
        self.assertFalse(audit["ledger_ok"])
        self.assertIn("LEDGER_CHAIN_BROKEN", {f["code"] for f in audit["findings"]})


class BudgetTests(Base):
    def test_rollover_and_boundary(self):
        now = [datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)]
        b = TokenBudget(self.root / "t.json", 1000, 330, clock=lambda: now[0])
        b.add(900)
        with self.assertRaises(BudgetExceeded):
            b.check(150)                       # 900 + 150 > 1000
        b.check(100)                           # exactly at the cap is allowed
        now[0] += timedelta(hours=24)
        self.assertEqual(b.used(), 0)          # new day, fresh allowance

    def test_cap_stops_work_and_it_resumes_next_run(self):
        over = {"daily_token_cap": 6000, "mock_usage": True, "audit_reserve_tokens": 0}
        h = Harness(self.root, mode="mock", overrides=over)
        s = h.run_day(15)
        self.assertTrue(s["halted_for_budget"])
        self.assertLessEqual(h.budget.used(), 6000)
        waiting = [c for c in h.cases.values() if c["state"] == "AWAITING_INTAKE"]
        self.assertTrue(waiting, "some applications should be waiting for tomorrow's budget")
        self.assertEqual(s["audit_s1"], 0)
        self.assertNotIn("TOKEN_ACCOUNTING_MISMATCH", {f["code"] for f in h.auditor.audit()["findings"]})
        # next run with room: the backlog is picked up first and nothing is lost
        h2 = Harness(self.root, mode="mock", overrides={"daily_token_cap": 5_000_000, "mock_usage": True})
        s2 = h2.run_day(0)
        self.assertFalse(s2["halted_for_budget"])
        self.assertFalse([c for c in h2.cases.values() if c["state"] == "AWAITING_INTAKE"])
        self.assertTrue(s2["gate_ok"])


class ReserveTests(Base):
    def test_reserve_is_limited_to_ten_percent_of_cap(self):
        self.assertEqual(Harness(self.root, overrides={"daily_token_cap": 20000}).reserve, 2000)
        self.assertEqual(Harness(make_root(), overrides={"daily_token_cap": 2_000_000}).reserve, 150000)


class _FakeAPI(http.server.BaseHTTPRequestHandler):
    script, seen = [], []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        type(self).seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        status, payload = type(self).script.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _ok(content, tokens=1234, finish="stop"):
    return 200, {"model": "gpt-5.4-test", "choices": [{"message": {"content": content}, "finish_reason": finish}],
                 "usage": {"total_tokens": tokens}}


RESOLVER_OK = json.dumps({"diagnosis": "d", "action": "ESCALATE", "changes": [], "missing_info": [],
                          "reason": "r", "confidence": 0.5})


class OpenAIPathTests(Base):
    def setUp(self):
        super().setUp()
        _FakeAPI.script, _FakeAPI.seen = [], []
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeAPI)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        os.environ["OPENAI_API_KEY"] = "test-key-not-real"
        self.h = Harness(self.root, mode="live",
                         overrides={"openai_base_url": f"http://127.0.0.1:{self.srv.server_address[1]}/v1"})

    def call(self):
        return self.h.gw.call("resolver", "resolver", {"case_id": "X"}, case_id="X")

    def test_request_shape_usage_and_retry(self):
        _FakeAPI.script = [(429, {"error": "slow down"}), _ok(RESOLVER_OK, 1234)]
        out = self.call()
        self.assertEqual(out["action"], "ESCALATE")
        self.assertEqual(len(_FakeAPI.seen), 2, "429 should be retried once")
        req = _FakeAPI.seen[-1]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["auth"], "Bearer test-key-not-real")
        self.assertEqual(req["body"]["model"], "gpt-5.4")
        self.assertEqual(req["body"]["response_format"], {"type": "json_object"})
        self.assertEqual(req["body"]["max_completion_tokens"], 4000)
        self.assertEqual(self.h.budget.used(), 1234)
        ev = [e for e in self.h.ledger.read_all() if e["action"] == "llm:resolver"][0]
        self.assertEqual((ev["tokens"], ev["model"]), (1234, "gpt-5.4-test"))
        self.assertNotIn("test-key-not-real", (self.root / "audit" / "ledger.jsonl").read_text())

    def test_cap_blocks_the_call_before_any_request(self):
        self.h.budget.cap = 3000                      # one resolver call may need ~4000+ tokens
        with self.assertRaises(BudgetExceeded):
            self.call()
        self.assertEqual(_FakeAPI.seen, [])

    def test_invalid_output_retries_once_then_fails_and_both_calls_are_counted(self):
        _FakeAPI.script = [_ok("not json", 100), _ok('{"diagnosis": 5}', 200)]
        with self.assertRaises(LLMError):
            self.call()
        self.assertEqual(self.h.budget.used(), 300)

    def test_truncated_output_still_counts_tokens(self):
        _FakeAPI.script = [_ok("{", 777, finish="length")]
        with self.assertRaises(LLMError):
            self.call()
        self.assertEqual(self.h.budget.used(), 777)

    def test_http_error_is_reported_without_leaking_the_key(self):
        _FakeAPI.script = [(404, {"error": {"message": "model not found"}})]
        with self.assertRaises(LLMError) as cm:
            self.call()
        self.assertIn("404", str(cm.exception))
        self.assertNotIn("test-key-not-real", str(cm.exception))

    def test_live_mode_requires_a_key(self):
        os.environ.pop("OPENAI_API_KEY", None)
        with self.assertRaises(LLMError):
            Harness(make_root(), mode="live")
        os.environ["OPENAI_API_KEY"] = "test-key-not-real"


if __name__ == "__main__":
    unittest.main()
