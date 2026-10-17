"""Every model call goes through LLMGateway.

The gateway (1) reserves room under the daily token cap BEFORE the call, (2) calls a backend
(OpenAI or mock), (3) validates the JSON against a schema, (4) records usage and the output in
the audit ledger. Agents never talk to the API or the ledger themselves.

The gateway is safe to call from several threads at once: the budget reservation is atomic and
the ledger serialises its own appends.
"""
import json
import os
import time
import urllib.error
import urllib.request

from . import mock_brain
from .budget import BudgetExceeded  # noqa: F401  (re-exported for callers)
from .prompts import SCHEMAS, SYSTEM
from .util import canon, sha


class LLMError(Exception):
    pass


def check(obj, spec, path="$"):
    """Tiny schema validator. Returns a list of error strings (empty means valid)."""
    if isinstance(spec, dict):
        if not isinstance(obj, dict):
            return [f"{path}: expected object"]
        errs = []
        for k, s in spec.items():
            if k not in obj:
                errs.append(f"{path}.{k}: missing")
            else:
                errs += check(obj[k], s, f"{path}.{k}")
        return errs
    if isinstance(spec, tuple):
        kind = spec[0]
        if kind == "enum":
            return [] if obj in spec[1] else [f"{path}: must be one of {spec[1]}"]
        if kind == "list":
            if not isinstance(obj, list):
                return [f"{path}: expected list"]
            errs = []
            for i, item in enumerate(obj):
                errs += check(item, spec[1], f"{path}[{i}]")
            return errs
        if kind == "map":
            if not isinstance(obj, dict):
                return [f"{path}: expected object"]
            errs = []
            for k, v in obj.items():
                errs += check(v, spec[1], f"{path}.{k}")
            return errs
        raise ValueError(f"bad spec {spec!r}")
    nullable = spec.endswith("?")
    base = spec.rstrip("?")
    if obj is None:
        return [] if nullable else [f"{path}: null not allowed"]
    types = {"str": str, "num": (int, float), "bool": bool, "list": list, "dict": dict}
    ok = isinstance(obj, types[base]) and not (base == "num" and isinstance(obj, bool))
    return [] if ok else [f"{path}: expected {base}"]


class MockBackend:
    """Offline stand-in. Deterministic; costs nothing. NOT evidence of how a real model performs."""
    name = "mock"
    io_bound = False        # instant and CPU-only: overlapping calls would only add thread overhead

    def __init__(self, settings):
        self.count_usage = bool(settings.get("mock_usage", False))

    def complete(self, task, system, payload, max_tokens):
        out = mock_brain.dispatch(task, payload)
        text = json.dumps(out)
        tokens = (len(system) + len(canon(payload)) + len(text)) // 4 if self.count_usage else 0
        return text, tokens, "mock"


class OpenAIBackend:
    """Chat Completions over plain urllib (no third-party packages)."""
    name = "openai"
    io_bound = True         # every call is seconds of waiting: overlap them

    def __init__(self, settings):
        self.s = settings
        self.key = os.environ.get(settings["api_key_env"], "")
        if not self.key:
            raise LLMError(f"environment variable {settings['api_key_env']} is not set")

    def complete(self, task, system, payload, max_tokens):
        body = {
            "model": self.s["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_tokens,
        }
        body.update(self.s.get("extra_params") or {})
        data = json.dumps(body).encode("utf-8")
        url = self.s["openai_base_url"].rstrip("/") + "/chat/completions"
        retries = int(self.s.get("http_retries", 3))
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, method="POST", headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
            })
            try:
                with urllib.request.urlopen(req, timeout=float(self.s.get("http_timeout_seconds", 120))) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                e.close()
                if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(min(2 ** attempt, 20) * (0.05 if os.environ.get("ENROLLSIM_FAST_RETRY") else 1))
                    continue
                raise LLMError(f"HTTP {e.code} from API: {detail}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 20) * (0.05 if os.environ.get("ENROLLSIM_FAST_RETRY") else 1))
                    continue
                raise LLMError(f"network error calling API: {e}")
        try:
            choice = raw["choices"][0]
            text = choice["message"].get("content") or ""
            tokens = int(raw.get("usage", {}).get("total_tokens", 0))
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, ValueError):
            raise LLMError("unexpected API response shape")
        if finish == "length":
            # Tokens were still spent, so report them via a special error the gateway can record.
            err = LLMError("output truncated (finish_reason=length); raise max_completion_tokens")
            err.tokens = tokens
            raise err
        return text, tokens, raw.get("model", self.s["model"])


class LLMGateway:
    def __init__(self, backend, ledger, budget, settings, sim_date_fn):
        self.backend, self.ledger, self.budget = backend, ledger, budget
        self.settings, self.sim_date = settings, sim_date_fn

    def call(self, agent, task, payload, case_id=None, keep=0):
        system = SYSTEM[task]
        max_tokens = int(self.settings["max_completion_tokens"][task])
        estimate = (len(system) + len(canon(payload))) // 3 + max_tokens
        last_err = None
        for attempt in range(2):
            held = self.budget.reserve(estimate, keep)  # raises BudgetExceeded before any spend
            sent = payload if attempt == 0 else dict(payload, _previous_output_was_invalid=last_err)
            try:
                text, tokens, model = self.backend.complete(task, system, sent, max_tokens)
            except LLMError as e:
                spent = getattr(e, "tokens", 0)
                self.budget.settle(held, spent, counted=bool(spent))
                self.ledger.append(sim_date=self.sim_date(), agent=agent, action=f"llm_error:{task}",
                                   case_id=case_id, tokens=spent, model=self.settings.get("model"),
                                   data={"input_sha": sha(payload), "error": str(e)[:300]})
                raise
            except BaseException:
                self.budget.settle(held, 0, counted=False)   # never leave a hold behind
                raise
            self.budget.settle(held, tokens)
            try:
                out = json.loads(text)
                errs = check(out, SCHEMAS[task])
            except ValueError:
                out, errs = None, ["output was not valid JSON"]
            self.ledger.append(sim_date=self.sim_date(), agent=agent, action=f"llm:{task}", case_id=case_id,
                               tokens=tokens, model=model,
                               data={"input_sha": sha(payload), "ok": not errs, "errors": errs,
                                     "output": out if not errs else text[:500]})
            if not errs:
                return out
            last_err = "; ".join(errs)[:300]
        raise LLMError(f"{task}: model output failed validation twice: {last_err}")
