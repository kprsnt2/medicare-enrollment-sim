"""Hard daily token cap, persisted on disk.

The check runs BEFORE every API call using a conservative estimate (prompt size plus the
call's max output), so the cap is a ceiling the code stops short of, not one it discovers
after the money is spent. `keep` holds back a reserve for the compliance and audit stages.

Calls may now overlap (see util.parallel_map), so "check then spend" is no longer atomic on its
own: two threads could both read `used` and both pass. `reserve` therefore does check-and-hold
in one locked step and `settle` releases the hold and books the real usage afterwards. The
invariant at every admission is `used + in-flight + this call + keep <= cap`, so the cap is still
a ceiling even with N calls in flight. Holds are process-local - a hold only means anything while
a call is in flight here - so they live in memory and cost no extra disk write. `check`/`add`
stay for single-call callers and tests.
"""
import threading
from datetime import datetime, timedelta, timezone

from .util import load_json, save_json


class BudgetExceeded(Exception):
    pass


class TokenBudget:
    def __init__(self, path, cap, offset_minutes=330, clock=None):
        self.path = path
        self.cap = int(cap)
        self.offset = timedelta(minutes=int(offset_minutes))
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._held = 0                      # tokens reserved by calls in flight in this process

    def day(self):
        return (self.clock() + self.offset).date().isoformat()

    def _load(self):
        return load_json(self.path, {}) or {}

    def _today(self, data, create=False):
        if create:
            return data.setdefault(self.day(), {"used": 0, "calls": 0})
        return data.get(self.day(), {}) or {}

    def used(self):
        with self._lock:
            return int(self._today(self._load()).get("used", 0))

    def calls(self):
        with self._lock:
            return int(self._today(self._load()).get("calls", 0))

    def reserved(self):
        """Tokens held by calls that are in flight right now."""
        with self._lock:
            return self._held

    def remaining(self):
        with self._lock:
            return max(0, self.cap - int(self._today(self._load()).get("used", 0)) - self._held)

    def check(self, estimate, keep=0):
        with self._lock:
            used = int(self._today(self._load()).get("used", 0))
            if used + self._held + int(estimate) + int(keep) > self.cap:
                raise BudgetExceeded(
                    f"daily token cap {self.cap:,} would be exceeded "
                    f"(used {used:,}, in flight {self._held:,}, this call may need ~{int(estimate):,}, "
                    f"reserve {int(keep):,})"
                )

    def reserve(self, estimate, keep=0):
        """Atomically check the cap and hold `estimate` until settle(). Returns the held amount."""
        self.check(estimate, keep)
        with self._lock:
            self._held += int(estimate)
            return int(estimate)

    def settle(self, held, tokens, counted=True):
        """Release a hold from reserve() and book what the call really used."""
        with self._lock:
            self._held = max(0, self._held - int(held or 0))
            data = self._load()
            today = self._today(data, create=True)
            today["used"] = int(today.get("used", 0)) + int(tokens or 0)
            if counted:
                today["calls"] = int(today.get("calls", 0)) + 1
            save_json(self.path, data)

    def add(self, tokens):
        with self._lock:
            data = self._load()
            today = self._today(data, create=True)
            today["used"] = int(today.get("used", 0)) + int(tokens or 0)
            today["calls"] = int(today.get("calls", 0)) + 1
            save_json(self.path, data)

    def release_all(self):
        """Drop every hold. Called once at startup: nothing can be in flight yet, so a hold is
        either in-memory residue or (from an older version) a `reserved` key left in the file."""
        with self._lock:
            self._held = 0
            data = self._load()
            today = self._today(data)
            if today.get("reserved"):
                today.pop("reserved", None)
                save_json(self.path, data)
