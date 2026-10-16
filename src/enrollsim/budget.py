"""Hard daily token cap, persisted on disk.

The check runs BEFORE every API call using a conservative estimate (prompt size plus the
call's max output), so the cap is a ceiling the code stops short of, not one it discovers
after the money is spent. `keep` holds back a reserve for the compliance and audit stages.
"""
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

    def day(self):
        return (self.clock() + self.offset).date().isoformat()

    def _load(self):
        return load_json(self.path, {}) or {}

    def used(self):
        return int(self._load().get(self.day(), {}).get("used", 0))

    def calls(self):
        return int(self._load().get(self.day(), {}).get("calls", 0))

    def remaining(self):
        return max(0, self.cap - self.used())

    def check(self, estimate, keep=0):
        used = self.used()
        if used + int(estimate) + int(keep) > self.cap:
            raise BudgetExceeded(
                f"daily token cap {self.cap:,} would be exceeded "
                f"(used {used:,}, this call may need ~{int(estimate):,}, reserve {int(keep):,})"
            )

    def add(self, tokens):
        data = self._load()
        today = data.setdefault(self.day(), {"used": 0, "calls": 0})
        today["used"] = int(today["used"]) + int(tokens or 0)
        today["calls"] = int(today["calls"]) + 1
        save_json(self.path, data)
