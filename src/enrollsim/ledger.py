"""Append-only, hash-chained audit ledger.

Only the harness holds a handle to this file. Agents never get a path or a writer:
they return values, and the harness records what they did. The auditor reads it.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from .util import canon, sha

GENESIS = "0" * 64
MAX_DATA_CHARS = 6000


def _clip(data):
    text = canon(data)
    if len(text) <= MAX_DATA_CHARS:
        return data
    return {"_truncated": True, "sha": sha(data), "head": text[:MAX_DATA_CHARS]}


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq, self.prev = self._tail()

    def _tail(self):
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0, GENESIS
        last = None
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = line
        if last is None:
            return 0, GENESIS
        rec = json.loads(last)
        return rec["seq"], rec["hash"]

    def append(self, *, sim_date, agent, action, case_id=None, data=None, tokens=0, model=None):
        self.seq += 1
        body = {
            "seq": self.seq,
            "sim_date": sim_date,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agent": agent,
            "action": action,
            "case_id": case_id,
            "tokens": int(tokens or 0),
            "model": model,
            "data": _clip(data if data is not None else {}),
            "prev": self.prev,
        }
        body["hash"] = sha(body)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(canon(body) + "\n")
        self.prev = body["hash"]
        return body

    def read_all(self):
        return read_ledger(self.path)

    @staticmethod
    def verify(path):
        """Return (ok, entries, message). Walks the whole chain."""
        p = Path(path)
        if not p.exists():
            return True, 0, "no ledger yet"
        prev, n = GENESIS, 0
        with open(p, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    return False, n, f"line {lineno}: not valid JSON"
                claimed = rec.pop("hash", None)
                if rec.get("prev") != prev:
                    return False, n, f"line {lineno}: prev hash does not match (chain broken)"
                if sha(rec) != claimed:
                    return False, n, f"line {lineno}: entry was modified (hash mismatch)"
                if rec.get("seq") != n + 1:
                    return False, n, f"line {lineno}: sequence gap"
                prev, n = claimed, n + 1
        return True, n, "chain intact"


def read_ledger(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out
