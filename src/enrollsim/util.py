"""Small pure helpers shared by every module."""
import hashlib
import json
import os
import re
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(obj):
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()


def norm(s):
    return re.sub(r"\s+", " ", str(s if s is not None else "")).strip().lower()


def contains_value(quote, value):
    """True if `value` appears in `quote` as a whole token sequence (case-insensitive)."""
    q, v = norm(quote), norm(value)
    if not v:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", q) is not None


def pdate(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def valid_date(s):
    try:
        pdate(s)
        return True
    except (ValueError, TypeError):
        return False


def first_of_next_month(x):
    if x.month == 12:
        return date(x.year + 1, 1, 1)
    return date(x.year, x.month + 1, 1)


def months_between(a, b):
    return (a.year * 12 + a.month) - (b.year * 12 + b.month)


def in_window(x, start_md, end_md):
    sm, sd = (int(p) for p in start_md.split("-"))
    em, ed = (int(p) for p in end_md.split("-"))
    return (sm, sd) <= (x.month, x.day) <= (em, ed)


def derive_effective_date(election_type, election_date, policy):
    rule = policy["effective_date_rules"].get(election_type)
    if not rule or not valid_date(election_date):
        return None
    d = pdate(election_date)
    if rule == "JAN_1_NEXT_YEAR":
        return date(d.year + 1, 1, 1).isoformat()
    if rule == "FIRST_OF_NEXT_MONTH":
        return first_of_next_month(d).isoformat()
    return None


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path, obj):
    """Atomic write so a crash never leaves a half-written state file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def parallel_map(items, fn, workers=1):
    """Yield (item, ok, value_or_exception) for each item, in the input order.

    At most `workers` calls run at once; with workers <= 1 this is a plain in-order loop.
    After the first item that raises, no further items are started (work already in flight is
    drained and its results are dropped), so a token-cap halt stops the stage instead of
    racing ahead. Exceptions are handed back to the caller rather than raised here, because
    the caller knows whether the failure is fatal (LLMError) or a normal stop (BudgetExceeded).

    Wall-clock matters: every model call is dead time, so independent calls overlap instead of
    queueing one behind the other. Results are applied in the original order, so the pipeline
    still behaves like the sequential one.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        for it in items:
            try:
                yield it, True, fn(it)
            except Exception as e:                       # noqa: BLE001 - handed to the caller
                yield it, False, e
        return
    with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="enrollsim") as pool:
        queue = deque()
        pending = iter(items)

        def fill():
            while len(queue) < int(workers):
                nxt = next(pending, None)
                if nxt is None:
                    return
                queue.append((nxt, pool.submit(fn, nxt)))

        fill()
        while queue:
            item, fut = queue.popleft()
            try:
                value, ok = fut.result(), True
            except Exception as e:                       # noqa: BLE001 - handed to the caller
                value, ok = e, False
            yield item, ok, value
            if not ok:
                for _, f in queue:
                    f.cancel()
                return
            fill()
