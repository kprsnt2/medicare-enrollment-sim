"""Reproduce a live day's wall clock locally, without an API key and without spending tokens.

A real model call takes seconds of dead time; the answers here come from the same deterministic
mock brain, but each call sleeps `--latency` seconds so the *timing* is realistic. The default
3.1s is the average call latency measured in the audit ledger of the GitHub Actions runs
(228s of wall clock for 73 calls on 2026-10-15).

    PYTHONPATH=src python tools/bench_speed.py                 # 1 worker vs 8 workers
    PYTHONPATH=src python tools/bench_speed.py --workers 1,4,8,16 --latency 3.1

Prints, per setting: wall clock for one simulated day, model calls, peak calls in flight, the
audit gate and the outcome accuracy. Accuracy must be identical across settings - only the
clock should move.
"""
import argparse
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enrollsim.llm import MockBackend                    # noqa: E402
from enrollsim.pipeline import Harness                   # noqa: E402


class SlowBackend(MockBackend):
    """Deterministic mock answers with real-API latency; records how many calls overlapped."""

    name = "slow-mock"
    io_bound = True

    def __init__(self, latency, settings=None):
        super().__init__(settings or {})
        self.latency = float(latency)
        self.inflight = self.peak = self.calls = 0
        self._lock = threading.Lock()

    def complete(self, task, system, payload, max_tokens):
        with self._lock:
            self.inflight += 1
            self.calls += 1
            self.peak = max(self.peak, self.inflight)
        try:
            time.sleep(self.latency)
            return super().complete(task, system, payload, max_tokens)
        finally:
            with self._lock:
                self.inflight -= 1


def run_once(workers, latency, new, quiet=False):
    root = Path(tempfile.mkdtemp(prefix="enrollsim_bench_"))
    shutil.copytree(ROOT / "config", root / "config")
    backend = SlowBackend(latency)
    try:
        h = Harness(root, mode="mock", backend=backend, overrides={"llm_concurrency": workers})
        if not quiet:
            h.progress = lambda m: print(m, file=sys.stderr, flush=True)
        started = time.perf_counter()
        s = h.run_day(new)
        wall = time.perf_counter() - started
        return {"workers": workers, "wall": wall, "calls": backend.calls, "peak": backend.peak,
                "gate_ok": s["gate_ok"], "accuracy": s["evaluation"]["outcome_accuracy"],
                "states": s["states"], "tokens": s["tokens_used_today"]}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latency", type=float, default=3.1, help="seconds per model call (default 3.1)")
    ap.add_argument("--new", type=int, default=30, help="new applications in the day (default 30)")
    ap.add_argument("--workers", default="1,8", help="comma-separated worker counts to compare")
    ap.add_argument("--quiet", action="store_true", help="hide per-stage progress")
    a = ap.parse_args(argv)

    settings_workers = [int(w) for w in a.workers.split(",") if w.strip()]
    rows = []
    print(f"one simulated day, {a.new} applications, {a.latency}s per model call\n", flush=True)
    for w in settings_workers:
        print(f"--- {w} concurrent call(s) ---", file=sys.stderr, flush=True)
        rows.append(run_once(w, a.latency, a.new, a.quiet))

    baseline = next((r for r in rows if r["workers"] == 1), rows[0])
    print(f"{'workers':>8} {'wall clock':>12} {'model calls':>12} {'peak in flight':>15} "
          f"{'gate':>6} {'accuracy':>9} {'speed-up':>9}")
    for r in rows:
        print(f"{r['workers']:>8} {r['wall']:>11.1f}s {r['calls']:>12} {r['peak']:>15} "
              f"{'PASS' if r['gate_ok'] else 'FAIL':>6} {str(r['accuracy']):>9} "
              f"{baseline['wall'] / r['wall']:>8.1f}x")
    print(f"\nsequential floor if calls never overlapped: {baseline['calls'] * a.latency:.0f}s "
          f"({baseline['calls']} calls x {a.latency}s)")
    acc = {r["accuracy"] for r in rows}
    states = {tuple(sorted(r["states"].items())) for r in rows}
    print("same decisions at every setting:", "yes" if len(acc) == 1 and len(states) == 1 else "NO - INVESTIGATE")
    return 0 if all(r["gate_ok"] for r in rows) and len(acc) == 1 and len(states) == 1 else 2


if __name__ == "__main__":
    sys.exit(main())
