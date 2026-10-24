# Medicare enrollment simulator (synthetic) with audited LLM agents

A working model of an enrollment operation, built from the process you described: applications
arrive on paper and digitally, a system submits them to a (mock) CMS and gets reply codes back,
failures fall into a **work basket**, an employee-agent works the **oldest item first** against a
10-day turnaround, members get letters, a compliance agent checks the rules, and a separate
**auditor** checks everything against a log the agents cannot write to.

Everything is **synthetic**: fictional people, `SYN`-prefixed IDs, invented plans.
The reply codes, election windows and letter wording in `config/policy.json` are my simplified
stand-ins, **not** the real CMS transaction codes or rules. You know the real ones: replace them.
Do not put real member data or any employer-confidential procedure in this repo.

## The agents

| Role | Does | Uses the model? |
|---|---|---|
| Applicant simulator | Generates paper (OCR-noisy text) and digital applications, with injected defects and a hidden answer key | No (seeded, free, reproducible) |
| Paper intake | Reads scan text into fields; every field needs a verbatim evidence quote | Yes |
| Digital intake | Validates and normalises the electronic form | No |
| Processor (the system) | Completeness check, submits to mock CMS, routes the reply: accept / duplicate / work basket | No |
| Resolver (the employee) | Takes the oldest basket item, diagnoses it, proposes a fix, asks the member, denies, or escalates | Yes |
| Notice | Writes member letters in plain language | Yes (template fallback) |
| Compliance | Checks letters, turnaround breaches and at-risk items | Rules (+ small advisory LLM review) |
| Auditor | Re-verifies every case from the log and stored files; reviews a sample | Rules decide; LLM is advisory only |

Defects injected (your list): demographic mismatch, eligibility-date mismatch, plan not in area,
wrong ZIP/address, wrong plan chosen, invalid election period (sometimes with a special-election
reason hidden in the notes), provider (PCP) not in network, duplicate, member ID typo, missing
signature, OCR errors.

## How "agents work in a folder and log everything" is enforced

```
agents ──(data in / data out only)──▶ harness ──▶ workspace/   inbox, letters   (agent-visible work product)
                                          │  ──▶ state/       cases, clock, token use
                                          └────▶ audit/       ledger.jsonl (hash-chained), reports/, gold/
auditor ◀── reads audit/ and state/ (read-only) ── re-derives the outcome, never trusts an agent's summary
```

* Agents hold no file handles, no API client and no ledger writer. Everything they do passes through
  the harness, which writes one hash-chained line per action to `audit/ledger.jsonl`
  (edit or delete any line and `verify-ledger` fails).
* A correction is applied only if its evidence quote exists in the source document and the value
  passes a hard check (the guard in `rules.py`). A made-up fix is rejected and logged.
* **DENY is the one action a model decides by judgement alone**, so it has a second deterministic gate:
  if the documents hint at a special-election reason, the denial is blocked and the case goes to a human
  (`ESCALATED`). In production every denial should get human sign-off; this gate stands in for that.
* `audit/gold/` holds the hidden answer key. It is used only to score accuracy and never reaches a prompt or the ledger (tested).

## Quick start (free, offline)

```bash
PYTHONPATH=src python -m unittest discover -s tests        # 46 tests, no network
PYTHONPATH=src python -m enrollsim run --days 5            # mock mode, one simulated day per day
PYTHONPATH=src python -m enrollsim status                  # queue, oldest first
PYTHONPATH=src python -m enrollsim audit                   # re-audit current state
PYTHONPATH=src python -m enrollsim verify-ledger
PYTHONPATH=src python -m enrollsim reset --yes             # start over
```
Reports land in `audit/reports/<date>_summary.md`. Mock mode uses deterministic parsers that were written
against the same defect list, so it scores ~100%. **That proves the plumbing, guards and audit, not model quality.**

## When a case and the answer key disagree

`diagnose.py` compares every case against the hidden key and says *why* each one differs - the gold
outcome, the state the run reached, the CMS reply code behind the basket reason, and that case's ledger
timeline. It writes `why_cases.txt` and prints a summary.

```bash
PYTHONPATH=src python diagnose.py                           # summary + every miss, explained
PYTHONPATH=src python diagnose.py --case APP-20261017-010   # one case, in full
PYTHONPATH=src python diagnose.py --all                     # also list the cases that agreed
```

Read the output in two parts. A case that is **still open** (`PENDING_MEMBER`, `ESCALATED`) is not wrong
yet - it is waiting on the RFI clock or a human, and the daily summary already counts it as open. Only a
**terminal** case that differs from gold is a real defect. An empty miss-list means every outcome and
every stored field matched.

The generator guarantees each case's expected outcome is reachable from its own paperwork: a `zip_typo`
always ships the proof-of-address attachment that proves the right ZIP, an `mbi_typo` always ships the
card, and an attachment only ever repeats the form's own values - a card showing the *correct* MBI would
quietly undo an MBI typo. `SynthGoldTests` enforces all of that, so a later tweak to the defect
catalogue cannot reintroduce a case nobody could have got right.

## Where the time goes, and why it is fast now

A live day is ~73 model calls. Each one is seconds of waiting on the network, and in the first version
they ran strictly one after another, so the day cost the sum of every call. Measured from the audit
ledger of the GitHub Actions runs on 2026-10-15/16: **228s and 262s of wall clock, 100% of it spent
inside `gw.call`** - intake 108s, resolver 60s, notices 52s, audit sample 8s. Nothing else in the day
(synthesis, CMS rules, letters, audit, dashboard) takes measurable time.

The calls are independent per case, so they now overlap:

* `config/settings.json` -> `llm_concurrency` (default **8**) is how many calls may be in flight at once.
  `--workers N` overrides it for one run; `--workers 1` gives the old strictly sequential behaviour.
* Results are still applied in the original case order, so the day decides the same things - only the
  clock changes. `ConcurrencyTests` and `LiveModeTests` assert that: one worker and eight workers must
  produce the same states, fields, letters, accuracy and audit gate.
* The ledger is still a single hash chain (one writer, one lock) and the token cap is still a hard
  ceiling: `TokenBudget.reserve` checks *and* holds the estimate atomically before the call, so N calls
  in flight cannot together cross the cap. Holds are released in `settle` (and on any exception).
* Mock mode stays sequential on purpose - an instant backend has nothing to wait for, and threads would
  only add overhead.
* Progress goes to stderr per stage (`--quiet` turns it off), because a silent four-minute terminal
  looks like a hang.

Reproduce the numbers locally with no API key and no cost - a fake backend with real-API latency:

```bash
PYTHONPATH=src python tools/bench_speed.py --workers 1,8,16   # 30 applications, 3.1s per call
```

Measured on this machine (3.1s per call, the average the Actions runs showed):

| concurrent calls | wall clock for the day | model calls | audit gate | accuracy |
|---|---|---|---|---|
| 1 (old behaviour) | 227.1s | 73 | PASS | 1.0 |
| 8 (new default)   | 40.7s  | 73 | PASS | 1.0 |
| 16                | 25.2s  | 73 | PASS | 1.0 |

The same day, the same decisions, 5.6x faster at the default setting - so the daily Actions job drops
from about 5 minutes to about 1.

## Live mode with the OpenAI API and a 2M-token daily cap

```bash
export OPENAI_API_KEY=sk-...
PYTHONPATH=src python -m enrollsim run --mode live --new 10     # start small
PYTHONPATH=src python -m enrollsim run --mode live --workers 8  # explicit concurrency
```

* Model name and endpoint are in `config/settings.json` (`"model": "gpt-5.4"`). I could not verify that
  name or run a real call from my environment, so if the API answers "model not found", change it there.
  Model-specific parameters (for example reasoning effort) go in `extra_params`.
* `daily_token_cap` (2,000,000) is checked **before every call** against a conservative estimate
  (prompt size + the call's max output), so it is a ceiling the code stops short of. Real usage from each
  response is added afterwards and saved in `state/tokens.json`. The day boundary follows IST (`budget_day_offset_minutes`).
* `audit_reserve_tokens` (150,000) is held back so compliance and audit can still run when intake and resolving hit the cap.
* When the cap is hit the run stops early, unprocessed applications wait in the queue for the next day, letters
  fall back to templates, and the deterministic audit still runs. The summary says "STOPPED EARLY".
* Reasoning models spend hidden tokens that count toward usage. Measure real consumption with a small first run
  before scaling `new_applications_per_day`.
* The code cap trusts its own state file. Set a **monthly spend limit in the OpenAI dashboard** as the backstop.

## GitHub Actions (daily, automatic)

1. Create a **private** GitHub repo and push this folder (the run commits state and logs back to it).
2. Settings -> Secrets and variables -> Actions -> new secret `OPENAI_API_KEY`.
3. Settings -> Actions -> General -> Workflow permissions -> **Read and write**.
4. Actions tab -> `daily-enrollment-sim` -> **Run workflow** (try `mock` first, then `live` with a small number).
5. It then runs daily at 09:00 IST (edit the cron in `.github/workflows/daily.yml`). Each run: tests, one simulated day,
   audit, then a commit of `state/`, `workspace/` and `audit/`. If the audit gate fails the job goes red but the evidence is still committed.

Runs are serialised (`concurrency`), so the token ledger never races. `state/cases.json` grows with the case count;
trim old terminal cases if the repo gets heavy.

## Dashboard (Vercel or GitHub Pages)

`python -m enrollsim dashboard` writes one self-contained file, `docs/index.html` (no server, no libraries). It shows
KPIs, accuracy and tokens by day, accuracy by defect, who did the work, the work basket, audit findings, and a case
explorer with each case's ledger timeline and letters. The daily workflow rebuilds and commits it.

* **Vercel (works with a private repo on the free plan):** New Project -> import the repo -> Framework "Other",
  leave the build command empty, set **Output Directory** to `docs`. Every daily commit redeploys it.
  Vercel's default URL is public, so turn on Vercel Authentication if you do not want it open. The data is synthetic, but the
  ledger shows model outputs.
* **GitHub Pages:** free accounts need a *public* repo. Settings -> Pages -> Deploy from branch -> `main` / `/docs`.

## Make it yours

* `config/policy.json`: reply codes, windows, TAT days, letter phrases, which fields an agent may change.
* `config/reference.json`: plans, service areas, networks.
* `config/settings.json`: `llm_concurrency` (calls in flight), `new_applications_per_day`, the token cap.
* `src/enrollsim/synth.py`: the defect catalogue and how often each occurs.
* Add a new work-basket reason: add a CMS check in `rules.evaluate`, a defect in `synth.py`, and a branch in
  `mock_brain.resolver` (live mode needs no code: the model reads the same payload).
* To add your own real-world cases, describe them to me and I will turn each into a defect + rule + test.

## Known limits

* Rules are simplified (for example one reply code at a time, no retroactive dates, no real SEP catalogue).
* The denial gate is a keyword list. It is a safety net for the simulation, not a substitute for human review.
* Live accuracy is unmeasured until you run it: compare `evaluation.outcome_accuracy` and `false_accepts` in the daily report.
* The OpenAI client is tested against a local fake server, not the real API.

## Showing the workflow

`python -m enrollsim workflow_export --root . --out workflow_export.json` writes one JSON file
that describes the pipeline as it actually ran. It reads only what the harness already wrote
(`config/`, `state/`, `workspace/`, `audit/`), verifies the ledger chain, and takes under a second.
No network, no model calls, no third-party packages.

* `meta` - simulated date and next run date, days run, mode and model, token cap and use, ledger
  status and entry count. `days` - one row per simulated day: new applications, states, accuracy,
  false accepts, audit gate S1/S2, tokens and stage timings.
* `stages` - the eight stages `Harness.run_day` executes (applicants, intake, processor + mock CMS,
  CMS reply, work basket, resolver, notices, compliance and audit), with the engine behind each one
  (model or deterministic), the artifact it writes, counts from the ledger, and one real example
  line. `loops` records that the work basket hands a fixed case back to submission.
* `cases` - a deterministic spread (the two most recent days plus the oldest open items) with the
  field changes, the evidence quote behind each fix, the letters as issued, and the ledger timeline.
  `kpis`, `agents`, `codes`, `before_after` and `audit_evidence` carry the totals, the per-agent
  ledger load, the reply-code catalogue, one before/after case, and the latest audit and compliance
  reports.
* The answer key in `audit/gold/gold.json` is read for `defect` and `expected_final` only - the same
  comparison `docs/index.html` publishes. `true_fields` is never read, so no answer value can reach
  the file.

This export is what the MEOP site's `/workflow` page renders as its second project tile. Re-run it
after each simulated day (the dashboard and the workflow page share the same source files).
