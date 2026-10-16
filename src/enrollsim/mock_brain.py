"""Deterministic stand-ins for the LLM tasks, used for offline runs, tests and CI.

These are simple parsers and templates, NOT a model. A clean mock run proves the plumbing,
the guards and the audit work end to end; it says nothing about how a real model would perform.
Use `--mode live` to measure that.
"""
import re

from .util import contains_value, derive_effective_date, norm

CONF = {"0": "O", "1": "l", "5": "S", "8": "B", "2": "Z"}
REV = {"O": "0", "l": "1", "I": "1", "S": "5", "B": "8", "Z": "2"}
DIGIT_FIELDS = {"dob": r"^\d{4}-\d{2}-\d{2}$", "zip": r"^\d{5}$", "part_a_eff": r"^\d{4}-\d{2}-\d{2}$",
                "part_b_eff": r"^\d{4}-\d{2}-\d{2}$", "phone": r"^\d{3}-\d{3}-\d{4}$",
                "election_date": r"^\d{4}-\d{2}-\d{2}$", "requested_effective_date": r"^\d{4}-\d{2}-\d{2}$"}
CUE_REASONS = [(r"\bmoved\b", "MOVED_TO_NEW_SERVICE_AREA"), (r"employer coverage ended", "LOST_EMPLOYER_COVERAGE"),
               (r"\bdiscontinued\b", "PLAN_CONTRACT_ENDED")]


def _fix_digits(v):
    return "".join(REV.get(c, c) for c in v)


def _confusable(read, truth):
    if len(read) != len(truth) or read == truth:
        return False
    return all(a == b or CONF.get(b) == a or REV.get(a) == b for a, b in zip(read, truth))


def _card(src):
    m = re.search(r"Attachment A \(Medicare card copy\):\s*(.+)", src)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split("/")]
    try:
        name = parts[0].split()
        return {"name": parts[0], "last_name": name[0], "first_name": name[1], "dob": parts[1].split()[1],
                "sex": parts[2].split()[1], "mbi": parts[3].split()[1], "part_a_eff": parts[4].split()[2],
                "part_b_eff": parts[5].split()[2], "_q": {
                    "last_name": parts[0], "first_name": parts[0], "dob": parts[1], "sex": parts[2],
                    "mbi": parts[3], "part_a_eff": parts[4], "part_b_eff": parts[5]}}
    except IndexError:
        return None


def _proof(src):
    m = re.search(r"Attachment B \(proof of address\):\s*(.+)", src)
    if not m:
        return None
    line = m.group(1).strip()
    return {"zip": line.split()[-1], "_q": {"zip": line}}


def _notes(src):
    m = re.search(r"^Notes:\s*(.*)$", src, re.M)
    return m.group(1).strip() if m and m.group(1).strip() != "-" else ""


def paper_intake(p):
    raw = p["raw_text"]
    fields, ev = {}, {}

    def grab(pattern, keys):
        m = re.search(pattern, raw, re.M)
        for i, k in enumerate(keys, 1):
            fields[k] = m.group(i).strip() if m else ""
            ev[k] = m.group(0).strip() if m else None

    grab(r"^Received:\s*(\S+)", ["election_date"])
    grab(r"^Name:\s*(.+?),\s*(.+?)\s*$", ["last_name", "first_name"])
    grab(r"^DOB:\s*(\S+)\s+Sex:\s*(\S+)", ["dob", "sex"])
    grab(r"^Medicare No:\s*(\S+)", ["mbi"])
    grab(r"^Part A effective:\s*(\S+)\s+Part B effective:\s*(\S+)", ["part_a_eff", "part_b_eff"])
    grab(r"^Address:\s*(.+?)\s*$", ["address"])
    grab(r"^City:\s*(.+?)\s{2,}State:\s*(\S+)\s+ZIP:\s*(\S+)\s+County:\s*(.+?)\s*$", ["city", "state", "zip", "county"])
    grab(r"^Phone:\s*(\S+)", ["phone"])
    grab(r"^Plan selected:\s*(.+?)\s*\(([^)]*)\)\s*$", ["plan_name", "plan_id"])
    grab(r"^PCP:\s*(.*?)\s*$", ["pcp_name"])
    grab(r"^Election:\s*(\S+)\s+Sep reason:\s*(\S+)\s+Signed:\s*(\S+)", ["election_type", "sep_reason", "signature"])
    grab(r"^Requested effective date:\s*(\S+)", ["requested_effective_date"])
    if fields["pcp_name"] == "-":
        fields["pcp_name"] = ""
    if fields["sep_reason"] == "-":
        fields["sep_reason"] = ""
    fields["signature"] = "Y" if fields["signature"].lower().startswith("y") else "N"
    ev["signature"] = ev["signature"] or None

    conf = {k: (0.95 if fields.get(k) else 0.0) for k in fields}
    for k, pat in DIGIT_FIELDS.items():
        v = fields.get(k, "")
        if v and not re.match(pat, v):
            cleaned = _fix_digits(v)
            if re.match(pat, cleaned):
                fields[k], conf[k] = cleaned, 0.55
            else:
                conf[k] = 0.3
    card = _card(raw)
    if card and fields.get("mbi") and _confusable(fields["mbi"], card["mbi"]):
        conf["mbi"] = 0.4
    return {"fields": {k: (v if v != "" else None) for k, v in fields.items()}, "confidence": conf,
            "evidence": ev, "notes": "mock parser"}


def resolver(p):
    f, src, reason, code = p["fields"], p["source_text"], p["basket_reason"], p.get("last_code")
    card, proof, notes = _card(src), _proof(src), _notes(src)
    policy, ref = p["policy"], p["reference"]

    def out(action, diagnosis, changes=None, missing=None):
        return {"diagnosis": diagnosis, "action": action, "changes": changes or [],
                "missing_info": missing or [], "reason": diagnosis, "confidence": 0.9}

    def ch(field, new, quote):
        return {"field": field, "new_value": new, "evidence_quote": quote}

    def from_card(keys):
        res = []
        if card:
            for k in keys:
                if norm(card[k]) != norm(f.get(k)):
                    res.append(ch(k, card[k], card["_q"][k]))
        return res

    if reason == "OCR_LOW_CONFIDENCE":
        res = from_card([k for k in p["flagged_fields"] if k in ("first_name", "last_name", "dob", "sex", "mbi", "part_a_eff", "part_b_eff")])
        if proof and "zip" in p["flagged_fields"] and proof["zip"] != f.get("zip"):
            res.append(ch("zip", proof["zip"], proof["_q"]["zip"]))
        return out("FIX", "OCR doubt resolved from the attached documents", res) if res else \
            out("CONFIRM", "Flagged values checked; no conflicting evidence")
    if reason == "INCOMPLETE":
        return out("REQUEST_INFO", "Required items are missing", missing=p.get("missing_items") or ["missing items"])
    if code == "REJ-DEMO":
        res = from_card(["first_name", "last_name", "dob", "sex"])
        return out("FIX", "Name/DOB/sex differs from the Medicare card copy", res) if res else \
            out("REQUEST_INFO", "No document to check identity against", missing=["a copy of the Medicare card"])
    if code == "REJ-NOENT":
        res = from_card(["mbi"])
        return out("FIX", "Member ID differs from the card copy", res) if res else \
            out("REQUEST_INFO", "Member ID not found and no card copy", missing=["a copy of the Medicare card"])
    if code == "REJ-ELIG-DATE":
        if "ntitlement" in (p.get("last_message") or ""):
            res = from_card(["part_a_eff", "part_b_eff"])
            return out("FIX", "Entitlement dates differ from the card copy", res) if res else \
                out("REQUEST_INFO", "Entitlement dates cannot be verified", missing=["a copy of the Medicare card"])
        return out("FIX", "Effective date does not follow the election period",
                   [ch("requested_effective_date", derive_effective_date(f["election_type"], f["election_date"], policy), "policy")])
    if code == "REJ-PLAN":
        hit = next((pid for pid, pl in ref["plans"].items() if norm(pl["name"]) in norm(src)), None)
        return out("FIX", "Plan ID is not offered; the plan name on the form matches a catalog plan",
                   [ch("plan_id", hit, f["plan_name"])]) if hit else \
            out("REQUEST_INFO", "Plan cannot be identified", missing=["the plan you want to join"])
    if code == "REJ-ADDR":
        return out("FIX", "ZIP differs from the proof of address", [ch("zip", proof["zip"], proof["_q"]["zip"])]) \
            if proof and proof["zip"] != f.get("zip") else \
            out("REQUEST_INFO", "ZIP does not match the county and no proof of address", missing=["proof of address"])
    if code == "REJ-AREA":
        return out("DENY", "County is outside the plan's service area")
    if code == "REJ-ELECTION":
        for pat, reason_code in CUE_REASONS:
            m = re.search(pat, notes, re.I)
            if m:
                return out("FIX", "Notes describe a special election reason",
                           [ch("election_type", "SEP", notes), ch("sep_reason", reason_code, notes)])
        return out("DENY", "No valid election period and no special reason given")
    if code == "REJ-PCP":
        plan = ref["plans"].get(f.get("plan_id", ""), {"network": []})
        hit = next((n for n in plan["network"] if contains_value(notes, n)), None)
        return out("FIX", "Notes name an in-network provider", [ch("pcp_name", hit, notes)]) if hit else \
            out("REQUEST_INFO", "Provider is not in the network", missing=["an in-network primary care provider"])
    return out("ESCALATE", "Unrecognised situation")


def notice(p):
    k, name, ph = p["kind"], p["member_name"], p["contact_placeholder"]
    if k == "ACCEPT":
        subject = "Your enrollment request was accepted"
        body = (f"Dear {name},\n\nWe received your enrollment request on {p['received_date']} and it has been "
                f"accepted for {p['plan_name']}. The effective date is {p['effective_date']}.\n\n"
                f"If you have questions, call {ph}.")
    elif k == "DENY":
        subject = "We could not process your enrollment request"
        body = (f"Dear {name},\n\nWe are unable to process your enrollment request received on "
                f"{p['received_date']}. The reason: {p['reason_text']}.\n\nYou have the right to ask us to "
                f"review this decision. Call {ph} for help.")
    elif k == "RFI":
        subject = "We need more information to finish your enrollment"
        items = "; ".join(p.get("missing_items") or ["some details"])
        body = (f"Dear {name},\n\nWe need more information to finish your enrollment request received on "
                f"{p['received_date']}: {items}.\n\nPlease respond by {p['respond_by']}. Call {ph} if you need help.")
    else:
        subject = "Your enrollment request has been closed"
        body = (f"Dear {name},\n\nWe have closed your enrollment request received on {p['received_date']} "
                f"because we did not receive the information we asked for. You may apply again. Call {ph}.")
    return {"subject": subject, "body": body}


def auditor_review(p):
    return {"verdict": "PASS", "findings": []}


def letter_review(p):
    return {"pass": True, "issues": []}


def dispatch(task, payload):
    return {"paper_intake": paper_intake, "resolver": resolver, "notice": notice,
            "auditor_review": auditor_review, "letter_review": letter_review}[task](payload)
