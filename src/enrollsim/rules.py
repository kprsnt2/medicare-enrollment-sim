"""Deterministic rules: the mock CMS reply, completeness checks, and the fix guard.

Everything here is exact logic with no model involved. The guard (`validate_change`) is the
anti-fabrication gate: an agent can propose a correction, but it is only applied if the
evidence quote really exists in the source document and the value passes a hard check.
"""
from .util import contains_value, derive_effective_date, in_window, months_between, norm, pdate, valid_date

CODES = {
    "ACC": "Accepted",
    "DUP": "Duplicate enrollment request",
    "REJ-NOENT": "Member ID not found on the master file",
    "REJ-DEMO": "Name, date of birth or sex does not match the master file",
    "REJ-ELIG-DATE": "Entitlement or effective date problem",
    "REJ-PLAN": "Plan ID is not offered",
    "REJ-ADDR": "ZIP code does not match the stated county",
    "REJ-AREA": "Member's county is outside the plan's service area",
    "REJ-ELECTION": "Election period is not valid for this request",
    "REJ-PCP": "Primary care provider is not in the plan network",
}

ELECTION_TYPES = ["AEP", "OEP", "IEP", "SEP"]


def election_valid(fields, member, policy):
    t = fields.get("election_type", "")
    if not valid_date(fields.get("election_date", "")):
        return False
    d = pdate(fields["election_date"])
    if t == "AEP":
        return in_window(d, *policy["election_windows"]["AEP"])
    if t == "OEP":
        return in_window(d, *policy["election_windows"]["OEP"]) and bool(member and member.get("current_plan"))
    if t == "IEP":
        if not valid_date(fields.get("dob", "")):
            return False
        b = pdate(fields["dob"])
        turns65 = b.replace(year=b.year + 65) if not (b.month == 2 and b.day == 29) else b.replace(year=b.year + 65, day=28)
        return abs(months_between(d, turns65)) <= policy["iep_window_months"]
    if t == "SEP":
        return fields.get("sep_reason", "") in policy["sep_reasons"]
    return False


def evaluate(fields, member, ref, policy, skip_duplicate=False):
    """Mock CMS reply. Checks run in a fixed order and the first failure is returned."""
    def res(code, msg=None):
        return {"code": code, "message": msg or CODES[code]}

    if not member:
        return res("REJ-NOENT")
    for k in ("first_name", "last_name", "sex"):
        if norm(fields.get(k)) != norm(member.get(k)):
            return res("REJ-DEMO")
    if fields.get("dob") != member.get("dob"):
        return res("REJ-DEMO")
    if fields.get("part_a_eff") != member.get("part_a_eff") or fields.get("part_b_eff") != member.get("part_b_eff"):
        return res("REJ-ELIG-DATE", "Entitlement dates do not match the master file")
    plan = ref["plans"].get(fields.get("plan_id", ""))
    if not plan:
        return res("REJ-PLAN")
    if ref["zip_to_county"].get(fields.get("zip", "")) != fields.get("county"):
        return res("REJ-ADDR")
    if fields.get("county") not in plan["counties"]:
        return res("REJ-AREA")
    if not skip_duplicate and member.get("current_plan") == fields.get("plan_id"):
        return res("DUP")
    if not election_valid(fields, member, policy):
        return res("REJ-ELECTION")
    derived = derive_effective_date(fields.get("election_type"), fields.get("election_date"), policy)
    if derived is None or fields.get("requested_effective_date") != derived:
        return res("REJ-ELIG-DATE", "Requested effective date is not consistent with the election period")
    if plan["type"] == "HMO" and fields.get("pcp_name") not in plan["network"]:
        return res("REJ-PCP")
    return res("ACC")


def completeness_issues(fields, ref, policy):
    missing = [k for k in policy["required_fields"] if not str(fields.get(k, "")).strip()]
    if str(fields.get("signature", "")).strip().upper() != "Y":
        missing.append("signature")
    plan = ref["plans"].get(fields.get("plan_id", ""))
    if plan and plan["type"] == "HMO" and not str(fields.get("pcp_name", "")).strip():
        missing.append("pcp_name")
    return missing


def validate_change(field, new_value, quote, after, source_text, ref, policy):
    """Guard for one proposed correction. Returns (ok, reason).

    `after` is the full field set with ALL of the proposed changes applied, so rules that
    depend on other fields (effective date vs election type, PCP vs plan) see the final state.
    """
    if field not in policy["fixable_fields"]:
        return False, f"field '{field}' may not be changed by an agent"
    if new_value is None or str(new_value).strip() == "":
        return False, "empty new value"
    if field == "requested_effective_date":
        want = derive_effective_date(after.get("election_type"), after.get("election_date"), policy)
        if want != new_value:
            return False, f"effective date must equal the policy-derived date ({want})"
        return True, "derived from policy"
    if not quote or norm(quote) not in norm(source_text):
        return False, "evidence quote is not present in the source document"
    if field == "plan_id":
        plan = ref["plans"].get(new_value)
        if not plan:
            return False, "plan id is not in the catalog"
        if norm(plan["name"]) not in norm(source_text):
            return False, "the plan's name does not appear in the source document"
        if norm(plan["name"]) not in norm(quote):
            return False, "evidence quote must contain the plan's name"
        return True, "plan name found in source and quoted"
    if field == "pcp_name":
        plan = ref["plans"].get(after.get("plan_id", ""))
        if not plan or new_value not in plan["network"]:
            return False, "provider is not in the plan network"
        if not contains_value(quote, new_value):
            return False, "evidence quote does not contain the provider name"
        return True, "in network and quoted"
    if field == "election_type":
        return (new_value in ELECTION_TYPES), "election type must be AEP, OEP, IEP or SEP"
    if field == "sep_reason":
        return (new_value in policy["sep_reasons"]), "reason must be on the approved SEP list"
    if not contains_value(quote, new_value):
        return False, "the new value does not appear inside the evidence quote"
    return True, "quoted"


def denial_check(code, fields, source_text, ref, policy):
    """Second, deterministic gate on DENY - the one action a model decides by judgement alone.

    Returns (ok, reason). If the documents hint that the member may qualify (a special-election
    cue) the denial is blocked and the case goes to a human. In production every denial should
    get human sign-off; this keyword gate is the simulator's stand-in for that safety net.
    """
    if code == "REJ-AREA":
        plan = ref["plans"].get(fields.get("plan_id", ""))
        if plan and fields.get("county") in plan["counties"]:
            return False, "county is inside the plan's service area"
        return True, "county outside service area"
    if code == "REJ-ELECTION":
        text = norm(source_text)
        hits = [k for k in policy["sep_cue_keywords"] if k in text]
        if hits:
            return False, f"documents mention a possible special-election reason ({', '.join(hits)})"
        return True, "no special-election cue found"
    return False, "denial is not allowed for this code"
