"""Synthetic applicant generator (the 'applicant' side of the simulation).

Deterministic for a given seed/date/index, so runs are reproducible and cost nothing.
Every application gets a hidden `gold` record (what SHOULD happen) that only the evaluation
step ever reads - no agent sees it. All people, IDs and plans are fictional ("SYN" prefixed).
"""
import random
from datetime import date, timedelta

from .util import derive_effective_date, first_of_next_month, in_window, pdate

FIRST = ["John", "Mary", "Robert", "Patricia", "James", "Linda", "Michael", "Barbara", "William", "Susan",
         "David", "Karen", "Richard", "Nancy", "Joseph", "Betty", "Thomas", "Helen", "Charles", "Dorothy"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Martinez", "Hernandez",
        "Lopez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson"]
STREETS = ["Oak St", "Maple Ave", "Pine Rd", "Cedar Ln", "Elm St", "Lakeview Dr", "Hill St", "Park Ave"]
ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
DEFECTS = [("clean", .30), ("demo_mismatch", .10), ("elig_date", .10), ("zip_typo", .07), ("out_of_area", .05),
           ("wrong_plan", .07), ("election_invalid", .07), ("pcp_mismatch", .06), ("duplicate", .04),
           ("mbi_typo", .04), ("missing_signature", .05), ("ocr_noise", .05)]
FIELDS = ["first_name", "last_name", "dob", "sex", "mbi", "part_a_eff", "part_b_eff", "address", "city", "state",
          "zip", "county", "phone", "plan_name", "plan_id", "pcp_name", "election_type", "sep_reason",
          "election_date", "requested_effective_date", "signature"]
CONF = {"0": "O", "1": "l", "5": "S", "8": "B", "2": "Z"}          # digit -> look-alike letter
CUES = {
    "MOVED_TO_NEW_SERVICE_AREA": "I moved to this county last month.",
    "LOST_EMPLOYER_COVERAGE": "My employer coverage ended last month.",
    "PLAN_CONTRACT_ENDED": "My old plan was discontinued.",
}
FILLER = ["Please send materials in large print.", "I prefer morning phone calls.", "Thank you for your help."]


def _weighted(rng, pairs):
    r, acc = rng.random(), 0.0
    for k, w in pairs:
        acc += w
        if r < acc:
            return k
    return pairs[-1][0]


def _swap_adjacent(s, rng):
    idx = [i for i in range(len(s) - 1) if s[i].lower() != s[i + 1].lower()]
    if not idx:
        return s + "e"
    i = rng.choice(idx)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def _ocr_noise(value, rng):
    chars = list(value)
    spots = [i for i, c in enumerate(chars) if c in CONF]
    if not spots:
        return value
    picks = [i for i in spots if rng.random() < 0.5] or [rng.choice(spots)]
    for i in picks:
        chars[i] = CONF[chars[i]]
    return "".join(chars)


def card_text(t):
    return (f"{t['last_name'].upper()} {t['first_name'].upper()} / DOB {t['dob']} / SEX {t['sex']} / "
            f"MBI {t['mbi']} / PART A {t['part_a_eff']} / PART B {t['part_b_eff']}")


def render_paper(app, notes, att, noise=None, rng=None):
    def n(k):
        return _ocr_noise(app[k], rng) if noise and k in noise else app[k]
    lines = [
        "MEDICARE ADVANTAGE ENROLLMENT REQUEST FORM (SYNTHETIC - NOT A REAL FORM)",
        f"Received: {n('election_date')}",
        f"Name: {app['last_name'].upper()}, {app['first_name'].upper()}",
        f"DOB: {n('dob')}   Sex: {app['sex']}",
        f"Medicare No: {n('mbi')}",
        f"Part A effective: {n('part_a_eff')}   Part B effective: {n('part_b_eff')}",
        f"Address: {app['address']}",
        f"City: {app['city']}   State: {app['state']}   ZIP: {n('zip')}   County: {app['county']}",
        f"Phone: {app['phone']}",
        f"Plan selected: {app['plan_name']} ({app['plan_id']})",
        f"PCP: {app['pcp_name'] or '-'}",
        f"Election: {app['election_type']}   Sep reason: {app['sep_reason'] or '-'}   "
        f"Signed: {'Yes' if app['signature'] == 'Y' else 'No'}",
        f"Requested effective date: {app['requested_effective_date']}",
        f"Notes: {notes or '-'}",
    ]
    if att.get("card_text"):
        lines.append(f"Attachment A (Medicare card copy): {att['card_text']}")
    if att.get("address_proof"):
        lines.append(f"Attachment B (proof of address): {att['address_proof']}")
    return "\n".join(lines)


def render_digital_text(payload):
    f = payload["form"]
    lines = [f"{k}: {f.get(k, '')}" for k in FIELDS]
    lines.append(f"Notes: {payload.get('notes') or '-'}")
    att = payload.get("attachments", {})
    if att.get("card_text"):
        lines.append(f"Attachment A (Medicare card copy): {att['card_text']}")
    if att.get("address_proof"):
        lines.append(f"Attachment B (proof of address): {att['address_proof']}")
    return "\n".join(lines)


def generate_case(sim_date, idx, S, ref, pol):
    rng = random.Random(f"{S['seed']}|{sim_date}|{idx}")
    D = pdate(sim_date)
    plans, z2c, cinfo = ref["plans"], ref["zip_to_county"], ref["county_info"]
    channel = "paper" if rng.random() < S["paper_share"] else "digital"
    defect = _weighted(rng, DEFECTS)
    if defect == "ocr_noise" and channel != "paper":
        defect = "clean"

    county = rng.choice(sorted(set(z2c.values())))
    zipc = rng.choice(sorted(z for z, c in z2c.items() if c == county))
    in_county = sorted(p for p, v in plans.items() if county in v["counties"])
    out_county = sorted(p for p, v in plans.items() if county not in v["counties"])
    hmos = [p for p in in_county if plans[p]["type"] == "HMO"]
    if defect == "pcp_mismatch" and not hmos:
        defect = "clean"
    if defect == "out_of_area" and not out_county:
        defect = "clean"
    pool = {"pcp_mismatch": hmos, "out_of_area": out_county}.get(defect, in_county)
    plan_id = rng.choice(pool)
    plan = plans[plan_id]

    iep = defect != "election_invalid" and rng.random() < 0.15
    if iep:
        b65 = D + timedelta(days=rng.randint(-60, 75))
        dob = date(b65.year - 65, b65.month, min(b65.day, 28))
    else:
        dob = date(rng.randint(1930, 1958), rng.randint(1, 12), rng.randint(1, 28))
    ent = date(dob.year + 65, dob.month, 1)

    sep = ""
    if iep:
        etype = "IEP"
    elif in_window(D, *pol["election_windows"]["AEP"]):
        etype = "AEP"
    elif in_window(D, *pol["election_windows"]["OEP"]):
        etype = "OEP"
    else:
        etype, sep = "SEP", rng.choice(pol["sep_reasons"])

    current = None
    if etype == "OEP":
        current = rng.choice(sorted(p for p in plans if p != plan_id))
    if defect == "duplicate":
        current = plan_id

    first, last = rng.choice(FIRST), rng.choice(LAST)
    true = {
        "first_name": first, "last_name": last, "dob": dob.isoformat(), "sex": rng.choice(["M", "F"]),
        "mbi": "SYN" + "".join(rng.choice(ALPHA) for _ in range(7)),
        "part_a_eff": ent.isoformat(), "part_b_eff": ent.isoformat(),
        "address": f"{rng.randint(10, 999)} {rng.choice(STREETS)}", "city": cinfo[county][0],
        "state": cinfo[county][1], "zip": zipc, "county": county,
        "phone": f"{rng.choice(['512', '713', '602', '312', '305'])}-555-{rng.randint(0, 9999):04d}",
        "plan_name": plan["name"], "plan_id": plan_id,
        "pcp_name": rng.choice(plan["network"]) if plan["type"] == "HMO" else "",
        "election_type": etype, "sep_reason": sep, "election_date": sim_date,
        "requested_effective_date": derive_effective_date(etype, sim_date, pol), "signature": "Y",
    }
    member = {k: true[k] for k in ("first_name", "last_name", "dob", "sex", "mbi", "part_a_eff", "part_b_eff")}
    member["current_plan"] = current

    app, notes, noise, variant = dict(true), rng.choice(FILLER) if rng.random() < .3 else "", None, None
    has_card = rng.random() < (0.7 if defect in ("demo_mismatch", "mbi_typo") else 0.85)
    has_proof = rng.random() < (0.7 if defect == "zip_typo" else 0.85)
    expected = "ACCEPTED"

    if defect == "demo_mismatch":
        which = rng.choice(["last_name", "dob", "sex"])
        variant = which
        if which == "last_name":
            app["last_name"] = _swap_adjacent(true["last_name"], rng)
        elif which == "dob":
            day = int(true["dob"][8:])
            app["dob"] = true["dob"][:8] + f"{day + 1 if day < 28 else day - 1:02d}"
        else:
            app["sex"] = "F" if true["sex"] == "M" else "M"
        expected = "ACCEPTED" if has_card else "CLOSED_INCOMPLETE"
    elif defect == "elig_date":
        variant = rng.choice(["entitlement", "effective"])
        if variant == "entitlement":
            app["part_b_eff"] = ent.replace(year=ent.year + 1).isoformat()
            has_card = rng.random() < 0.7
            expected = "ACCEPTED" if has_card else "CLOSED_INCOMPLETE"
        else:
            app["requested_effective_date"] = D.replace(day=1).isoformat()
    elif defect == "zip_typo":
        app["zip"] = rng.choice(sorted(z for z, c in z2c.items() if c != county))
        expected = "ACCEPTED" if has_proof else "CLOSED_INCOMPLETE"
    elif defect == "out_of_area":
        expected = "DENIED"
    elif defect == "wrong_plan":
        app["plan_id"] = plan_id[:-1] + "X"
    elif defect == "election_invalid":
        claim = "OEP" if in_window(D, *pol["election_windows"]["AEP"]) else "AEP"
        app["election_type"], app["sep_reason"] = claim, ""
        app["requested_effective_date"] = derive_effective_date(claim, sim_date, pol)
        if rng.random() < 0.4:
            reason = rng.choice(sorted(CUES))
            notes = CUES[reason]
            true["election_type"], true["sep_reason"] = "SEP", reason
            true["requested_effective_date"] = derive_effective_date("SEP", sim_date, pol)
            variant = "cue"
        else:
            expected, variant = "DENIED", "no_cue"
    elif defect == "pcp_mismatch":
        parts = true["pcp_name"].split()
        parts[-1] = _swap_adjacent(parts[-1], rng)
        app["pcp_name"] = " ".join(parts)
        if rng.random() < 0.6:
            notes = f"My regular doctor is {true['pcp_name']} at the clinic."
            variant = "cue"
        else:
            expected, variant = "CLOSED_INCOMPLETE", "no_cue"
    elif defect == "duplicate":
        expected = "CLOSED_DUPLICATE"
    elif defect == "mbi_typo":
        wrong = rng.choice([c for c in ALPHA if c != true["mbi"][-1]])
        app["mbi"] = true["mbi"][:-1] + wrong
        expected = "ACCEPTED" if has_card else "CLOSED_INCOMPLETE"
    elif defect == "missing_signature":
        app["signature"], expected = "N", "CLOSED_INCOMPLETE"
    elif defect == "ocr_noise":
        noise = rng.sample(["dob", "zip", "mbi"], 2)
        has_card = has_proof = True

    att = {}
    if has_card:
        att["card_text"] = card_text(true)
    if has_proof:
        att["address_proof"] = f"{true['address']} {true['city']} {true['state']} {true['zip']}"

    item = {
        "case_id": f"APP-{sim_date.replace('-', '')}-{idx:03d}", "channel": channel,
        "member": member,
        "gold": {"defect": defect, "variant": variant, "expected_final": expected,
                 "true_fields": {k: true[k] for k in FIELDS}},
    }
    if channel == "paper":
        item["raw"] = render_paper(app, notes, att, noise, rng)
        item["source_text"] = item["raw"]
    else:
        item["payload"] = {"form": {k: app[k] for k in FIELDS}, "notes": notes, "attachments": att}
        item["source_text"] = render_digital_text(item["payload"])
    return item


def generate_day(sim_date, n, S, ref, pol):
    return [generate_case(sim_date, i + 1, S, ref, pol) for i in range(int(n))]
