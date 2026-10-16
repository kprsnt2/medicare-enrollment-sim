import json,sys
cases=json.load(open("state/cases.json"))
gold=json.load(open("audit/gold/gold.json"))
ids=set()
for cid,c in cases.items():
    g=gold.get(cid)
    if not g: continue
    if c["state"] in ("PENDING_MEMBER","ESCALATED","IN_BASKET") and g["defect"] in ("clean","duplicate","elig_date","zip_typo"): ids.add(cid)
    if c["state"]=="ACCEPTED":
        for k in ("first_name","last_name","dob","sex","mbi","part_a_eff","part_b_eff","zip","plan_id","pcp_name","election_type","sep_reason","requested_effective_date"):
            if str(c["fields"].get(k)).lower()!=str(g["true_fields"].get(k)).lower(): ids.add(cid); break
out=[]
CRIT=("first_name","last_name","dob","sex","mbi","part_a_eff","part_b_eff","zip","plan_id","pcp_name","election_type","sep_reason","requested_effective_date")
for cid,c in sorted(cases.items()):
    g=gold.get(cid)
    if g and c["state"]=="ACCEPTED":
        for k in CRIT:
            if str(c["fields"].get(k)).strip().lower()!=str(g["true_fields"].get(k)).strip().lower():
                out.append(f'MISMATCH {cid} ({g["defect"]}) {k}: stored={c["fields"].get(k)!r} correct={g["true_fields"].get(k)!r} original={c["original_fields"].get(k)!r}')
for l in open("audit/ledger.jsonl",encoding="utf-8"):
    e=json.loads(l)
    if e["case_id"] in ids and e["action"] in ("intake_parsed","to_basket","decision","fix_applied","action_rejected","rfi_requested","llm:paper_intake","llm_failed","llm_error:paper_intake"):
        out.append(f'{e["case_id"]} #{e["seq"]} {e["agent"]} {e["action"]} {json.dumps(e["data"])[:700]}')
open("why_cases.txt","w",encoding="utf-8").write("\n".join(out))
print(len(ids),"cases,",len(out),"lines")
