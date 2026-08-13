import json
from rag import retrieve

GOLD = {
    "P1": "dimensional-weight", "P2":"facility-directory",
    "P3": "declared-value-insurance", "P4": " hazmat-restrictions",
    "P5": "claim-eligibilty", "P6": "customs-documentation",
    "P7": "fuel-surchange", "P8" : None,
}

qs = json.load(open("questions.json"))
print(f"{'id':4}{'gold@1':7} {'in top5':8} {'top1':>7}{'margin':>8} top1 doc")

hits1 = hits5 = total = 0
for q in qs:
    hits = retrieve(q["question"], 5)
    docs = [c.doc for c, _ in hits]
    scores = [s for _, s in hits]
    g = GOLD[q["id"]]
    if g: 
        total +=1
        hits1 += docs[0] == g
        hits5 += g in docs 
    at1 = str(docs[0] == g) if g else "n/a"
    at5 = str(g in docs) if g else "n/a"
    print (f"{q['id']:4}  {at1:7} {scores[0]:7.2f} { scores[0]- scores[1]:8.2f} {docs[0]}")

print(f"\n gold doc at rank 1:{hits1}/{total} in top 5: {hits5}/{total}")