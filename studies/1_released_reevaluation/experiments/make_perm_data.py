"""Layer 2E data: MedQA 5-option test with option contents permuted (seeded), gold remapped.
Writes external/MedAgents/datasets/MedQA_perm<p>/test.jsonl for p = 1..3 (same question order)."""
import json
import os
import random

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = f"{ROOT}/external/MedAgents/datasets/MedQA/test.jsonl"
QA = [json.loads(l) for l in open(SRC)]
for p in (1, 2, 3):
    rng = random.Random(1000 + p)
    out = f"{ROOT}/external/MedAgents/datasets/MedQA_perm{p}"
    os.makedirs(out, exist_ok=True)
    moved = 0
    with open(f"{out}/test.jsonl", "w") as f:
        for q in QA:
            keys = sorted(q["options"]); vals = [q["options"][k] for k in keys]
            order = list(range(len(vals))); rng.shuffle(order)
            new = {keys[j]: vals[order[j]] for j in range(len(keys))}
            gold = keys[order.index(keys.index(q["answer_idx"]))]
            assert new[gold] == q["options"][q["answer_idx"]]
            moved += gold != q["answer_idx"]
            f.write(json.dumps({**q, "options": new, "answer_idx": gold, "answer": new[gold]}) + "\n")
    print(f"perm{p}: gold letter changed for {moved}/{len(QA)} questions")
