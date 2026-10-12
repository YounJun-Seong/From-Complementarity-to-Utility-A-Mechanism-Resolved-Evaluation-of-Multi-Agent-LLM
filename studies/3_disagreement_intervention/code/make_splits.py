"""Phase III splits (PHASE3_FREEZE.md §2). Question text only; no outcomes are read.
Pool = MedMCQA validation (4,183, 4-option) minus Part I R2 (MedMCQA_r500) and Phase II MedMCQA-G, after
dropping exact duplicates (normalized question + options). Seed 51.

  python studies/3_disagreement_intervention/code/make_splits.py  ->  studies/3_disagreement_intervention/results/splits/{dev,pilot,final,reserve}.json + SPLITS.sha256
"""
import collections
import hashlib
import json
import os
import random
import re

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
D = f"{ROOT}/data/splits"
OUT = f"{ROOT}/studies/3_disagreement_intervention/results/splits"
SIZES = (("dev", 200), ("pilot", 300), ("final", 1000))


def norm(r):
    return re.sub(r"\s+", " ", (r["question"] + "||" + json.dumps(r["options"], sort_keys=True)).lower()).strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(f"{D}/MedMCQA/test.jsonl")]
    used = {norm(json.loads(l)) for l in open(f"{D}/MedMCQA_r500/test.jsonl")}
    used |= {norm(rows[i]) for i in json.load(open(f"{ROOT}/studies/2_complementarity_router/results/splits/medmcqa_g.json"))["idx"]}
    keys = [norm(r) for r in rows]
    cnt = collections.Counter(keys)
    pool = [i for i, k in enumerate(keys) if k not in used and cnt[k] == 1]
    rng = random.Random(51)
    rng.shuffle(pool)
    start, out = 0, {}
    for name, n in SIZES:
        out[name] = sorted(pool[start:start + n]); start += n
    out["reserve"] = sorted(pool[start:])
    lines = []
    for name, idx in out.items():
        p = f"{OUT}/{name}.json"
        json.dump({"source": "MedMCQA/test.jsonl (validation split)", "idx": idx}, open(p, "w"))
        lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {name}.json")
        print(f"{name:8s} {len(idx)}")
        ds = f"{D}/MedMCQA_p3{name}"
        os.makedirs(ds, exist_ok=True)
        with open(f"{ds}/test.jsonl", "w") as f:            # generation input: no gold fields
            for i in idx:
                r = rows[i]
                f.write(json.dumps({"question": r["question"], "options": r["options"], "qid": i}) + "\n")
        with open(f"{OUT}/{name}_labels.json", "w") as f:    # gold kept apart from generation inputs
            json.dump({str(i): rows[i]["answer_idx"] for i in idx}, f)
    open(f"{OUT}/SPLITS.sha256", "w").write("\n".join(lines) + "\n")
    assert sum(len(v) for v in out.values()) == len(pool) and len(set().union(*map(set, out.values()))) == len(pool)


if __name__ == "__main__":
    main()
