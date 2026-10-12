"""Phase V data (PHASE5_FREEZE.md §1). Eval inputs = released PubMedQA test file (pubid, question, contexts; no gold);
labels kept apart. Token probe = 30 PQA-L questions not among the 498 (sorted pubids, seed 61). Run with a Python
that has pandas + huggingface_hub.

  python make_data.py <MedARC repo dir>
"""
import hashlib
import json
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUT = f"{ROOT}/studies/5_medarc_transfer/results/data"


def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(sys.argv[1], "data", "pubmedqa_test.jsonl")
    rows = [json.loads(l) for l in open(src) if l.strip()]
    with open(f"{OUT}/eval_inputs.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps(dict(idx=i, pubid=r["pubid"], question=r["question"], contexts=r["context"]["contexts"])) + "\n")
    json.dump({str(r["pubid"]): r["final_decision"] for r in rows}, open(f"{ROOT}/studies/5_medarc_transfer/results/labels.json", "w"))
    test_ids = {r["pubid"] for r in rows}
    df = pd.read_parquet(hf_hub_download("qiaojin/PubMedQA", "pqa_labeled/train-00000-of-00001.parquet", repo_type="dataset"))
    pool = sorted(int(p) for p in df["pubid"] if int(p) not in test_ids)
    pick = set(random.Random(61).sample(pool, 30))
    with open(f"{OUT}/probe_inputs.jsonl", "w") as f:
        k = 0
        for _, r in df.sort_values("pubid").iterrows():
            if int(r["pubid"]) in pick:
                f.write(json.dumps(dict(idx=k, pubid=int(r["pubid"]), question=r["question"],
                                        contexts=list(r["context"]["contexts"]))) + "\n"); k += 1
    lines = [f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, ROOT)}" for p in
             (src, f"{OUT}/eval_inputs.jsonl", f"{ROOT}/studies/5_medarc_transfer/results/labels.json", f"{OUT}/probe_inputs.jsonl")]
    open(f"{OUT}/DATA.sha256", "w").write("\n".join(lines) + "\n")
    print(f"eval {len(rows)}, probe {k}, PQA-L pool outside test {len(pool)}")


if __name__ == "__main__":
    main()
