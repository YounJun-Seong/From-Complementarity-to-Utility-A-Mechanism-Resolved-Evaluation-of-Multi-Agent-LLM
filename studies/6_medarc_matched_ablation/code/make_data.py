"""Phase VI data (PHASE6_FREEZE.md §1): PQA-L minus the Phase V eval (498) and probe (30) pubids -> 472 questions.
Inputs (pubid, question, contexts; no gold) and labels kept apart."""
import hashlib
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUT = f"{ROOT}/studies/6_medarc_matched_ablation/results/data"


def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(hf_hub_download("qiaojin/PubMedQA", "pqa_labeled/train-00000-of-00001.parquet", repo_type="dataset"))
    used = {json.loads(l)["pubid"] for f in ("eval_inputs", "probe_inputs") for l in open(f"{ROOT}/studies/5_medarc_transfer/results/data/{f}.jsonl")}
    rows = [r for r in df.sort_values("pubid").itertuples() if int(r.pubid) not in used]
    assert len(rows) == 472
    with open(f"{OUT}/inputs.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps(dict(idx=i, pubid=int(r.pubid), question=r.question, contexts=list(r.context["contexts"]))) + "\n")
    json.dump({str(int(r.pubid)): r.final_decision for r in rows}, open(f"{ROOT}/studies/6_medarc_matched_ablation/results/labels.json", "w"))
    lines = [f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, ROOT)}"
             for p in (f"{OUT}/inputs.jsonl", f"{ROOT}/studies/6_medarc_matched_ablation/results/labels.json")]
    open(f"{OUT}/DATA.sha256", "w").write("\n".join(lines) + "\n")
    print(len(rows))


if __name__ == "__main__":
    main()
