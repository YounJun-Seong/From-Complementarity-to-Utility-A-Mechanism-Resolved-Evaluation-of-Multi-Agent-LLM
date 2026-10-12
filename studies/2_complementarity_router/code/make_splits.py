"""Phase II splits (PHASE2_FREEZE.md §3.1). Uses question text only; no outcomes are read.

  python studies/2_complementarity_router/code/make_splits.py   ->  studies/2_complementarity_router/results/splits/{dev,pilot,heldout,medmcqa_g}.json
"""
import json
import os
import random

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
D = f"{ROOT}/data/splits"
OUT = f"{ROOT}/studies/2_complementarity_router/results/splits"


def load(path):
    return [json.loads(l) for l in open(path)]


def main():
    os.makedirs(OUT, exist_ok=True)
    medqa = load(f"{D}/MedQA/test.jsonl")
    pos = {r["question"]: i for i, r in enumerate(medqa)}
    dev = set(range(200))
    for s in "ABC":
        dev |= {pos[r["question"]] for r in load(f"{D}/MedQA_rand{s}/test.jsonl")}
    fresh = sorted(set(range(len(medqa))) - dev)
    pilot = sorted(random.Random(31).sample(fresh, 300))
    heldout = sorted(set(fresh) - set(pilot))

    mcqa = load(f"{D}/MedMCQA/test.jsonl")
    mpos = {r["question"]: i for i, r in enumerate(mcqa)}
    r500 = {mpos[r["question"]] for r in load(f"{D}/MedMCQA_r500/test.jsonl")}
    medmcqa_g = sorted(random.Random(41).sample(sorted(set(range(len(mcqa))) - r500), 500))

    for name, idx in [("dev", sorted(dev)), ("pilot", pilot), ("heldout", heldout), ("medmcqa_g", medmcqa_g)]:
        json.dump({"source": "MedMCQA/test.jsonl" if name == "medmcqa_g" else "MedQA/test.jsonl",
                   "idx": idx}, open(f"{OUT}/{name}.json", "w"))
        print(f"{name:10s} {len(idx)}")
    assert not (set(pilot) & set(heldout)) and not (dev & set(pilot)) and not (dev & set(heldout))


if __name__ == "__main__":
    main()
