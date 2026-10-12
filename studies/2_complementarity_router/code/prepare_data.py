"""Write the PILOT and HELD-OUT questions (split order) as a MedAgents-format dataset: datasets/MedQA_p2{pilot,heldout}/test.jsonl.
Row j of a file is MedQA test question <split>.json["idx"][j]. No outcomes are read.

  python studies/2_complementarity_router/code/prepare_data.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
D = f"{ROOT}/data/splits"


def main():
    rows = open(f"{D}/MedQA/test.jsonl").read().splitlines()
    for split, name in (("pilot", "MedQA_p2pilot"), ("heldout", "MedQA_p2heldout")):
        idx = json.load(open(f"{ROOT}/studies/2_complementarity_router/results/splits/{split}.json"))["idx"]
        os.makedirs(f"{D}/{name}", exist_ok=True)
        with open(f"{D}/{name}/test.jsonl", "w") as f:
            for i in idx:
                f.write(rows[i] + "\n")
        print(f"{name}: {len(idx)} questions")


if __name__ == "__main__":
    main()
