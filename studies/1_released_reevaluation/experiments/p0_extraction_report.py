"""P0-5 report: extractor agreement breakdown on runs/p0_extraction.jsonl (no API calls)."""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    df = pd.DataFrame([json.loads(l) for l in open(f"{ROOT}/runs/p0_extraction.jsonl")])
    n = len(df)
    ok = df["E1"] != ""
    explicit = df["E1_src"].isin(["explicit", "bold"])
    print(f"n={n}; reasoning finish=length {int((df['finish_reason'] == 'length').sum())}")
    print(f"E1 success {ok.sum()}/{n} ({ok.mean():.1%}); source {df['E1_src'].value_counts().to_dict()}")
    print(f"no explicit letter choice in reasoning (option_text/none): {(~explicit).sum()}")
    for c in ("E2", "E3", "E3f"):
        dis = ok & (df[c] != df["E1"])
        print(f"E1 <-> {c:3s} disagreement: {dis.sum():3d} ({dis.sum() / ok.sum():.2%} of E1-parsed); "
              f"on explicit-letter cases {int((dis & explicit).sum())}")
    r3, rf = df["E3"] == df["gold"], df["E3f"] == df["gold"]
    e = df["gold"] == "E"
    print(f"\nE3 -> E3f accuracy: all {r3.mean():.4f} -> {rf.mean():.4f} ({rf.mean() - r3.mean():+.4f}); "
          f"gold=E (n={e.sum()}) {r3[e].mean():.4f} -> {rf[e].mean():.4f}; "
          f"gold!=E {r3[~e].mean():.4f} -> {rf[~e].mean():.4f}")
    print(f"E3 wrong -> E3f right: {int((~r3 & rf).sum())};  E3 right -> E3f wrong: {int((r3 & ~rf).sum())}")
    eE = explicit & (df["E1"] == "E")
    print(f"\nexplicit answer E (n={eE.sum()}): E3 picks non-E {int((eE & (df['E3'] != 'E')).sum())}, "
          f"E3f non-E {int((eE & (df['E3f'] != 'E')).sum())}, E2 non-E {int((eE & (df['E2'] != 'E')).sum())}")
    print(f"prediction rate of E: E1 {np.mean(df['E1'] == 'E'):.3f} E2 {np.mean(df['E2'] == 'E'):.3f} "
          f"E3 {np.mean(df['E3'] == 'E'):.3f} E3f {np.mean(df['E3f'] == 'E'):.3f}  gold {e.mean():.3f}")
    cols = ["idx", "gold", "E1", "E1_src", "E2", "E3", "E3f", "finish_reason"]
    bad = df[ok & ((df["E3"] != df["E1"]) | (df["E3f"] != df["E1"]) | (df["E2"] != df["E1"]))]
    if len(bad):
        print("\ndisagreements:\n" + bad[cols].to_string(index=False))


if __name__ == "__main__":
    main()
