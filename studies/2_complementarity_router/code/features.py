"""Pre-decision router features (PHASE2_FREEZE.md §3.4). Only S run 1, the 4-sample probe (first four
samples of SC run 1) and the question text are used; no gold label, no M1 output.

  build(s_rows, probe_votes, idx, dataset) -> (Q x 8) array, column names in NAMES
"""
import json
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
NAMES = ["extractors_agree", "s_truncated", "log_reasoning_chars",
         "probe_entropy", "probe_max_share", "probe_agrees_with_S",
         "log_question_tokens", "n_numeric_with_units"]
UNIT_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s?(?:mm\s?Hg|mg/dL|mEq/L|mmol/L|g/dL|U/L|IU/L|/mm3|/mm\^3|/min|bpm|"
    r"mg|mcg|μg|µg|kg|g|mL|dL|L|cm|mm|°C|°F|%)(?![A-Za-z])")
_QTOK = None


def question_tokens(dataset):
    global _QTOK
    if _QTOK is None:
        _QTOK = json.load(open(f"{ROOT}/studies/2_complementarity_router/results/features/question_tokens.json"))
    return _QTOK[dataset]


def entropy(votes):
    _, c = np.unique(votes, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def majority(votes):
    labs, first, cnt = np.unique(votes, return_index=True, return_counts=True)
    top = np.flatnonzero(cnt == cnt.max())
    return labs[top[np.argmin(first[top])]]


def build(s_rows, probe_votes, idx, questions, dataset="MedQA"):
    """s_rows: S run-1 records (fields E1, E2, E3, finish_reason, reasoning_chars) aligned with idx;
    probe_votes: (Q x 4) answers; questions: question texts aligned with idx."""
    qtok = question_tokens(dataset)
    X = []
    for r, pv, i, q in zip(s_rows, probe_votes, idx, questions):
        pv = np.asarray(pv)
        X.append([
            float(r["E1"] == r["E2"] == r["E3"]),
            float(r["finish_reason"] == "length"),
            np.log1p(r["reasoning_chars"]),
            entropy(pv),
            float(np.unique(pv, return_counts=True)[1].max() / len(pv)),
            float(majority(pv) == r["E3"]),
            np.log(qtok[i]),
            float(len(UNIT_RE.findall(q))),
        ])
    return np.array(X)
