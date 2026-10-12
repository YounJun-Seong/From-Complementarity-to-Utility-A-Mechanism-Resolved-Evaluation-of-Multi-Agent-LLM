"""Frozen V_C router policy (PHASE2_FREEZE.md §6, Amendment 2). Applies studies/2_complementarity_router/results/router/router_frozen.json.

  q_C(x) = P[collaboration is utility-positive over the better cheap action]   (logistic regression)
  q_A(x) = P[SAMPLE is utility-positive over STOP]                             (logistic regression)
  action = COLLABORATE if q_C > 0.5, else SAMPLE if q_A > 0.5, else STOP       (ties at exactly 0.5 -> cheaper)
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
FROZEN = f"{ROOT}/studies/2_complementarity_router/results/router/router_frozen.json"
STOP, SAMPLE, COLLABORATE = 0, 1, 2


def load(path=FROZEN):
    return json.load(open(path))


def _prob(p, X):
    z = (X - np.array(p["mean"])) / np.array(p["scale"]) @ np.array(p["coef"]) + p["intercept"]
    return 1 / (1 + np.exp(-z))


def act(frozen, X):
    qC, qA = _prob(frozen["params"]["q_C"], X), _prob(frozen["params"]["q_A"], X)
    return np.where(qC > 0.5, COLLABORATE, np.where(qA > 0.5, SAMPLE, STOP)), qC, qA
