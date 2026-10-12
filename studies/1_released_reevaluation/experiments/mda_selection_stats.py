"""Question-clustered tests of the MDAgents selection / coordination findings (no API calls).
Difficulty proxy = strong single S accuracy per question (mean over 5 runs, questions 0–199).
All CIs resample QUESTIONS (10k, seed 0) and keep every MDAgents run of a resampled question.

 (1) selection:  question-level discussion frequency (share of the 5 runs that were intermediate,
     completed, and actually discussed = not noAns) vs S accuracy — Spearman rho, cluster CI.
 (2) interaction: gap(MDAgents − S) on discussion units minus gap on no-discussion intermediate
     units — cluster-bootstrap CI.
 (3) routing: question-level mean route level (basic 0 / intermediate 1 / advanced 2) vs
     S accuracy — Spearman rho with cluster CI; S accuracy by modal route with cluster CIs.
 Also a difficulty-bin table and a per-question CSV for the figure.

  python experiments/mda_selection_stats.py F2_final_v2 F3_final_v2
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from mda_analyze import extract, load_runs   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.default_rng(0)
B = 10000
S_RUNS = np.array([[int(json.loads(l)["E3"] == json.loads(l)["gold"]) for l in open(f"{ROOT}/runs/mda/S5/run{r}.jsonl")]
                   for r in range(1, 6)])
S = S_RUNS.mean(0)
LEVEL = {"basic": 0, "intermediate": 1, "advanced": 2}


def ci(x):
    return np.percentile(x, [2.5, 97.5])


def analyze(cond):
    runs = list(load_runs(cond).values())
    Q = 200
    disc = np.zeros((len(runs), Q)); inter_nd = np.zeros((len(runs), Q)); corr = np.zeros((len(runs), Q))
    level = np.full((len(runs), Q), np.nan)
    for k, rows in enumerate(runs):
        for i in range(Q):
            r = rows[i]
            ok = r["error"] is None
            corr[k, i] = ok and extract(r["response"])[0] == r["label"]
            if r["difficulty"] in LEVEL:
                level[k, i] = LEVEL[r["difficulty"]]
            if ok and r["difficulty"] == "intermediate":
                disc[k, i] = not r["noAns"]
                inter_nd[k, i] = r["noAns"]
    dfreq = disc.mean(0); rlev = np.nanmean(level, 0)

    def interaction(idx):
        d, n, c = disc[:, idx].ravel().astype(bool), inter_nd[:, idx].ravel().astype(bool), corr[:, idx].ravel()
        s = np.tile(S[idx], len(runs))
        if d.sum() == 0 or n.sum() == 0:
            return np.nan, np.nan, np.nan
        gd = c[d].mean() - s[d].mean(); gn = c[n].mean() - s[n].mean()
        return gd, gn, gd - gn

    boot_rho_d, boot_rho_r, boot_int = [], [], []
    for _ in range(B):
        idx = RNG.integers(0, Q, Q)
        boot_rho_d.append(stats.spearmanr(S[idx], dfreq[idx]).statistic)
        boot_rho_r.append(stats.spearmanr(S[idx], rlev[idx]).statistic)
        boot_int.append(interaction(idx))
    boot_int = np.array(boot_int)
    rho_d = stats.spearmanr(S, dfreq).statistic; rho_r = stats.spearmanr(S, rlev).statistic
    gd, gn, gi = interaction(np.arange(Q))
    print(f"=== {cond} (pre-registered scorer; crash = wrong; CIs cluster-bootstrapped over questions) ===")
    print(f"(1) discussion frequency vs S accuracy: Spearman rho = {rho_d:+.3f} [{ci(boot_rho_d)[0]:+.3f}, {ci(boot_rho_d)[1]:+.3f}]")
    print(f"(2) gap with discussion {gd:+.3f} [{ci(boot_int[:, 0])[0]:+.3f}, {ci(boot_int[:, 0])[1]:+.3f}] | "
          f"gap without (intermediate noAns) {gn:+.3f} [{ci(boot_int[:, 1])[0]:+.3f}, {ci(boot_int[:, 1])[1]:+.3f}]")
    print(f"    difference (discussion − no discussion) = {gi:+.3f} [{np.nanpercentile(boot_int[:, 2], 2.5):+.3f}, "
          f"{np.nanpercentile(boot_int[:, 2], 97.5):+.3f}]")
    print(f"(3) mean route level vs S accuracy: Spearman rho = {rho_r:+.3f} [{ci(boot_rho_r)[0]:+.3f}, {ci(boot_rho_r)[1]:+.3f}]")
    modal = np.array([np.bincount(level[:, i][~np.isnan(level[:, i])].astype(int), minlength=3).argmax() for i in range(Q)])
    for name, lv in LEVEL.items():
        m = modal == lv
        if m.sum() < 2:
            print(f"    modal route {name:12s} n_q={m.sum():3d}"); continue
        bs = [S[np.flatnonzero(m)][RNG.integers(0, m.sum(), m.sum())].mean() for _ in range(2000)]
        print(f"    modal route {name:12s} n_q={m.sum():3d}  S acc {S[m].mean():.3f} [{ci(bs)[0]:.3f}, {ci(bs)[1]:.3f}]")
    print("    difficulty bins (by S accuracy over 5 runs):")
    print(f"    {'bin':22s} {'n_q':>4s} {'disc rate':>9s} {'S acc':>6s} {'MDA acc':>7s} {'gap':>7s}")
    for lab, m in (("hard   (S <= 0.4)", S <= 0.4), ("medium (0.4 < S < 1)", (S > 0.4) & (S < 1)), ("easy   (S = 1)", S == 1)):
        print(f"    {lab:22s} {m.sum():4d} {dfreq[m].mean():9.3f} {S[m].mean():6.3f} {corr[:, m].mean():7.3f} "
              f"{corr[:, m].mean() - S[m].mean():+7.3f}")
    # regression-to-the-mean guard: bin with S runs 1-2, evaluate S on held-out runs 3-5
    Sbin, Sev = S_RUNS[:2].mean(0), S_RUNS[2:].mean(0)
    print("    held-out bins (bin: S runs 1-2; S acc: runs 3-5):")
    for lab, m in (("hard (both wrong)", Sbin == 0), ("split", Sbin == 0.5), ("easy (both right)", Sbin == 1)):
        print(f"    {lab:22s} {m.sum():4d} {dfreq[m].mean():9.3f} {Sev[m].mean():6.3f} {corr[:, m].mean():7.3f} "
              f"{corr[:, m].mean() - Sev[m].mean():+7.3f}")
    out = f"{ROOT}/runs/mda/selection_{cond}.csv"
    with open(out, "w") as f:
        f.write("idx,S_acc,discussion_freq,mean_route_level,mda_acc\n")
        for i in range(Q):
            f.write(f"{i},{S[i]:.3f},{dfreq[i]:.3f},{rlev[i]:.3f},{corr[:, i].mean():.3f}\n")
    print(f"    per-question data -> {os.path.relpath(out, ROOT)}\n")


for c in sys.argv[1:] or ["F2_final_v2", "F3_final_v2"]:
    analyze(c)
