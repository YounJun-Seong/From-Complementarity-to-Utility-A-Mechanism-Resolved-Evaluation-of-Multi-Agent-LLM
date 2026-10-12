"""Figure: Layer 2A capability curve. x = strong-single accuracy (S), y = M1 − S with 95% CI,
one point per Qwen2.5 model; direct labels carry size and M1 crash rate. Reads summary.csv.

  python experiments/fig_2a_capability.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SERIES, INK, INK2, MUTED, GRID, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

rows = list(csv.DictReader(open(f"{ROOT}/runs/layer2/2A/summary.csv")))
x = [float(r["S_acc"]) for r in rows]
y = [float(r["gap"]) for r in rows]
lo = [float(r["gap"]) - float(r["ci_lo"]) for r in rows]
hi = [float(r["ci_hi"]) - float(r["gap"]) for r in rows]

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.axhline(0, color=INK2, lw=1, zorder=1)
ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=SERIES, elinewidth=2, capsize=0, zorder=2)
ax.plot(x, y, "o", ms=8, color=SERIES, mec=SURF, mew=2, zorder=3)
for r, xi, yi in zip(rows, x, y):
    size = r["model"].split("-")[1]
    crash = float(r["crash"])
    ax.annotate(f"{size}" + (f"\ncrash {crash:.1%}" if crash >= 0.005 else ""), (xi, yi),
                xytext=(8, -3), textcoords="offset points", fontsize=8, color=INK2, va="top")
ax.set_xlabel("Strong single-agent accuracy (S)", color=INK2)
ax.set_ylabel("Multi-agent gain (M1 − S accuracy)", color=INK2)
ax.set_title("Qwen2.5-Instruct, MedQA 0–199, 3 runs", fontsize=9, color=INK, loc="left")
ax.set_xlim(0.30, 0.82); ax.set_ylim(-0.12, 0.08)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED)
fig.tight_layout()
out = f"{ROOT}/runs/layer2/2A/fig_capability"
fig.savefig(out + ".png"); fig.savefig(out + ".pdf")
print("saved", out + ".png/.pdf")
