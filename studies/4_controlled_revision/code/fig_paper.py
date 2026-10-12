"""Paper version of the Phase IV figure from the sealed RESULTS.json (no recomputation).
Control → Blinded critique → Opponent claim is the intervention ladder (connected); the matched irrelevant
claim is a separate robustness control (unconnected, set apart, shaded).

  python studies/4_controlled_revision/code/fig_paper.py  ->  figures/fig_phase4_revision.{pdf,png}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
R = json.load(open(f"{ROOT}/studies/4_controlled_revision/results/main/RESULTS.json"))["by_condition"]
INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(3.35, 2.6))
ladder, xs = ("control", "blinded", "opponent"), [0, 1, 2]
for key, col, mk, lab in (("BRR", "#2a78d6", "o", "wrong → correct (BRR)"), ("HRR", "#eb6834", "s", "correct → wrong (HRR)")):
    v = [R[k][key] for k in ladder]
    ax.errorbar(xs, [x[0] for x in v], yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], color=col, marker=mk,
                ms=5, lw=2, capsize=3, label=lab)
    iv = R["irrelevant"][key]
    ax.errorbar([3.4], [iv[0]], yerr=[[iv[0] - iv[1]], [iv[2] - iv[0]]], color=col, marker=mk, ms=5, mfc="white",
                ls="none", capsize=3)
ax.axvspan(3.0, 3.8, color=GRID, alpha=0.5, lw=0)
ax.text(3.4, 0.43, "robustness\ncontrol", ha="center", va="top", fontsize=5.8, color=MUTED)
ax.set_xticks([0, 1, 2, 3.4], ["Control\n(self-review)", "Blinded\ncritique", "Opponent\nclaim", "Irrelevant\nclaim"])
di = R["opponent"]["DI"]
ax.text(2.0, 0.06, f"Opponent DI = {di[0]*100:+.1f} pp\n95% CI [{di[1]*100:+.1f}, {di[2]*100:+.1f}]\nnon-discriminating",
        ha="center", va="bottom", fontsize=6.3, color=INK2, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))
ax.set_xlim(-0.4, 3.8); ax.set_ylim(0, 0.45)
ax.set_ylabel("Revision rate", color=INK2)
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=6.3, loc="upper left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{ROOT}/figures/fig3_revision.{ext}", dpi=300)
