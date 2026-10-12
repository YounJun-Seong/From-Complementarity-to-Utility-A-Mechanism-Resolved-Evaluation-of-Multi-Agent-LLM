"""Paper Figure 2: complementary information does not imply realized interaction utility.
(a) controlled study: oracle headroom H -> frozen router (C_realized) -> disagreement-triggered TDC;
(b) external validation, MedARC: C* -> G_int. All values read from the sealed RESULTS files (no recomputation).

  python experiments/figs/fig_cug.py  ->  figures/fig2_cug.{pdf,png}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
J = lambda p: json.load(open(f"{ROOT}/{p}"))
p2 = J("studies/2_complementarity_router/results/pilot/RESULTS.json")["Qwen2.5-14B-Instruct"]
p5 = J("studies/5_medarc_transfer/results/RESULTS.json")
A = [("Oracle\nheadroom $H$", "MedQA pilot", p2["H"], "comp"),
     ("Frozen\nrouter", "MedQA held-out", J("studies/2_complementarity_router/results/heldout/RESULTS.json")["B"]["diff"], "real"),
     ("Disagreement\ntrigger (TDC)", "MedMCQA reserve", J("studies/3_disagreement_intervention/results/reserve/RESULTS.json")["P1"]["diff"], "real")]
B = [("Complementarity\n$C^*$", "", p5["G2"]["C_star"], "comp"),
     ("Compute-adjusted\ngain $G_{int}$", "", p5["G1"]["G_int"], "real")]
fmt = lambda x: "0.0" if abs(x * 100) < 0.05 else f"{x*100:+.1f}"
COL = dict(comp="#2a78d6", real="#eb6834")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "mathtext.default": "regular"})

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), sharey=True, gridspec_kw=dict(width_ratios=[3, 2], wspace=0.08))
for ax, rows, title in ((axes[0], A, "(a) Controlled study (Qwen2.5-14B)"),
                        (axes[1], B, "(b) External validation: MedARC")):
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for x, (lab, sub, v, kind) in enumerate(rows):
        ax.errorbar(x, v[0] * 100, yerr=[[(v[0] - v[1]) * 100], [(v[2] - v[0]) * 100]], fmt="o", ms=6, color=COL[kind],
                    mec="white", mew=1.2, capsize=3, lw=1.8, zorder=3)
        ax.text(x + 0.12, v[0] * 100 + 0.35, f"{v[0]*100:+.1f}", va="bottom", ha="left", fontsize=7.5, color=INK,
                fontweight="bold", bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=4)
        ax.text(x + 0.12, v[0] * 100 - 0.35, f"[{fmt(v[1])}, {fmt(v[2])}]", va="top", ha="left", fontsize=5.8,
                color=INK2, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=4)
        if sub:
            ax.text(x, -11.6, sub, ha="center", va="bottom", fontsize=6.3, color=MUTED, style="italic")
    if len(rows) == 3:   # a bracket, not arrows: the three estimates come from distinct populations
        ax.plot([0.75, 0.75, 2.25, 2.25], [7.6, 8.4, 8.4, 7.6], color=MUTED, lw=0.8)
        ax.text(1.5, 9.0, "prospective realization tests", ha="center", fontsize=6.5, color=MUTED)
    else:
        ax.plot([0, 0, 1, 1], [7.6, 8.4, 8.4, 7.6], color=MUTED, lw=0.8)
        ax.text(0.5, 9.0, "same system, same questions", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_title(title, loc="left", fontsize=8, color=INK, pad=4)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylim(-12, 11)
axes[0].set_ylabel("percentage points (95% CI)", color=INK2)
handles = [plt.Line2D([], [], marker="o", ls="", color=COL[k], label=l) for k, l in
           (("comp", "complementarity evidence"), ("real", "realized utility vs compute-matched sampling"))]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.25)
for ext in ("pdf", "png"):
    fig.savefig(f"{ROOT}/figures/fig2_cug.{ext}", dpi=300)
