"""Paper Figure 1: the six-gate evaluation ladder (conceptual; contains no results).

  python experiments/figs/fig_framework.py  ->  figures/fig1_framework.{pdf,png}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import FancyBboxPatch       # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUE, BLUE_BG = "#2a78d6", "#eef4fc"
ORANGE, ORANGE_BG = "#eb6834", "#fdf1ec"
AQUA, AQUA_BG = "#1baf7a", "#ecf8f3"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.default": "regular"})

GATES = [  # (id, name, question, quantity, group)
    ("G1", "Controlled\ngain", "Is the reported\ngain an interaction\ngain?", "$G_{base}$, $G_{int}$", 0),
    ("G2", "Complemen-\ntarity", "Does interaction\ncarry information\nsampling lacks?", "$C^{*}=H-H_{ref}$", 1),
    ("G3", "Identifi-\ncation", "Can its use be\ndecided before the\nanswer is known?", "$C^{*}-C_{real}$", 1),
    ("G4", "Coverage", "Is the correct\nanswer among the\ncandidates?", "$\\kappa_{gen}$, $\\Delta\\kappa$", 2),
    ("G5", "Truth-sensitive\nrevision", "Does a challenge\ncorrect more than\nit corrupts?", "$DI=BRR-HRR$", 2),
    ("G6", "Realization", "Do local revisions\nsurvive\naggregation?", "$R$", 2),
]
GROUPS = [("Interaction gain?", BLUE, BLUE_BG),
          ("Complementarity & exploitation", ORANGE, ORANGE_BG),
          ("Mechanism: where is it lost?", AQUA, AQUA_BG)]

W, H = 6.8, 2.1
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
x0, bw, gap, y0, bh = 0.66, 0.80, 0.11, 0.12, 1.42
xs = [x0 + i * (bw + gap) for i in range(6)]

# group bands (above the boxes)
spans = {}
for i, g in enumerate(GATES):
    spans.setdefault(g[4], []).append(xs[i])
for k, (label, col, bg) in enumerate(GROUPS):
    a, b = min(spans[k]), max(spans[k]) + bw
    ax.plot([a, a, b, b], [y0 + bh + 0.10, y0 + bh + 0.17, y0 + bh + 0.17, y0 + bh + 0.10], color=col, lw=1.0)
    ax.text((a + b) / 2, y0 + bh + 0.22, label, ha="center", va="bottom", fontsize=6.2, color=col, fontweight="bold")

for i, (gid, name, q, quant, grp) in enumerate(GATES):
    col, bg = GROUPS[grp][1], GROUPS[grp][2]
    x = xs[i]
    ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.06", fc=bg, ec=col, lw=1.0))
    ax.text(x + bw / 2, y0 + bh - 0.08, gid, ha="center", va="top", fontsize=7, color=col, fontweight="bold")
    ax.text(x + bw / 2, y0 + bh - 0.25, name, ha="center", va="top", fontsize=6.5, color=INK, fontweight="bold", linespacing=1.0)
    ax.text(x + bw / 2, y0 + 0.62, q, ha="center", va="center", fontsize=5.4, color=INK2, linespacing=1.15)
    ax.plot([x + 0.08, x + bw - 0.08], [y0 + 0.27, y0 + 0.27], color=LINE, lw=0.6)
    ax.text(x + bw / 2, y0 + 0.14, quant, ha="center", va="center", fontsize=6.0, color=INK)
    if i < 5:
        ax.annotate("", xy=(xs[i + 1] - 0.01, y0 + bh / 2), xytext=(x + bw + 0.01, y0 + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, mutation_scale=7))

# ends
ax.text(x0 - 0.08, y0 + bh / 2, "Reported\nMAS gain", ha="right", va="center", fontsize=6.5, color=INK, fontweight="bold")
ax.annotate("", xy=(x0 - 0.01, y0 + bh / 2), xytext=(x0 - 0.07, y0 + bh / 2),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, mutation_scale=7))
xe = xs[-1] + bw
ax.annotate("", xy=(xe + 0.08, y0 + bh / 2), xytext=(xe + 0.01, y0 + bh / 2),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, mutation_scale=7))
ax.text(xe + 0.10, y0 + bh / 2, "Realized\ncollaboration\nutility", ha="left", va="center", fontsize=6.5, color=INK, fontweight="bold")

for ext in ("pdf", "png"):
    fig.savefig(f"{ROOT}/figures/fig1_framework.{ext}", dpi=300)
