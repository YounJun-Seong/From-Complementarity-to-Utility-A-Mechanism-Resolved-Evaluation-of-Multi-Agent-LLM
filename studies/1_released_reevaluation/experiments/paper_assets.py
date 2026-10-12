"""Paper tables and figures from the frozen results (no LLM calls). Writes to paper_assets/.

  Table 1  framework summary (released baseline, strong baseline, MAS, MAS − strong [CI], tokens ratio)
  Table 2  robustness / intervention summary (numbers from the frozen result files)
  Fig 2    MAS − released baseline vs MAS − strong single, per framework x backbone (95% CI)
  Fig 3    accuracy–compute Pareto (Qwen2.5-7B, Qwen2.5-14B, Llama-3.1-8B)

  python experiments/paper_assets.py
"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import synthesis as SY            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = f"{ROOT}/paper_assets"
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, INK, INK2, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})


def ci(d):
    lo, hi = SY.boot(np.asarray(d, float))
    return float(np.mean(d)), float(lo), float(hi)


def llama_medla():
    Q = 1273
    def L(s):
        d = {x["idx"]: x for x in SY.jl(f"{ROOT}/runs/layer3/{s}/run1.jsonl")}
        return [d[i] for i in range(Q)]
    L0, L1 = L("L0"), L("L1")
    S4 = SY.jl(f"{ROOT}/runs/layer3/S4/run1.jsonl")
    m = np.array([x["final_decision"] == x["label"] for x in L0], float)
    b = np.array([x["final_decision"] == x["label"] for x in L1], float)
    s = np.array([x["E3"] == x["gold"] for x in S4], float)
    tok = np.mean([x["prompt_tokens"] + x["completion_tokens"] for x in L0])
    stok = np.mean([x["S_prompt_tokens"] + x["S_completion_tokens"] for x in S4])
    return dict(M=m, S=s, base=b, tok=tok, stok=stok)


def parse_consumption(path):
    """Rows of paper_assets/table_compute_consumption.md -> {model: [dict]}"""
    out, cur = {}, None
    for line in open(path):
        if line.startswith("**"):
            cur = line.split("**")[1]; out[cur] = []
        elif line.startswith("| ") and cur and not line.startswith("| Method") and not line.startswith("|---"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            name = {"S": "S (zero-shot CoT)"}.get(c[0], c[0])
            out[cur].append(dict(method=name, acc=float(c[1]), input=float(c[2].replace(",", "")),
                                 output=float(c[3].replace(",", "")), total=float(c[4].replace(",", "")), calls=float(c[5])))
    return out


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def main():
    ma, md, ml, ll = SY.medagents(), SY.mdagents("F2_final_v2"), SY.medla(), llama_medla()
    rows = [  # name, backbone, data, released label, released acc, strong acc, MAS acc, MAS-strong, MAS-released, tokens ratio, confounder
        ("MedAgents", "GPT-4o-mini", "5-opt, 1,273 q, 1 run", "B1 (300-tok CoT)", ma["base"][0].mean(), ma["S"][0].mean(),
         ma["M"][0].mean(), ci(ma["M"][0] - ma["S"][0]), ci(ma["M"][0] - ma["base"][0]), ma["tok"] / ma["stok"],
         "restrictive generation budget of the released single baseline"),
        ("MDAgents (F2)", "GPT-4o-mini*", "5-opt, 200 q, 5 runs", "—", np.nan, md["S"].mean(), md["M"].mean(),
         ci(md["M"].mean(0) - md["S"].mean(0)), None, md["tok"] / md["stok"], "parser brittleness; routing / router-backbone sensitivity"),
        ("MedLA", "GPT-4o-mini", "4-opt, 1,273 q, L0 2 / S4, L1 3 runs", "L1 (released single mode)", ml["base"].mean(),
         ml["S"].mean(), ml["M"].mean(), ci(ml["M"].mean(0) - ml["S"].mean(0)), ci(ml["M"].mean(0) - ml["base"].mean(0)),
         ml["tok"] / ml["stok"], "weak released single baseline"),
        ("MedLA", "Llama-3.1-8B", "4-opt, 1,273 q, 1 run", "L1 (released single mode)", ll["base"].mean(), ll["S"].mean(),
         ll["M"].mean(), ci(ll["M"] - ll["S"]), ci(ll["M"] - ll["base"]), ll["tok"] / ll["stok"], "weak released single baseline"),
    ]
    f = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
    fci = lambda t: "—" if t is None else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    with open(f"{OUT}/table1_framework_summary.md", "w") as fh:
        fh.write("| Framework | Backbone | Data | Released baseline | Released acc. | Strong single acc. | MAS acc. | "
                 "MAS − released [95% CI] | MAS − strong [95% CI] | Tokens vs strong | Main confounder |\n|" + "---|" * 11 + "\n")
        for r in rows:
            fh.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {f(r[4])} | {f(r[5])} | {f(r[6])} | {fci(r[8])} | {fci(r[7])} | "
                     f"{r[9]:.0f}× | {r[10]} |\n")
        fh.write("\n*MDAgents routes with gpt-3.5-turbo and answers with gpt-4o-mini (released allocation); no released "
                 "single-agent mode. Strong single = zero-shot CoT (700 tok) + extractor; paired bootstrap over questions.\n")
    with open(f"{OUT}/table1_framework_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["framework", "backbone", "released_acc", "strong_acc", "mas_acc", "mas_minus_strong", "lo", "hi",
                     "mas_minus_released", "lo_r", "hi_r", "tokens_ratio"])
        for r in rows:
            w.writerow([r[0], r[1], r[4], r[5], r[6], *r[7], *(r[8] or (np.nan,) * 3), r[9]])

    # ---------------- Figure 2: baseline choice changes the apparent MAS gain (pp, 95% CI)
    fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=220)
    fig.patch.set_facecolor(SURF); style(ax)
    labels = [f"{r[0].replace(' (F2)', '')} / {r[1].rstrip('*')}" for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=INK2, lw=1.4, ls=(0, (4, 3)), zorder=1)
    for yi, r in zip(y, rows):
        st = [100 * v for v in r[7]]
        if r[8] is not None:
            rl = [100 * v for v in r[8]]
            ax.plot([st[0], rl[0]], [yi, yi], color=MUTED, lw=1, zorder=1)            # shift connector
            ax.plot([rl[1], rl[2]], [yi, yi], color=ORANGE, lw=5, alpha=0.35, solid_capstyle="round", zorder=2)
            ax.plot(rl[0], yi, "s", ms=7, color=ORANGE, mec=SURF, mew=1.5, zorder=3)
            ax.annotate(f"{rl[0]:+.1f}", (rl[0], yi), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=INK2)
        else:
            ax.annotate("released: N/A", (st[0], yi), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5, color=MUTED)
        ax.plot([st[1], st[2]], [yi, yi], color=BLUE, lw=5, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.plot(st[0], yi, "o", ms=7, color=BLUE, mec=SURF, mew=1.5, zorder=3)
        ax.annotate(f"{st[0]:+.1f}", (st[0], yi), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=7, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels, color=INK2)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlabel("MAS − baseline accuracy (percentage points; bands = 95% CI)", color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.plot([], [], "s", color=ORANGE, label="vs released single baseline")
    ax.plot([], [], "o", color=BLUE, label="vs strong single-agent control")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.2, labelcolor=INK2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig2_released_vs_strong.{ext}")

    # ---------------- Figure 3 + Appendix A1-A3 from the consumption table
    C = parse_consumption(f"{OUT}/table_compute_consumption.md")
    models = [("Qwen2.5-7B-Instruct", "(a) Qwen2.5-7B"), ("Qwen2.5-14B-Instruct", "(b) Qwen2.5-14B"), ("Llama-3.1-8B-Instruct", "(c) Llama-3.1-8B")]
    for key, xlab, fname in (("total", "Total tokens per question (log)", "fig3_pareto"),
                             ("output", "Output tokens per question (log)", "figA1_pareto_output_tokens"),
                             ("calls", "Model calls per question (log)", "figA2_pareto_calls")):
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), dpi=220)
        fig.patch.set_facecolor(SURF)
        for ax, (tag, title) in zip(axes, models):
            style(ax)
            R = [r for r in C[tag] if r["method"] != "SC-7"]
            pts = [(r[key], r["acc"], r["method"]) for r in R]
            front = sorted([p for p in pts if not any(q[1] >= p[1] and q[0] <= p[0] and (q[1] > p[1] or q[0] < p[0]) for q in pts)])
            ax.plot([p[0] for p in front], [p[1] for p in front], color=BLUE, lw=1, zorder=2)
            for x_, y_, name in pts:
                if name == "MedAgents M1":
                    ax.plot(x_, y_, "X", ms=9, color=ORANGE, mec=SURF, mew=1, zorder=4)
                    ax.annotate("MedAgents", (x_, y_), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
                else:
                    mk = "D" if name == "Reflection" else "o"
                    ax.plot(x_, y_, mk, ms=5, color=BLUE if name != "Reflection" else MUTED, mec=SURF, mew=1, zorder=3)
                    ax.annotate(name.replace(" (zero-shot CoT)", ""), (x_, y_), xytext=(0, 6 if name != "Reflection" else -10),
                                textcoords="offset points", ha="center", fontsize=6.3, color=INK2)
            ax.set_xscale("log"); ax.minorticks_off()
            if key == "calls":
                ax.set_xticks([2, 5, 10, 20]); ax.set_xticklabels(["2", "5", "10", "20"])
            else:
                ticks = [500, 1000, 2000, 5000, 10000, 20000]
                ax.set_xticks(ticks); ax.set_xticklabels(["0.5k", "1k", "2k", "5k", "10k", "20k"])
            ys = [p[1] for p in pts]; ax.set_ylim(min(ys) - 0.045, max(ys) + 0.035)
            ax.set_title(title, fontsize=8.5, color=INK, loc="left"); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
            ax.set_xlabel(xlab, color=INK2, fontsize=7.5)
            if key == "total":
                m1 = next(r for r in R if r["method"] == "MedAgents M1"); sc = next(r for r in R if r["method"] == "SC-5")
                note = f"SC-5 vs MedAgents: {100 * (sc['acc'] - m1['acc']):+.1f} pp, {100 * (1 - sc['total'] / m1['total']):.0f}% fewer tok."
                ax.set_title(f"{title}\n", fontsize=8.5, color=INK, loc="left")
                ax.text(0.0, 1.02, note, transform=ax.transAxes, fontsize=6.3, color=INK2, va="bottom")
        axes[0].set_ylabel("Accuracy (MedQA, 200 q, 3 runs)", color=INK2)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{OUT}/{fname}.{ext}")
    # A3: input / output composition, Qwen2.5-14B
    R = [r for r in C["Qwen2.5-14B-Instruct"]]
    order = ["S (zero-shot CoT)", "SC-5", "Reflection", "SC-10", "SC-20", "MedAgents M1"]
    R = sorted(R, key=lambda r: order.index(r["method"]))
    fig, ax = plt.subplots(figsize=(5.0, 2.4), dpi=220); fig.patch.set_facecolor(SURF); style(ax)
    yy = np.arange(len(R))[::-1]
    ax.barh(yy, [r["input"] / 1000 for r in R], color=BLUE, height=0.6, label="input tokens", edgecolor=SURF, linewidth=2)
    ax.barh(yy, [r["output"] / 1000 for r in R], left=[r["input"] / 1000 for r in R], color="#9ec2ee", height=0.6,
            label="output tokens", edgecolor=SURF, linewidth=2)
    for yi, r in zip(yy, R):
        ax.text(r["total"] / 1000 + 0.3, yi, f"{r['total'] / 1000:.1f}k ({100 * r['input'] / r['total']:.0f}% input)",
                va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(yy); ax.set_yticklabels([r["method"].replace(" (zero-shot CoT)", "") for r in R], color=INK2)
    ax.set_xlabel("Tokens per question (thousands), Qwen2.5-14B", color=INK2); ax.set_xlim(0, 30)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=7, labelcolor=INK2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/figA3_token_composition.{ext}")

    # ---------------- Table 2: robustness / interventions (values from frozen result files)
    t2 = [
        ("Budget sweep (P0, GPT-4o-mini)", "B1 → B3 +0.104; 103% of MedAgents' gain over B1 lies on the 279 B1-truncated questions; answer-only continuation recovers as much as a fresh 700-token run"),
        ("Budget sweep (2B, Qwen2.5)", "B3 − B1 = +0.18 / +0.15 / +0.42 (3B / 7B / 14B), ~100% on B1-truncated questions; Qwen-14B (0.272) scores below Qwen-1.5B (0.342) under the 300-token baseline"),
        ("Internal MedAgents budget expansion (2C)", "expert-analysis truncation 100% → 41% / 18% (7B / 14B); accuracy change −0.022 / 0.000"),
        ("Parser stabilization (MDAgents F1 → F2)", "completion 0.70 → 1.00, accuracy +0.20; gap to strong single −0.120 [−0.155, −0.085] remains"),
        ("Router backbone (MDAgents F2 → F3)", "route agreement 0.47 → 0.21, accuracy −0.06"),
        ("Answer-order permutation (2E, Qwen-14B)", "M1 − S per order −0.020 … +0.020; pooled −0.001 [−0.037, +0.035]"),
        ("Cross-model (2A)", "M1 − S ≤ 0 for Qwen2.5 1.5B / 3B / 7B / 14B; Llama-3.1-8B −0.095 [−0.150, −0.040]"),
        ("Compute allocation (2D)", "MedAgents off the accuracy–token Pareto frontier for all three models; SC-5 − M1 = +0.028 / +0.043 / +0.127 (Qwen-7B / Qwen-14B / Llama-8B)"),
    ]
    with open(f"{OUT}/table2_robustness.md", "w") as fh:
        fh.write("| Analysis | Finding |\n|---|---|\n")
        for a, b in t2:
            fh.write(f"| {a} | {b} |\n")
    print(open(f"{OUT}/table1_framework_summary.md").read())
    print("written:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
