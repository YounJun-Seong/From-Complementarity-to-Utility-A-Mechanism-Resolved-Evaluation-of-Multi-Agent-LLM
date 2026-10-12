# Post-freeze reviewer-robustness additions (frozen 2026-09-26, before any of these results)

Added after `final-freeze` to address three anticipated reviewer objections. Exploratory open-weight
layer (local vLLM, BF16, `--generation-config vllm`, max_model_len 32768); no API calls; never pooled
with the GPT-4o-mini results. Code: MedAgents `3da1947`+ (crash isolation), scripts listed below.

## R1 — first-200 vs random subsets (MedQA 5-option)
Three seeded random 200-question subsets of the full 1,273-question test (seeds 11 / 12 / 13;
`external/MedAgents/datasets/MedQA_rand{A,B,C}`; sampled from all questions, overlap with the
first 200 allowed). Systems: S (strong single) and M1 (MedAgents) on Qwen2.5-14B and Llama-3.1-8B,
1 run per subset. Report M1 − S per subset (paired bootstrap) and pooled over the 600 draws;
the question is only whether the first-200 conclusion (M1 − S ≤ 0) holds on random subsets.

## R2 — external dataset (MedMCQA dev, 4-option)
500 questions sampled with seed 21 from the 4,183-question labelled split
(`external/MedAgents/datasets/MedMCQA_r500`). Systems: S, M1 (MedAgents, MedMCQA mode), SC-5.
Models: Qwen2.5-14B and Llama-3.1-8B, 1 run each. Report M1 − S and SC-5 − M1 (paired bootstrap),
tokens / calls per question.

## R3 — total compute (existing + new logs)
Per method: input, output and total tokens per question, calls per question; GPT-4o-mini $ cost.
Pareto re-plotted on total tokens. Wall-clock latency is not reported for the local runs (shared,
batched serving makes it incomparable across methods); MedLA / MDAgents wall times are descriptive.

## R4 — strong-single budget robustness (existing P0 data, no new runs)
M1 − base_cot[role, budget] for budgets 512 / 700 / 1024 / 1536 and both roles (paired CI), plus
the pre-registered S (zero-shot CoT 700).

## R5 — approximately compute-matched comparisons
Qwen2.5-14B: M1 (~22k tokens) vs SC-20 (~23k), existing runs. Llama-3.1-8B: new SC-7 (≈ M1's
7.6k tokens), first-200 questions, 3 runs, same protocol as 2D.

## R7 — MDAgents parser decomposition (existing F0 / F2 outputs)
Both-valid accuracy, share of F0-invalid (crash) units that become valid in F2, and their accuracy.

## Amendment (2026-09-26, before any R1 result): SC-5 added to R1
To include "a representative compute-scaled single-agent condition" on the random subsets, SC-5
(same protocol as 2D, seed 1, 1 run) is added for each random subset A / B / C on Qwen2.5-14B
(primary) and Llama-3.1-8B. Output: `runs/postfreeze/R1/<model>/SC5_<subset>.jsonl`.

## Analysis note (2026-09-26): overlapping random subsets
Subsets A / B / C were sampled independently from the full test, so they overlap (A–B 21, A–C 28,
B–C 30 questions; 523 unique questions in 600 draws). Pooled R1 confidence intervals are therefore
computed with a bootstrap clustered by unique MedQA question id (all repeated occurrences of a
question are resampled together); per-subset intervals are unaffected. The earlier row-level pooled
interval was superseded. MedMCQA: the 4,183-question labelled file is the MedMCQA validation (dev)
split; 500 questions were drawn uniformly without replacement (seed 21) before any outcome was seen.
