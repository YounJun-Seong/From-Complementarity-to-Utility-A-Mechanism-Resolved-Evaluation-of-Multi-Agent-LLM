# Final results (frozen 2026-09-25, tag `final-freeze`)

All numbers are recomputed from the result files by the scripts named in each section; every run
passed `experiments/validate_runs.py` (expected questions, api_failed 0, no harness error, valid
predictions; framework crashes scored wrong as pre-registered). Model families are never pooled.

## Layer 1 — confirmatory, GPT-4o-mini-2024-07-18 (`runs/SYNTHESIS.txt`, section A)
| framework | data | MAS − strong single [95% CI] | MAS − released baseline | execution failure | main sensitivity | tokens vs S |
|---|---|---|---|---|---|---|
| MedAgents (P0) | MedQA 5-opt, 1,273 q | −0.005 [−0.027, +0.016]; equivalent within the pre-registered 3-pp margin (TOST vs S p=.011, vs B3 p=.003; not within 2 pp vs S, p=.086) | +0.107 (vs B1, 300-tok) | 0% | released single-agent generation budget | 16× ($ 11×) |
| MDAgents F2 | 200 q × 5 runs | −0.120 [−0.155, −0.085] | n/a | 0.1% (released F0: 29%) | parser robustness; router backbone (F2→F3) | 80× |
| MedLA | MedQA 4-opt, 1,273 q | −0.048 [−0.065, −0.030] | +0.007 (vs L1) | 0% | none detected | 148× ($ 111×) |

MedAgents mechanism (P0): B1 → B3 = +0.104; 103% of M1's gain over B1 lies on the 279 B1-truncated
questions; answer-only continuation recovers as much as a fresh 700-token run (0.620 vs 0.620).

## Layer 2 — exploratory, open weights (local vLLM, BF16)
- 2A capability (200 q × 3): M1 − S = −0.012 (1.5B), −0.033 (3B), −0.003 (7B), −0.040 (14B),
  **−0.095 [−0.150, −0.040] (Llama-3.1-8B)**; M1 crash 7.5% / 8.8% at 1.5B / 3B, ≤ 0.7% otherwise.
- 2B budget: B3 − B1 = +0.18 (3B), +0.15 (7B), +0.42 (14B); ~100% of the gain on B1-truncated
  questions; under the 300-token baseline Qwen-14B (0.272) scores below Qwen-1.5B (0.342).
- 2C internal budgets (7B / 14B): expert-analysis truncation 100% → 41% / 18%, accuracy change
  −0.022 / 0.000; the deficit is not internal truncation.
- 2D compute-matched: across the evaluated models MedAgents was off the accuracy–compute Pareto frontier; a
  single-agent alternative achieved higher point-estimate accuracy with lower or comparable inference consumption
  (SC-5 − M1 = +0.028 [CI incl. 0] / +0.043 / +0.127 for Qwen-7B / Qwen-14B / Llama-8B; `runs/layer2/2D/PARETO.txt`).
- 2E option order (Qwen-14B): M1 − S per order −0.020 … +0.020, pooled −0.001 [−0.037, +0.035];
  14B is reported as "no gain", not "worse".

## Layer 3 — MedLA with its paper backbone Llama-3.1-8B (`runs/layer3/RESULTS.txt`)
L0 0.590, L1 0.563, S4 0.700 (paper: MedLA 62.6, CoT 54.9; L0 is in a similar range to the paper's MedLA, not a strict reproduction — serving stack, vLLM version and hardware differ). L0 − L1 = +0.027 [−0.003, +0.057];
**L0 − S4 = −0.110 [−0.139, −0.081]**; L0 uses 174× the tokens of S4.

## Cross-framework analyses (`runs/SYNTHESIS.txt`, B–D)
- Win/loss: MAS-only-correct questions are fewer than single-only-correct in every framework
  (MDAgents 5 vs 18, MedLA 71 vs 93, MedAgents 88 vs 95) and are hard, low-stability questions.
- Difficulty (held-out bins): the MAS deficit concentrates on questions the strong single agent
  solves reliably; positive values on "hard" bins are not interpreted as coordination gains
  (a deterministic T=0 single agent stays below chance on its own failure set; stochastic systems
  regress toward chance).
- MDAgents: discussion is triggered more on harder questions (ρ = −0.32 / −0.19, F2 / F3); the gap
  does not narrow with discussion; router labels barely track difficulty (ρ CIs include 0).

## Deviations and amendments (all recorded in the respective pre-registration files)
1. MDAgents F2 implementation correction and restart (list markers, preamble) — Amendment 1.
2. OpenAI credit exhaustion (15:51, 09-24): failed questions re-run in the same condition.
3. MedLA pilot (questions 0–199) was unrepresentative (full-test gap −0.053 vs −0.005 on q0–199).
4. MedLA L0 run 3 skipped for budget after run 2 (rule required runs 2–3) — Amendment 3.
5. Layer 2A: MedAgents harness crash isolation added mid-run (fill pass for 7B) — Amendment 1.
6. Layer 3: first L0 run served with a 32k context; 1,043 affected questions re-run with 128k
   (configuration defect, not a model/framework result) — Amendment 1; execution later split
   across two GPUs (no methodological change).
7. vLLM `generation_config` defaults disabled (`--generation-config vllm`) after the first local
   validation; the affected validation cache was discarded.

## Post-freeze robustness additions (POSTFREEZE_ROBUSTNESS.md; tag `final-freeze-v2`)
- R1 random subsets (3 × 200, seeds 11/12/13, overlapping; pooled CIs clustered by question):
  Qwen2.5-14B M1 − S = −0.030 / +0.025 / −0.040 per subset, pooled −0.015 [−0.054, +0.024];
  Llama-3.1-8B pooled −0.113 [−0.159, −0.068]; SC-5 − M1 pooled +0.017 (Qwen) / +0.133 (Llama).
- R2 MedMCQA (validation split, 500 q, seed 21): Qwen2.5-14B S 0.648, M1 0.636, SC-5 0.642
  (M1 − S −0.012 [−0.048, +0.026]; M1 uses 19.2k vs SC-5 4.4k total tokens); Llama-3.1-8B S 0.608,
  M1 0.542, SC-5 0.648 (M1 − S −0.066 [−0.112, −0.020]).
- R3 inference consumption (`paper_assets/table_compute_consumption.md`): M1 is input-dominated
  (Qwen 84% input tokens) while SC-k is balanced; M1 stays off the frontier on total tokens, and on
  output tokens SC-5 (Qwen) or S (Llama) still dominates it.
- R4 budget robustness: M1 − single baseline within [−0.006, +0.004] for budgets 512–1536 × 2 roles,
  all equivalent within 3 pp (TOST p ≤ .009).
- R5 compute-matched: Llama SC-7 (7.8k tok) − M1 (7.6k) = +0.133 [+0.082, +0.185]; Qwen-14B
  SC-20 (23.4k) − M1 (22.1k) = +0.032 [−0.010, +0.075].
- R7 MDAgents parser: on units valid in both F0 and F2, accuracy 0.730 vs 0.711; the F1→F2 gain comes
  from the 290 crashed units (99.7% valid in F2, accuracy 0.633).
