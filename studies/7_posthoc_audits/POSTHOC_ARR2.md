# Post-hoc analyses for the ARR revision, round 2 (2026-10-12): ARR2-E1 … E4

Status: **post-hoc, descriptive**. Definitions are fixed and committed here before any of the analyses
below is run. **No new model calls**: every analysis reads existing Phase II / V / VI logs. No frozen result,
threshold, label or verdict is changed, and no number here replaces a sealed number.
Prefix "ARR2-" avoids a clash with the E1–E4 of POSTHOC_EXPERIMENTS.md.
CIs: 95%, 10,000 bootstrap resamples over questions (cluster bootstrap where units are q×run), seed 0.
Scripts: `experiments/posthoc_arr2_phase2.py` (E1–E3), `experiments/posthoc_arr2_medarc.py` (E4).
Outputs: `runs/posthoc/arr2/`.

## ARR2-E1. Compute-corrected C\* (Phase II PILOT; Qwen2.5-14B primary, Llama-3.1-8B secondary)

Problem (POSTHOC_TRANSITION.md §2): Phase II SC was one request with n = 20 samples, so the generation
prompt was charged once per request. The frozen per-sample cost therefore undercharges SC, and k\* = 22
(Qwen) is too large under per-sample accounting (k\*′ = 18).

Accountings:
- **frozen**: SC per-sample = (prompt + completion) / 20 over SC runs 1–3 (as in `pilot_analysis.py`).
- **corrected**: generation prompt charged per sample. gen_p = (20·mean S prompt − mean SC prompt) / 19
  (S = gen + E3 prompt; SC = gen + 20·E3 prompt). SC per-sample′ = per-sample + gen_p·19/20.
  HET (one request per recruited role, n = samples of that role): per-sample′ = per-sample +
  gen_p·(21 − n_roles)/21, where n_roles = calls − 2 − 21 per record; this slightly undercounts (it ignores
  the role system prompt). S and M1 are separate requests and are unchanged.
- k\*′ = `oracle_analysis.k_star(M1 tokens, per-sample′)`; k\*′_het the same with HET per-sample′.

Quantities, computed with the **unchanged** `oracle_analysis.analyze` (pools, cross-fitting, tie rules,
placebo split exactly as sealed) at k ∈ {k\*′, …, k\*}: SC-k accuracy (eval pool E), H, H_plc,
**C\* = H − H_plc**, n_C and G3 contrast, oracle accuracies and their mean token cost under **both**
accountings. HET at the same k: H_het, C\*_het = H − H_het (code as in `het_analysis.py`).
Reproduction check: at k = 22 the frozen H, H_plc, C\* must be reproduced exactly (asserted).
Headline row: k = k\*′ with corrected costs.

HELD-OUT sensitivity (Qwen, n = 339, one run): SC-k\*′ = majority of the first k\*′ of the 22 votes (a
prefix of one n = 22 request is a valid k\*′-sample draw; ties → earliest). Frozen router actions are
unchanged; SAMPLE and Ours-Sampling use SC-k\*′. Per-question corrected SC cost: per-sample′ × k, with
gen_p estimated per question from that question's S and SC prompts. Report Criterion B (Ours-Collab −
Ours-Sampling, CI, token ratio) and SC-k\*′ vs M1. This is a sensitivity analysis, not a re-evaluation of
the sealed criterion.

## ARR2-E2. Candidate-budget robustness of C\* (Phase II PILOT; replication on Phase V MedARC)

Candidate budget = k, the number of samples given to each non-interactive alternative (SC, placebo, HET).
M1 is fixed. Grid: Phase II k ∈ {1, 3, 5, 7, 10, 15, 18, 20, 22}; Phase V k ∈ {1, 3, 5, 7, 10, 15, 20, 25}.
The eval pool E has 20 (Phase II) / 25 (Phase V) samples; for k above that, draws are with replacement,
exactly as the sealed code does at k = 22. The placebo halves have 30 (37/38) samples.

At each k:
- **Cross-fitted (sealed definition)**: H(k), H_plc(k), C\*(k) = H − H_plc; Phase II also H_het(k), C\*_het(k).
- **Uncorrected oracle headroom (no cross-fitting)**: selection and evaluation on the same estimates, all
  runs pooled (p_S over 3 runs, p_SC over pools A ∪ E, p_M over 3 runs); H_naive = mean over q of
  max(p_S, p_SC, p_M) − max(p_S, p_SC); H_naive,plc the same with M replaced by the placebo (all 60 / 75
  samples); C\*_naive = H_naive − H_naive,plc.
- **Union oracle (gold-oracle on the eval half)**: U(k) = mean[max(eS, eSC, eM)] − mean[max(eS, eSC)], and
  U_plc with the placebo eval half. With fractional eSC these are expectations.
- **Complementarity/diversity on the eval half**: P(M right, SC-k wrong) = mean[eM(1 − eSC)] and
  P(M wrong, SC-k right) = mean[(1 − eM)eSC].
- Token cost of SC-k (frozen and corrected accounting for Phase II) as a ratio to M1.
Phase V uses the sealed Phase V mapping (S = B_strong, M = A_paper_no_reinit, SC pools 1–3 / placebo 4–6,
pool split 2k\*_V = 50). Reproduction check at the sealed k (22; 25) is asserted.
Reading rule: C\* is a diagnostic of a fixed candidate construction; no constancy across k is expected.
The curve is reported whatever its shape.

## ARR2-E3. Router: rate–accuracy–cost trade-off and routing baselines (Phase II, Qwen2.5-14B)

The frozen router chose COLLABORATE on 4/339 HELD-OUT questions. These analyses separate "this frozen
policy fails" from "the complementarity cannot be identified by pre-decision signals".

Escalation scores (all pre-decision; non-escalated questions keep the frozen router's STOP/SAMPLE action):
- **q_C**: frozen model on HELD-OUT; on PILOT, out-of-fold q_C from the same 5-fold cross-fitting as
  `router.py::crossfit_vc` (λ = 0, KFold seed 0).
- **Uncertainty**: probe vote entropy, descending (ties: lower probe_max_share first, then question index).
- **Difficulty**: log S reasoning length, descending (ties: question index).
- **Random**: 1,000 uniform permutations, seed 0.

Collaboration-rate grid r ∈ {0, 1, 2, 5, 10, 15, 20, 30, 50, 100}%. Top ⌈r·Q⌉ questions by score are
escalated. For each selector and r, report Ours-Collab (escalated → M1), Ours-Sampling (escalated → SC-22),
and on PILOT also Ours-Hetero; accuracy, mean tokens (frozen accounting; corrected ratio also given),
**Collab − Sampling** with paired CI. PILOT values are expectations over runs (Y as in `router.py`).
HELD-OUT values use the single run of each arm.

- **PILOT curve** (router-development data, the "DEV" of the review): fully cross-fitted, so it is
  out-of-sample with respect to the router fit.
- **PILOT-selected threshold, applied once to HELD-OUT** ("quasi-prospective"): τ\* = the q_C threshold in
  {0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025} that maximises PILOT cross-fitted Collab − Sampling with
  the token ratio in [0.9, 1.1]; ties → higher τ. It is selected on PILOT only and applied to HELD-OUT q_C. The
  frozen HELD-OUT result was already known when this rule was written, so the result is labelled
  post-hoc and is **not** a prospective result.
- **HELD-OUT curves over r** are **exploratory** (the rate is not selected on HELD-OUT). No HELD-OUT
  value is reported as prospective performance, and no threshold is chosen from HELD-OUT.

## ARR2-E4. MedARC failure trace (Phase V n = 498; Phase VI arms A/B/C n = 472; 3 runs each)

Answers: controlled parser (`aggregate.controlled`). Final: `A_paper_no_reinit` (sealed); R2 majority
(`a_majority(rec, 2)`) secondary. G = agent's answer equals gold; unparsed counts as W.

- **T1 Agent-level stage transitions.** For each (q, run, agent), the trajectory over rounds 0 → 1 → 2 in
  {G, W}³. Report the distribution of the 8 patterns and the 2×2 transition matrices for 0→1 and 1→2:
  P(G→W) (abandon) and P(W→G) (adopt).
- **T2 Holder-count transitions (q×run).** #gold holders at R0 (0–3) × #gold holders at R2 (0–3), with the
  share of units in each cell and P(final correct) per cell.
- **T3 Lone-correct fate.** Units with exactly one gold holder at R0. Holder's state at r = 1 and r = 2 and
  the unit's fate: *realized* (final correct), *selection miss* (gold ∈ R2, final wrong), *lost*
  (gold ∉ R2). Split by **peer configuration at R0**: *unanimous wrong* (both peers give the same wrong
  label) vs *split wrong* (different wrong labels, or one unparsed); and by gold label (yes/no/maybe).
  Arms A, B, C and Phase V; A − B and A − C contrasts on P(lone holder abandons gold at r = 1) (same R0 in
  all Phase VI arms).
- **T4 Gate mapping.** One row per arm: G2 avail = P(gold ∈ R0); G4 preserved | avail; G5 agent-level
  P(W→G) and P(G→W) over both rounds, and the selectivity difference; G6 selection miss | preserved.
- **Case studies.** Selected **at random (seed 0) among eligible units**, not hand-picked. Eligibility:
  (a) lone-correct, unanimous wrong peers, lost in arm A, and the same (q, run) in arm B keeps gold;
  (b) lone-correct in arm A with final correct (success); (c) gold ∈ R2 but final wrong in arm A
  (selection miss); (d) Phase V lone-correct lost. Two cases per category. For each case: question
  (truncated), gold, R0/R1/R2 answers + confidences per agent, the summary the lone holder read at r = 1
  (truncated to 700 chars), the holder's round-1 turn (truncated to 500 chars), and the final answer.
  Output: `runs/posthoc/arr2/E4_CASES.md`.

Interpretation limits: question-level description; the peer configuration is observed, not
manipulated, except for the across-arm contrast (identical R0 in Phase VI). No causal mediation claim.

## Results (2026-10-12; `runs/posthoc/arr2/E{1,2,3,4}.{txt,json}`, `E4_CASES.md`)

Reproduction checks passed: frozen H / H_plc / C\* / G4 AUC / C\*_het at k = 22 (Qwen) and 7 (Llama); Phase V C\* at k = 25;
OOF router actions equal `crossfit_vc`; HELD-OUT SC-22 first-k equals the sealed pred; E4 final accuracies equal the
sealed values (V .598, A .581, B .714, C .649).

Departures from the spec above (disclosed, not hidden): (i) E1 reports the SC/M1 token ratio under both
accountings but not the mean cost of each oracle policy. (ii) E3 curves use only the frozen accounting; the corrected
Criterion-B token ratio is in E1 (1.010). (iii) In E2/E1, k = 21–22 draw from the 20-sample eval pool E with replacement
(this is also what the sealed code does), which explains the small SC_E dip at k ≥ 21.

- **E1.** Qwen: gen prompt ≈ 238 tok, so SC/sample goes from 1158 to 1384 and k\* from 22 to **k\*′ = 18** (SC-18/M1 tokens 0.997).
  At k\*′ = 18: H +3.4 [+1.1, +5.8], H_plc +0.0 [−0.8, +0.8], **C\* +3.4 [+1.0, +6.0]**, C\*_het +2.9 [+0.5, +5.5], n_C 16.
  C\* lies between +3.3 and +3.4 for every k in 18–22. Llama: k\* goes from 7 to 5, and C\* +3.7 [+1.7, +6.0] at k = 5.
  HELD-OUT with SC-18 and corrected cost: M1 − SC −3.2 [−7.7, +1.2]; Criterion B +0.3 [−0.6, +1.5], token ratio 1.010
  (unchanged).
- **E2.** Phase II C\* over k = 1–22 lies between +3.3 and +3.8 pp, and every CI excludes 0. H_plc ≤ +0.3 throughout.
  Without cross-fitting (H_naive 4.3–4.6, C\*_naive 3.8–4.0) or as a gold union oracle (U 5.0–5.5, U − U_plc 4.1–4.3),
  the headroom is larger by 1–2 pp. P(M right, SC wrong) falls from .094 to .058 as k grows, and P(M wrong, SC right) stays
  ≈ .084. Phase V MedARC: C\* +1.6 to +1.9 over k = 1–25, all CIs > 0; C\*_naive 2.4–2.8.
- **E3.** PILOT cross-fitted: no selector (q_C, uncertainty, difficulty, random) at any rate 1–50% gives Collab − Sampling
  CI > 0. The best point estimate is +0.3, and at 100% it is −3.9 [−7.6, −0.2]. τ\* (PILOT) = 0.15. Applied once to HELD-OUT
  (quasi-prospective): COLLAB 57/339 (16.8%), Collab − Sampling −1.2 [−4.1, +1.8], ratio 1.094. HELD-OUT exploratory curves:
  no selector-rate pair has CI > 0, and q_C is no better than the uncertainty, difficulty or random selectors.
- **E4.** Lone-correct units (15% of avail): abandon at r = 1 is 62.6% in A vs 21.9% in B and 15.5% in C
  (A − B +40.6 [+30.9, +50.4]). Fate in A: realized 41.3%, lost 48.4%, selection miss 10.3%. In B: 69.7 / 20.0 / 10.3.
  89% of lone-correct units have unanimous-wrong peers, and the A − B abandon gap is the same for unanimous peers (+41.3)
  and split peers (+35.3, n = 17). Gate rows: G4 preserved | avail A 85.6% vs B 94.5%. G5 adopt W→G A 10.3% vs B 21.2%;
  abandon G→W 10.1% vs 5.9%. G6 selection miss | preserved is 8.7–11.4% in all arms. In arm A, lone-correct gold = yes is
  lost in 64.3% of cases vs no 30.4%. Case studies (random, seed 0): `E4_CASES.md`. Observation not in the spec: in sampled
  failure cases the unanimous wrong peers often answer "maybe". Not tabulated, so not reported as a finding.
