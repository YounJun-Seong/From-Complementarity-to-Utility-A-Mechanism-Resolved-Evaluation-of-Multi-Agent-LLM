# Post-hoc audit (2026-10-11): question-level G2→G4→G6 transitions + SC token-accounting sensitivity

Status: **post-hoc, descriptive**. Definitions below were fixed and committed before the script was run.
No new model calls. No frozen result, threshold or label is changed; nothing here re-labels a frozen verdict.

## 1. Question-level transitions (MedARC; Phase V n=498, Phase VI arms A/B/C n=472)

Unit = (question q, MedARC run r), 3 runs per question; CIs = 10,000 cluster bootstrap over questions (seed 0).
Answers use the controlled parser (`aggregate.controlled`), exactly as in the sealed G4/G5 analyses.

- `R0(q,r)` = set of round-0 answers of the 3 agents; `R2(q,r)` = set of round-2 answers.
- **avail** (G2 proxy, candidate available): gold ∈ R0.
- **comp** (complementary candidate): avail AND the compute-matched independent baseline fails on q,
  i.e. `p_SC25(q) < 0.5` (the same p_SC used for G_int; pools = SC runs 1–3). Secondary: B_strong
  majority over its 3 runs is wrong.
- **preserved** (G4): gold ∈ R2; **lost** = avail AND not preserved; **gained** = not avail AND preserved.
- **final** (G6): `A_paper_no_reinit` correct (frozen aggregator). Secondary: R2MV correct.

Because the final answer is always one of the round-2 answers, *lost ⇒ final wrong* holds by construction.
The analysis therefore reports a decomposition (an accounting identity), not an association test:

    among avail:  realized (preserved ∧ correct) + selection miss (preserved ∧ wrong) + lost  = 1

Reported per data set / arm: P(avail), P(comp); on avail and on comp: realized / selection-miss / lost shares;
share of all final failures that had gold in R0 and split of those into lost vs selection miss; on not-avail:
gained and gained∧correct. Contrasts (Phase VI, same questions, identical R0 across arms): A − B and A − C
on comp-realized and comp-lost.

Interpretation limits: question-level description of observed transitions; not a causal mediation analysis;
"comp" depends on the choice of independent baseline (both variants reported).

## 2. SC compute-matching sensitivity (Phase II, Qwen2.5-14B)

Phase II SC was one request with n=20 (PILOT) / n=22 (HELD-OUT), so the generation prompt is charged once
per request while E3 extraction and M1 calls are charged per call. Alternative accounting = generation
prompt charged per sample. Generation-prompt length is recovered from the records
(S = gen + E3 prompt; SC = gen + 20 × E3 prompt). k*′ = integer k closest to M1 tokens under the
alternative accounting. Report SC-k accuracy for k ∈ {k*′, …, k*} on PILOT (pool run 3, all subsets
averaged by `p_sc`) and HELD-OUT (first-k votes and the subset average), and M1 − SC-k.
Phase V/VI need no sensitivity (SC = k* separate n=1 requests; realized ratio reported in RESULTS).

## 3. Results (2026-10-11; `runs/posthoc/TRANSITION.{txt,json}`, script `experiments/posthoc_transition.py`)

Transitions (comp = gold in R0 ∧ SC-25 wrong; units q×run):
- Phase V MedARC (144 units / 81 q): realized 44.4% [34.9, 54.3], selection miss 16.0%, lost 39.6% [29.7, 49.6].
  Of all 600 final failures, 38.7% [33.5, 43.9] had gold in R0 (lost 23.8% + selection miss 14.8%).
- Phase VI A (147 / 81): realized 39.5%, lost 46.9%. B no-summary: realized 68.7%, lost 19.0%. C placebo: 58.5% / 21.1%.
  A − B realized −29.3 pp [−40.6, −16.9], lost +27.9 pp [+18.4, +37.5]; A − C realized −19.0 pp [−30.4, −7.3].
- Same pattern with comp defined against B_strong.

Additional audit checks (same day, no protocol change):
- Phase V and Phase VI sealed analyses re-run from committed code/data into scratch: RESULTS.json bit-identical.
- Sealed scripts last modified in their pre-output code commits (2eac884, c24633e, da3992d).
- Router freeze manifest: 16/16 sha256 OK.
- Phase II SC token accounting: gen prompt ≈ 238 tok charged once per n=20 request → per-request k*′ = 18 (frozen 22).
  M1 − SC unchanged in sign: PILOT M1 .701 vs SC-18 .739 / SC-22 .740; HELD-OUT M1 .646 vs SC-18 .678 / SC-22 .687.
- Phase III RESERVE SampleMore (n=4 one request; gen prompt ≈ 101 tok → +303 tok/q under per-request accounting,
  TDC/SM ratio ≈ 0.90). TDC − SM with k_add = 3: −4.4 pp [−8.0, −0.9]; k_add = 2: −2.4 pp [−5.7, +0.9].
