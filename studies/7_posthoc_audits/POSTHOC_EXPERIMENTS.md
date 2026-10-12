# Post-hoc experiments for the ARR revision (2026-10-11)

Status: **post-hoc, descriptive**. Definitions fixed and committed before any of these analyses were run.
No frozen result, threshold or label is changed. Existing data only: Phase V (MedARC, n=498) and Phase VI
(arms A interactive / B no-summary / C placebo summary; identical round-0 turns; n=472), 3 runs each.
CIs: 10,000 cluster bootstrap over questions, seed 0. Answers: controlled parser (`aggregate.controlled`).
Builds on POSTHOC_TRANSITION.md (same definitions of avail / comp / preserved / lost / realized).

## E1. Complementary-candidate conversion matrix

comp = gold ∈ R0 ∧ p_SC25 < 0.5. Four cells per (q, run): Realized (gold ∈ R2, final correct),
Preserved-unrealized (gold ∈ R2, final wrong), Lost (gold ∉ R2, final wrong), Other (gold ∉ R2, final correct;
must be 0 — consistency check, asserted). Reported at two units: q×run (pooled ratio) and question
(per-question cell shares averaged over the question's comp runs, then averaged over questions).
Relative contribution: Lost − Preserved-unrealized share (pp, CI). Phase V; Phase VI A, B, C; A−B, A−C contrasts.

## E2 → superseded by E2′ (below) before any run. E3, E4: not run (user decision 10-11: one added analysis only).

## E2′. Summary-level information preservation and uptake (v2, fixed before running)

Data: Phase VI arm A (primary; identical round-0 with B/C) and Phase V MedARC (replication). Unit of a *summary
event* e = (q, run, round r ∈ {1,2}, agent i). The summary that agent i reads at round r is a summary of the other
two agents' round r−1 turns; "Response 1" = lower-index other agent, "Response 2" = the other (as in the code).

**Judge.** Qwen3-32B-FP8 (a different and larger model than the Qwen2.5-14B agents), non-thinking mode, T=0, JSON.
The judge never sees the gold label or which answer is correct, and is told to judge only by comparing texts.
Prompts are fixed in `experiments/posthoc_summary_judge.py`.

Stage J1 — preservation, one judgment per (e, source response j):
inputs = question, the original round r−1 turn of source j, the summary. Outputs:
`final_answer_original` (yes/no/maybe/unclear), `answer_in_summary` (yes/no/maybe/unstated),
`evidence_preserved` (full/partial/none: are the original's key reasons for its final answer conveyed),
`misrepresented` (summary attributes a conclusion/reasoning the original does not contain, or frames it as
supporting a different answer).
Stage J2 — uptake, for each (e, j) where J1 says the summary conveys a label L (answer_in_summary = L) that
differs from agent i's own parsed r−1 answer: inputs = summary, agent i's round-r turn, L. Output
`engagement` ∈ {adopted, considered_rejected, ignored}.

**Derived states** of a source j in event e (gold used only in analysis): *conveyed+evidence* (answer_in_summary =
source label and evidence full/partial), *answer only* (label conveyed, evidence none), *not conveyed*
(unstated or another label); misrepresented flagged separately. Source labels come from the controlled parser;
J1's `final_answer_original` agreement with the parser is reported as a sanity check.

**Analyses** (cluster bootstrap over questions):
- P1 Preservation: distribution of states for gold-carrying vs wrong-carrying sources; also restricted to
  disagreement events (the two sources differ).
- P2 Uptake (agent i wrong at r−1, ≥ 1 gold source): P(agent i gold at r) by best state of the gold source(s);
  J2 engagement distribution.
- P3 Wrong-information path (agent i gold at r−1): P(agent i leaves gold at r) when the summary conveys a wrong
  label and no gold label vs otherwise; incorrect convergence (R2 unanimous wrong | gold ∈ R0) for A, B, C.
- P4 Lost-candidate trace (headline): for each lost (q, run) (gold ∈ R0, gold ∉ R2) find the round t at which the
  last gold holder(s) dropped it. Classify each drop event (holder agent h, gold at t−1, not at t) by the summary h
  read at t: **erased** (a source carried gold but the summary did not convey it), **conveyed-but-dropped** (summary
  conveyed gold), **wrong-only** (no source carried gold; summary conveyed a wrong label), **no-label** (no source
  carried gold; summary conveyed no label). Case-level shares = mean over the case's drop events. Reported for
  all lost cases and for comp cases.
- Answer-mention-only reading vs evidence reading are both reported, so the gap between "label mentioned" and
  "reasons conveyed" is visible.

**Human check.** 80 J1 items from Phase VI A (40 gold-carrying, 40 wrong-carrying sources, seed 0), blind: no gold,
no judge output. Sheet `runs/posthoc/human_check/sheet.csv`; key kept separately. Agreement (Cohen's κ per field)
is computed only after the user fills the sheet. Until then the judge results are reported as unvalidated.

Interpretation: observational associations inside arm A; drop-event classification describes what the agent
read when it dropped gold, not why. No causal claim about summary content.

## E2′ additions decided after seeing J1 descriptives (labelled post-hoc-of-post-hoc)

- **Judge validity control** (`experiments/posthoc_judge_control.py`): 500 VI_A J1 items re-judged with the summary of
  a different question. Added because J1 rated ~99% of responses as conveyed with full evidence.
- **Holder-exposure contrast across A/B/C** (no judge; in `posthoc_summary_analysis.exposure_abc`): gold holder at
  r−1; P(leave gold at r) by whether both other agents were wrong at r−1. r = 1 only is the clean comparison
  (conditions on round-0, byte-identical across arms); r = 1+2 reported for completeness.

## E2′ results (2026-10-11; `runs/posthoc/SUMMARY_E2.{txt,json}`, judge outputs `judge_j1/j2.jsonl`)

- Judge sanity: answer agreement with parser 1.000. Control: label = source 0.988 (true summary) vs 0.186 (mismatched);
  evidence none 0/500 vs 389/500; misrepresented 0.002 vs 0.674.
- P1 (VI_A): gold-carrying responses conveyed with label + evidence 99.1% (evidence full 97.0%); wrong-carrying 99.6%.
  Same in disagreement events (98.7% / 99.7%) and in Phase V. → no evidence of erasure at the summary step.
- P4 lost-candidate trace (VI_A, 147 lost q×run): erased 0.3%, gold conveyed but holder dropped 28.9%,
  lone holder whose summary conveyed only wrong labels 63.9% (comp subset: 73.9%). Phase V: 0.3 / 29.7 / 66.4%.
- P3: holder whose summary conveys only wrong labels leaves gold 64.0% vs 7.3% otherwise (Phase V 62.8% vs 7.7%).
- Holder exposure, r = 1 (same 155 events in every arm): P(leave gold | both others wrong) A 62.6% [53.9, 70.6],
  B 21.9% [15.2, 29.1], C 15.5% [9.9, 21.6]; A − B +40.6 pp [+30.7, +50.3], A − C +47.1 pp [+37.2, +56.6].
  With ≥1 other holder: A 9.6%, B 6.7%, C 5.6%.
- Incorrect convergence (R2 unanimous wrong | gold ∈ R0): A 12.9% [10.3, 15.6] vs B 4.4%, C 4.3%; Phase V 11.6%.
- P2 uptake: agent wrong at r−1 with conveyed gold switches to gold 36.0% (VI_A), 36.3% (V); J2: adopted 701,
  considered-and-rejected 558, ignored ≈ 0 (VI_A).
- Human check sheet (80 items, blind) written; agreement pending.

## Human check (2026-10-11; `runs/posthoc/human_check/sheet_labeled.csv`, `AGREEMENT.json`; 80 blind items, 1 annotator)

- final_answer_original 80/80, answer_in_summary 80/80 agree (κ = 1.0).
- evidence_preserved: judge "full" on all 80; human full 34 / partial 46 / none 0 → agreement .425, κ = 0 (judge has
  no variance). The collapsed category used in P1/P4 (label conveyed ∧ evidence ∈ {full, partial}) agrees 80/80.
  Reading: the judge cannot separate full from partial; the 97% "evidence full" figure is not reported.
- Human, by stratum: gold-carrying full 26 / partial 14; wrong-carrying full 8 / partial 32 (Fisher exact p = 9.2e-5).
- misrepresented: human true on 1 item (numeric error in a summary's Differences), judge false on all; agreement .9875.
