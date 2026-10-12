# Phase IV — Does targeted interaction tell correction from misleading counterclaims?

Status: **frozen** (tag `phase4-freeze`, 2026-10-05). No Phase IV LLM call had been made at the time of
tagging.

## 0. Isolation, motivation, and what had been seen

**Read-only:** Part I, Phase II and Phase III, together with their tags.

**Motivation.** This phase was designed after post-hoc Phase III analysis (`runs/cud/CUD.md`), and
that analysis is stated openly here. Over 220 Phase III representative revisions:
- 49.5% of representatives switched answer.
- **P(C→W | started correct) = .406**, and P(W→C | started wrong) = .265.
- Per-revision net utility was positive (BRR .182 vs HRR .127). Even so, TDC lost at the system level
  (rescue 7 vs harm 16).
- In 41 of 111 triggered questions, gold was outside {X, Y}.

The Phase III failure therefore has several **candidate failure modes**: susceptibility to
counterclaims, coverage (gold outside the debated pair), and aggregation. They overlap, and no
additive decomposition of the −8.1 pp is claimed.

**Question.** Does targeted interaction distinguish corrective evidence from misleading
counterclaims? And how does local revision behaviour relate to system-level utility?

The aim is to measure truth discrimination, not to prove persuasion.

## 1. Data: the reassigned, unopened split

The 1,000 questions were originally designated Phase III FINAL. That split **remained completely
unopened**: no LLM call, and neither its labels nor its generation file were read. The reason is
that Phase III stopped at its pre-specified NO-GO criterion. Before any access, its use was
reassigned prospectively to Phase IV.
- The original hash is unchanged: `runs/phase3/splits/final.json`, sha256 `4dfd0831…`. It is
  recorded in full in `runs/phase4/splits/ALIAS.sha256`, together with the generation-file and
  label-file hashes.
- RESERVE (1,683) stays untouched. Any follow-up experiment ("Phase IV-2") must use RESERVE, never
  these 1,000 questions.
- Labels are read only by `experiments/phase4/sealed_analysis.py`.

## 2. Procedure (backbone Qwen2.5-14B-Instruct; local vLLM as in Phase III)

1. **Probe, state, trigger, top-2 and representatives.** These are exactly the Phase III rules
   (§3.1–3.3 of PHASE3_FREEZE.md): the same prompts, T, token limits, seed formula and E3.
2. **Disagreement extraction.** Exactly as in Phase III (`extract_disagreement.txt`, T = 0), once per
   triggered question that has two parsed answers.
3. **Four revision conditions** per representative, rep(X) and rep(Y). Each uses T = 0, max 700
   tokens, and E3 on the last 2,500 characters. A revision whose E3 fails keeps the original
   answer, and this is counted.

| Condition | What follows the agent's own original turn (question, its full reasoning) |
|---|---|
| **Control** (self-review) | `revise_control.txt`: re-examine your reasoning, then give your final answer |
| **Blinded critique** | `revise_blinded.txt`: your reasoning may contain an error; re-examine it, then give your final answer |
| **Opponent claim** | Phase III `targeted_revision.txt`, unchanged, with this question's CORE and the opposing claim |
| Matched irrelevant claim (robustness control) | the same template as Opponent, but with the CORE and the opposing claim of the **next triggered question in qid order** (cyclic) |

- **Independence.** Each condition starts independently from the identical original representative
  context. No revision generated in one condition is visible to any other condition. Every call is
  a separate, stateless request with its own cache key.
- Conditions run in the fixed order Control, Blinded, Opponent, Irrelevant for every question. This
  matters only for reproducibility, because there is no carry-over.
- The Irrelevant condition varies both the relevance and the correctness of the challenge. It
  answers only one question: "is arbitrary challenge content enough to induce switching?" It plays
  no role in the primary analysis or the verdicts.

## 3. Populations

- **Primary mechanism population: discordant pairs.** These are triggered questions with two
  parsed answers where exactly one of X and Y equals gold. Each such question contributes one
  correct-side representative and one wrong-side representative.
- **Coverage population:** all triggered questions. This population is descriptive only: the
  shares with gold ∈ {X, Y}, gold ∉ {X, Y} (both representatives wrong), and both correct
  (impossible, since X ≠ Y); and the state distribution.

## 4. Estimands and analysis (sealed; discordant pairs unless stated)

For each condition c:
- BRR_c = P(after = gold | wrong-side representative)
- HRR_c = P(after ≠ gold | correct-side representative)

All intervals are 95% bootstrap intervals clustered by question (both sides of a pair are resampled
together), with 10,000 resamples and seed 0.

**Primary: DI_opp = BRR_opp − HRR_opp.**

**Key secondary:**
- Opponent − Control: ΔBRR and ΔHRR.
- Opponent − Blinded: ΔBRR and ΔHRR.
- Blinded − Control: ΔBRR and ΔHRR (the generic-challenge effect).
- Switch rates for each condition, plus GenericChallengeEffect = Switch(Blinded) − Switch(Control)
  and SpecificClaimEffect = Switch(Opponent) − Switch(Blinded).

**Additional control:** DI, BRR, HRR and the switch rate under the Irrelevant condition.

**Descriptive:**
- DI_opp by state (weak / split / fragmented);
- the coverage shares (§3);
- revision E3 failure rates;
- extraction parse rates.

**Main figure.** BRR and HRR (with CIs) as two lines across Control → Blinded → Opponent, with the
Irrelevant condition shown as separate markers.

**Verdicts.** Two layers, both fixed now.

1. *Truth discrimination* (from DI_opp):
   - **truth-discriminating:** the CI lower bound is > 0;
   - **non-discriminating:** the whole CI lies within [−0.10, +0.10], an equivalence margin fixed
     now;
   - **inconclusive:** anything else.
2. *Effect of the specific counterclaim* (Opponent vs Control):
   - **corrective:** the ΔBRR CI lower bound is > 0, and the ΔHRR CI upper bound is ≤ 0 or the
     ΔHRR CI includes 0;
   - **corruptive:** the ΔHRR CI lower bound is > 0, and the ΔBRR CI includes 0 or is negative;
   - **mixed:** the CI lower bounds of both ΔBRR and ΔHRR are > 0;
   - **no detectable effect:** anything else.

Named readings:
- *Persuasion-consistent* = non-discriminating AND (corruptive OR mixed).
- *Correction-sensitive* = truth-discriminating AND corrective.
- Every other combination is reported by its two labels, for example "truth-discriminating +
  mixed". An inconclusive DI is never relabelled as persuasion.

**Local versus system-level utility (exploratory).** These analyses are labelled as exploratory
sensitivity analyses. Neither of them is used to claim that a single mechanism caused the
Phase III loss.
- On the Phase IV triggered questions, recompute the Phase III TDC final answer from the Opponent
  revisions and the frozen aggregation rule, and report its accuracy next to the per-revision
  BRR − HRR.
- On Phase III data, report how sensitive the final outcome is to the aggregation rule (for
  example, revised votes only).

## 5. Order and commitments

1. Tag `phase4-freeze`. This document and its prompts are in it, and the alias manifest records
   the hashes.
2. Commit the code: `sealed_analysis.py` and every runner.
3. DEV smoke test on at most 10 DEV questions with indices 100–199 (not used by Phase III). Labels
   are not loaded. Checks: formats, independence of contexts, token logging.
4. One full run on the 1,000 questions.
5. Stop the servers.
6. Run `sealed_analysis.py` once.

**After the results:** no prompt, condition, population or threshold is changed, and no condition
is added, on these 1,000 questions. Any follow-up uses RESERVE under its own freeze.

## Amendments

### Amendment 1 (2026-10-05) — implementation only

This amendment was written before any Phase IV model output existed.
- The temporary global prompt-path mutation was replaced by an explicit prompt-path renderer
  (`run_revisions.prompt4`). The old approach could have caused a multithreading race. Its output
  was verified to be identical to the original Phase III prompt rendering.
- `check_invariants.py` was added. It rebuilds every revision request from the stored probe and
  extraction, then compares it with the request actually sent (cache). It checks:
  - that every condition is independent and starts from the original representative turn;
  - that opponent information does not leak into Control or Blinded;
  - that the Irrelevant content comes from the next triggered qid;
  - that E3, the token fields and truncation are recorded.

The DEV smoke test (DEV 100–109, labels not loaded) passed all checks: 24 revision requests,
0 failures. Nothing in the protocol, prompts, populations or thresholds was changed.

### Note (2026-10-05, after the full run and before `sealed_analysis.py`; labels not read)

`check_invariants.py` on the full run checked 2,848 revision requests. Every one matched the frozen
construction: no request differed from what the rules specify.

It flagged 11 items:
- **3 truncations** (finish = length), 0.1% of requests. These are recorded as specified, not
  protocol violations.
- **8 "leak" flags in 4 questions** (113, 367, 2299, 2725). These are false positives of the string
  check. In each case the extracted "opposing claim" occurs verbatim in that agent's *own* reasoning,
  so the Control and Blinded requests contain it through the agent's own turn and nothing else. The
  extractor attributed one of the agent's own sentences to the opponent. This is an extraction-quality
  case and is reported as a diagnostic.

Nothing was rerun or changed.
