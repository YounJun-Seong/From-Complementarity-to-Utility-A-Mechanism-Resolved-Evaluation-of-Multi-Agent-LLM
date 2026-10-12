# Phase III — Targeted Disagreement Collaboration (TDC) vs sampling more (DRAFT pilot freeze)

Status: **frozen** (tag `phase3-pilot-freeze`, 2026-10-05). No Phase III LLM call had been made at the
time of tagging. Later changes are
logged as dated amendments that state what had been seen.

## 0. Isolation

- Part I tags (`p0-freeze`, `final-freeze`, `final-freeze-v2`) and Phase II tags (`phase2-freeze`,
  `phase2-router-freeze`) are read-only. No Phase I/II question, output or result is used to develop,
  tune or select anything in Phase III. This covers MedQA in full, MedMCQA R2, MedMCQA-G, and all
  PILOT/HELD-OUT outputs.
- Namespaces: code `experiments/phase3/`, outputs `runs/phase3/`, LLM cache `data/phase3_cache`.
- Generation inputs contain no gold labels. Labels live only in `runs/phase3/splits/<split>_labels.json`,
  and only `sealed_analysis.py` reads them.

## 1. Question

**RQ.** When independent reasoning ends in unresolved disagreement, does targeted interaction resolve it
better than spending the same additional inference compute on independent sampling?

**Primary hypothesis.** On questions whose 4-sample probe is not unanimous, TDC is more accurate than
SampleMore at approximately matched incremental tokens. HeteroMore is a secondary control that
separates role heterogeneity from interaction.

Phase III builds no router and no learned trigger. The trigger is the observed disagreement itself.

## 2. Data (fixed; `experiments/phase3/make_splits.py`, seed 51)

**Pool.** MedMCQA validation split (4,183 questions, 4 options), minus:
- Part I R2 (`MedMCQA_r500`),
- Phase II MedMCQA-G, which stays reserved for Phase II Criterion C,
- exact duplicates (normalized question + options).

That leaves 3,183 questions, none of which has ever been run.

| Split | n | Use |
|---|---|---|
| DEV | 200 | §5 steps 2–3 only: smoke test (≤ 20 questions) and token probe. No accuracy is computed. |
| PILOT | 300 | GO / NO-GO (§6) |
| FINAL | 1,000 | untouched; run only after a PILOT GO |
| RESERVE | 1,683 | untouched; any use requires an amendment |

Hashes are in `runs/phase3/splits/SPLITS.sha256`. The generation files are
`datasets/MedMCQA_p3<split>/test.jsonl` (question, options, qid; no gold).

## 3. Protocol (backbone Qwen2.5-14B-Instruct only; local vLLM, BF16, `--generation-config vllm`)

All prompts are frozen verbatim in `experiments/phase3/prompts/`. Answer extraction everywhere is E3:
`extract_answer.txt`, T=0, 20 tokens, JSON `{"choice": …}`. Its input is the first 2,500 characters of
a reasoning. The one exception is revisions, where the input is the **last** 2,500 characters,
because the revised answer is at the end.

### 3.1 Probe (shared by every arm)

- 4 samples of `solve.txt`, drawn as one request with n = 4, T = 0.7, max 700 tokens,
  seed = 51·10⁶ + qid.
- E3 is applied per sample.
- Each sample is stored with its answer, reasoning, raw response, and input/output tokens.

### 3.2 Disagreement state and trigger

A probe answer that E3 fails to parse is a **parse failure**. With n = number of parsed answers (0–4)
and m = the count of the most common parsed answer:

- **unanimous:** n = 4 and m = 4. All four probe answers parse and agree. The action is STOP and the
  answer is that letter.
- **weak:** n = 4 and m = 3 (3-1).
- **split:** n = 4, m = 2, and exactly 2 distinct answers (2-2).
- **fragmented:** n = 4 and any other pattern (2-1-1, 1-1-1-1), **or 1 ≤ n ≤ 3**. Any parse failure
  triggers. For example, A A A fail and A fail fail fail are both fragmented.
- **failure:** n = 0. There is no valid probe answer, so the question is scored wrong in every arm.
  It does not trigger, and it is excluded from the triggered set and reported separately.

Parse failures are never counted as votes. The probe parse success rate is reported (§6).

**Trigger:** any state other than unanimous. Only triggered questions receive the arms below.

### 3.3 TDC (treatment)

1. **Top-2 answers.**
   - Rank the distinct parsed answers by count (descending), breaking ties by earliest first
     occurrence in probe order.
   - X = rank 1 and Y = rank 2. Lower-ranked answers stay in the vote but take no part in the
     interaction.
   - If only one distinct answer parsed (for example A A A fail), there is no opposing answer. TDC
     makes no calls, its final answer is that answer, and its incremental tokens are 0.
   - In that case SampleMore and HeteroMore still run as specified, so the comparison stays paired.
     The number of such questions is reported.
2. **Representatives.** rep(X) and rep(Y) are the first probe samples whose answers are X and Y.
3. **Disagreement extraction.**
   - One call to `extract_disagreement.txt`, T = 0, max 300 tokens.
   - The fields `CLAIM_X`, `CLAIM_Y` and `CORE_DISAGREEMENT` are parsed line by line.
   - If CORE is missing, the stripped raw response (first 1,000 characters) is used in its place. A
     missing claim becomes "(not extracted)". The parse success rate is reported.
   - The extractor never chooses an answer.
4. **Targeted revision (one round).**
   - For each of rep(X) and rep(Y), one call to `targeted_revision.txt`, T = 0, max 700 tokens.
   - The call contains the agent's own original turn (question, its full reasoning, its answer), the
     CORE disagreement, and **only the opposing claim**. The opposing reasoning is not shown.
   - The revised answer is E3 on the revision. If it fails to parse, the revised answer is the
     original answer.
   - There is no further round.
5. **Final aggregation.**
   - Pool = the parsed probe answers (in probe order), then revised(X), then revised(Y).
   - The answer with the highest count wins.
   - Ties among the top answers are resolved in order:
     1. if revised(X) = revised(Y) and that answer is among the tied ones, it wins;
     2. otherwise the probe-majority answer (count, then earliest), if it is among the tied ones;
     3. otherwise the tied answer that occurs earliest in pool order.

   For example, A A C C with revisions A→A and C→A gives the pool A A C C A A, so the answer is A. With
   revisions A→A and C→C, the pool is A A C C A C (a 3–3 tie). Rule 1 does not apply. In rule 2 the
   probe majority is A, because the probe is 2–2 and A occurs first, so the answer is A.

### 3.4 SampleMore (primary control)

- k_add more samples of `solve.txt`, drawn as one request with n = k_add, T = 0.7,
  seed = 52·10⁶ + qid. E3 is applied per sample.
- Final answer = majority over the probe answers followed by the additional answers, with ties going
  to the earliest in that order.

### 3.5 HeteroMore (secondary control)

This is the Phase II HET protocol, applied incrementally:
1. MedAgents' released expert recruitment (5 question + 2 option domains, T = 0), parsed as M1
   parses it. The recruitment tokens are charged to HeteroMore.
2. k_het more samples, assigned round-robin over the recruited roles. Each uses the MedAgents
   analyzer system prompt for its role and the `solve.txt` user prompt, T = 0.7, 700 tokens,
   seed = 53·10⁶ + qid·10 + role.
3. E3 per sample. Final answer = majority over the probe answers followed by the HET answers, with
   ties going to the earliest.

There is no interaction in this arm.

### 3.6 Full MAS (secondary, efficiency only)

MedAgents M1, released protocol, one run on the triggered questions. It answers on its own and is
not pooled with the probe. It is reported with its full token cost.

## 4. Compute accounting and matching

**Incremental tokens** of an arm on a triggered question are all input and output tokens of its calls
after the probe, E3 calls included. The probe (4 samples + 4 E3 calls) is common to every arm and is
reported separately.

Both k values are fixed on the DEV token probe (§5 step 3) from **token counts only**:
- T_TDC = mean incremental TDC tokens.
- t_s = mean tokens per additional sample, including its E3 call, measured on the probe samples.
- t_r = mean HeteroMore recruitment tokens; t_h = mean tokens per role sample.
- **k_add** = argmin_{k ≥ 1} |k·t_s − T_TDC| and **k_het** = argmin_{k ≥ 1} |t_r + k·t_h − T_TDC|.
  Ties go to the smaller k.

Both values are written to `runs/phase3/K_FROZEN.json` and committed before PILOT starts. The file also
stores how they were computed: `n_token_probe`, `T_TDC`, `t_s`, `t_r`, `t_h`, `k_add`, `k_het`,
`predicted_ratio_add` = T_TDC / (k_add·t_s), and `predicted_ratio_het` = T_TDC / (t_r + k_het·t_h).
**If P2 fails on PILOT, k_add and k_het are not recalibrated using PILOT outcomes or token counts.**

## 5. Execution order (fixed)

0. This document is reviewed, committed and tagged `phase3-pilot-freeze`. The code in §9 is committed
   before step 2.
1. Splits and hashes (done by `make_splits.py`; committed with this document).
2. **DEV smoke test**, ≤ 20 DEV questions, all arms. The checks are only these: the parser, the
   extraction format, revision parsing, crashes, and token logging. No accuracy is computed, because
   labels are not loaded. Fixes here are limited to software defects such as parsing, crashes and
   formatting. Prompts are not edited unless a prompt fails to produce the frozen output format; any
   such edit becomes an amendment before step 3.
3. **DEV token probe.** Run the probe on DEV questions in order until 20 are triggered, then run TDC,
   HeteroMore recruitment and one role sample on those 20. Only tokens are read. k_add and k_het are
   fixed (§4) and committed.
4. **PILOT probe** on all 300.
5. **PILOT arms.** On every triggered PILOT question, run TDC, SampleMore (k_add), HeteroMore (k_het)
   and Full MAS, one run each. All arms use the same questions, so every comparison is paired.
6. All vLLM servers are stopped.
7. The pre-committed `sealed_analysis.py` is run once.
8. §6 decides. On GO, FINAL runs with the identical protocol and k values, and the same
   `sealed_analysis.py` is used with the split changed. On NO-GO, Phase III ends: there is no
   prompt, trigger or aggregation rescue, and no FINAL run.

## 6. Endpoints and decision (PILOT, triggered questions only)

| | Definition | Requirement |
|---|---|---|
| **P1 (primary)** | Δ = Acc(TDC) − Acc(SampleMore), paired bootstrap over questions (10,000, seed 0) | Δ ≥ +2.0 pp and 95% CI lower bound > 0 |
| **P2 (compute)** | mean incremental tokens TDC / SampleMore | ratio in [0.9, 1.1] |
| P3 (mechanism) | Rescue = SampleMore wrong & TDC right; Harm = SampleMore right & TDC wrong | Rescue > Harm (reported; sanity check) |

- **GO** requires P1 and P2. P3 is reported in every case.
- The +2.0 pp in P1 is the **minimum observed effect for progression**, a screening threshold on the
  point estimate. It is not a superiority margin that the CI must exceed: the CI lower bound only
  has to be > 0. PILOT is a progression gate, and FINAL provides the prospective confirmation. The
  paper keeps these roles separate.
- P2 can fail without P1 failing. In that case the result is reported as "comparison not
  compute-matched", and the decision is NO-GO.

**Secondary (pre-declared, no decision role):**
- Acc(TDC) − Acc(HeteroMore) and Acc(HeteroMore) − Acc(SampleMore), with CIs.
- Full MAS accuracy and tokens.
- Δ by state (weak / split / fragmented), each with n and CI. The primary endpoint is never replaced
  by a state subgroup.
- **Implementation diagnostics**, each reported separately:
  - probe parse success rate, and the counts of each state including failure;
  - disagreement-extraction success (all three fields parsed);
  - CORE fallback rate and missing-CLAIM rate;
  - revision E3 failure rate;
  - the number of triggered questions with only one parsed answer.
- **Revision flow**, computed without gold:
  - X stays X / X→Y / X→other;
  - Y stays Y / Y→X / Y→other;
  - both converge on X / both on Y / both on a third answer / disagreement remains / swap
    (X→Y and Y→X);
  - P(convergence) overall and by state (weak / split / fragmented).
- Overall accuracy over all questions for the TDC system (STOP on unanimous questions) versus the
  SampleMore system.

**Output tables** (fixed, in this order):
1. Condition × (accuracy, incremental tokens/q, Δ vs SampleMore, 95% CI) for SampleMore, HeteroMore,
   TDC and Full MAS.
2. State × (n, SampleMore, TDC, Δ).
3. Transitions SampleMore → TDC: rescue, harm, both correct, both wrong.

## 7. FINAL (only after GO)

FINAL uses the same protocol, prompts, k values and analysis, on 1,000 questions. Its primary endpoint
and requirements are P1 and P2 as in §6. Backbone or dataset extensions come only after FINAL, each
by its own amendment.

## 8. Conventions

- One run per arm. A crash or empty answer is scored wrong unless a rule above says otherwise.
- Every run must pass `validate_runs.py`: expected qids, no API failures, the stored fields present.
- Token accounting follows Part I R3. Servers run only for the duration of a step.

## 9. Code (to be committed before §5 step 2)

| File | Role |
|---|---|
| `experiments/phase3/prompts/*.txt` | frozen prompts |
| `run_probe.py` | probe and state classification |
| `run_tdc.py` | TDC |
| `run_sample_more.py` | SampleMore |
| `run_hetero_more.py` | HeteroMore |
| `aggregate.py` | the §3.3 and §3.4 rules, with unit tests of the examples above |
| `token_match.py` | §4 |
| `validate_runs.py` | run checks |
| `sealed_analysis.py` | §6; the only reader of labels |

Full MAS reuses MedAgents `run.py`. Because that script requires a gold field, it receives a
placeholder label, and its outputs are scored by `sealed_analysis.py` from the label file.
- The placeholder is a constant identical for every question: `answer_idx = "A"` and
  `answer = "PLACEHOLDER"`. The runner asserts
  `len({r["answer_idx"] for r in rows}) == 1 and len({r["answer"] for r in rows}) == 1` before
  launch.
- The placeholder is verified not to enter any prompt, recruitment decision, generation branch or
  other model-visible computation. Checked 2026-10-05:
  - `run.py` reads `answer_idx` only to pass it to `fully_decode`;
  - `utils.fully_decode` only writes it into the output record (`gold_answer`, line 132);
  - the `data_utils` reference builders that read `answer` are never called by `run.py`.

## Amendments

(none)

### Phase III Amendment — RESERVE Confirmatory Evaluation (2026-10-06)

**Rationale.** Phase III terminated at the frozen PILOT-stage NO-GO decision, so the pre-allocated
FINAL split was never executed under the Phase III protocol. That split was later reassigned to the
Phase IV mechanism study (`PHASE4_FREEZE.md` §1) and is no longer an untouched confirmatory
evaluation set.

Disclosure: Phase IV ran the same probe procedure and the same targeted-revision prompt on those
1,000 questions. Its sealed analysis computed, as an exploratory quantity, the accuracy of the TDC
aggregation on the FINAL triggered questions (.396 vs probe majority .419). No Phase III comparator
(SampleMore) was run on FINAL.

**Action.** The frozen Phase III disagreement-triggered policy is evaluated on the previously unused
RESERVE partition (N = 1,683; sha256 `8b2c8e47…`, unchanged since the Phase III freeze).

**No protocol changes.** The following stay exactly as frozen:
- the four-probe generation procedure;
- seed construction;
- the non-unanimous trigger;
- top-two candidate selection;
- disagreement extraction;
- the targeted revision prompt;
- deterministic aggregation;
- the SampleMore comparator with k_add = 4 and HeteroMore with k_het = 3 (`K_FROZEN.json`);
- Full MAS;
- the compute-matching definition;
- the primary analysis and bootstrap procedure (`sealed_analysis.py --split reserve`, run once).

All four Phase III arms are run so that the frozen analysis script applies unchanged. The primary
endpoint is P1 (TDC − SampleMore on triggered questions), together with the realized token ratio
(P2) and rescue/harm (P3). Secondary: weak / split / fragmented. The GO/NO-GO mapping is not used to
reopen Phase III. This evaluation is reported as an additional frozen confirmatory evaluation.

This amendment changes only the evaluation partition. No outcome-dependent retuning or modification
of the Phase III intervention is permitted.
