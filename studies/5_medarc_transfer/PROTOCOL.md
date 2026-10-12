# Phase V — External validation of the six-gate framework on MedARC (DRAFT, not frozen)

Status: **frozen** (tag `phase5-freeze`, 2026-10-05). No Phase V LLM call had been made at the time of tagging.

## 0. Purpose and scope

Phase V is an external validation of the evaluation ladder in `FRAMEWORK.md` on an independently
developed, published, contemporary medical MAS: MedARC (Miao et al., Int J Med Inform 206:106136, 2026).
It is neither a reproduction attempt nor a method-improvement study.

**No performance-based progression criterion is used.** All 498 questions are evaluated once, and every
pre-specified gate estimate is reported regardless of its direction. After Phase V no further MAS is
added.

**Provenance** (see `MEDARC_AUDIT.md`)

| Layer | Content |
|---|---|
| Paper-confirmed | Structured inter-agent summarization (the summarizer matches the backbone in homogeneous settings); confidence-aware aggregation over **final-round** opinions with threshold 0.85, re-initiation if none passes, then highest confidence; T = 2 debate rounds by default; zero-shot / CoT baselines; DeepSeek-V3 headline PubMedQA 0.729 → 0.772; heterogeneous and round analyses; the authors note that self-consistency was not compared |
| Released-code-confirmed (`asdmiao/MedARC@3b6b5a6`) | 3 agents × (1 initial + 2 debate) = 15 calls per question; exact prompt strings; summary of the other agents' latest turns; released evaluator = highest stated confidence among **all 9** turns with no threshold; released yes/no parser; PubMedQA test file (498) |
| Phase V controlled choices | Qwen2.5-14B-Instruct for every agent and the summarizer (homogeneous); T = 1.0, max_tokens = 1,024 (unspecified in the release; declared here); controlled parser; offline aggregations (§4); token-matched SC; strong single baseline |

The configuration is named "**a released homogeneous MedARC configuration under a controlled common
backbone**".

## 1. Data

- **Evaluation set:** the released `data/pubmedqa_test.jsonl` (498 questions, unique pubids).
  - Generation inputs are pubid, question and contexts only.
  - Gold labels (`final_decision`) go to `runs/phase5/labels.json`, which only `sealed_analysis.py`
    reads.
  - The file's sha256 is recorded at freeze.
- **Token probe** (k\* only, no labels loaded): 30 PubMedQA PQA-L questions (`qiaojin/PubMedQA`,
  `pqa_labeled`) whose pubids are not among the 498. They are drawn with seed 61 from the
  sorted-pubid list. This project has never used PubMedQA.

## 2. Arms (Qwen2.5-14B-Instruct, local vLLM, BF16, `--generation-config vllm`)

Prompts and message construction are copied verbatim from the release: initial prompt, summarizer
system and user text, re-analysis prefix, and the history accumulation of `gen_pubmedqa_deepseek.py`.
Only the client, model name and the declared T and max_tokens change.

| Arm | Definition | Runs |
|---|---|---|
| **M** = MedARC | 3 agents; round 0 independent; rounds 1–2: each agent receives `generate_summary` (same model) of the other two agents' latest turns plus the re-analysis prefix; full history kept; T = 1.0, max 1,024 | **exactly 3** independent runs |
| B_released | released zero-shot (`gen_pubmedqa_deepseek_danlun.py`, 1 agent × 1 round), T = 1.0, max 1,024 | 3 |
| B_strong | released CoT template (zero-shot + "Let's think step by step."), **T = 0**, max 1,024; controlled parser | exactly 3 executions, kept for structural symmetry with Phase II. They are deterministic under T = 0 and are **not** treated as independent stochastic replicates. |
| SC-k\* | k\* independent samples of the released initial prompt (= independent round-0 agents), T = 1.0, max 1,024; controlled parser per sample; majority vote, ties to the earliest | **exactly 6** runs of k\* samples: runs 1–2 = pool A, run 3 = pool E, runs 4–6 = placebo pool P (the Phase II structure) |

Request seeds are deterministic: run·10⁶ + index·100 + call number. They make the runs reproducible
and do not change the method.

## 3. Answer and confidence extraction

**Released parser:** the verbatim released functions.
- `\boxed{…}` is taken from the first match. If there is none, the whole turn is used.
- `parse_answer` returns the first of "yes", "no", "maybe" found as a **substring**, in that order.
- Confidence is the first `confidence: N` match, or 0 if there is none.

**Controlled parser:**
- *Answer:* take the content of the **last** `\boxed{…}` in the turn. Inside it, the answer is the
  first whole word among yes / no / maybe. If there is no box, the answer is the last match of
  `answer\s*(?:is|:)\s*\**\s*(yes|no|maybe)\b`. Otherwise the turn counts as unparsed, and an
  unparsed turn scores as wrong.
- *Confidence:* the last match of `confidence\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(%?)` gives a number x.
  It is normalized to c ∈ [0, 1] as follows:
  - if a % sign follows: c = x / 100;
  - else if 1 < x ≤ 100: c = x / 100;
  - else if 0 ≤ x ≤ 1: c = x;
  - otherwise (or with no match) the confidence is invalid and c = 0.

  The release's prompt asks for "a confidence score (0%-100%)". The 0.85 threshold is applied to c.
  The released parser keeps its raw, unnormalized number, as released.
- **Unit tests before any LLM call.** Both parsers are tested on hand-written synthetic strings. The
  rules are never edited after any Phase V model output has been seen.

## 4. Aggregations (computed offline from the same generations; nothing is regenerated)

| Name | Rule |
|---|---|
| A_released | released evaluator behaviour: all 9 turns, highest released-extracted confidence (`argsort[::-1][0]`), released parser |
| **A_paper_no-reinit** (primary MedARC answer) | The paper-specified final-round confidence aggregation applies when at least one round-2 response reaches c ≥ 0.85: the highest c wins, and ties go to the lowest agent index. The paper's re-initiation procedure is not specified well enough for controlled reproduction, so no regeneration is performed. When no response reaches 0.85, the highest-c final-round response is used as a pre-specified deterministic fallback. The none-pass rate is reported as a sensitivity limitation and is not acted on. Controlled parser. |
| A_R2MV | majority of the round-2 answers; ties go to the highest confidence among the tied, then the lowest agent index. Controlled parser. |
| A_R0MV | the same rule on round-0 answers |

## 5. Gates and estimands

**Conventions.**
- Per-question accuracy is the mean over the 3 runs for M, B_released and B_strong. For SC-k\* it is
  p_SC(k\*), estimated from pools A ∪ E (runs 1–3) as in Phase II.
- Intervals are paired bootstraps over questions (10,000 resamples, seed 0). Multi-level estimands
  are clustered by question.

| Gate | Estimand |
|---|---|
| G1 | G_app = Acc(A_released) − Acc(B_released, released zero-shot evaluator). This is the *released-pipeline apparent gain under the common backbone*. It is **not** the paper's reported gain (+4.3 pp, DeepSeek-V3), which is cited only as a descriptive reference. **G_base = Acc(A_paper_no-reinit) − Acc(B_strong)**; **G_int = Acc(A_paper_no-reinit) − Acc(SC-k\*)**; G_het is N/A (MedARC has no roles) |
| G2 | Exactly the Phase II definition, computed by calling `experiments/phase2/oracle_analysis.analyze` **unchanged**. A test checks before any Phase V run that this call reproduces the frozen Phase II `RESULTS.json` (H, H_plc, H − H_plc) from the Phase II inputs. Mapping to the arguments: `S` = B_strong correctness (3 × Q; runs 1–2 = selection, run 3 = evaluation); `M` = A_paper_no-reinit correctness (3 × Q; same split); `sc_votes` = the SC answers of runs 1–3 concatenated (Q × 3k\*), with `pool_split` = 2k\* (pool A = runs 1–2 for selection, pool E = run 3 for evaluation); `placebo_votes` = runs 4–6 (Q × 3k\*), split in half inside `analyze`; `k` = k\*; `X` = a constant feature matrix (the G4 AUC output of `analyze` is not used in Phase V). Selection and evaluation always use disjoint runs or pools. **C\* = H − H_plc**. H and H_plc are reported too. |
| G3 | N/A (always-on; no selector) |
| G4 | κ_gen = P(gold ∈ round-0 answers); κ_post = P(gold ∈ round-2 answers); **Δκ = κ_post − κ_gen = κ_gain − κ_loss**, where κ_gain = P(gold ∉ A⁰, gold ∈ A²) and κ_loss = P(gold ∈ A⁰, gold ∉ A²). Computed per run and averaged; cluster bootstrap by question. Δκ is a difference in prevalence; κ_gain and κ_loss are reported so that it is not read as "new correct hypotheses". |
| G5 | For agent i and transition r → r+1, r ∈ {0, 1}: W_r means A_i^(r) ≠ gold and some other agent j has A_j^(r) = gold; C_r means A_i^(r) = gold and some other agent j has A_j^(r) ≠ gold. DI_r = P(A_i^(r+1) = gold \| W_r) − P(A_i^(r+1) ≠ gold \| C_r). **Primary for G5: pooled DI** over both transitions, all agents and all runs, with the bootstrap resampling questions (all observations of a question together). DI_0→1, DI_1→2 and the switch rates are secondary. The truth-discrimination label uses the Phase IV rule: truth-discriminating if the CI lower bound > 0; non-discriminating if the whole CI lies within ±0.10; inconclusive otherwise. |
| G6 | R_int = Acc(A_R2MV) − Acc(A_R0MV) (the population change through interaction); Acc(A_paper_no-reinit) − Acc(A_R2MV) and Acc(A_released) − Acc(A_R2MV) (the aggregation choice); parser sensitivity = A_released under the released vs the controlled parser; count of questions with no final-round turn ≥ 85 |

**Descriptive:**
- Acc(A_released) next to the paper's DeepSeek-V3 0.772, with no claim of replication;
- tokens and calls per question for every arm;
- unparsed rates under both parsers.

## 6. Compute matching (token probe; no labels)

On the 30 probe questions, run M once (15 calls each) and a single round-0 sample per question.
- T_M = mean MedARC tokens per question, including summaries.
- t_s = mean tokens of one independent sample.
- **k\*** = argmin_{k ≥ 1} |k·t_s − T_M|.

`runs/phase5/K_FROZEN.json` stores n_probe, T_M, t_s, k\*, the predicted SC tokens and the predicted
ratio. It is committed before the 498 are run. The realized token ratio is reported but does not
gate anything, and k\* is never recalibrated.

## 7. Order

1. Review, then tag `phase5-freeze`.
2. Commit the code: the runner ported verbatim, parsers, aggregations, `sealed_analysis.py`, and
   an invariant check that compares the rendered prompts with the release. Run the synthetic parser
   unit tests and the Phase II reproduction test for `analyze`.
3. Smoke test on 5 probe questions; labels are not loaded.
4. Token probe, then commit K_FROZEN.
5. Full run: all arms on the 498.
6. Stop the servers and validate.
7. Run `sealed_analysis.py` once.

Nothing is changed after the results. No further MAS experiment follows.

## Notes (implementation clarifications, written before any Phase V model output)

- **2026-10-05.** Two cases the frozen text left open are settled here.
  - Controlled-parser failures in SC samples enter the vote as an empty label. That is the
    behaviour of `oracle_analysis.p_sc`, which G2 must call unchanged, so an empty-label majority can
    never be correct. G1 uses the same `p_sc`.
  - Unparsed MedARC turns count as wrong answers in G4 and G5.
- **Released-code fact, recorded for provenance.** The released zero-shot prompt
  (`gen_pubmedqa_deepseek_danlun.py`) does not request a confidence score, whereas the MedARC
  initial prompt does. The paper says baselines use the initial-response template. B_released
  follows the code.
- **Verification before smoke.** The ported strings were confirmed byte-identical to the release
  with `test_released_strings.py` (AST constants). Parser and aggregation unit tests pass
  (`parsers.py test`, `aggregate.py test`).

## Amendments

### Amendment 1 (2026-10-05) — controlled confidence extraction (implementation only; approved by the user)

*Seen at the time of writing:* the DEV smoke outputs on 5 probe questions, by **format only**. No
labels were loaded, and no accuracy or aggregation result was computed.

Qwen2.5-14B states confidence as "Confidence score: 70%", "a confidence score of 60%", "My confidence
score for this answer is 75%" or "\textbf{60\%}". The frozen controlled regex
`confidence\s*[:\-]?\s*(\d+…)` required the number to follow "confidence" directly. It therefore
extracted **no** confidence (0/15 SC samples), which would have silently disabled the
confidence-aware aggregation. The frozen intent is "extract the model's stated confidence", and the
prompt asks for "a confidence score (0%-100%)".

**Change** (`parsers.CONF`):
- The number is the first one within 60 characters after "confidence" or "confident", on the same
  line. Matches of "confidence interval(s)" are excluded.
- An optional `%` or `\%` follows the number.
- The last such match in the turn is used.
- Normalization (§3) and the 0.85 threshold are unchanged.

**Tests.** The 15 observed formats and 2 "confidence interval" cases were added to the synthetic
tests, and all tests pass.

**Re-smoke** (the same cached generations re-parsed; still no labels):
- controlled confidence extracted for 45/45 MedARC turns and 15/15 SC samples;
- MedARC: 15 calls per question (9 agent, 6 summary), no truncation, well-formed summaries;
- the **released** extractor still finds 0/45. This is kept verbatim and reported under G6 as
  released evaluation behaviour under this backbone.

Nothing else was changed.

### Amendment 2 (2026-10-05) — SC request batching (implementation only; before any evaluation-set output)

*Seen at the time of writing:* the token-probe outputs, tokens only, and the smoke outputs, format
only. No labels had been loaded, and no output on the 498 questions existed.

**Problem.** The frozen rule is k\* = argmin |k·t_s − T_M|, where t_s is the cost of one independent
sample. It was measured with one request per sample (n = 1), so each sample includes its prompt of
about 400 tokens. The SC runner, however, drew the k\* samples in a single request (n = k\*), and
vLLM then counts the prompt only once. The SC arm would have received about 12k tokens per question
against MedARC's 21k, which breaks the frozen matching.

**Change.** SC issues k\* separate n = 1 requests (seeds run·10⁶ + idx·100 + j). Every sample now
pays its own prompt tokens, matching both t_s and MedARC's per-call accounting. The k\* value from
the probe is unchanged: k\* = 25, with predicted ratio T_M / (k\*·t_s) = 0.991.

**Re-smoke.** The SC arm was re-run on 5 probe questions with k = 3: 3 requests per question, 765 tokens per sample (858 on the 30-question probe). Token logging was checked; no
labels were loaded.

*Clarification to Amendment 2.*
- The official frozen quantities are **t_s = 858** (30-question token probe) and **k\* = 25**.
- The 765 tokens per sample in the re-smoke is an implementation check that independent-request
  accounting is applied. It is not a new t_s.
- The amendment was made before any evaluation-set (498) output had been generated.
- After the full run, the realized ratio T_SC25 / T_M is reported as is; k\* is never re-matched.
