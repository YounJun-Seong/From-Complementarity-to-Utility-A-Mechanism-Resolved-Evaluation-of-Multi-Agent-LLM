# Phase II — When is collaboration worth its compute?

Status: **frozen** (tag `phase2-freeze`, 2026-10-04), before any Phase II run and before any Phase II
analysis of existing outputs. Changes after the tag are logged
as amendments at the bottom, with the date and whether any Phase II result had been seen.

## 0. Isolation from Part I (the frozen audit)

- Tags `p0-freeze`, `final-freeze`, `final-freeze-v2` and every file they cover are read-only. Phase II
  never rewrites `runs/` outside `runs/phase2/`, never edits Part I scripts, and its numbers never
  enter Part I tables or figures.
- New namespaces: code `experiments/phase2/`, outputs `runs/phase2/`, LLM caches `data/phase2_cache/`
  (no cache sharing with Part I, so Part I outputs cannot be silently reused or overwritten).
- Part I outputs may be *read* for the retrospective calibration in §3.1 only.
- Whatever Phase II finds, the Part I paper stands unchanged; a NO-GO is itself reported (§4).

## 1. Research question and hypotheses

**RQ.** Can we predict, from signals observable before the gold answer is known, when inter-agent
collaboration adds enough marginal value over cheaper inference strategies to justify its extra
compute?

Notation: for question x and action a, `Correct(a, x)` ∈ {0,1}, `Cost(a, x)` = total tokens
(input + output) per question, utility `U(a, x) = Correct(a, x) − λ·Cost(a, x)`. The collaboration
value is `V_C(x) = E[U(COLLAB, x) − U(A_cheap, x)]`, where `A_cheap` is the better of STOP and SAMPLE
on x.

- **H1 (selective vs always-on).** Selective collaboration achieves a better accuracy–compute
  trade-off than always-on collaboration.
- **H2 (adaptive vs fixed).** Choosing among STOP / SAMPLE / COLLABORATE per question improves the
  accuracy–token frontier over every fixed strategy (S, SC-k, always-on MAS).
- **H3 (interaction vs diversity; primary).** Under the *same* router and approximately matched mean
  token budgets, sending the routed questions to collaboration beats sending them to independent
  sampling (Ours-Collab > Ours-Sampling). Matching is on the mean tokens per question over the
  evaluated set (k* chosen so that SC-k* ≈ M1 on average), not per question; Criterion B's
  token-ratio window [0.9, 1.1] is the safeguard.

H3 is the hypothesis that addresses Part I's main limitation: the audit removed alternative
explanations but never showed that interaction itself adds value beyond inference diversity.

## 2. System and action space (minimal version)

```
question → S (strong single, always run) → probe → estimator V̂_C(x) → { STOP | SAMPLE | COLLABORATE }
```

| Action | Definition (exactly the Part I protocols) |
|---|---|
| STOP | answer of S (zero-shot CoT, 700 tok, T=0, extractor E3) |
| SAMPLE | SC-k: k samples of the S prompt at T=0.7, 700 tok, E3 per sample, majority vote (ties → earliest) |
| COLLABORATE | MedAgents M1, released configuration (as in Part I Layer 2A) |

- **Probe** (charged to every routed method): the first 4 SC samples (T=0.7). SAMPLE continues the
  same sample stream, so probe samples count toward its k.
- The contribution is the estimator/router, not a new multi-agent architecture. No action other
  than these three is added in Phase II (no new MAS, no prompt changes to M1).
- Router features are restricted to the pre-decision list in §3.4. M1 outputs, gold labels and
  any per-question accuracy are never router inputs.

## 3. Empirical-oracle pilot (the next and only experiment until the Go/No-Go)

### 3.1 Data splits (fixed before any outcome is seen)

MedQA-US test has 1,273 questions. 634 have been run on open weights in Part I (first 200 ∪ R1
subsets A/B/C). The remaining **639 never-run questions** are split once, seed 31:

| Split | Questions | Use |
|---|---|---|
| DEV | first 200 + R1 A/B/C (634) | retrospective calibration of the analysis code only; never used for Go/No-Go |
| PILOT | 300 sampled from the 639 (seed 31) | Empirical-oracle pilot → Go/No-Go; later router training |
| HELD-OUT | the other 339 | final router evaluation only; not opened until the router is frozen |
| MedMCQA-G | 500 from MedMCQA validation excluding `MedMCQA_r500` (seed 41) | generalization (Criterion C) only |

The split files (`runs/phase2/splits/*.json`) are committed with this freeze.

### 3.2 Retrospective calibration (no new compute)

On the DEV first-200 using existing outputs (S ×3, SC-20 ×3 = 60 vote samples per question, M1 ×3;
Qwen2.5-7B / 14B, Llama-3.1-8B): implement and debug every §3.3 quantity. These numbers are reported
as exploratory and play **no role** in the decision. (They are not clean: first-200 was shown to be
unrepresentative, and the SC-k pool is shared between the selection and evaluation halves.)

### 3.3 Prospective pilot runs

- **Backbones:** Qwen2.5-14B-Instruct (primary; the decision is taken on this model alone) and
  Llama-3.1-8B-Instruct (secondary; reported, not decision-relevant). Local vLLM, BF16,
  `--generation-config vllm`; servers up only for the run and stopped right after.
- **Per question in PILOT:**
  - S × 3 runs (T=0; repeats measure only serving nondeterminism)
  - SC samples: 6 runs × 20 samples (T=0.7). Pools: **A** = runs 1–2 (40 samples), **E** = run 3
    (20), **P** = runs 4–6 (60, placebo collaborator)
  - M1 × 3 runs (released; T=0 in its agent calls, so runs differ only by serving nondeterminism)
- **Compute matching:** `k*` = the integer k for which mean SC-k tokens per question is closest to
  M1's mean tokens per question on PILOT. It is computed from token counts only, before any
  correctness is computed, and logged. Expected ≈ 20 (Qwen-14B) and ≈ 7 (Llama-8B).
- **Cost check before the full pilot:** a 20-question timing probe; if the projected wall-clock is
  above 48 GPU-hours per backbone, P is cut to 2 runs (40 samples). This is the only allowed design
  change, and it is decided on timing alone.

### 3.4 Quantities

Per question x, all estimated from the pools named:

- `p_S(x)`: mean S correctness over the 3 runs.
- `p_SC(x)`: probability that a k*-sample majority vote is correct, estimated by 2,000 draws of k*
  samples without replacement from the pool. If the pool has fewer than k* samples, draws are made
  with replacement.
- `p_M(x)`: mean M1 correctness over the runs named.

**Question types.** These are computed on the selection half (S runs 1–2, SC pool A, M1 runs 1–2)
with threshold 0.5. They form a partition:

| Type | Rule | Meaning |
|---|---|---|
| A | p_M ≥ .5 and p_S ≥ .5 | collaboration unnecessary |
| B | p_M ≥ .5, p_S < .5, p_SC ≥ .5 | sampling suffices |
| **C** | p_M ≥ .5, p_S < .5, p_SC < .5 | **collaboration-only rescue** |
| D | p_M < .5 and max(p_S, p_SC) ≥ .5 | collaboration harms |
| E | all < .5 | everything fails |

**Cross-fitted empirical oracle headroom.** This is the primary Go quantity. "Oracle" here never
means choosing with gold outcomes on the evaluated runs: actions are chosen from noisy finite-run
estimates on the selection half and scored on independent runs. It is cross-fitted so that a noisy
per-question maximum cannot create headroom (winner's curse).
- *Selection.* For each x, pick the action with the highest selection-half estimate:
  `a*_CS(x)` = argmax over {S, SC-k*, M}, and `a*_S(x)` = argmax over {S, SC-k*}. Ties go to the
  cheaper action.
- *Evaluation.* Score the picked action on the held-out half: S run 3, SC pool E, M1 run 3.
- `H = Acc(a*_CS) − Acc(a*_S)` is the headroom that empirical-oracle routing with collaboration
  adds over empirical-oracle routing among the cheap actions only.
- **Placebo:** repeat the same computation with M replaced by a second, independent sampler at the
  same k*. Pool P is split into a selection half and an evaluation half. The result is
  `H_plc`, the headroom that is pure additional diversity at approximately matched mean tokens.
- Report the token cost of each empirical-oracle policy next to its accuracy.

**Pre-decision features (router inputs, fixed list).**
- From S: E3 answer; agreement of extractors E1/E2/E3; `finish_reason == length`; reasoning
  length.
- From the 4-sample probe: vote entropy; max vote share; agreement of the probe majority with S.
- From the question: token length; number of numeric values with units (regex fixed in code at
  freeze).

### 3.5 Go / No-Go (Qwen2.5-14B, PILOT, 95% bootstrap over questions, 10,000 resamples, seed 0)

| Criterion | Test | Pass if |
|---|---|---|
| G1 headroom | H | H ≥ 2.0 pp and CI lower bound > 0 |
| G2 beyond diversity | H − H_plc (paired over questions) | ≥ 1.5 pp and CI lower bound > 0 |
| G3 replication of C | on type-C questions (selection half), mean of [M1 run 3 − SC pool E at k*] | n_C ≥ 15 and CI lower bound > 0 |
| G4 predictability | out-of-fold AUC (5-fold, logistic regression on the §3.4 features) for y = 1[p_M − max(p_S, p_SC) > 0] on the selection half | AUC ≥ 0.65 and CI lower bound ≥ 0.55 |

- G3 is a replication check, not primary evidence of a collaboration benefit; G1–G2 carry the
  inferential weight. The n_C threshold is an operational minimum for deciding whether a
  collaboration-beneficial subgroup is sufficiently prevalent to motivate routing, rather than a
  power-derived inferential threshold.
- G4 uses an accuracy-based label on purpose, so that the decision does not depend on λ (which is
  only chosen later, on PILOT). **Secondary diagnostic (no decision role):** utility-positive
  collaboration, y_U(λ) = 1[(p_M − max(p_S, p_SC)) − λ·(T_M − T_cheap)/10^4 > 0], where T are mean
  tokens per question of M1 and of the cheaper action attaining the max, on the selection half.
  Out-of-fold AUC is reported for λ ∈ {0, 0.5, 1, 2, 5} pp per 10k tokens, with the same features,
  model and folds as G4.
- **GO** = G1 ∧ G2 ∧ G3 ∧ G4. Router development starts (§5, §6).
- **NO-GO** = G1 fails or G2 fails. No adaptive MAS is built. The result is written up as an
  audit finding: *apparent collaboration benefits are largely recovered by inference diversity at
  equal compute.* The type distribution, H and H_plc are reported for both backbones.
- **INCONCLUSIVE** = G1 ∧ G2 pass, but G3 or G4 fails. Exactly one pre-specified extension is
  allowed: run the same pilot on 150 more questions taken from HELD-OUT, which shrinks HELD-OUT to
  189. Then decide once more with the same criteria on all 450 questions. There is no second
  extension; failing again counts as NO-GO.

Thresholds are fixed here. They are not revised after seeing DEV (§3.2) or PILOT numbers.

### 3.6 Execution order (fixed)

Freeze → DEV code validation → 20-question timing probe → full PILOT (both backbones) → sealed
Go/No-Go analysis.

- DEV: checks only that p_SC, cross-fitting, bootstrap, the type partition and the placebo run
  correctly. No research decision (thresholds, features, design) is taken from DEV numbers.
- Timing probe: only wall-clock and token usage are inspected; accuracy is not computed. If the
  48 GPU-hour rule triggers, P is cut from 60 to 40 samples as a pre-specified branch (logged, not an
  amendment).
- PILOT is run in full on both Qwen2.5-14B and Llama-3.1-8B. Llama is run regardless of Qwen results;
  the GO decision uses Qwen2.5-14B only.
- The sealed analysis computes and reports exactly G1–G4 (plus the secondary diagnostic and the
  §4 tables) and maps them to GO / INCONCLUSIVE / NO-GO by §3.5.

## 4. Reporting regardless of outcome

The paper reports the type table (A–E shares with CIs), H, H_plc, the G3 contrast and the G4 AUC for
both backbones, whichever branch is taken. A NO-GO becomes a Part II section in the audit paper:
*Does interaction add value beyond sampling? — cross-fitted empirical-oracle analysis*. A GO becomes
the bridge to the adaptive system.

## 5. Baselines for the router phase (fixed now, used only after GO)

All methods run on HELD-OUT. Compute is measured in total tokens per question and reported both
as a mean and per question.

| Method | Role | Definition |
|---|---|---|
| Strong Single (S) | minimum compute | §2 STOP for every question |
| SC-k | independent diversity | k ∈ {5, 10, k*}; SC-k* is the compute-matched one |
| Always-on MAS | existing collaboration | M1 on every question |
| Random routing | controls for the effect of adaptivity itself | same action proportions as Ours, assigned uniformly at random (1,000 draws) |
| Difficulty routing | simple adaptive inference | escalate when probe max vote share is below τ: to SC-k* (variant a) or to M1 (variant b) |
| Ours-Sampling | same router, no collaboration | Ours with COLLABORATE replaced by SC-k*; same questions escalated |
| **Ours-Collab** | final system | interpretable router (§6) over STOP / SAMPLE / COLLABORATE |
| Empirical-oracle router | attainable reference (not a true upper bound) | cross-fitted empirical oracle of §3.4 |

λ (and τ) are chosen on PILOT and then fixed. Results are reported as a curve over a λ grid, with
one pre-declared operating point: the λ whose PILOT mean cost is closest to 0.5× always-on M1.

## 6. Router (only after GO)

1. An interpretable model first: logistic regression or a depth-≤3 tree over the §3.4 features.
   It predicts V_C(x) (the utility difference), not difficulty. It is trained on PILOT with
   labels from all runs.
2. A neural router is out of scope unless the interpretable router passes Criterion A and fails
   Criterion B for reasons the features plausibly miss. Even then, it needs an amendment before
   any HELD-OUT run.
3. HELD-OUT (339) is never used for router, feature, model or λ selection. Features, model and λ are
   fixed on PILOT only, the router is frozen (tag `phase2-router-freeze`), and only then are
   HELD-OUT and MedMCQA-G run, once.

## 7. Success criteria (fixed now)

- **Criterion A (efficiency, H1/H2).** At the pre-declared operating point, Ours-Collab costs at
  most 0.5× the mean tokens of always-on M1. Its accuracy must be non-inferior to always-on M1
  with a 1 pp margin (TOST, α = .05).
- **Criterion B (collaboration value, H3; primary).** Ours-Collab − Ours-Sampling > 0 with CI
  lower bound > 0. The mean token ratio of the two must lie within [0.9, 1.1].
- **Criterion C (generalization).** The frozen router keeps the direction of A and B (point
  estimates) on MedMCQA-G (Qwen2.5-14B) and on HELD-OUT with Llama-3.1-8B. CIs are reported; no
  significance is required.

How the outcomes are read:
- A only: an efficient adaptive-inference result. It is not reported as evidence for
  collaboration.
- A + B: selective collaboration.
- A + B + C: the strongest form.
- B failing while A passes: reported as "adaptivity, not interaction, drives the gain". This is
  consistent with Part I.

## 8. Analysis conventions

Paired bootstrap over questions; seeds as listed; crashes are scored wrong; model families are
never pooled; every run must pass `validate_runs.py`-style checks (expected questions,
api_failed 0, valid predictions). The tokenizer and token accounting are the same as in Part I R3.

## Amendments

### Amendment 1 (2026-10-04) — non-interactive heterogeneous control (HET)

*Seen at the time of writing:* the sealed PILOT result (`runs/phase2/pilot/RESULTS.txt`: Qwen2.5-14B GO,
Llama-3.1-8B INCONCLUSIVE). The GO decision and all §3.5 thresholds are unchanged. This amendment only
adds a stricter control. The reason: the §3.4 placebo is a second draw from the same SC distribution,
so G2 shows that M1 adds value beyond *more of the same sampling*. It cannot separate *interaction*
from *being a different, decorrelated pipeline*.

**HET-k\* (definition).** Per question:
1. Run MedAgents' own expert-recruitment step: the question-domain prompt (5 domains) and the
   option-domain prompt (2 domains), both released prompts at T=0. Their tokens are charged to HET.
2. Each recruited domain d gets MedAgents' released analyzer system prompt ("You are a medical expert
   in the domain of {d}. From your area of specialization, …").
3. Every sample answers the S user prompt (question + options + "Let's think step by step.") at
   T=0.7 with 700 tokens. Its answer is extracted with E3, and the samples are combined by majority
   vote (ties go to the earliest sample).
4. Roles are assigned round-robin over the 7 domains in recruitment order.

No sample sees another sample's output. There is no synthesis, consensus or revision step. HET
therefore keeps M1's heterogeneity of expertise and removes inter-agent interaction.

*Clarification (2026-10-04, after a 3-question smoke test and before any full HET run; no HET accuracy
was computed).* MedAgents' recruitment output is parsed exactly as M1 parses it, so HET uses the experts
that M1 itself uses. On the PILOT M1 outputs, this parse gives the full 5 + 2 well-formed domains in
93.7% of question-runs for Qwen2.5-14B. For Llama-3.1-8B it does so in 0.3%: Llama ignores the format,
and M1 runs with about 2 "domains" that are fragments of free text. HET keeps this behavior on purpose,
because it must mirror M1's actual heterogeneity. Each run therefore draws 21 samples round-robin over
however many roles were recruited. When a role needs extra samples, they are generated and the surplus
is discarded.

**PILOT runs.**
- 3 runs × 21 samples per question, seeds 201–203, on both backbones.
- Pools: selection = runs 1–2 (42 samples), evaluation = run 3 (21 samples).
- k\* is the same as in §3.3 (22 for Qwen2.5-14B, 7 for Llama-3.1-8B).
- Draws of k\* samples from a pool follow §3.4.

**PILOT diagnostic (reported, no effect on GO).**
- `H_het`: the cross-fitted empirical-oracle headroom of {S, SC, HET} over {S, SC}, computed exactly
  like H.
- Primary contrast: `H − H_het`, paired over questions, with a 95% CI. This is the headroom
  attributable to interaction beyond heterogeneous non-interactive sampling.
- Also reported: on type-C questions, HET pool E − SC pool E, and M1 run 3 − HET pool E.

**Router phase (additions to §5 and §7).**
- New baseline **Ours-Hetero**: the same router, with COLLABORATE replaced by HET-k\* on the
  escalated questions. It is run on HELD-OUT with one run of k\* samples per question.
- New criterion **B′ (interaction beyond heterogeneity)**: Ours-Collab − Ours-Hetero > 0 with CI
  lower bound > 0, and the mean token ratio of the two must lie in [0.9, 1.1].
- How the outcomes are read:
  - B and B′ pass: an interaction-specific benefit.
  - B passes and B′ fails: the gain over sampling is explained by heterogeneous expertise, not
    interaction.
  - B fails: §7 as written.

*Transparency note.* HET was added **after** the sealed PILOT Go/No-Go result had been seen. It was added
to test a stricter alternative explanation of that result. It makes the hypothesis harder to confirm;
it is not a favourable comparison.

### Amendment 2 (2026-10-04) — router choice, λ, HELD-OUT protocol; tag `phase2-router-freeze`

*Seen at the time of writing:* the PILOT results (`RESULTS.txt`, `HET_RESULTS.txt`, `HET_COMPARE.txt`)
and the PILOT cross-fitted router diagnostics (`runs/phase2/router/PILOT_CROSSFIT.txt`). HELD-OUT has
not been read or run.

**Router.** Two designs were built on PILOT, and both are recorded in `PILOT_CROSSFIT.txt`:
- (a) per-action correctness models. These were committed in code before fitting, but they do not
  predict V_C as §6 requires. Cross-fitted on PILOT: Ours-Collab .730 vs Ours-Sampling .744.
- (b) a direct V_C design, conformant with §6. Cross-fitted on PILOT: .736 vs .736, with
  COLLABORATE chosen on 1.3% of questions.

**(b) is frozen** because §6 takes precedence over the implementation candidate. The choice does not
depend on which design looks better. After this point there is no further feature, model or
threshold search.
- **λ:** the pre-declared rule selects the grid value whose PILOT cross-fitted mean cost is closest
  to 0.5 × always-on M1 (12,496 tokens). For (b), cost falls monotonically with λ, and the grid
  starts at 0. λ = 0 gives the maximum cost on the grid (8,420), so **λ = 0**. This cost is within
  Criterion A's 0.5× bound.
- **Frozen artifact:** `runs/phase2/router/router_frozen.json`. It contains the feature list, the
  coefficients of q_C and q_A, the rule (COLLABORATE if q_C > .5, else SAMPLE if q_A > .5, else
  STOP), k\* = 22, the 4-sample probe, the difficulty-routing τ (= 0.5625, cost-matched to (b) on
  PILOT) and the random-routing shares. It is applied by `experiments/phase2/router_policy.py`.

**HELD-OUT protocol** (Qwen2.5-14B only; Llama is not a decision backbone because of its recruitment
failure). Each arm runs exactly once on the 339 HELD-OUT questions:
- S: the frozen protocol.
- SC-22: one request with n = 22 independent samples, E3 per sample, majority vote. The probe is its
  first 4 samples.
- HET-22: 22 samples, as defined in Amendment 1.
- M1: the released protocol.

Ours-Collab, Ours-Sampling and Ours-Hetero are built **offline** from these same outputs, so no
policy gets its own LLM run. Per-question cost is the measured token count; the probe is charged as
4/22 of that question's SC-22 tokens.

**Fixed analysis order** (`experiments/phase2/heldout_analysis.py`):
1. Criterion B (primary): paired CI and token ratio.
2. Criterion A: one-sided 95% non-inferiority with a 1 pp margin, and ≤ 0.5× M1 tokens.
3. Criterion B′.
4. Fixed strategies and routing baselines.
5. Descriptive diagnostics.

No threshold is revisited after (1) is read.

**Pre-declared readings.**
- A + B: selective collaboration.
- A only: adaptive inference saves compute, but interaction gives no prospective incremental value
  beyond sampling.
- B only: an interaction signal without practical efficiency.
- Neither: the adaptive method is dropped, and the paper reports the chain "average collaboration
  hurts → sparse empirical-oracle headroom → not explained by heterogeneous sampling →
  pre-decision signals insufficient to exploit it".
- If B passes, B′ is reported as in Amendment 1.

**Hashes:** see `runs/phase2/router/FREEZE_MANIFEST.sha256`.
