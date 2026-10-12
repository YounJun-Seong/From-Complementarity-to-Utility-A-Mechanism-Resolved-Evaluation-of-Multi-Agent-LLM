# Phase VI (P0) — Matched interaction ablation on MedARC (frozen before any run)

Status: **frozen** (tag `phase6-freeze`, 2026-10-08). No Phase VI LLM call had been made at the time of
tagging. Read-only: every earlier phase and tag.

## 0. Purpose

Phase VI tests whether letting agents see one another's work changes the outcome when everything else is
held fixed: the model, the questions, the generation settings and the initial (round-0) answers. It
separates the effect of the interaction intervention from differences in prompting and candidate
generation.

It is **not** a method study. A ≤ B is as informative a result as A > B. A − B is reported as the effect
of *allowing interaction from identical initial states*. It is not called a pure causal effect of
information generation.

## 1. Data

- **Questions.** PubMedQA PQA-L (`qiaojin/PubMedQA`, `pqa_labeled`, 1,000 questions), minus the 498 Phase V
  evaluation questions and the 30 Phase V token-probe questions. That leaves **472 questions**.
- **Overlap check.** None of the 472 questions duplicates the text of a used question. No Phase I–V
  run has used any of them; within this project, PubMedQA appears only in Phase V.
- **Disclosure.** While verifying the split, the 472-question gold-label marginal was printed (yes 262 /
  no 163 / maybe 47). No per-question label and no model output was seen.
- **Files.** `runs/phase6/data/inputs.jsonl` holds pubid, question and contexts, with no gold. Labels are
  in `runs/phase6/labels.json`, which only `sealed_analysis.py` reads. Hashes are in
  `runs/phase6/data/DATA.sha256`.

## 2. Arms (Qwen2.5-14B-Instruct, homogeneous, local vLLM; T = 1.0, max_tokens 1,024)

All MedARC arms use the Phase V verbatim port (`experiments/phase5/released.py`): 3 agents, round 0 plus
debate rounds 1–2, the summarizer from the same model, and full history.

| Arm | Rounds 1–2: message appended to agent *i*'s history |
|---|---|
| **A. Interactive** | the released message: `Summary of other agents' responses:\n{S}\n\nPlease re-analyze the question using these provided insights. Provide your updated answer to the question: {q}\nPresent your reasoning before final answer.` Here S = `generate_summary` of the other two agents' latest turns on **this** question. |
| **B. No-communication** | the same template with the summary block and the clause about provided insights removed: `Please re-analyze the question. Provide your updated answer to the question: {q}\nPresent your reasoning before final answer.` There is no summary call. |
| **C. Placebo communication** | the A template, but S is a `generate_summary` call on the latest turns of the other two agents of **arm A on a different question**. That question is the next one in idx order (cyclic), with the same run, the same round and the same agent positions. The source is real, length-natural agent text about an unrelated question. |
| **D. SC-25** | 25 independent round-0 samples, each sent as a separate n = 1 request; majority vote with the controlled parser. This is the Phase V protocol. k\* = 25 is taken from the Phase V `K_FROZEN.json` and is not re-probed. |
| S (for C\*) | B_strong: the released CoT template at T = 0. |

**B prompt.** B deliberately does **not** use the released fallback `Please verify your answer…`.
Instead it keeps A's template minus the summary, so that A and B differ only in the presence of
another agent's information. Two residual differences are limitations: the dropped words "using these
provided insights", and B's lack of the 6 summarizer calls, which means B uses fewer tokens.

**Identical initial states.**
- Seeds are run·10⁶ + idx·100 + call number. Round-0 calls are calls 0–2 in every MedARC arm.
- A **round-0 pre-pass** generates the round-0 turns for all 472 questions × 3 runs into the shared
  cache before any arm runs. Arms A, B and C then read the identical round-0 turns from that cache.
- `check_invariants.py` verifies byte-identical round-0 turns across A, B and C. It also verifies:
  - that B's messages contain no summary;
  - that C's summary source is the designated other question;
  - that A's summary source is the same question.

**Runs.** **Exactly 3 runs** of each of A, B and C. B_strong also has 3 executions; under T = 0 they are
deterministic and are not treated as independent replicates. SC has **exactly 6 runs** of 25 samples:
runs 1–2 form pool A, run 3 pool E, and runs 4–6 the placebo pool. This is the Phase V layout. Arm C
runs after arm A.

## 3. Extraction and aggregation

Identical to Phase V (`parsers.py`, including Amendment 1; `aggregate.py`):
- The controlled parser is used for every arm.
- The MedARC answer of record is **A_paper_no-reinit**: the final round, confidence ≥ 0.85, highest
  confidence, with the deterministic fallback. It is applied to A, B and C alike.
- Unparsed turns are wrong.
- Unparsed SC samples enter the vote as an empty label, as in Phase V.

## 4. Estimands (sealed `experiments/phase6/sealed_analysis.py`, run once)

**Conventions.**
- Per-question accuracy is the mean over 3 runs.
- Intervals are paired bootstraps over questions (10,000, seed 0). Multi-level quantities are clustered
  by question.

| Role | Estimand |
|---|---|
| **Primary** | **Acc(A) − Acc(B)**, A_paper_no-reinit, paired 95% CI |
| Key secondary | κ_post(A) − κ_post(B); κ_gain and κ_loss per arm; Δκ(A) − Δκ(B), where κ_gen is identical by construction |
| Secondary | BRR, HRR and DI per arm (Phase V G5 definitions, pooled over transitions 0→1 and 1→2); G_int of each arm vs SC-25; C\* of each arm; Acc(R2MV) per arm |
| Robustness | Acc(A) − Acc(C) and Acc(B) − Acc(C) |
| Descriptive | tokens per question per arm; parse and none-pass rates; switch rates |

**C\* of each arm** is computed by `experiments/phase2/oracle_analysis.analyze`, unchanged, with this
argument mapping:
- `S` = B_strong (3 × Q);
- `M` = that arm's A_paper_no-reinit correctness (3 × Q; runs 1–2 for selection, run 3 for evaluation);
- `sc_votes` = SC runs 1–3, with `pool_split` = 50;
- `placebo_votes` = SC runs 4–6;
- `k` = 25;
- `X` = constant.

The three C\* values share S, SC and the placebo and differ only in M. The difference between arms is
reported descriptively, without a CI: `analyze` gives each arm's interval separately, and no joint
estimator is specified.

**No decision rule and no GO/NO-GO.** Every estimand is reported in every case. No hypothesis,
threshold, arm or prompt is added or changed after any Phase VI output has been seen.

## 5. Order

1. Tag `phase6-freeze`.
2. Code commit and unit checks: the A arm must render Phase V's exact strings, B differs only as
   specified, and C uses the designated source.
3. Smoke test on 5 Phase V probe questions; no labels are read.
4. Round-0 pre-pass, then arms A, B, B_strong and SC, then arm C.
5. Stop the servers.
6. Label-free validation, including round-0 identity.
7. Commit.
8. Run `sealed_analysis.py` once.
