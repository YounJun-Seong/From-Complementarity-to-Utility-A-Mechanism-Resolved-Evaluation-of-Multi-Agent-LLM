"""E2′ judge (POSTHOC_EXPERIMENTS.md): summary-level preservation (J1) and agent uptake (J2) on MedARC summaries.
Gold is never read here. Judge = Qwen3-32B-FP8, non-thinking, T=0; cached by payload hash in cache/posthoc.

  TMC_BASE_URL=http://127.0.0.1:PORT/v1 python experiments/posthoc_summary_judge.py j1|j2|sheet
  ->  studies/7_posthoc_audits/results/judge_j1.jsonl, judge_j2.jsonl, human_check/{sheet.csv,key.json}
"""
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, f"{HERE}/../../5_medarc_transfer/code")
import aggregate as AG   # noqa: E402

MODEL = "Qwen/Qwen3-32B-FP8"
CACHE = f"{ROOT}/cache/posthoc"
OUT = f"{ROOT}/studies/7_posthoc_audits/results"
SYS = "You compare texts precisely and answer only with JSON."

J1 = """A medical expert answered a research question whose possible answers are "yes", "no" or "maybe". Another model then wrote a summary comparing this response with a second response. In the summary, this response is called "Response {j}".

Question: {question}

=== Original response (called "Response {j}" in the summary) ===
{original}

=== Summary ===
{summary}

Judge ONLY by comparing the summary with the original response. Do not use your own knowledge of which answer is correct.
1. final_answer_original: the final answer of the original response ("yes", "no", "maybe", or "unclear").
2. answer_in_summary: the final answer the summary attributes to Response {j} ("yes", "no", "maybe", or "unstated" if the summary does not make it clear).
3. evidence_preserved: are the key reasons the original response gives for its final answer conveyed in the summary? "full" (the main reasons are conveyed), "partial" (only some), or "none".
4. misrepresented: does the summary attribute to Response {j} a conclusion or reasoning that the original response does not contain, or present its reasoning as supporting a different answer? true or false.
Answer with JSON only: {{"final_answer_original": "...", "answer_in_summary": "...", "evidence_preserved": "...", "misrepresented": true/false}}"""

J2 = """An expert answered a research question whose possible answers are "yes", "no" or "maybe". The expert then read a summary of other experts' responses and wrote a new response. According to the summary, one of the other experts concluded "{label}".

Question: {question}

=== Summary the expert read ===
{summary}

=== The expert's new response ===
{new}

Does the new response engage with the position that the answer is "{label}" or with the reasons the summary gives for it?
"adopted": the new response takes "{label}" as its final answer.
"considered_rejected": the new response explicitly discusses that position or its reasons but keeps a different final answer.
"ignored": the new response does not engage with that position or its reasons.
Answer with JSON only: {{"engagement": "adopted|considered_rejected|ignored"}}"""

_client = None


def chat(user, max_tokens):
    global _client
    payload = dict(model=MODEL, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                   temperature=0.0, max_tokens=max_tokens, seed=0,
                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]
    path = f"{CACHE}/{key}.json"
    if os.path.exists(path):
        return json.load(open(path))["text"]
    if _client is None:
        _client = OpenAI(base_url=os.environ["TMC_BASE_URL"], api_key="EMPTY", timeout=900)
    for attempt in range(6):
        try:
            r = _client.chat.completions.create(**payload)
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 * 2 ** attempt)
    os.makedirs(CACHE, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    json.dump(dict(text=r.choices[0].message.content), open(tmp, "w")); os.replace(tmp, path)
    return r.choices[0].message.content


def js(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def norm(v, allowed):
    v = str(v).strip().lower()
    return v if v in allowed else None


def jl(p):
    return [json.loads(l) for l in open(p)]


def sources():
    """Yield (meta, records) per data set; records keep the full MedARC transcript."""
    q5 = {json.loads(l)["pubid"]: json.loads(l)["question"] for l in open(f"{ROOT}/studies/5_medarc_transfer/results/data/eval_inputs.jsonl")}
    q6 = {json.loads(l)["pubid"]: json.loads(l)["question"] for l in open(f"{ROOT}/studies/6_medarc_matched_ablation/results/data/inputs.jsonl")}
    for tag, pat, qs in (("VI_A", f"{ROOT}/studies/6_medarc_matched_ablation/results/main/A_run{{}}.jsonl", q6), ("V", f"{ROOT}/studies/5_medarc_transfer/results/main/medarc_run{{}}.jsonl", q5)):
        for run in (1, 2, 3):
            for rec in jl(pat.format(run)):
                yield tag, run, qs[rec["pubid"]], rec


def j1_items():
    for tag, run, question, rec in sources():
        T = AG.turns(rec)
        ans = AG.controlled(rec)[0]
        for s in rec["summaries"]:
            r, i = s["round"], s["agent"]
            others = [k for k in range(3) if k != i]
            for j, src in enumerate(others, 1):
                yield dict(set=tag, run=run, idx=rec["idx"], pubid=rec["pubid"], round=r, agent=i, j=j, src_agent=src,
                           src_label=ans[src][r - 1], question=question, original=T[src][r - 1], summary=s["text"])


def run_j1(it):
    d = js(chat(J1.format(j=it["j"], question=it["question"], original=it["original"], summary=it["summary"]), 120))
    mis = d.get("misrepresented")
    mis = mis if isinstance(mis, bool) else {"true": True, "false": False}.get(str(mis).strip().lower())
    out = {k: it[k] for k in ("set", "run", "idx", "pubid", "round", "agent", "j", "src_agent", "src_label")}
    return out | dict(final_answer_original=norm(d.get("final_answer_original"), ("yes", "no", "maybe", "unclear")),
                      answer_in_summary=norm(d.get("answer_in_summary"), ("yes", "no", "maybe", "unstated")),
                      evidence_preserved=norm(d.get("evidence_preserved"), ("full", "partial", "none")),
                      misrepresented=mis)


def run_j2(it):
    d = js(chat(J2.format(label=it["label"], question=it["question"], summary=it["summary"], new=it["new"]), 40))
    out = {k: it[k] for k in ("set", "run", "idx", "pubid", "round", "agent", "j", "label")}
    return out | dict(engagement=norm(d.get("engagement"), ("adopted", "considered_rejected", "ignored")))


def dump(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    stage = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    if stage == "j1":
        items = list(j1_items())
        with ThreadPoolExecutor(64) as ex:
            rows = list(ex.map(run_j1, items))
        dump(rows, f"{OUT}/judge_j1.jsonl")
        ok = sum(r["answer_in_summary"] is not None and r["evidence_preserved"] is not None and r["misrepresented"] is not None for r in rows)
        print(f"J1 {len(rows)} items, fully parsed {ok / len(rows):.4f}")
    elif stage == "j2":
        j1 = {(r["set"], r["run"], r["idx"], r["round"], r["agent"], r["j"]): r for r in jl(f"{OUT}/judge_j1.jsonl")}
        items = []
        for tag, run, question, rec in sources():
            T, ans = AG.turns(rec), AG.controlled(rec)[0]
            for s in rec["summaries"]:
                r, i = s["round"], s["agent"]
                for j in (1, 2):
                    L = j1[(tag, run, rec["idx"], r, i, j)]["answer_in_summary"]
                    if L in ("yes", "no", "maybe") and L != ans[i][r - 1]:
                        items.append(dict(set=tag, run=run, idx=rec["idx"], pubid=rec["pubid"], round=r, agent=i, j=j,
                                          label=L, question=question, summary=s["text"], new=T[i][r]))
        with ThreadPoolExecutor(64) as ex:
            rows = list(ex.map(run_j2, items))
        dump(rows, f"{OUT}/judge_j2.jsonl")
        print(f"J2 {len(rows)} items, parsed {sum(r['engagement'] is not None for r in rows) / max(len(rows), 1):.4f}")
    elif stage == "sheet":
        # blind human check: 40 gold-carrying + 40 wrong-carrying J1 items from VI_A. Gold is read ONLY to stratify;
        # neither gold nor judge output appears on the sheet.
        gold = json.load(open(f"{ROOT}/studies/6_medarc_matched_ablation/results/labels.json"))
        items = [it for it in j1_items() if it["set"] == "VI_A" and it["src_label"]]
        rng = random.Random(0)
        g = [it for it in items if it["src_label"] == gold[str(it["pubid"])]]
        w = [it for it in items if it["src_label"] != gold[str(it["pubid"])]]
        pick = rng.sample(g, 40) + rng.sample(w, 40)
        rng.shuffle(pick)
        os.makedirs(f"{OUT}/human_check", exist_ok=True)
        with open(f"{OUT}/human_check/sheet.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["item", "response_label_in_summary", "question", "original_response", "summary",
                         "final_answer_original", "answer_in_summary", "evidence_preserved", "misrepresented", "notes"])
            for n, it in enumerate(pick):
                wr.writerow([n, f"Response {it['j']}", it["question"], it["original"], it["summary"], "", "", "", "", ""])
        json.dump([{k: it[k] for k in ("set", "run", "idx", "pubid", "round", "agent", "j")} for it in pick],
                  open(f"{OUT}/human_check/key.json", "w"), indent=0)
        print("sheet: 80 items")


if __name__ == "__main__":
    main()
