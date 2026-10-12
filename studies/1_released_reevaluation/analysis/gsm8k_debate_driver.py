"""Resilient driver to finish GSM8K debate under a saturated daily API quota.
single_cot and self_consistency are already cached; here we drive the debate method
to completion, retrying rate-limited questions and sleeping between passes so the
rolling quota can recover. All completed LLM calls are cached, so progress is never
lost even if the process is killed and restarted."""
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import gsm8k_reasoning as G
from openai import RateLimitError
from datasets import load_dataset

N = 200
SLEEP = 600  # seconds to wait when the quota is hard-capped
OUT = "data/RESULTS_gsm8k_full.txt"
RNG = np.random.default_rng(0)


def ci(x):
    b = [RNG.choice(x, len(x), replace=True).mean() for _ in range(10000)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def gain(a, b):
    d = b - a
    bs = [RNG.choice(d, len(d), replace=True).mean() for _ in range(10000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def log(msg):
    with open(OUT, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def main():
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(N))
    Q = [r["question"] for r in ds]
    Gold = [G.norm(G.gold(r["answer"])) for r in ds]

    single = np.array([int(G.norm(G.single_cot(q)) == g) for q, g in zip(Q, Gold)])
    scv = np.array([int(G.norm(G.self_consistency(q)) == g) for q, g in zip(Q, Gold)])
    log(f"single_cot={single.mean():.3f}  self_consistency={scv.mean():.3f}")

    preds = [None] * N
    passno = 0
    while any(p is None for p in preds):
        passno += 1
        todo = [i for i in range(N) if preds[i] is None]
        advanced = 0
        capped = False
        for i in todo:
            try:
                preds[i] = G.debate(Q[i])
                advanced += 1
            except RateLimitError:
                capped = True
                break
            except Exception as e:
                log(f"[warn] q{i} failed non-rate-limit: {type(e).__name__}; skipping as wrong")
                preds[i] = ""  # count as wrong rather than block forever
        done = sum(p is not None for p in preds)
        log(f"[pass {passno}] debate done {done}/{N} (+{advanced} this pass)")
        if any(p is None for p in preds):
            if capped and advanced == 0:
                log(f"[pass {passno}] hard-capped; sleeping {SLEEP}s")
                time.sleep(SLEEP)
            # else: quota trickling, loop again immediately

    deb = np.array([int(G.norm(p) == g) for p, g in zip(preds, Gold)])
    log("\n=== GSM8K final (n=%d) ===" % N)
    for nm, x in [("single_cot", single), ("self_consistency", scv), ("debate", deb)]:
        m, lo, hi = ci(x)
        log(f"  {nm:18s} {m:.3f} [{lo:.3f},{hi:.3f}]")
    g, lo, hi = gain(single, scv)
    log(f"  SC - single     = {g:+.3f} [{lo:+.3f},{hi:+.3f}]  "
        f"{'POS. CONTROL PASSED' if lo > 0 else 'n.s.'}")
    g, lo, hi = gain(single, deb)
    log(f"  debate - single = {g:+.3f} [{lo:+.3f},{hi:+.3f}]  "
        f"{'debate helps' if lo > 0 else 'TIE (no debate gain)'}")
    g, lo, hi = gain(deb, scv)
    log(f"  SC - debate     = {g:+.3f} [{lo:+.3f},{hi:+.3f}]  "
        f"{'cheap SC beats debate' if lo > 0 else 'tie'}")
    log("GSM8K_FULL_DONE")


if __name__ == "__main__":
    open(OUT, "w").close()
    main()
