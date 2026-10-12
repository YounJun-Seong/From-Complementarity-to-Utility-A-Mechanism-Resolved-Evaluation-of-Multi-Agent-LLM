"""Phase V offline aggregations (PHASE5_FREEZE.md §4). Pure functions on one MedARC record."""
import numpy as np

import parsers as P
import released as R

THRESH = 0.85


def turns(rec):
    """turns[i][r] = assistant text of agent i at round r."""
    return [[m["content"] for m in ctx if m["role"] == "assistant"] for ctx in rec["agent_contexts"]]


def controlled(rec):
    T = turns(rec)
    return [[P.answer(t) for t in a] for a in T], [[P.confidence(t) for t in a] for a in T]


def a_released(rec):
    return R.released_select(rec["agent_contexts"])


def a_released_controlled_parser(rec):
    """Released turn selection (all 9 turns, released confidence, argsort[::-1][0]); answer via the controlled parser."""
    flat = [t for a in turns(rec) for t in a]
    conf = [R.extract_confidence(t) for t in flat]
    return P.answer(flat[int(np.argsort(conf)[::-1][0])]) if flat else None


def a_paper_no_reinit(rec, r=2):
    ans, conf = controlled(rec)
    c = [conf[i][r] for i in range(len(ans))]
    passing = [i for i in range(len(ans)) if c[i] >= THRESH]
    pool = passing if passing else list(range(len(ans)))
    best = max(pool, key=lambda i: (c[i], -i))          # highest confidence, ties -> lowest agent index
    return ans[best][r], not passing


def a_majority(rec, r):
    ans, conf = controlled(rec)
    votes = [(ans[i][r], conf[i][r], i) for i in range(len(ans)) if ans[i][r]]
    if not votes:
        return None
    cnt = {}
    for a, c, i in votes:
        n, mc, mi = cnt.get(a, (0, -1.0, 99))
        cnt[a] = (n + 1, max(mc, c), min(mi, i))
    return max(cnt, key=lambda a: (cnt[a][0], cnt[a][1], -cnt[a][2]))


def _test():
    mk = lambda texts: {"agent_contexts": [[{"role": "user", "content": "q"}] + [{"role": "assistant", "content": t} for t in a] for a in texts]}
    r = mk([["\\boxed{yes} confidence: 70%", "\\boxed{yes} confidence: 80%", "\\boxed{no} confidence: 90%"],
            ["\\boxed{no} confidence: 60%", "\\boxed{no} confidence: 70%", "\\boxed{no} confidence: 86%"],
            ["\\boxed{maybe} confidence: 99%", "\\boxed{yes} confidence: 75%", "\\boxed{yes} confidence: 80%"]])
    assert a_paper_no_reinit(r) == ("no", False)           # round 2: 90 and 86 pass -> agent 0 (90) -> no
    assert a_released(r) == "maybe"                         # all 9 turns: 99 is round 0 of agent 2
    assert a_majority(r, 2) == "no"                         # round 2: no, no, yes
    assert a_majority(r, 0) == "maybe"                      # round 0: 1-1-1 tie -> highest confidence (0.99)
    r2 = mk([["\\boxed{yes} confidence: 50%"] * 3, ["\\boxed{no} confidence: 60%"] * 3, ["\\boxed{yes} confidence: 40%"] * 3])
    assert a_paper_no_reinit(r2) == ("no", True)            # none passes -> highest final-round confidence (fallback)
    assert a_released_controlled_parser(r) == "maybe"
    print("aggregate: all tests passed")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["test"]:
        _test()
