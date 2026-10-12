"""Frozen decision rules (PHASE3_FREEZE.md §3.2–3.4). Pure functions; `python aggregate.py test` runs unit tests."""
from collections import Counter


def state(answers):
    """answers: 4 probe answers, '' = parse failure."""
    parsed = [a for a in answers if a]
    n = len(parsed)
    if n == 0:
        return "failure"
    m = max(Counter(parsed).values())
    if n == 4 and m == 4:
        return "unanimous"
    if n == 4 and m == 3:
        return "weak"
    if n == 4 and m == 2 and len(set(parsed)) == 2:
        return "split"
    return "fragmented"


def ranked(answers):
    """Distinct parsed answers by count desc, ties by earliest first occurrence."""
    parsed = [a for a in answers if a]
    c = Counter(parsed)
    return sorted(c, key=lambda a: (-c[a], parsed.index(a)))


def top2(answers):
    r = ranked(answers)
    return (r[0], r[1]) if len(r) >= 2 else (r[0] if r else "", None)


def majority(pool):
    """Majority over a list of answers ('' ignored); ties -> earliest in pool order."""
    pool = [a for a in pool if a]
    if not pool:
        return ""
    c = Counter(pool)
    best = max(c.values())
    return next(a for a in pool if c[a] == best)


def tdc_final(probe, rev_x, rev_y):
    """§3.3 step 5. probe: 4 answers ('' = failure); rev_x / rev_y: revised answers of the representatives."""
    pool = [a for a in probe if a] + [a for a in (rev_x, rev_y) if a]
    c = Counter(pool)
    best = max(c.values())
    tied = [a for a in c if c[a] == best]
    if len(tied) == 1:
        return tied[0]
    if rev_x and rev_x == rev_y and rev_x in tied:                       # rule 1
        return rev_x
    pm = ranked(probe)[0] if ranked(probe) else ""                     # rule 2: probe majority (count, earliest)
    if pm in tied:
        return pm
    return next(a for a in pool if a in tied)                          # rule 3


def flow(x, y, rev_x, rev_y):
    """Revision-flow labels (no gold)."""
    fx = "X stays X" if rev_x == x else "X→Y" if rev_x == y else "X→other"
    fy = "Y stays Y" if rev_y == y else "Y→X" if rev_y == x else "Y→other"
    if rev_x == x and rev_y == y:
        joint = "disagreement remains"
    elif rev_x == y and rev_y == x:
        joint = "swap"
    elif rev_x == rev_y == x:
        joint = "converge X"
    elif rev_x == rev_y == y:
        joint = "converge Y"
    elif rev_x == rev_y:
        joint = "converge third"
    else:
        joint = "disagreement remains"
    return fx, fy, joint


def _test():
    S = lambda s: [c if c != "-" else "" for c in s]
    assert state(S("AAAA")) == "unanimous" and state(S("AAAB")) == "weak" and state(S("AABB")) == "split"
    assert state(S("AABC")) == "fragmented" and state(S("ABCD")) == "fragmented"
    assert state(S("AAA-")) == "fragmented" and state(S("A---")) == "fragmented" and state(S("----")) == "failure"
    assert top2(S("AACC")) == ("A", "C") and top2(S("CAAB")) == ("A", "C") and top2(S("ABCA")) == ("A", "B")
    assert top2(S("AAA-")) == ("A", None)
    assert tdc_final(S("AACC"), "A", "A") == "A"           # freeze example 1
    assert tdc_final(S("AACC"), "A", "C") == "A"           # freeze example 2 (rule 2, earliest)
    assert tdc_final(S("CCAA"), "C", "A") == "C"
    assert tdc_final(S("AACC"), "C", "C") == "C"           # 4 C vs 2 A
    assert tdc_final(S("AABC"), "B", "B") == "B"           # 3 B vs 2 A
    assert tdc_final(S("ABCD"), "C", "C") == "C"
    assert tdc_final(S("AAAB"), "B", "B") == "B"           # tie A3 B3; revisions agree on B -> rule 1
    assert tdc_final(S("AAAB"), "A", "B") == "A"           # 4 A vs 2 B
    assert tdc_final(S("ABCD"), "A", "B") == "A"           # tie A2 B2; rule 2: probe majority (earliest) A
    assert tdc_final(S("AABC"), "C", "C") == "C"
    assert majority(S("AACC") + ["C", "A", "C"]) == "C" and majority(S("AACC") + ["C", "A"]) == "A"
    assert flow("A", "C", "A", "C")[2] == "disagreement remains" and flow("A", "C", "C", "A")[2] == "swap"
    assert flow("A", "C", "A", "A")[2] == "converge X" and flow("A", "C", "B", "B")[2] == "converge third"
    assert flow("A", "C", "B", "C")[:2] == ("X→other", "Y stays Y")
    print("aggregate: all tests passed")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["test"]:
        _test()
