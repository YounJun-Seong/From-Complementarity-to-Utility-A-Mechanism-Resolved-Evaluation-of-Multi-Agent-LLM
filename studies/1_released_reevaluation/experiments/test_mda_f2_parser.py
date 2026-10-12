"""F2 parser-robustness unit test (no API calls).

1. Canonical forms: all variants of a member line parse to the same member.
2. Replay every advanced-path recruiter output in the cache (incl. every F2_buggy/F3_buggy
   crash) through normalization -> released group split -> released parse_group_info, and
   count groups with zero members (the IndexError in Group.interact).

  MDA_STRICT_PARSE unset:  python experiments/test_mda_f2_parser.py
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "external", "MDAgents"))
os.environ.pop("MDA_STRICT_PARSE", None)
import utils  # noqa: E402

VARIANTS = ["Member 1: Cardiology", "**Member 1:** Cardiology", "- Member 1: Cardiology",
            "- **Member 1:** Cardiology", "* **Member 1:** Cardiology"]


def groups_of(recruited):
    """Released advanced-path parsing, with the F2 normalization in front."""
    recruited = utils._normalize_recruited(recruited)
    groups = [g.strip() for g in recruited.split("Group") if g.strip()]
    return [utils.parse_group_info("Group " + g) for g in groups]


def main():
    ref = None
    for v in VARIANTS:
        g = utils.parse_group_info("Group 1 - Initial Assessment Team (IAT)\n" + utils._strip_markdown(v))
        got = [(m["role"], m["expertise_description"]) for m in g["members"]]
        print(f"  {v!r:32s} -> {got}")
        ref = ref or got
        assert got == ref == [("Cardiology", "")], v
    print("canonical forms: OK")

    crashed = set()
    for C in ("F2_buggy", "F3_buggy"):
        for f in glob.glob(f"{ROOT}/runs/mda/{C}/run*/shard*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                if r["error"] and r["error"].startswith("IndexError") and r["difficulty"] == "advanced":
                    crashed.add(f"mda:run{r['seed']}:rng-{r['rng']}:strict0:unify{int(r['unify'])}:adaptive:{r['idx']}:")
    outs, crash_outs = [], 0
    for f in glob.glob(f"{ROOT}/data/llm_cache/*.json"):
        s = open(f).read()
        if "You should organize" not in s:
            continue
        r = json.loads(s)
        if "You should organize" not in r["request"]["messages"][-1]["content"]:
            continue
        n = r["request"].get("nonce", "")
        is_crash = any(n.startswith(c) for c in crashed)
        crash_outs += is_crash
        outs.append((n, is_crash, r["response"]))
    empty, empty_crash, bad = 0, 0, []
    for n, is_crash, resp in outs:
        gs = groups_of(resp)
        e = sum(1 for g in gs if not g["members"])
        if e:
            empty += 1
            empty_crash += is_crash
            bad.append((n, [g["group_goal"][:40] for g in gs if not g["members"]], resp[:300]))
    print(f"advanced recruiter outputs replayed: {len(outs)} (of which from F2/F3_buggy crashes: {crash_outs}/{len(crashed)})")
    print(f"outputs still yielding an empty group: {empty} (from crashes: {empty_crash})")
    for n, goals, head in bad[:5]:
        print("  ", n, goals, "|", head.replace("\n", " / ")[:250])
    assert crash_outs == len(crashed), "missing recruiter outputs for some crashes"
    # residual (out of F2 scope, reported not repaired): numbered member lists without
    # "Member N:", and the word "Group" inside member text
    assert all(not re.match(r"^\s*$", h) for _, _, h in bad), "unexpected"
    assert empty <= 5, "formatting failures remain beyond the known out-of-scope residual"
    print(f"replay: OK ({empty} residual out-of-scope outputs listed above)")


if __name__ == "__main__":
    main()
