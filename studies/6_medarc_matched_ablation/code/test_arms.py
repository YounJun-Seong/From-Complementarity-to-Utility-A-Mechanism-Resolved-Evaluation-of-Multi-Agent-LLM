"""Unit checks before any LLM call: arm A renders exactly the Phase V request sequence; B differs only by the
no-summary template (no summary calls, same agent-call seeds); C summarises the designated source."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "5_medarc_transfer", "code"))
import run_arms as RA          # noqa: E402
import run_ablation as P6      # noqa: E402

LOG = []


def fake(messages, temperature, max_tokens, seed, n=1):
    LOG.append((json_dump(messages), temperature, max_tokens, seed, n))
    return dict(texts=[f"resp seed {seed}"], finish=["stop"], input_tokens=1, output_tokens=1)


def json_dump(m):
    import json
    return json.dumps(m)


def main():
    d = dict(idx=7, pubid=1, question="Q?", contexts=["c1", "c2"])
    RA.chat = fake; P6.chat = fake
    LOG.clear(); RA.medarc(d, 2); ref = list(LOG)
    LOG.clear(); P6.medarc(d, 2, "A"); a = list(LOG)
    assert a == ref, "arm A differs from Phase V"
    LOG.clear(); b_out = P6.medarc(d, 2, "B"); b = list(LOG)
    agent_a = [x for x in a if '"role": "system"' not in x[0]]
    assert len(b) == 9 and [x[3] for x in b] == [x[3] for x in agent_a], "B agent seeds must match A"
    assert b[:3] == agent_a[:3], "round 0 must be identical"
    assert all("Summary of other agents" not in x[0] for x in b)
    src = dict(idx=8, agent_contexts=[[{"role": "user", "content": "u"}, {"role": "assistant", "content": f"SRC{i}"}] for i in range(3)])
    LOG.clear(); c_out = P6.medarc(d, 2, "C", src)
    first_summary = [x for x in LOG if '"role": "system"' in x[0]][0][0]
    assert "SRC1" in first_summary and "SRC2" in first_summary and "SRC0" not in first_summary
    assert c_out["summaries"][0]["source_idx"] == 8 and LOG[:3] == ref[:3]
    print("phase6 arm checks: all passed")


if __name__ == "__main__":
    main()
