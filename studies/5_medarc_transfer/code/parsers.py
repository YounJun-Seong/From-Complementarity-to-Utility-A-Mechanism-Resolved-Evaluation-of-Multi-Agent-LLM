"""Controlled parser (PHASE5_FREEZE.md §3). `python parsers.py test` runs the synthetic unit tests."""
import re

WORDS = ("yes", "no", "maybe")
BOX = re.compile(r"\\boxed\{([^{}]*)\}")
ANS = re.compile(r"answer\s*(?:is|:)\s*\**\s*(yes|no|maybe)\b", re.I)
# Amendment 1 (2026-10-05): the number is the first one within 60 characters after "confidence"/"confident"
# on the same line (e.g. "Confidence score: 70%", "a confidence score of 60%", "\\textbf{60\\%}").
CONF = re.compile(r"confiden(?:ce|t)(?!\s*intervals?)[^\d\n]{0,60}?(\d+(?:\.\d+)?)\s*(\\?%)?", re.I)


def answer(text):
    text = text or ""
    boxes = BOX.findall(text)
    if boxes:
        m = re.search(r"\b(yes|no|maybe)\b", boxes[-1].lower())
        return m.group(1) if m else None
    m = ANS.findall(text)
    return m[-1].lower() if m else None


def confidence(text):
    m = CONF.findall(text or "")
    if not m:
        return 0.0
    x, pct = float(m[-1][0]), m[-1][1]
    if pct:
        c = x / 100
    elif 1 < x <= 100:
        c = x / 100
    elif 0 <= x <= 1:
        c = x
    else:
        return 0.0
    return c if 0 <= c <= 1 else 0.0


def _test():
    A = answer
    assert A("... \\boxed{yes} <confidence: 85%>") == "yes"
    assert A("\\boxed{No}") == "no" and A("\\boxed{Maybe.}") == "maybe"
    assert A("first \\boxed{no} then revised \\boxed{yes}") == "yes"           # last box
    assert A("\\boxed{not known}") is None                                      # whole word only ("not" != "no")
    assert A("\\boxed{The answer is no; it is not yes}") == "no"                 # first whole word inside the box
    assert A("Therefore, the answer is **maybe**.") == "maybe"                  # no box: structured expression
    assert A("Final answer: yes") == "yes"
    assert A("I think yes, but no box and no structured expression") is None
    assert A("") is None and A(None) is None
    C = confidence
    assert C("<confidence: 85%>") == 0.85 and C("Confidence: 90") == 0.9 and C("confidence: 0.9") == 0.9
    assert C("confidence - 75 %") == 0.75 and C("Confidence: 1") == 1.0 and C("confidence: 150") == 0.0
    assert C("confidence: 60% ... updated confidence: 80%") == 0.8                # last match
    assert C("no score here") == 0.0
    # Amendment 1: formats observed in the smoke output (format only; no labels)
    for t, v in (("**Confidence Score: 70%**", .70), ("accompanied by a confidence score of 60%.", .60),
                 ("Confidence score: 60%. The uncertainty stems", .60), ("My confidence score for this answer is 75%, as", .75),
                 ("Confidence score: 85%\n\nThe rationale behind the confidence score is", .85), ("confidence score: 75%", .75),
                 ("My confidence score for this conclusion is 80%. The reason for not reaching a higher confidence is", .80),
                 ("with a confidence score of \\textbf{60\\%}.", .60), ("My confidence score for this answer is roughly 80%.", .80),
                 ("My confidence score for this judgment is 85% due to", .85), ("The confidence score for this recommendation would be around 70%", .70),
                 ("<confidence: 85%>", .85), ("I am 90% confident", 0.0),
                 ("Confidence score: 70%. The OR was 2.1 (95% confidence interval 1.2-3.4).", .70),
                 ("HR 0.8, 95% confidence interval 0.6 to 0.9", 0.0)):
        assert abs(C(t) - v) < 1e-9, (t, C(t), v)
    print("parsers: all tests passed")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["test"]:
        _test()
