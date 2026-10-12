"""Verify every cached response behind a run file was generated in the P0 epoch.
Records written before metadata logging was added carry no generated_at; for those the
cache file's mtime is used (files are written once, never rewritten).

  python experiments/epoch_check.py [--since 2026-09-24T00:00:00]
"""
import argparse
import collections
import datetime as dt
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-24T00:00:00")
    a = ap.parse_args()
    since = dt.datetime.fromisoformat(a.since)
    files = glob.glob(f"{ROOT}/data/llm_cache/*.json")
    fp, src, early, models, fin = collections.Counter(), collections.Counter(), [], collections.Counter(), collections.Counter()
    t_min = t_max = None
    for f in files:
        r = json.load(open(f))
        if r.get("generated_at"):
            t = dt.datetime.strptime(r["generated_at"][:19], "%Y-%m-%dT%H:%M:%S"); src["logged"] += 1
        else:
            t = dt.datetime.fromtimestamp(os.path.getmtime(f)); src["mtime"] += 1
        t_min = t if t_min is None or t < t_min else t_min
        t_max = t if t_max is None or t > t_max else t_max
        if t < since:
            early.append(f)
        fp[r.get("system_fingerprint")] += 1
        models[r.get("served_model") or r["request"]["model"]] += 1
        fin[r.get("finish_reason")] += 1
    print(f"{len(files)} cached responses, generated {t_min} .. {t_max} (timestamp source {dict(src)})")
    print(f"before epoch start {a.since}: {len(early)}")
    print(f"models: {dict(models)}")
    print(f"system_fingerprint: {dict(fp)}")
    print(f"finish_reason: {dict(fin)}")
    print("EPOCH OK" if not early else "EPOCH VIOLATION")


if __name__ == "__main__":
    main()
