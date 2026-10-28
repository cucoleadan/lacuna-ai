"""Labelled pairs in data/work.db -> data/dataset/{train,val,test}.jsonl, split by CV (80/10/10).

Each example: cv text, job text, the four pair labels, and the job's requirement lines with a per-line label,
found by matching each DeepSeek requirement to the posting line it came from. The line head is what powers
"top gaps": must-have lines labelled `no`.
"""
import argparse
import json
import random
import sys
import re
import sqlite3
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cvtext import normalize_cv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work.db"
OUT = ROOT / "data" / "dataset"
V1_MAX_CV = 1973  # CVs of dataset v1; later CVs are split by id

VERDICT = ["apply", "stretch", "skip"]
LOCATION = ["yes", "unclear", "no"]
SENIORITY = ["below", "at", "above"]
LINE_MET = ["none", "yes", "partial", "no"]  # none = not a requirement line
STOP = set("a an and or the of to in for with on at by as is are be your you we our will have has from this that "
           "it its their they who what which experience years year strong ability skills knowledge".split())


def words(text):
    return {w for w in re.findall(r"[a-z0-9+#.]+", text.lower()) if w not in STOP and len(w) > 1}


def job_lines(job):
    """Character spans of non-empty lines in the job text."""
    spans, pos = [], 0
    for line in job.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        if len(line.strip(" -*•\t")) >= 15:
            spans.append((start, end))
    return spans


def align(job, requirements):
    """Match each requirement to its best posting line by word overlap; unmatched lines are 'none'."""
    spans = job_lines(job)
    lw = [words(job[a:b]) for a, b in spans]
    met = ["none"] * len(spans)
    must = [0] * len(spans)
    matched = 0
    for r in requirements:
        rw = words(r.get("text", ""))
        if not rw:
            continue
        scores = [len(rw & w) / len(rw) for w in lw]
        if scores and max(scores) >= 0.5:
            i = scores.index(max(scores))
            met[i] = r.get("met") if r.get("met") in LINE_MET else "none"
            must[i] = int(r.get("kind") == "must")
            matched += 1
    return [{"span": s, "met": m, "must": k} for s, m, k in zip(spans, met, must)], matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--raw-cv", action="store_true", help="skip CV normalisation (datasets v1/v2 were built without it)")
    a = ap.parse_args()
    out_dir = Path(a.out)
    con = sqlite3.connect(WORK)
    rows = con.execute("""select p.cv_id, p.kind, c.cv, s.text, p.label, p.derived from pairs p
                          join cvs c on c.id = p.cv_id join postings s on s.id = p.posting_id
                          where p.label is not null""").fetchall()
    cv_ids = sorted({r[0] for r in rows})
    v1 = [c for c in cv_ids if c <= V1_MAX_CV]  # the v1 split stays frozen so test scores compare across versions
    random.Random(0).shuffle(v1)
    split = {cid: "train" if i < 0.8 * len(v1) else "val" if i < 0.9 * len(v1) else "test" for i, cid in enumerate(v1)}
    split.update({c: "train" if c % 10 < 8 else "val" if c % 10 == 8 else "test" for c in cv_ids if c > V1_MAX_CV})
    n = len(cv_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {k: open(out_dir / f"{k}.jsonl", "w") for k in ("train", "val", "test")}
    stats, req_total, req_matched = Counter(), 0, 0
    for cv_id, kind, cv, job, label, derived in rows:
        lab, d = json.loads(label), json.loads(derived)
        lines, matched = align(job, lab.get("requirements", []))
        req_total += len(lab.get("requirements", []))
        req_matched += matched
        ex = {"cv_id": cv_id, "kind": kind, "cv": cv if a.raw_cv else normalize_cv(cv), "job": job,
              "verdict": VERDICT.index(d["verdict"]),
              "location": LOCATION.index(lab["location_eligible"]) if lab.get("location_eligible") in LOCATION else 1,
              "seniority": SENIORITY.index(d["candidate_vs_role"]),
              "must_met": int(bool(d["must_haves_met"])),
              "lines": lines}
        files[split[cv_id]].write(json.dumps(ex) + "\n")
        stats[split[cv_id]] += 1
    for f in files.values():
        f.close()
    (out_dir / "labels.json").write_text(json.dumps({"verdict": VERDICT, "location": LOCATION, "seniority": SENIORITY,
                                                 "line_met": LINE_MET}, indent=1))
    print(f"CVs {n}, examples {dict(stats)}")
    print(f"requirements matched to a posting line: {req_matched}/{req_total} = {req_matched / max(1, req_total):.0%}")


if __name__ == "__main__":
    main()
