"""Unlabelled CV × posting pairs for distillation -> <dataset>/unlabelled.jsonl.

  python scripts/distil_pairs.py --per-cv 8

Train-split CVs only (val/test CVs never appear, so test scores stay clean). Per CV, new postings drawn like the
labelled kinds: similar jobs at a nearby level, one level up, two+ levels away, a location mismatch, and a few
broader matches. Pairs already in work.db are skipped. No API calls: the ModernBERT-large teacher labels these
(train/teach.py), and the student trains on its soft predictions.
"""
import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import job_lines  # noqa: E402
from cvtext import normalize_cv  # noqa: E402
from pipeline import LEVELS, WORK, Index, can_work  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--per-cv", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    train_cvs = {json.loads(l)["cv_id"] for l in open(Path(a.data) / "train.jsonl")}
    con = sqlite3.connect(WORK)
    seen = set(con.execute("select cv_id, posting_id from pairs").fetchall())
    ix = Index(con)
    cvs = con.execute("select id, persona, facts, cv from cvs").fetchall()
    stats = Counter()
    with open(Path(a.data) / "unlabelled.jsonl", "w") as f:
        for cv_id, persona, facts, cv in cvs:
            if cv_id not in train_cvs:
                continue
            p, cl = json.loads(persona), LEVELS.index(json.loads(facts)["level"])
            own = {r["posting_id"] for r in p["roles"]}
            ranked = [x for x in ix.similar(p["target_id"]) if x[0] not in own and (cv_id, x[0]) not in seen]
            ok = [x for x in ranked if can_work(p["location"], x[3])]
            pools = [("near", [x for x in ok if abs(LEVELS.index(x[2]) - cl) <= 1][:60], 3),
                     ("up", [x for x in ok if LEVELS.index(x[2]) == cl + 1][:40], 1),
                     ("far", [x for x in ok if abs(LEVELS.index(x[2]) - cl) >= 2][:60], 1),
                     ("away", [x for x in ranked if not can_work(p["location"], x[3])][:60], 1),
                     ("broad", ok[100:1500], a.per_cv - 6)]
            picked = set()
            for kind, pool, k in pools:
                for x in rng.sample(pool, min(k, len(pool))):
                    if x[0] in picked:
                        continue
                    picked.add(x[0])
                    job = x[6]
                    f.write(json.dumps({"cv_id": cv_id, "posting_id": x[0], "kind": f"d-{kind}", "cv": normalize_cv(cv),
                                        "job": job, "lines": [{"span": s} for s in job_lines(job)]}) + "\n")
                    stats[kind] += 1
    print(f"{sum(stats.values())} pairs from {len(train_cvs)} train CVs: {dict(stats)}")


if __name__ == "__main__":
    main()
