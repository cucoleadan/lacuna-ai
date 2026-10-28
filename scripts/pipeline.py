"""Training data pipeline: postings -> synthetic CVs -> pairs -> labels.

  python scripts/pipeline.py postings          # clean + tag postings from data/pages_snapshot.db
  python scripts/pipeline.py cvs --n 20         # sample careers, DeepSeek writes the CVs
  python scripts/pipeline.py pairs --per-cv 4   # per CV: its target job, a near one, a far-level one, a location mismatch
  python scripts/pipeline.py extra              # targeted pairs for rare classes (over-qualified, stretch)
  python scripts/pipeline.py label              # DeepSeek labels pairs, Python derives verdicts
  python scripts/pipeline.py report             # label distribution per head, spend

Everything lives in data/work.db (gitignored). Every stage resumes. Total spend across runs is
kept in data/spend.json and capped at --cap dollars (default 30).
"""
import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import LEVELS, TODAY, Client, derive, fill, prompt_parts, roles_list, scrub, title_level, union_years  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SNAP = DATA / "pages_snapshot.db"
WORK = DATA / "work.db"
SPEND = DATA / "spend.json"

FUNCTIONS = [  # first match wins; order matters
    ("design", r"\b(designer|ux|ui/ux|ui designer|design lead|product design)"),
    ("product", r"\b(product manager|product owner|head of product|product lead|\bpm\b)"),
    ("data", r"\b(data scien|data analy|analytics|business intelligence|bi analyst|data engineer|machine learning|ml engineer|ai engineer|research scientist)"),
    ("engineering", r"\b(engineer|developer|devops|sre\b|architect|programmer|software|frontend|front-end|backend|back-end|full.?stack|mobile|qa\b|tester|cto)"),
    ("marketing", r"\b(marketing|growth|seo|content|brand|communications|social media|copywriter|demand gen|pr manager)"),
    ("sales", r"\b(sales|account executive|account manager|business development|sdr\b|bdr\b|customer success|partnerships)"),
    ("operations", r"\b(operations|recruit|talent|people|hr\b|human resources|finance|accountant|controller|legal|counsel|support|office manager|project manager|program manager)"),
]
CAREER_FUNCTIONS = ["engineering", "data", "design", "product", "marketing", "sales", "operations"]

LOCATIONS = [  # (location, native language, weight): Europe-heavy, the rest of the world for coverage
    ("Bucharest, Romania", "Romanian", 6), ("Warsaw, Poland", "Polish", 6), ("Berlin, Germany", "German", 6),
    ("Madrid, Spain", "Spanish", 5), ("Lisbon, Portugal", "Portuguese", 5), ("Amsterdam, Netherlands", "Dutch", 4),
    ("Paris, France", "French", 5), ("Milan, Italy", "Italian", 4), ("London, United Kingdom", "English", 6),
    ("Dublin, Ireland", "English", 3), ("Stockholm, Sweden", "Swedish", 3), ("Prague, Czech Republic", "Czech", 3),
    ("Athens, Greece", "Greek", 2), ("Kyiv, Ukraine", "Ukrainian", 2), ("Istanbul, Turkey", "Turkish", 2),
    ("New York, USA", "English", 6), ("Austin, USA", "English", 4), ("Toronto, Canada", "English", 4),
    ("Bengaluru, India", "Hindi", 5), ("São Paulo, Brazil", "Portuguese", 3), ("Mexico City, Mexico", "Spanish", 3),
    ("Buenos Aires, Argentina", "Spanish", 2), ("Manila, Philippines", "Filipino", 2), ("Lagos, Nigeria", "English", 2),
    ("Cape Town, South Africa", "English", 2), ("Sydney, Australia", "English", 2), ("Singapore", "English", 2),
]
YEARS_BANDS = [((0, 0), 8), ((1, 3), 25), ((3, 6), 30), ((6, 10), 25), ((10, 20), 12)]
FORMATS = ["markdown with headings", "plain text with ALL-CAPS section names",
           "text extracted from a PDF (broken line wraps, a repeated header line, bullets as odd characters)",
           "a one-page compact layout"]
QUALITY = [("strong (specific results with numbers)", 3), ("average (mix of specific and generic bullets)", 5),
           ("weak (vague duties, a few typos, no numbers, skills list longer than the evidence)", 2)]
EXTRAS = ["a short profile summary at the top", "a long skills section", "nothing extra", "nothing extra",
          "an employment gap explained in one line", "a career change: say so in the summary"]


def wchoice(rng, items):
    return rng.choices([i[0] if isinstance(i, tuple) else i for i in items],
                       weights=[i[-1] for i in items])[0]


def db():
    con = sqlite3.connect(WORK)
    con.executescript("""
    create table if not exists postings (id integer primary key, url text unique, company text, title text,
        location text, workplace text, function text, level text, text text);
    create table if not exists cvs (id integer primary key, persona text, facts text, cv text);
    create table if not exists pairs (id integer primary key, cv_id int, posting_id int, kind text,
        label text, derived text, usage text, unique(cv_id, posting_id));
    """)
    return con


def spent():
    return json.loads(SPEND.read_text())["total"] if SPEND.exists() else 0.0


def add_spend(amount, what):
    d = json.loads(SPEND.read_text()) if SPEND.exists() else {"total": 0.0, "runs": []}
    d["total"] = round(d["total"] + amount, 4)
    d["runs"].append({"date": date.today().isoformat(), "what": what, "usd": round(amount, 4)})
    SPEND.write_text(json.dumps(d, indent=1))


def tag_function(title):
    t = title.lower()
    for fn, pat in FUNCTIONS:
        if re.search(pat, t):
            return fn
    return "other"


# ---------- stages ----------

def cmd_postings(args):
    con, src = db(), sqlite3.connect(f"file:{SNAP}?mode=ro", uri=True)
    seen_sha, seen_ct, n = set(), set(), 0
    rows = src.execute("""select url, company, title, location, workplace, sha256, markdown from postings
                          where english=1 and chars between 800 and 20000""")
    for url, company, title, location, workplace, sha, md in rows:
        key = ((company or "").lower(), (title or "").lower())
        if sha in seen_sha or key in seen_ct:  # exact dupes and reposts
            continue
        seen_sha.add(sha)
        seen_ct.add(key)
        header = "\n".join(f"{k}: {v}" for k, v in
                           zip(("Title", "Company", "Location", "Workplace"), (title, company, location, workplace)) if v)
        fn = tag_function(title or "")
        con.execute("insert or ignore into postings values (null,?,?,?,?,?,?,?,?)",
                    (url, company, title, location, workplace, fn, title_level(title or "", 3), scrub(header + "\n\n" + md)))
        n += 1
    con.commit()
    print(f"kept {n} postings")
    print(Counter(r[0] for r in con.execute("select function from postings")).most_common())
    print(Counter(r[0] for r in con.execute("select level from postings")).most_common())


def level_for(years_at_start):
    return "junior" if years_at_start < 2 else "mid" if years_at_start < 5 else "senior" if years_at_start < 9 else "lead"


YEARS_FOR_LEVEL = {"intern": (0, 0), "junior": (0.5, 2), "mid": (2, 5), "senior": (5, 9), "lead": (8, 12), "principal": (10, 18)}
COVERAGE = [("strong: the CV shows evidence for about 90% of the target's requirements", 4),
            ("partial: the CV shows evidence for about 60% of the target's requirements", 4),
            ("weak: the CV shows evidence for about 30% of the target's requirements", 2)]


class Index:
    """TF-IDF over postings (title ×3 + top of posting), for 'similar job' lookups."""

    def __init__(self, con):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.posts = con.execute("""select id, function, level, coalesce(location,'') || ' ' || coalesce(workplace,''),
                                    company, title, text from postings""").fetchall()
        self.idx = {r[0]: i for i, r in enumerate(self.posts)}
        print("tf-idf over postings…", flush=True)
        self.X = TfidfVectorizer(max_features=200_000, sublinear_tf=True, stop_words="english", min_df=3).fit_transform(
            (r[6].split("\n", 1)[0] + "\n") * 3 + r[6][:1500] for r in self.posts)

    def similar(self, pid, k=3000):
        sims = (self.X @ self.X[self.idx[pid]].T).toarray().ravel()
        return [self.posts[i] for i in sims.argsort()[::-1][1:k + 1]]


def sample_persona(rng, ix):
    target = rng.choice([r for r in ix.posts if r[1] in CAREER_FUNCTIONS])
    tid, fn, tlevel, tloc = target[:4]
    ok_locs = [l for l in LOCATIONS if can_work(l[0], tloc, strict=True)] or [l for l in LOCATIONS if can_work(l[0], tloc)]
    pool = ok_locs if ok_locs and rng.random() < 0.85 else LOCATIONS
    loc, native, _ = rng.choices(pool, weights=[w for *_, w in pool])[0]
    english = "native" if native == "English" else rng.choice(["fluent", "professional", "professional", "basic"])
    langs = [f"{native} (native)"] if native == "English" else [f"{native} (native)", f"English ({english})"]
    if rng.random() < 0.25:
        langs.append(f"{rng.choice(['German', 'French', 'Spanish', 'Italian'])} ({rng.choice(['basic', 'professional'])})")

    lo, hi = YEARS_FOR_LEVEL[tlevel] if rng.random() < 0.7 else wchoice(rng, YEARS_BANDS)
    asked = re.search(r"\b(\d{1,2})\+?\s*(?:-|to)?\s*(?:\d{1,2}\s*)?\+?\s*years", target[6], re.I)
    if asked and hi and rng.random() < 0.7 and int(asked.group(1)) <= 15:  # meet the posting's stated years
        lo, hi = int(asked.group(1)), int(asked.group(1)) + 3
    years = round(rng.uniform(lo, hi), 1) if hi else 0.0
    end_month = TODAY.year * 12 + TODAY.month - 1
    if rng.random() < 0.12:
        end_month -= rng.randint(2, 12)  # currently between jobs
    similar = [r for r in ix.similar(tid, 400) if r[1] == fn][:200] or [target]
    roles, months_left = [], int(years * 12)
    if years == 0:  # student or new graduate: one internship
        roles.append(role(rng.choice(similar[:20]), end_month - rng.randint(3, 6), end_month, "internship", fn))
    else:
        n_roles = 1 if years < 1.5 else rng.randint(1, 2) if years < 3 else rng.randint(2, 3) if years < 6 else rng.randint(3, 4)
        cuts = sorted(rng.sample(range(1, max(2, months_left)), min(n_roles - 1, max(0, months_left - 2))))
        spans = [y - x for x, y in zip([0] + cuts, cuts + [months_left])]
        cursor = end_month
        career_change = rng.random() < 0.1 and n_roles > 1
        for i, span in enumerate(spans):  # most recent first; later roles less similar
            years_before = (months_left - sum(spans[: i + 1])) / 12
            want = level_for(years_before)
            if career_change and i == len(spans) - 1:
                other = rng.choice([f for f in CAREER_FUNCTIONS if f != fn])
                cands = [r for r in ix.posts[:20000] if r[1] == other] or similar
                role_fn = other
            else:
                window = similar[: 10 + 30 * i]
                cands = [r for r in window if r[2] == want] or window
                role_fn = fn
            roles.append(role(rng.choice(cands), cursor - span, cursor,
                              "full-time" if rng.random() > 0.12 else "freelance", role_fn))
            cursor -= span
    edu = rng.choices(["a bachelor's degree relevant to the field", "a master's degree relevant to the field",
                       "a bachelor's degree in an unrelated field", "a bootcamp certificate, no degree",
                       "no degree"], weights=[5, 3, 2, 1, 1])[0]
    target_text = target[6].split("\n\n", 1)[-1]
    return {"target_id": tid, "function": fn, "years_target": years, "location": loc, "languages": langs,
            "relocation": "yes" if rng.random() < 0.15 else "not stated", "education": edu, "roles": roles,
            "coverage": wchoice(rng, COVERAGE), "target_requirements": target_text[-2500:],
            "format": rng.choice(FORMATS), "quality": wchoice(rng, QUALITY),
            "length": rng.choice(["one page (~400 words)", "two pages (~800 words)"]),
            "extra": "a career change: say so in the summary" if any(r["function"] != fn for r in roles) else rng.choice(EXTRAS)}


def role(posting, start, end, employment, fn):
    pid, _, _, _, company, title, text = posting
    company = (company or "").replace("-", " ").replace("_", " ").title()  # ATS slugs -> readable names
    body = text.split("\n\n", 1)[-1][:1200]
    return {"posting_id": pid, "title": title, "company": company, "start": f"{start // 12}-{start % 12 + 1:02d}",
            "end": "present" if end >= TODAY.year * 12 + TODAY.month - 1 else f"{end // 12}-{end % 12 + 1:02d}",
            "employment": employment, "function": fn, "excerpt": body}


def facts_from(persona):
    roles = persona["roles"]
    years = union_years(roles)
    by_fn = {}
    for fn in sorted({r["function"] for r in roles}):
        rs = [r for r in roles if r["function"] == fn]
        y = union_years(rs)
        by_fn[fn] = {"years": y, "level": max((title_level(r["title"], y) for r in rs), key=LEVELS.index)}
    level = max((title_level(r["title"], years) for r in roles), key=LEVELS.index)
    return {"total_years": years, "level": level, "by_function": by_fn, "location": persona["location"],
            "languages": persona["languages"], "relocation": persona["relocation"]}


def cmd_cvs(args):
    con = db()
    rng = random.Random(args.seed + con.execute("select count(*) from cvs").fetchone()[0])
    ix = Index(con)
    personas = [sample_persona(rng, ix) for _ in range(args.n)]
    client = Client(args.cap - spent())
    sys_p, user_p = prompt_parts("synth_cv.md")

    def write(p):
        roles = "\n".join(f"- {r['title']} at {r['company']}, {r['start']} to {r['end']} ({r['employment']})\n"
                          f"  Posting excerpt: {r['excerpt']}" for r in p["roles"])
        out, _ = client.chat(sys_p, fill(user_p, location=p["location"], languages=", ".join(p["languages"]),
                                         relocation=p["relocation"], education=p["education"], roles=roles,
                                         format=p["format"], quality=p["quality"], length=p["length"], extra=p["extra"],
                                         coverage=p["coverage"], target=p["target_requirements"]),
                             max_tokens=2500)
        return p, out.get("cv", "")

    done = 0
    with ThreadPoolExecutor(8) as ex:
        for p, cv in ex.map(lambda p: _safe(write, p), personas):
            if cv:
                for r in p["roles"]:
                    r.pop("excerpt", None)
                p.pop("target_requirements", None)
                con.execute("insert into cvs values (null,?,?,?)", (json.dumps(p), json.dumps(facts_from(p)), cv))
                done += 1
                if done % 50 == 0:
                    con.commit()
    con.commit()
    add_spend(client.spent, f"cvs n={args.n}")
    print(f"wrote {done}/{args.n} CVs, ~${client.spent:.3f} (total ${spent():.2f})")


def _safe(fn, x):
    try:
        return fn(x)
    except Exception as e:
        print("error:", str(e)[:150])
        return x, None


# where each persona can work, from the posting's Location + Workplace fields (the labeller reads the body too)
EU = ["romania", "poland", "germany", "spain", "portugal", "netherlands", "france", "italy", "ireland", "sweden",
      "czech", "greece"]
REGION = {
    "europe": r"\b(europe|emea|eu\b|european union|cet\b|cest\b)",
    "latam": r"\b(latam|latin america|south america|americas)\b",
    "apac": r"\b(apac|asia|asia.pacific)\b",
    "africa": r"\b(africa|emea)\b",
}
US_MARK = r"(united states|\busa\b|\bu\.s\.|\bus\b|, (?:al|az|ca|co|ct|dc|fl|ga|il|ma|md|mi|mn|nc|nj|ny|oh|or|pa|tn|tx|ut|va|wa|wi)\b|new york|san francisco|seattle|austin|boston|chicago|los angeles|denver|atlanta|miami)"
COUNTRY = {  # persona location -> (own markers, regions it belongs to)
    "Romania": (r"romania|bucharest|cluj", ["europe"]), "Poland": (r"poland|warsaw|krak", ["europe"]),
    "Germany": (r"germany|berlin|munich|hamburg|deutschland", ["europe"]), "Spain": (r"spain|madrid|barcelona", ["europe"]),
    "Portugal": (r"portugal|lisbon|porto", ["europe"]), "Netherlands": (r"netherlands|amsterdam|rotterdam", ["europe"]),
    "France": (r"france|paris|lyon", ["europe"]), "Italy": (r"italy|milan|rome", ["europe"]),
    "United Kingdom": (r"united kingdom|\buk\b|london|england|manchester", ["europe"]),
    "Ireland": (r"ireland|dublin", ["europe"]), "Sweden": (r"sweden|stockholm", ["europe"]),
    "Czech Republic": (r"czech|prague", ["europe"]), "Greece": (r"greece|athens", ["europe"]),
    "Ukraine": (r"ukraine|kyiv", ["europe"]), "Turkey": (r"turkey|türkiye|istanbul", ["europe"]),
    "USA": (US_MARK, []), "Canada": (r"canada|toronto|vancouver|montreal", []),
    "India": (r"india|bengaluru|bangalore|hyderabad|pune|mumbai|delhi|gurgaon|chennai", ["apac"]),
    "Brazil": (r"brazil|brasil|são paulo|sao paulo", ["latam"]), "Mexico": (r"mexico|méxico", ["latam"]),
    "Argentina": (r"argentina|buenos aires", ["latam"]), "Philippines": (r"philippines|manila", ["apac"]),
    "Nigeria": (r"nigeria|lagos", ["africa"]), "South Africa": (r"south africa|cape town|johannesburg", ["africa"]),
    "Australia": (r"australia|sydney|melbourne", ["apac"]), "Singapore": (r"singapore", ["apac"]),
}
GLOBAL = r"\b(worldwide|anywhere|global|fully remote|remote.first|work from anywhere)\b"
ANY_PLACE = "|".join(m for m, _ in COUNTRY.values()) + "|" + "|".join(REGION.values())


def can_work(persona_loc, loc_text, strict=False):
    """Cheap pre-filter for pairing: True if the posting's header suggests this person may be eligible.
    strict: only an explicit match (own country/city, region, worldwide); bare "Remote" is often US-only in the body."""
    country = persona_loc.split(", ")[-1]
    own, regions = COUNTRY[country]
    t = loc_text.lower()
    if re.search(own, t) or re.search(GLOBAL, t) or any(re.search(REGION[r], t) for r in regions):
        return True
    return not strict and "remote" in t and not re.search(ANY_PLACE, t)


def cmd_pairs(args):
    con = db()
    rng = random.Random(args.seed)
    ix = Index(con)
    n = 0
    todo = con.execute("select id, persona, facts from cvs where id not in (select cv_id from pairs)").fetchall()
    for cv_id, persona, facts in todo:
        p = json.loads(persona)
        own = {r["posting_id"] for r in p["roles"]}
        li = LEVELS.index(ix.posts[ix.idx[p["target_id"]]][2])
        ranked = [x for x in ix.similar(p["target_id"]) if x[0] not in own]
        ok = [x for x in ranked if can_work(p["location"], x[3])]
        near = [x for x in ok if abs(LEVELS.index(x[2]) - li) <= 1][:20]
        far = [x for x in ok if abs(LEVELS.index(x[2]) - li) >= 2][:40]
        away = [x for x in ranked if not can_work(p["location"], x[3])][:40]
        picks = [("target", [ix.posts[ix.idx[p["target_id"]]]]), ("near", near), ("seniority", far), ("location", away)]
        for kind, pool in picks[: args.per_cv]:
            if pool:
                pid = rng.choice(pool)[0]
                n += con.execute("insert or ignore into pairs (cv_id, posting_id, kind) values (?,?,?)", (cv_id, pid, kind)).rowcount
    con.commit()
    print(f"added {n} pairs; total {con.execute('select count(*) from pairs').fetchone()[0]}")


def cmd_extra(args):
    """Targeted pairs for the rare classes in v1 (DECISIONS.md, dataset v2):
    over    = similar job 2+ levels below the CV (over-qualified, seniority `above`), senior+ CVs only
    stretch = similar job exactly one level above the CV (mostly `stretch`/`apply` verdicts)"""
    con = db()
    rng = random.Random(args.seed)
    ix = Index(con)
    n = 0
    todo = con.execute("""select id, persona, facts from cvs where id not in
                          (select cv_id from pairs where kind in ('over', 'stretch'))""").fetchall()
    for cv_id, persona, facts in todo:
        p, cl = json.loads(persona), LEVELS.index(json.loads(facts)["level"])
        own = {r["posting_id"] for r in p["roles"]}
        ok = [x for x in ix.similar(p["target_id"]) if x[0] not in own and can_work(p["location"], x[3])]
        picks = [("over", [x for x in ok if LEVELS.index(x[2]) <= cl - 2][:40]),
                 ("stretch", [x for x in ok if LEVELS.index(x[2]) == cl + 1][:20])]
        for kind, pool in picks:
            if pool:
                pid = rng.choice(pool)[0]
                n += con.execute("insert or ignore into pairs (cv_id, posting_id, kind) values (?,?,?)", (cv_id, pid, kind)).rowcount
    con.commit()
    print(f"added {n} targeted pairs; total {con.execute('select count(*) from pairs').fetchone()[0]}")


def domestic_override(lab, persona_loc, job_loc):
    """The labeller often ignores the domestic-relocation rule; if the posting's own location names the
    candidate's country, 'no' becomes 'unclear' (synthetic personas are authorized in their own country)."""
    own = COUNTRY[persona_loc.split(", ")[-1]][0]
    if lab.get("location_eligible") == "no" and re.search(own, job_loc.lower()):
        lab["location_eligible"], lab["location_override"] = "unclear", True


def cmd_label(args):
    con = db()
    todo = con.execute("""select p.id, c.cv, c.facts, c.persona, s.text,
                                 coalesce(s.location,'') || ' ' || coalesce(s.workplace,'') from pairs p join cvs c on c.id = p.cv_id
                          join postings s on s.id = p.posting_id where p.label is null order by p.cv_id limit ?""",
                       (args.limit or -1,)).fetchall()
    client = Client(args.cap - spent())
    sys_p, user_p = prompt_parts("label_pair.md")

    def run(row):
        pid, cv, facts, persona, job, job_loc = row
        f, roles = json.loads(facts), json.loads(persona)["roles"]
        lab, u = client.chat(sys_p, fill(user_p, total_years=f["total_years"], roles_list=roles_list(roles),
                                         location=f["location"], open_to_relocation=f["relocation"],
                                         work_authorization="not stated", languages=", ".join(f["languages"]),
                                         today=TODAY.isoformat(), cv_text=cv, job_text=job))
        domestic_override(lab, f["location"], job_loc)
        return pid, lab, derive(lab, roles), u

    done = 0
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(lambda r: _safe(run, r), todo):
            if len(res) == 4:
                pid, lab, d, u = res
                con.execute("update pairs set label=?, derived=?, usage=? where id=?",
                            (json.dumps(lab), json.dumps(d), json.dumps(u), pid))
                done += 1
                if done % 50 == 0:
                    con.commit()
    con.commit()
    add_spend(client.spent, f"label n={done}")
    print(f"labelled {done}/{len(todo)}, ~${client.spent:.3f} (total ${spent():.2f})")


def cmd_report(args):
    con = db()
    rows = [(k, json.loads(l), json.loads(d), json.loads(u)) for k, l, d, u in
            con.execute("select kind, label, derived, usage from pairs where label is not null")]
    print(f"CVs {con.execute('select count(*) from cvs').fetchone()[0]}, labelled pairs {len(rows)}, total spend ${spent():.2f}")
    if not rows:
        return
    for head, get in [("verdict", lambda l, d: d["verdict"]), ("location", lambda l, d: l.get("location_eligible")),
                      ("seniority", lambda l, d: d["candidate_vs_role"]), ("must_haves_met", lambda l, d: d["must_haves_met"]),
                      ("role_function", lambda l, d: l.get("role_function"))]:
        print(f"{head:15}", dict(Counter(get(l, d) for _, l, d, _ in rows).most_common()))
    for kind in sorted({k for k, *_ in rows}):
        print(f"  {kind:11} verdicts", dict(Counter(d["verdict"] for k, _, d, _ in rows if k == kind)))
    tin = sum(u["prompt_tokens"] for *_, u in rows) / len(rows)
    tout = sum(u["completion_tokens"] for *_, u in rows) / len(rows)
    print(f"tokens per pair: in {tin:.0f}, out {tout:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["postings", "cvs", "pairs", "extra", "label", "report"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--per-cv", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cap", type=float, default=30.0)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    globals()[f"cmd_{args.stage}"](args)


if __name__ == "__main__":
    main()
