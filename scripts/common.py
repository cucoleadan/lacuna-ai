"""Shared pieces: prompt files, the LLM client, CV date arithmetic and the rubric thresholds.

The labeller (DeepSeek V4.1 Flash) is reached through any OpenAI-compatible endpoint; the endpoint, credentials
and local paths are private settings in the gitignored local.json (see local.example.json).
"""
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402  private endpoint, accounts and paths (local.json)

ROOT = Path(__file__).resolve().parent.parent
PRICE_IN, PRICE_OUT = 0.30 / 1e6, 1.20 / 1e6  # list price; cache hits are cheaper, so this overestimates
TODAY = date(2026, 10, 22)  # fixed so "present" resolves the same way for every label

LEVELS = ["intern", "junior", "mid", "senior", "lead", "principal"]
LEVEL_WORDS = [
    ("lead", r"\b(founder|co-founder|cto|ceo)\b"),  # small-company founder titles do not screen as principal
    ("principal", r"\b(principal|head of|director|vp|vice president|chief|distinguished)\b"),
    ("lead", r"\b(staff|lead|engineering manager|senior manager|sr\.? manager)\b"),
    ("senior", r"\b(senior|sr\.?)\b"),
    ("junior", r"\b(junior|jr\.?|graduate|associate|assistant)\b"),
    ("intern", r"\b(intern|internship|trainee|apprentice|student)\b"),
]


# ---------- prompts ----------

def prompt_parts(name):
    md = (ROOT / "prompts" / name).read_text()
    system = md.split("## System", 1)[1].split("\n## ", 1)[0].strip()
    user = md.split("## User", 1)[1].split("\n## ", 1)[0].strip()
    return system, user


def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template


# ---------- text ----------

def html_text(path):
    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser")
    for t in soup(["style", "script", "svg"]):
        t.decompose()
    txt = soup.get_text("\n")
    return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", txt)).strip()


def scrub(text):
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", text)
    return re.sub(r"\+?\d[\d ()-]{8,}\d", "[phone]", text)


# ---------- LLM endpoint (OpenAI-compatible) ----------

class Client:
    def __init__(self, budget):
        self.budget, self.spent, self.lock = budget, 0.0, threading.Lock()
        self._refresh()

    def _refresh(self):  # API access tokens expire after 1 h; the full run died with 401s at that mark
        self.token, self.token_at = subprocess.check_output(config.get("token_cmd"), text=True).strip(), time.time()

    def chat(self, system, user, max_tokens=1200):
        with self.lock:
            if self.spent >= self.budget:
                raise RuntimeError(f"budget ${self.budget} reached")
        with self.lock:
            if time.time() - self.token_at > 2700:
                self._refresh()
        body = {
            "model": config.get("llm_model"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }
        req = urllib.request.Request(config.get("llm_url"), json.dumps(body).encode(), {
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json", **config.CFG.get("llm_headers", {})})
        for attempt in range(3):  # the endpoint occasionally stalls; 19/119 timed out at 8 threads
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.load(r)
                break
            except (TimeoutError, OSError):
                if attempt == 2:
                    raise
        u = d["usage"]
        with self.lock:
            self.spent += u["prompt_tokens"] * PRICE_IN + u["completion_tokens"] * PRICE_OUT
        text = d["choices"][0]["message"]["content"]
        return json.loads(text[text.find("{"): text.rfind("}") + 1]), u


# ---------- CV facts ----------

def ym(s):
    if not s:
        return None
    if str(s).lower() == "present":
        return TODAY.year * 12 + TODAY.month - 1
    m = re.match(r"(\d{4})(?:-(\d{1,2}))?", str(s))
    return int(m.group(1)) * 12 + int(m.group(2) or 1) - 1 if m else None


def title_level(title, years):
    t = title.lower().replace("member of technical staff", "")  # MTS is a mid-level title
    for level, pat in LEVEL_WORDS:
        if re.search(pat, t):
            return level
    return "mid" if years >= 2 else "junior"


def union_years(roles):
    spans = []
    for r in roles:
        if (r.get("employment") or "") == "internship":
            continue
        a, b = ym(r.get("start")), ym(r.get("end"))
        if a is not None and b is not None and b >= a:
            spans.append((a, b + 1))
    spans.sort()
    months, cur = 0, None
    for a, b in spans:
        if cur and a <= cur[1]:
            cur = (cur[0], max(cur[1], b))
        else:
            if cur:
                months += cur[1] - cur[0]
            cur = (a, b)
    if cur:
        months += cur[1] - cur[0]
    return round(months / 12, 1)


def top_level(roles, years):
    levels = [title_level(r.get("title", ""), years) for r in roles] or ["junior"]
    return max(levels, key=LEVELS.index)


def computed(facts):
    """Total years and level, plus the same per role function (relevant experience)."""
    roles = facts.get("roles", [])
    years = union_years(roles)
    by_fn = {}
    for fn in sorted({r.get("function") or "other" for r in roles}):
        rs = [r for r in roles if (r.get("function") or "other") == fn]
        y = union_years(rs)
        by_fn[fn] = {"years": y, "level": top_level(rs, y)}
    return years, top_level(roles, years), by_fn


# ---------- data ----------

def roles_list(roles):
    return "\n".join(f"  {i}. {r.get('title')} at {r.get('company')}, {r.get('start')} to {r.get('end')}"
                     f" ({r.get('employment') or 'full-time'})" for i, r in enumerate(roles, 1)) or "  none"


def derive(lab, roles):
    """Apply the rubric thresholds (prompts/label_pair.md, 'Python, after labelling')."""
    picked = [roles[i - 1] for i in lab.get("relevant_roles") or [] if isinstance(i, int) and 0 < i <= len(roles)]
    years = union_years(picked)
    level = top_level(picked, years) if picked else "junior"
    rl = lab.get("role_level") if lab.get("role_level") in LEVELS else "mid"
    steps = LEVELS.index(rl) - LEVELS.index(level)  # >0: role above candidate
    try:
        req = float(lab.get("required_years")) if lab.get("required_years") not in (None, "null", "") else None
    except (TypeError, ValueError):
        req = None
    gap_years = (req - years) if req is not None else steps * 2
    if steps <= -2 and (req is None or years >= req + 5):  # over-qualified needs real extra years too
        vs = "above"
    elif gap_years >= 2 or (req is None and steps >= 1):
        vs = "below"
    else:
        vs = "at"
    if years == 0 and rl == "intern":
        vs = "at"
    musts = [r for r in lab.get("requirements", []) if r.get("kind") == "must"]
    score = sum({"yes": 1, "partial": 0.5}.get(r.get("met"), 0) for r in musts) / len(musts) if musts else 1.0
    gates_ok = all(r.get("met") == "yes" for r in musts if r.get("gate"))
    other_no = sum(r.get("met") == "no" for r in musts if not r.get("gate"))
    far_below = gap_years >= 3 or (req is None and steps >= 2)
    loc = lab.get("location_eligible")
    if loc == "no" or not gates_ok or far_below or score < 0.5:
        verdict = "skip"
    elif loc in ("yes", "unclear") and score >= 2 / 3 and (vs == "at" or (vs == "below" and gap_years <= 2 and score == 1)):
        verdict = "apply"
    else:
        verdict = "stretch"
    return {"relevant_years": years, "relevant_level": level, "candidate_vs_role": vs, "gap_years": gap_years, "must_score": round(score, 2),
            "must_haves_met": gates_ok and other_no <= 1, "verdict": verdict}
