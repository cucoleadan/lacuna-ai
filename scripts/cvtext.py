"""One canonical CV text, used for training CVs and for real CVs at inference (port to web/ as-is).

  html_to_text(html)   block-aware HTML -> text, like a browser's innerText (inline tags do not break lines)
  normalize_cv(text)   any CV text (HTML-, PDF- or LLM-produced) -> one layout: ALL-CAPS headings, "- " bullets,
                       "Label: value" contact lines, wrapped lines joined, contact details scrubbed

Rule-based and deterministic on purpose: the model only ever sees this layout, and the browser runs the same rules.
"""
import re
import unicodedata

from bs4 import BeautifulSoup, Comment, Doctype, NavigableString

BLOCK = {"p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "section", "article",
         "header", "footer", "aside", "main", "nav", "dl", "dt", "dd", "blockquote", "pre", "hr", "address"}
HEADINGS = {"summary", "profile", "professional summary", "about", "about me", "objective", "experience",
            "work experience", "professional experience", "employment", "employment history", "career history",
            "education", "skills", "technical skills", "core skills", "key skills", "technical proficiencies",
            "proficiencies", "competencies", "core competencies", "projects", "selected projects", "certifications",
            "certificates", "languages", "awards", "publications", "volunteering", "interests", "contact",
            "achievements", "leadership", "tools", "stack", "tech stack", "courses", "training", "references"}
LABELS = {"email", "e-mail", "phone", "mobile", "tel", "linkedin", "github", "website", "web", "portfolio",
          "location", "address", "twitter", "x"}
BULLET = re.compile(r"^\s*(?:[-*•·▪◦●○■□►▶➢✓✔–—]|\d{1,2}[.)])\s+")


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["style", "script", "svg", "noscript", "head"]):
        t.decompose()
    out = []

    def walk(node):
        for c in node.children:
            if isinstance(c, (Comment, Doctype)):
                continue
            if isinstance(c, NavigableString):  # HTML whitespace (source line breaks included) is one space
                out.append(re.sub(r"\s+", " ", str(c)))
            elif c.name == "br":
                out.append("\n")
            elif c.name in BLOCK:
                out.append("\n")
                if c.name == "li":
                    out.append("- ")
                walk(c)
                out.append("\n")
            else:
                walk(c)
    walk(soup)
    return "".join(out)


def _heading(line):
    s = line.strip().rstrip(":").strip()
    bare = re.sub(r"^[#\s]+|[*_]+", "", s).strip()
    if bare.lower() in HEADINGS:
        return bare.upper()
    if 2 < len(bare) <= 40 and bare.isupper() and len(bare.split()) <= 4 and not re.search(r"[|@,.\d]", bare):
        return bare
    return None


def normalize_cv(text):
    text = unicodedata.normalize("NFKC", text).replace("\r", "")
    text = re.sub(r"[ \t ]+", " ", text)
    lines = [l.strip() for l in text.split("\n")]
    out, pending_bullet = [], False
    for line in lines:
        if not line or re.fullmatch(r"(?:page\s*)?\d{1,3}(?:\s*(?:/|of)\s*\d{1,3})?", line, re.I):  # blanks, page numbers
            continue
        if line in ("-", "*", "•", "·", "–"):  # a bullet marker split from its text
            pending_bullet = True
            continue
        if pending_bullet:
            line, pending_bullet = "- " + line, False
        line = re.sub(r"^#{1,6}\s*", "", line)  # markdown heading marks
        h = _heading(line)
        if h:
            out += ["", h] if out else [h]
            continue
        line = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), line)
        is_bullet = bool(BULLET.match(line))
        if is_bullet:
            line = "- " + BULLET.sub("", line)
        prev = out[-1] if out else ""
        if prev.rstrip(":").strip().lower() in LABELS and not is_bullet:  # "Email" / "x@y.com" -> one line
            out[-1] = prev.rstrip(":").strip() + ": " + line
            continue
        # a wrapped line: previous line is text without closing punctuation, this one continues it
        if (prev and not is_bullet and not _heading(prev) and prev[-1] not in ".:;!?|" and len(prev) > 40
                and (line[0].islower() or line[0] in "(&,")):
            out[-1] = prev + " " + line
            continue
        out.append(line)
    text = "\n".join(out).strip()
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", text)
    text = re.sub(r"\+?\d[\d ()-]{8,}\d", "[phone]", text)
    return re.sub(r"\n{3,}", "\n\n", text)
