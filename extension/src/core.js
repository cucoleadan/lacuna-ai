// Lacuna core: the same text rules and encoding as training (scripts/cvtext.py, scripts/build_dataset.py, train/train.py),
// plus turning model outputs into what the panel shows. No browser APIs here, so test/parity.mjs runs it in Node.

const BLOCK = new Set(["p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "section", "article",
  "header", "footer", "aside", "main", "nav", "dl", "dt", "dd", "blockquote", "pre", "hr", "address"]);
const HEADINGS = new Set(["summary", "profile", "professional summary", "about", "about me", "objective", "experience",
  "work experience", "professional experience", "employment", "employment history", "career history",
  "education", "skills", "technical skills", "core skills", "key skills", "technical proficiencies",
  "proficiencies", "competencies", "core competencies", "projects", "selected projects", "certifications",
  "certificates", "languages", "awards", "publications", "volunteering", "interests", "contact",
  "achievements", "leadership", "tools", "stack", "tech stack", "courses", "training", "references"]);
const LABELS = new Set(["email", "e-mail", "phone", "mobile", "tel", "linkedin", "github", "website", "web", "portfolio",
  "location", "address", "twitter", "x"]);
const BULLET = /^\s*(?:[-*•·▪◦●○■□►▶➢✓✔–—]|\p{Nd}{1,2}[.)])\s+/u;

// Python str.isupper / str.islower: all cased characters in that case, and at least one cased character
const isUpper = (s) => s === s.toUpperCase() && s !== s.toLowerCase();
const isLower = (s) => s === s.toLowerCase() && s !== s.toUpperCase();

/** Block-aware HTML -> text (cvtext.html_to_text). `doc` is a parsed Document (DOMParser in the browser). */
export function htmlToText(doc) {
  const out = [];
  const skip = new Set(["style", "script", "svg", "noscript", "head"]);
  const walk = (node) => {
    for (const c of node.childNodes) {
      if (c.nodeType === 3) out.push(c.nodeValue.replace(/\s+/g, " "));
      else if (c.nodeType !== 1) continue;
      else {
        const name = c.nodeName.toLowerCase();
        if (skip.has(name)) continue;
        if (name === "br") out.push("\n");
        else if (BLOCK.has(name)) {
          out.push("\n");
          if (name === "li") out.push("- ");
          walk(c);
          out.push("\n");
        } else walk(c);
      }
    }
  };
  walk(doc.body || doc);
  return out.join("");
}

function heading(line) {
  const s = line.trim().replace(/:+$/, "").trim();
  const bare = s.replace(/^[#\s]+|[*_]+/g, "").trim();
  if (HEADINGS.has(bare.toLowerCase())) return bare.toUpperCase();
  if (bare.length > 2 && bare.length <= 40 && isUpper(bare) && bare.split(/\s+/).length <= 4 && !/[|@,.\p{Nd}]/u.test(bare)) return bare;
  return null;
}

/** Any CV text -> the one layout the model was trained on (cvtext.normalize_cv). */
export function normalizeCv(text) {
  text = text.normalize("NFKC").replace(/\r/g, "").replace(/[ \t ]+/g, " ");
  const out = [];
  let pendingBullet = false;
  for (let line of text.split("\n").map((l) => l.trim())) {
    if (!line || /^(?:page\s*)?\p{Nd}{1,3}(?:\s*(?:\/|of)\s*\p{Nd}{1,3})?$/iu.test(line)) continue;
    if (["-", "*", "•", "·", "–"].includes(line)) { pendingBullet = true; continue; }
    if (pendingBullet) { line = "- " + line; pendingBullet = false; }
    line = line.replace(/^#{1,6}\s*/, "");
    const h = heading(line);
    if (h) { if (out.length) out.push("", h); else out.push(h); continue; }
    line = line.replace(/\*\*(.+?)\*\*|__(.+?)__/g, (_, a, b) => a ?? b);
    const isBullet = BULLET.test(line);
    if (isBullet) line = "- " + line.replace(BULLET, "");
    const prev = out.length ? out[out.length - 1] : "";
    const prevLabel = prev.replace(/:+$/, "").trim();
    if (LABELS.has(prevLabel.toLowerCase()) && !isBullet) { out[out.length - 1] = prevLabel + ": " + line; continue; }
    if (prev && !isBullet && !heading(prev) && !".:;!?|".includes(prev[prev.length - 1]) && prev.length > 40
        && (isLower(line[0]) || "(&,".includes(line[0]))) { out[out.length - 1] = prev + " " + line; continue; }
    out.push(line);
  }
  return out.join("\n").trim()
    .replace(/[\p{L}\p{N}_.+-]+@[\p{L}\p{N}_-]+\.[\p{L}\p{N}_.-]+/gu, "[email]")
    .replace(/\+?\p{Nd}[\p{Nd} ()-]{8,}\p{Nd}/gu, "[phone]")
    .replace(/\n{3,}/g, "\n\n");
}

/** Character spans of the job's lines that can hold a requirement (build_dataset.job_lines). */
export function jobLines(job) {
  const spans = [];
  let pos = 0;
  for (const line of job.split("\n")) {
    const start = pos, end = pos + line.length;
    pos = end + 1;
    if ([...line.replace(/^[ \-*•\t]+|[ \-*•\t]+$/g, "")].length >= 15) spans.push([start, end]);  // code points, as Python
  }
  return spans;
}

// The byte-level pre-tokenizer's split (GPT-2 pattern). BPE never merges across these pieces, so encoding a run of
// whole pieces gives the same ids as encoding the full text, and each job line's tokens are known exactly.
const PRETOKEN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

const addedCache = new WeakMap();
function addedTokenPattern(tok) {  // longest first, like the tokenizer's splitter
  if (!addedCache.has(tok)) {
    const words = tok.added_tokens.map((t) => t.content).filter((c) => c && !tok.special_tokens.includes(c))
      .sort((a, b) => b.length - a.length).map((c) => c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    addedCache.set(tok, words.length ? new RegExp(words.join("|"), "g") : null);
  }
  return addedCache.get(tok);
}

/** [CLS] cv [SEP] job [SEP], truncated longest-first to maxLen, plus each job line's token range. */
export function encodePair(tok, cv, job, maxLen) {
  const ids = (t) => (t ? tok.encode(t, { add_special_tokens: false }).ids : []);
  const cvIds = ids(cv);
  const lines = jobLines(job);
  // the tokenizer first cuts out added tokens (this vocabulary has some for runs of spaces), then applies PRETOKEN
  // to the text between them
  const pieces = [];
  const added = addedTokenPattern(tok);
  let at = 0;
  const pre = (s, e) => {
    for (const m of job.slice(s, e).matchAll(PRETOKEN)) pieces.push({ s: s + m.index, e: s + m.index + m[0].length, ws: !/\S/.test(m[0]) });
  };
  for (const m of added ? job.matchAll(added) : []) {
    pre(at, m.index);
    pieces.push({ s: m.index, e: m.index + m[0].length, ws: !/\S/.test(m[0]) });
    at = m.index + m[0].length;
  }
  pre(at, job.length);
  let li = 0;
  for (const p of pieces) {  // a piece inside a line's span belongs to it (indentation too); "\n " across lines does not
    while (li < lines.length && lines[li][1] <= p.s) li++;
    p.line = li < lines.length && p.s >= lines[li][0] && p.e <= lines[li][1] ? li : -1;
  }
  const jobIds = [], lineTok = [];
  for (let i = 0; i < pieces.length;) {
    // whitespace pieces alone: "\n\n" before text is two pieces ("\n", "\n"), but encoded as one string it is one
    let j = i + 1;
    while (!pieces[i].ws && j < pieces.length && !pieces[j].ws && pieces[j].line === pieces[i].line) j++;
    const start = jobIds.length;
    jobIds.push(...ids(job.slice(pieces[i].s, pieces[j - 1].e)));
    const last = lineTok[lineTok.length - 1];
    if (pieces[i].line >= 0 && last?.line === pieces[i].line) last.b = jobIds.length;
    else if (pieces[i].line >= 0) lineTok.push({ line: pieces[i].line, a: start, b: jobIds.length });
    i = j;
  }
  const cls = tok.token_to_id("[CLS]"), sep = tok.token_to_id("[SEP]");
  let nc = cvIds.length, nj = jobIds.length;
  while (nc + nj + 3 > maxLen) { if (nc > nj) nc--; else nj--; }  // HF "longest_first": trim the longer one
  const off = nc + 2;
  const inputIds = [cls, ...cvIds.slice(0, nc), sep, ...jobIds.slice(0, nj), sep];
  const spans = lineTok.filter((l) => l.a < nj && l.b > l.a)
    .map((l) => ({ line: l.line, span: lines[l.line], a: off + l.a, b: off + Math.min(l.b, nj) }));
  return { inputIds, lines: spans, truncated: nc < cvIds.length || nj < jobIds.length };
}

const softmax = (xs, t = 1) => {
  const m = Math.max(...xs), e = xs.map((x) => Math.exp((x - m) / t)), s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
};
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Shown as a separate, softer state: a US-only remote posting is "no" for most non-US readers (the training rule),
// but some apply anyway as contractors.
const US_REMOTE = /\b(?:remote\s*[-–(,]?\s*(?:us|usa|u\.s\.|united states)|(?:us|usa|u\.s\.|united states)\s*[-–(,]?\s*remote)\b/i;

/**
 * Model outputs (plain arrays) -> what the panel shows.
 * out: {verdict, location, seniority, must_met: number[]; line_met: number[seq][4]; line_must: number[seq]}
 */
export function interpret(out, enc, job, cfg, settings = {}) {
  const T = cfg.temperatures, L = cfg.labels;
  const p = {};
  for (const h of ["verdict", "location", "seniority", "must_met"]) p[h] = softmax(out[h], T[h]);
  const fit = p.verdict[0] + p.verdict[1];
  let verdict = L.verdict[p.verdict.indexOf(Math.max(...p.verdict))];
  let location = L.location[p.location.indexOf(Math.max(...p.location))];
  const pNo = p.location[2];
  const usRemote = US_REMOTE.test(job.slice(0, 600));
  // hide only when the model is confident; otherwise flag it and cap the verdict
  let locationState = pNo >= 0.8 ? "no" : location === "yes" ? "yes" : "unclear";
  if (locationState === "no" && usRemote && settings.acceptUsRemote) locationState = "us-remote";
  if (locationState !== "yes" && verdict === "apply") verdict = "stretch";
  if (locationState === "no") verdict = "skip";

  const reqs = enc.lines.map((l) => {
    const n = l.b - l.a;
    const met = [0, 0, 0, 0];
    let must = 0;
    for (let i = l.a; i < l.b; i++) { for (let k = 0; k < 4; k++) met[k] += out.line_met[i][k] / n; must += out.line_must[i] / n; }
    const pm = softmax(met);
    return { text: job.slice(l.span[0], l.span[1]).trim(), met: L.line_met[pm.indexOf(Math.max(...pm))], pMet: pm, pMust: sigmoid(must) };
  }).filter((r) => r.met !== "none");
  return {
    fit, verdict, location: locationState, pLocationNo: pNo, usRemote,
    seniority: L.seniority[p.seniority.indexOf(Math.max(...p.seniority))], pSeniority: p.seniority,
    mustMet: p.must_met[1] >= 0.5, pMustMet: p.must_met[1],
    gaps: reqs.filter((r) => r.met === "no" && r.pMust >= 0.5),
    partial: reqs.filter((r) => r.met === "partial" && r.pMust >= 0.5),
    met: reqs.filter((r) => r.met === "yes"),
    nice: reqs.filter((r) => r.pMust < 0.5 && r.met !== "yes"),
    truncated: enc.truncated, probs: p,
  };
}

/** One key per job, so a posting is scored once: drops the #fragment, tracking parameters and an ATS's apply step
 *  (Ashby /application, Lever and Workable /apply, Workday /apply/applyManually), and maps LinkedIn's many job URLs
 *  (search, collections, /jobs/view/title-123) to /jobs/view/<id>/. The key is still a link to the posting. */
export function jobKey(href) {
  const u = new URL(href);
  u.hash = "";
  for (const k of [...u.searchParams.keys()]) {
    if (/^(utm_.*|ref|refid|trk|trackingid|source|src|gh_src|lever-source|lever-origin)$/i.test(k)) u.searchParams.delete(k);
  }
  const li = u.hostname.endsWith("linkedin.com") && (u.searchParams.get("currentJobId") || u.pathname.match(/\/jobs\/view\/(?:[^/]*-)?(\d+)/)?.[1]);
  if (li) return `https://www.linkedin.com/jobs/view/${li}/`;
  u.pathname = u.pathname.replace(/\/(apply|application)(\/[^/]*)?\/?$/i, "").replace(/\/+$/, "") || "/";
  return u.href;
}
