// Injected into the job page (chrome.scripting.executeScript, files): returns {url, title, company, location, workplace,
// text}, with text in the training layout: "Title: …\nCompany: …\nLocation: …\nWorkplace: …\n\n" + the posting as markdown.
// Self-contained: no imports. The last expression is the result.
(() => {
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  const q = (sel) => { for (const s of sel) { const e = document.querySelector(s); if (e && clean(e.innerText)) return e; } return null; };

  // schema.org JobPosting (most applicant tracking systems publish one)
  let ld = null;
  for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
    try {
      const j = JSON.parse(s.textContent);
      const all = [j, ...(Array.isArray(j) ? j : []), ...(j["@graph"] || [])].flat();
      ld = all.find((x) => x && [].concat(x["@type"]).includes("JobPosting")) || ld;
    } catch { /* not JSON */ }
  }
  const place = (l) => [].concat(l || []).map((x) => {
    const a = x.address || x;
    return typeof a === "string" ? a : [a.addressLocality, a.addressRegion, a.addressCountry?.name || a.addressCountry].filter(Boolean).join(", ");
  }).filter(Boolean).join("; ");

  // site markup, when there is no JSON-LD (LinkedIn) or it lacks fields
  const SITE = {
    title: [".job-details-jobs-unified-top-card__job-title", ".jobs-unified-top-card__job-title", ".top-card-layout__title", "h1"],
    company: [".job-details-jobs-unified-top-card__company-name", ".jobs-unified-top-card__company-name", ".topcard__org-name-link"],
    location: [".job-details-jobs-unified-top-card__primary-description-container .tvm__text", ".jobs-unified-top-card__bullet", ".topcard__flavor--bullet"],
    body: ["#job-details", ".jobs-description__content", ".jobs-description-content__text", ".description__text", ".job__description",
           "#content", "[class*='descriptionText']", ".posting-page .section-wrapper",
           "[data-automation-id='jobPostingDescription']", "[class*='job-description']", "[id*='job-description']", "[class*='jobDescription']"],
  };
  let body = null;
  if (ld?.description) {
    const d = new DOMParser().parseFromString(ld.description, "text/html");
    // some sites entity-encode the HTML twice
    body = /<\w/.test(d.body.textContent) && !d.body.querySelector("p,li,br,div") ? new DOMParser().parseFromString(d.body.textContent, "text/html").body : d.body;
  }
  if (!body || clean(body.textContent).length < 300) {
    body = q(SITE.body);
    if (!body || clean(body.innerText).length < 300) {  // largest plausible block
      body = [...document.querySelectorAll("main, article, section, div")].filter((e) => { const n = clean(e.innerText).length; return n > 500 && n < 40000; })
        .sort((a, b) => clean(b.innerText).length - clean(a.innerText).length)[0] || document.body;
    }
  }

  // HTML -> markdown like the training postings: ## headings, **bold**, "- " bullets, blank lines between blocks
  const BLOCK = /^(P|DIV|SECTION|ARTICLE|HEADER|FOOTER|MAIN|ASIDE|TABLE|TR|BLOCKQUOTE|PRE|DL|DT|DD|FIGURE)$/;
  const md = (node, depth = 0) => {
    let s = "";
    for (const c of node.childNodes) {
      if (c.nodeType === 3) { s += c.nodeValue.replace(/\s+/g, " "); continue; }
      if (c.nodeType !== 1) continue;
      const t = c.tagName;
      if (/^(SCRIPT|STYLE|NOSCRIPT|SVG|BUTTON|IMG|IFRAME|FORM|INPUT|SELECT)$/.test(t)) continue;
      if (/^H[1-6]$/.test(t)) s += `\n\n${"#".repeat(Math.min(+t[1], 3))} ${clean(c.textContent)}\n\n`;
      else if (t === "BR") s += "\n";
      else if (t === "LI") s += `\n${"  ".repeat(Math.max(0, depth - 1))}- ${md(c, depth).trim()}`;
      else if (t === "UL" || t === "OL") s += `\n${md(c, depth + 1)}\n`;
      else if (t === "STRONG" || t === "B") { const x = clean(c.textContent); s += x ? ` **${x}** ` : ""; }
      else if (BLOCK.test(t)) s += `\n\n${md(c, depth)}\n\n`;
      else s += md(c, depth);
    }
    return s;
  };
  const text = md(body).split("\n").map((l) => l.replace(/[ \t]+/g, " ").replace(/^ (?=\S)/, "").replace(/\s+$/, "")
    .replace(/\*\* +/g, "** ").replace(/^(\s*- ) +/, "$1")).join("\n").replace(/\n{3,}/g, "\n\n").trim();

  const title = clean(ld?.title) || clean(q(SITE.title)?.innerText) || clean(document.title);
  const company = clean(ld?.hiringOrganization?.name || (typeof ld?.hiringOrganization === "string" ? ld.hiringOrganization : "")) || clean(q(SITE.company)?.innerText);
  const remoteReq = place(ld?.applicantLocationRequirements);
  const location = [place(ld?.jobLocation), remoteReq].filter(Boolean).join("; ") || clean(q(SITE.location)?.innerText);
  const workplace = [].concat(ld?.jobLocationType || []).includes("TELECOMMUTE") ? "Remote" : "";
  const header = [["Title", title], ["Company", company], ["Location", location], ["Workplace", workplace]]
    .filter(([, v]) => v).map(([k, v]) => `${k}: ${v}`).join("\n");
  return { url: document.location.href, title, company, location, workplace,
           text: header + "\n\n" + text, chars: text.length };
})();
