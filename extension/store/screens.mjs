// Chrome Web Store screenshots: the built extension scores fictional postings (store/demo/) with a fictional CV,
// in headless Chromium, and saves raw captures of the panel and the job page to ../export/store-raw/.
//   node build.mjs && node store/screens.mjs
import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { chromium } from "playwright";

const root = new URL("../../", import.meta.url).pathname;
const demo = root + "extension/store/demo/";
const out = root + "export/store-raw/";
mkdirSync(out, { recursive: true });
const cv = readFileSync(demo + "cv.txt", "utf8");
const jobs = JSON.parse(readFileSync(demo + "jobs.json", "utf8"));
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

function md(body) {  // the tiny subset the demo postings use: "## " headings, "- " bullets, paragraphs
  const out = [];
  let list = [];
  const flush = () => { if (list.length) out.push(`<ul>${list.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`); list = []; };
  for (const line of body.split("\n")) {
    if (line.startsWith("- ")) { list.push(line.slice(2)); continue; }
    flush();
    if (line.startsWith("## ")) out.push(`<h2>${esc(line.slice(3))}</h2>`);
    else if (line.trim()) out.push(`<p>${esc(line)}</p>`);
  }
  flush();
  return out.join("\n");
}
function page(j) {
  const html = md(j.body);
  const ld = { "@context": "https://schema.org", "@type": "JobPosting", title: j.title, hiringOrganization: { name: j.company },
    jobLocation: { address: { addressLocality: j.location } }, description: html,
    ...(j.workplace === "Remote" ? { jobLocationType: "TELECOMMUTE" } : {}) };
  return `<!doctype html><html><head><meta charset="utf-8"><title>${esc(j.title)} · ${esc(j.company)}</title>
<script type="application/ld+json">${JSON.stringify(ld)}</script>
<style>
body{margin:0;font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:#1d2330;background:#f4f5f8}
.demo{background:#fff4d6;color:#7a5a00;text-align:center;font-size:12px;padding:6px}
.wrap{max-width:720px;margin:28px auto;background:#fff;border-radius:12px;padding:32px 40px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.logo{width:44px;height:44px;border-radius:10px;background:#3b4a6b;color:#fff;display:grid;place-items:center;font-weight:700;font-size:20px}
.head{display:flex;gap:14px;align-items:center;margin-bottom:6px}.co{color:#5b6477}
h1{font-size:24px;margin:14px 0 4px}.meta{color:#5b6477;margin-bottom:18px}
.apply{display:inline-block;background:#2f6fde;color:#fff;border-radius:8px;padding:9px 18px;font-weight:600;margin-bottom:10px}
h2{font-size:16px;margin:22px 0 6px}ul{padding-left:20px;margin:6px 0}li{margin:3px 0}
</style></head><body><div class="demo">Example posting · fictional company</div><div class="wrap">
<div class="head"><div class="logo">${esc(j.company[0])}</div><div class="co">${esc(j.company)} · Careers</div></div>
<h1>${esc(j.title)}</h1><div class="meta">${esc(j.location)} · ${esc(j.workplace)} · Full-time</div>
<span class="apply">Apply for this job</span>${html}</div></body></html>`;
}

const server = createServer((req, res) => {
  const m = req.url.match(/^\/job\/([\w-]+)\.html$/);
  const j = m && jobs.find((x) => x.slug === m[1]);
  if (j) { res.writeHead(200, { "content-type": "text/html; charset=utf-8" }); return res.end(page(j)); }
  if (req.url.startsWith("/model/")) {
    try {
      const f = readFileSync(root + "export/lacuna-celer/" + req.url.slice(7));
      res.writeHead(200, { "content-length": f.length, "access-control-allow-origin": "*" });
      return res.end(f);
    } catch { /* 404 below */ }
  }
  res.writeHead(404); res.end();
}).listen(0, "127.0.0.1");
await new Promise((r) => server.once("listening", r));
const base = `http://127.0.0.1:${server.address().port}`;

const dist = root + "extension/dist-test";
rmSync(dist, { recursive: true, force: true });
cpSync(root + "extension/dist", dist, { recursive: true });
const man = JSON.parse(readFileSync(dist + "/manifest.json", "utf8"));
man.host_permissions.push("http://127.0.0.1/*");
writeFileSync(dist + "/manifest.json", JSON.stringify(man));

const ctx = await chromium.launchPersistentContext("", { channel: "chromium", headless: true, deviceScaleFactor: 2,
  args: [`--disable-extensions-except=${dist}`, `--load-extension=${dist}`] });
const sw = ctx.serviceWorkers()[0] || (await ctx.waitForEvent("serviceworker"));
const id = new URL(sw.url()).host;
const panel = await ctx.newPage();
await panel.setViewportSize({ width: 400, height: 860 });
await panel.goto(`chrome-extension://${id}/panel.html`);
await panel.evaluate(async ({ cv, base }) => {
  await chrome.storage.local.set({ cv, history: [], settings: { tier: "celer", repos: { celer: base + "/model/" }, acceptUsRemote: true, autoScore: false } });
}, { cv, base });
await panel.reload();

const results = {};
for (const j of jobs) {
  const job = await ctx.newPage();
  await job.setViewportSize({ width: 900, height: 860 });
  await job.goto(`${base}/job/${j.slug}.html`);
  await job.screenshot({ path: `${out}job-${j.slug}.png` });
  await panel.evaluate(async (url) => {  // the panel scores the "active tab": point it at this job's tab
    globalThis.realQuery ||= chrome.tabs.query.bind(chrome.tabs);
    const [t] = await realQuery({ url });
    chrome.tabs.query = async () => [t];
  }, `${base}/job/${j.slug}.html`);
  await panel.click("[data-tab=page]");
  await panel.evaluate(() => { document.getElementById("result").hidden = true; });
  await panel.click("#scoreBtn");
  await panel.waitForFunction(() => !document.getElementById("result").hidden || document.querySelector(".status.error"), null, { timeout: 600000 });
  await panel.evaluate(() => { document.getElementById("timing").textContent = ""; const d = document.getElementById("metBox"); if (d) d.open = false; });
  results[j.slug] = await panel.evaluate(() => ({ fit: document.getElementById("fitPct").textContent, verdict: document.getElementById("verdict").textContent,
    chips: [...document.querySelectorAll(".chips li")].map((l) => l.textContent), gaps: [...document.querySelectorAll("#gaps li")].map((l) => l.textContent) }));
  await panel.screenshot({ path: `${out}panel-${j.slug}.png` });
  await job.close();
  console.log(j.slug, JSON.stringify(results[j.slug]));
}
await panel.click("[data-tab=ranked]");
await panel.waitForTimeout(300);
await panel.screenshot({ path: `${out}panel-ranked.png` });
await panel.click("[data-tab=settings]");
await panel.evaluate(() => { document.getElementById("repo").value = "lacuna-ai/lacuna-celer"; });
await panel.waitForTimeout(300);
await panel.screenshot({ path: `${out}panel-settings.png` });
writeFileSync(out + "results.json", JSON.stringify(results, null, 1));
await ctx.close();
server.close();
rmSync(dist, { recursive: true, force: true });
