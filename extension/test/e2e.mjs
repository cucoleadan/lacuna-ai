// End-to-end smoke test: built extension in headless Chromium scores a local job page (CPU/WASM: no GPU here).
//   node build.mjs && node test/e2e.mjs
// FROM_HUB=1 downloads the published model from Hugging Face instead. Serves export/lacuna-celer/ as the model and a job page built from a validation pair (data/parity/, private).
import { cpSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { chromium } from "playwright";

const root = new URL("../../", import.meta.url).pathname;
const pair = JSON.parse(readFileSync(root + "data/parity/pairs.json", "utf8"))[0];
const [head, ...rest] = pair.job.split("\n\n");
const field = (k) => (head.match(new RegExp(`^${k}: (.*)$`, "m")) || [])[1] || "";
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const bodyHtml = rest.join("\n\n").split(/\n{2,}/).map((b) => (/^- /m.test(b)
  ? `<ul>${b.split("\n").map((l) => `<li>${esc(l.replace(/^\s*- /, ""))}</li>`).join("")}</ul>` : `<p>${esc(b)}</p>`)).join("\n");
const ld = { "@context": "https://schema.org", "@type": "JobPosting", title: field("Title"), hiringOrganization: { name: field("Company") },
  jobLocation: { address: { addressCountry: field("Location") } }, description: bodyHtml };
const page = `<!doctype html><html><head><title>${esc(field("Title"))}</title>
<script type="application/ld+json">${JSON.stringify(ld)}</script></head><body><h1>${esc(field("Title"))}</h1>${bodyHtml}</body></html>`;

const types = { ".json": "application/json", ".onnx": "application/octet-stream", ".html": "text/html" };
const server = createServer((req, res) => {
  if (req.url === "/job.html") { res.writeHead(200, { "content-type": "text/html" }); return res.end(page); }
  if (req.url.startsWith("/model/")) {
    try {
      const f = readFileSync(root + "export/lacuna-celer/" + req.url.slice(7));
      res.writeHead(200, { "content-type": types[req.url.slice(req.url.lastIndexOf("."))] || "application/octet-stream",
                           "content-length": f.length, "access-control-allow-origin": "*" });
      return res.end(f);
    } catch { /* fall through */ }
  }
  res.writeHead(404); res.end();
}).listen(0, "127.0.0.1");
await new Promise((r) => server.once("listening", r));
const base = `http://127.0.0.1:${server.address().port}`;

// test build: same extension, plus access to the local server
const dist = root + "extension/dist-test";
rmSync(dist, { recursive: true, force: true });
cpSync(root + "extension/dist", dist, { recursive: true });
const man = JSON.parse(readFileSync(dist + "/manifest.json", "utf8"));
man.host_permissions.push("http://127.0.0.1/*");
writeFileSync(dist + "/manifest.json", JSON.stringify(man));

const ctx = await chromium.launchPersistentContext("", { channel: "chromium", headless: true,
  args: [`--disable-extensions-except=${dist}`, `--load-extension=${dist}`] });
let [sw] = ctx.serviceWorkers();
sw ||= await ctx.waitForEvent("serviceworker");
const id = new URL(sw.url()).host;
const job = await ctx.newPage();
await job.goto(base + "/job.html");
const panel = await ctx.newPage();
const logs = [];
panel.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));
panel.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
await panel.goto(`chrome-extension://${id}/panel.html`);
await panel.evaluate(async ({ cv, base, hub }) => {
  await chrome.storage.local.set({ cv, settings: { tier: "celer", repos: hub ? {} : { celer: base + "/model/" }, acceptUsRemote: false, autoScore: false } });
}, { cv: pair.cv, base, hub: !!process.env.FROM_HUB });
await panel.reload();
// the panel scores the active tab; in this test the panel is itself a tab, so point it at the job tab
await panel.evaluate(async () => {
  const [t] = await chrome.tabs.query({ url: "http://127.0.0.1/*" });
  chrome.tabs.query = async () => [t];
});
const t0 = Date.now();
await panel.click("#scoreBtn");
await panel.waitForFunction(() => !document.getElementById("result").hidden || document.querySelector(".status.error"), null, { timeout: 600000 });
const out = await panel.evaluate(() => ({
  error: document.querySelector(".status.error")?.textContent,
  title: document.getElementById("jobTitle").textContent, meta: document.getElementById("jobMeta").textContent,
  fit: document.getElementById("fitPct").textContent, verdict: document.getElementById("verdict").textContent,
  chips: [...document.querySelectorAll(".chips li")].map((l) => l.textContent),
  gaps: [...document.querySelectorAll("#gaps li")].map((l) => l.textContent),
  met: document.querySelectorAll("#met li").length, timing: document.getElementById("timing").textContent,
}));
await panel.click("[data-tab=ranked]");
const ranked = await panel.locator("#rankList li").count();
console.log(JSON.stringify({ ...out, ranked, seconds: (Date.now() - t0) / 1000 }, null, 1));
if (logs.length) console.log(logs.filter((l) => !l.startsWith("debug")).slice(0, 15).join("\n"));
await ctx.close();
server.close();
rmSync(dist, { recursive: true, force: true });
process.exit(out.error ? 1 : 0);
