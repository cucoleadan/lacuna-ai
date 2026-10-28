import { encodePair, htmlToText, interpret, normalizeCv } from "./core.js";
import { TIERS, clearCache, importFolder, isCached, load, run } from "./model.js";

const $ = (id) => document.getElementById(id);
const store = {
  get: (k) => chrome.storage.local.get(k).then((r) => r[k]),
  set: (k, v) => chrome.storage.local.set({ [k]: v }),
};
const DEFAULTS = { tier: "celer", repos: {}, acceptUsRemote: false, autoScore: true };
let settings = { ...DEFAULTS };

// ---------- tabs ----------
function show(tab) {
  for (const b of document.querySelectorAll("[role=tab]")) b.setAttribute("aria-selected", b.dataset.tab === tab);
  for (const v of document.querySelectorAll(".view")) v.hidden = v.id !== tab;
  if (tab === "ranked") renderRanked();
}
document.querySelectorAll("[role=tab]").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
document.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => show(b.dataset.goto)));

function status(msg, error = false) {
  const s = $("status");
  s.hidden = !msg;
  s.textContent = msg || "";
  s.classList.toggle("error", error);
}

// ---------- scoring ----------
const repoFor = (tier) => settings.repos[tier] || TIERS[tier].repo;
let busy = false;

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab;
}

async function readPage(tab) {
  const [{ result }] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["extract.js"] });
  return result;
}

async function score() {
  if (busy) return;
  const cv = await store.get("cv");
  $("nocv").hidden = !!cv;
  if (!cv) return;
  busy = true;
  $("scoreBtn").disabled = true;
  $("permNote").hidden = true;
  try {
    const tab = await activeTab();
    // tab.url is hidden on sites Lacuna has no access to; try anyway (activeTab may cover it)
    if (!tab || (tab.url && !tab.url.startsWith("http"))) { status("Open a job posting, then score it."); return; }
    let page;
    try { page = await readPage(tab); } catch (e) {
      status("Lacuna can't read this page yet.", true);
      $("permNote").hidden = false;
      return;
    }
    if (!page || page.chars < 200) { status("No job posting found on this page."); return; }
    const tier = settings.tier, repo = repoFor(tier), file = TIERS[tier].file;
    if (!(await isCached(repo, file))) status(`Downloading the ${TIERS[tier].label} model (~${TIERS[tier].mb} MB, once)…`);
    const m = await load(repo, file, (p) => {
      if (p.phase) status("Starting the model…");
      else if (p.total) status(`Downloading the ${TIERS[tier].label} model: ${Math.round(p.got / 1e6)} / ${Math.round(p.total / 1e6)} MB`);
    });
    status("Scoring…");
    const t0 = performance.now();
    const enc = encodePair(m.tok, cv, page.text, m.cfg.max_len);
    const out = await run(m, enc.inputIds);
    const r = interpret(out, enc, page.text, m.cfg, settings);
    const ms = performance.now() - t0;
    status("");
    render(page, r, `${enc.inputIds.length.toLocaleString()} tokens, ${(ms / 1000).toFixed(1)} s on ${m.device === "webgpu" ? "GPU" : "CPU"} · ${TIERS[tier].label} model`);
    await remember(page, r, tier);
  } catch (e) {
    console.error(e);
    status(`Something went wrong: ${e.message || e}`, true);
  } finally {
    busy = false;
    $("scoreBtn").disabled = false;
  }
}

const VERDICT = { apply: "Good fit: apply", stretch: "Stretch: worth a look", skip: "Weak fit" };
const LOCATION = {
  yes: ["Location OK", "good"], unclear: ["Location unclear", "warn"], no: ["Not eligible (location)", "bad"],
  "us-remote": ["US-remote (contract)", "warn"],
};
const SENIORITY = { below: ["Asks for more seniority", "warn"], at: ["Seniority matches", "good"], above: ["Pitched below your level", "warn"] };

function chip(el, [text, cls], title) {
  el.textContent = text;
  el.className = cls;
  el.title = title || "";
}

function list(id, items) {
  const ul = $(id);
  ul.replaceChildren(...items.map((r) => Object.assign(document.createElement("li"), { textContent: r.text.replace(/^[-*•\s]+|\*\*/g, "") })));
  ul.closest(".list").hidden = !items.length;
}

function render(page, r, timing) {
  $("result").hidden = false;
  $("jobTitle").textContent = page.title || "Job posting";
  $("jobMeta").textContent = [page.company, page.location].filter(Boolean).join(" · ");
  const pct = Math.round(r.fit * 100);
  $("fitPct").textContent = `${pct}`;
  $("fitBar").style.width = `${pct}%`;
  $("verdict").textContent = VERDICT[r.verdict];
  $("verdict").className = `verdict ${r.verdict}`;
  chip($("chipLocation"), LOCATION[r.location], `P(not eligible) = ${r.pLocationNo.toFixed(2)}`);
  chip($("chipSeniority"), SENIORITY[r.seniority]);
  chip($("chipMust"), r.gaps.length ? [`Gaps: ${r.gaps.length} must-have${r.gaps.length > 1 ? "s" : ""}`, "bad"] : r.mustMet ? ["No gaps in must-haves", "good"] : ["Gaps in must-haves", "bad"]);
  list("gaps", r.gaps);
  list("partial", r.partial);
  list("met", r.met);
  $("metCount").textContent = `(${r.met.length})`;
  $("truncNote").hidden = !r.truncated;
  $("timing").textContent = timing;
}

async function remember(page, r, tier) {
  const hist = (await store.get("history")) || [];
  const row = { url: page.url.split("#")[0], title: page.title, company: page.company, location: page.location, fit: r.fit,
                verdict: r.verdict, loc: r.location, gaps: r.gaps.map((g) => g.text).slice(0, 5), tier, at: Date.now() };
  await store.set("history", [row, ...hist.filter((h) => h.url !== row.url)].slice(0, 2000));
}

async function renderRanked() {
  const hist = ((await store.get("history")) || []).sort((a, b) => b.fit - a.fit);
  $("rankEmpty").hidden = !!hist.length;
  $("rankList").replaceChildren(...hist.map((h) => {
    const li = document.createElement("li");
    const s = Object.assign(document.createElement("span"), { className: "score", textContent: Math.round(h.fit * 100) });
    const body = document.createElement("div");
    const a = Object.assign(document.createElement("a"), { href: h.url, textContent: h.title || h.url, target: "_blank" });
    const sub = Object.assign(document.createElement("div"), { className: "sub",
      textContent: [h.company, LOCATION[h.loc]?.[0], h.gaps.length ? `gaps: ${h.gaps.length}` : ""].filter(Boolean).join(" · ") });
    body.append(a, sub);
    li.append(s, body);
    return li;
  }));
}
$("clearHist").addEventListener("click", async () => { await store.set("history", []); renderRanked(); });

// ---------- settings ----------
function renderTiers() {
  $("tiers").replaceChildren(...Object.entries(TIERS).map(([k, t]) => {
    const l = document.createElement("label");
    l.className = "tier";
    const i = Object.assign(document.createElement("input"), { type: "radio", name: "tier", value: k, checked: settings.tier === k });
    i.addEventListener("change", async () => { settings.tier = k; await saveSettings(); $("repo").value = repoFor(k); });
    const d = document.createElement("div");
    d.innerHTML = `<strong></strong> <span class="muted small"></span><small></small>`;
    d.querySelector("strong").textContent = t.label;
    d.querySelector("span").textContent = `~${t.mb} MB`;
    d.querySelector("small").textContent = t.note;
    l.append(i, d);
    return l;
  }));
  $("repo").value = repoFor(settings.tier);
}

async function saveSettings() { await store.set("settings", settings); }

$("repo").addEventListener("change", async () => {
  const v = $("repo").value.trim();
  if (v && v !== TIERS[settings.tier].repo) settings.repos[settings.tier] = v; else delete settings.repos[settings.tier];
  await saveSettings();
});
$("folder").addEventListener("change", async (e) => {
  const files = [...e.target.files].filter((f) => /\.(onnx|json)$/.test(f.name));
  if (!files.some((f) => f.name === TIERS[settings.tier].file)) { status(`That folder has no ${TIERS[settings.tier].file}.`, true); show("page"); return; }
  settings.repos[settings.tier] = await importFolder(files, settings.tier);
  await saveSettings();
  renderTiers();
});
$("clearModels").addEventListener("click", async () => { await clearCache(); $("device").textContent = "Downloaded models deleted."; });
$("usRemote").addEventListener("change", async (e) => { settings.acceptUsRemote = e.target.checked; await saveSettings(); });
$("autoScore").addEventListener("change", async (e) => { settings.autoScore = e.target.checked; await saveSettings(); });

$("cvFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const raw = await f.text();
  $("cvText").value = /\.html?$/i.test(f.name) ? htmlToText(new DOMParser().parseFromString(raw, "text/html")) : raw;
});
$("saveCv").addEventListener("click", async () => {
  const cv = normalizeCv($("cvText").value);
  if (cv.length < 200) { $("cvSaved").textContent = "That looks too short for a CV."; return; }
  await store.set("cv", cv);
  $("cvText").value = cv;
  $("cvSaved").textContent = `Saved (${cv.length.toLocaleString()} characters, contact details removed).`;
  $("nocv").hidden = true;
});

$("deleteCv").addEventListener("click", async () => {
  await chrome.storage.local.remove("cv");
  $("cvText").value = "";
  $("cvSaved").textContent = "CV deleted from this browser.";
  $("nocv").hidden = false;
});

$("scoreBtn").addEventListener("click", score);
$("allowAll").addEventListener("click", async () => {
  if (await chrome.permissions.request({ origins: ["https://*/*"] })) score();
});

// re-score when the user lands on another page (only where Lacuna may read it)
async function maybeAuto() {
  if (!settings.autoScore || busy) return;
  const tab = await activeTab();
  if (tab?.url?.startsWith("http") && (await chrome.permissions.contains({ origins: [new URL(tab.url).origin + "/*"] }))) score();
}
chrome.tabs.onActivated.addListener(maybeAuto);
chrome.tabs.onUpdated.addListener((id, info, tab) => { if (info.status === "complete" && tab.active) maybeAuto(); });

// ---------- start ----------
(async () => {
  settings = { ...DEFAULTS, ...((await store.get("settings")) || {}) };
  if (!TIERS[settings.tier]) settings.tier = DEFAULTS.tier;  // settings from before the tier rename
  const cv = await store.get("cv");
  if (cv) { $("cvText").value = cv; $("cvSaved").textContent = `Saved (${cv.length.toLocaleString()} characters).`; }
  $("usRemote").checked = settings.acceptUsRemote;
  $("autoScore").checked = settings.autoScore;
  renderTiers();
  $("device").textContent = navigator.gpu ? "WebGPU available: the model runs on your GPU." : "No WebGPU: the model runs on the CPU (slower).";
  if (!cv) { $("nocv").hidden = false; return; }
  maybeAuto();
})();
