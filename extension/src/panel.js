import { encodePair, htmlToText, interpret, normalizeCv } from "./core.js";
import { TIERS, clearCache, importFolder, isCached, load, run } from "./model.js";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const store = {
  get: (k) => chrome.storage.local.get(k).then((r) => r[k]),
  set: (k, v) => chrome.storage.local.set({ [k]: v }),
};
const DEFAULTS = { tier: "celer", repos: {}, acceptUsRemote: false, autoScore: true };
let settings = { ...DEFAULTS };
const ICON = {
  check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>',
  x: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>',
  minus: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 12h12"/></svg>',
};
const mb = (bytes) => Math.round(bytes / 1e6);
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

// ---------- views ----------
let current = "page", back = "page";
function show(view) {
  if (view === "settings" && ["page", "ranked"].includes(current)) back = current;
  current = view;
  for (const b of document.querySelectorAll("[role=tab]")) b.setAttribute("aria-selected", b.dataset.tab === view);
  for (const v of document.querySelectorAll(".view")) v.hidden = v.id !== view;
  $("nav").hidden = !["page", "ranked"].includes(view);
  if (view === "ranked") renderRanked();
  if (view === "settings") renderSettings();
  if (view === "setupModel") renderTiers();
  scrollTo(0, 0);
}
document.querySelectorAll("[data-tab],[data-goto]").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab || b.dataset.goto)));
$("settingsBack").addEventListener("click", () => show(back));

function status(msg, error = false) {
  const s = $("status");
  s.hidden = !msg;
  s.textContent = msg || "";
  s.classList.toggle("error", error);
}

// which block of "This job" is showing
function phase(p) {
  $("ready").hidden = p !== "ready";
  $("progress").hidden = p !== "progress";
  $("result").hidden = p !== "result";
}

// ---------- scoring ----------
const repoFor = (tier) => settings.repos[tier] || TIERS[tier].repo;
let busy = false, scored = false;

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab;
}

async function readPage(tab) {
  const [{ result }] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["extract.js"] });
  return result;
}

const STEPS = ["Read the posting", "Load the model", "Score it"];
const stepNote = [];
function progress(i, note = "", pct = [12, 40, 85][i]) {
  stepNote[i] = note;
  $("steps").replaceChildren(...STEPS.map((label, k) => el("li", { className: k < i ? "done" : k === i ? "active" : "" },
    el("span", { className: "mark", innerHTML: k < i ? ICON.check : "" }),
    el("span", { className: "grow", textContent: label }),
    el("span", { className: "mono faint", textContent: k <= i ? stepNote[k] || "" : "" }))));
  $("progBar").style.width = `${pct}%`;
}

const initial = (page) => (page.company || page.title || "?").trim().charAt(0).toUpperCase();
const meta = (page) => [page.company, page.location].filter(Boolean).join(" · ");

async function score() {
  if (busy) return;
  const cv = await store.get("cv");
  if (!cv) return show("welcome");
  busy = true;
  $("scoreBtn").disabled = $("rescoreBtn").disabled = true;
  $("permNote").hidden = true;
  status("");
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
    $("progInitial").textContent = initial(page);
    $("progTitle").textContent = page.title || "Job posting";
    $("progMeta").textContent = meta(page);
    phase("progress");
    stepNote.length = 0;
    stepNote[0] = `${page.text.split(/\s+/).length.toLocaleString()} words`;
    progress(1);
    const tier = settings.tier, repo = repoFor(tier), file = TIERS[tier].file;
    const m = await load(repo, file, (p) => {
      if (p.phase) progress(1, "starting", 75);
      else if (p.total) progress(1, `${mb(p.got)} / ${mb(p.total)} MB`, 12 + (60 * p.got) / p.total);
    });
    progress(2);
    const t0 = performance.now();
    const enc = encodePair(m.tok, cv, page.text, m.cfg.max_len);
    const out = await run(m, enc.inputIds);
    const r = interpret(out, enc, page.text, m.cfg, settings);
    const s = (performance.now() - t0) / 1000;
    render(page, r, `${enc.inputIds.length.toLocaleString()} tokens · ${s.toFixed(1)} s · ${m.device === "webgpu" ? "GPU" : "CPU"} · ${TIERS[tier].label}`);
    await remember(page, r, tier);
  } catch (e) {
    console.error(e);
    status(`Something went wrong: ${e.message || e}`, true);
  } finally {
    busy = false;
    $("scoreBtn").disabled = $("rescoreBtn").disabled = false;
    if (!$("progress").hidden) phase(scored ? "result" : "ready");
  }
}

const VERDICT = { apply: ["Apply", "Good fit", "check"], stretch: ["Stretch", "Worth a look", "minus"], skip: ["Skip", "Weak fit", "x"] };
const LOCATION = {
  yes: ["Eligible", "good"], unclear: ["Unclear", "warn"], no: ["Not eligible", "bad"], "us-remote": ["US-remote", "warn"],
};
const SENIORITY = { below: ["More senior", "warn"], at: ["Matches", "good"], above: ["Below you", "warn"] };
const SENIORITY_TITLE = { below: "Asks for more seniority than your CV shows", above: "Pitched below your level" };

function chip(id, [text, cls], title) {
  const li = $(id);
  li.className = cls;
  li.title = title || "";
  li.querySelector("strong").textContent = text;
}

function list(id, items, icon) {
  const ul = $(id);
  ul.replaceChildren(...items.map((r) => el("li", {},
    ...(icon ? [el("span", { className: "ico", innerHTML: ICON[icon] })] : []),
    el("span", { textContent: r.text.replace(/^[-*•\s]+|\*\*/g, "") }))));
  ul.closest(".list").hidden = !items.length;
}

function render(page, r, timing) {
  scored = true;
  phase("result");
  $("jobInitial").textContent = initial(page);
  $("jobTitle").textContent = page.title || "Job posting";
  $("jobMeta").textContent = meta(page);
  $("fitPct").textContent = `${Math.round(r.fit * 100)}`;
  $("gauge").style.setProperty("--p", r.fit.toFixed(3));
  const [label, head, icon] = VERDICT[r.verdict];
  $("verdict").className = `pill ${r.verdict}`;
  $("verdict").replaceChildren(el("span", { className: "ico", innerHTML: ICON[icon] }), label);
  $("verdictHead").textContent = head;
  const covered = r.met.filter((x) => x.pMust >= 0.5).length, must = covered + r.gaps.length + r.partial.length;
  $("verdictSub").textContent = !must ? "No clear must-haves in this posting."
    : `Your CV covers ${covered} of ${plural(must, "must-have")}${r.partial.length ? `, ${r.partial.length} partly` : ""}.`;
  chip("chipLocation", LOCATION[r.location], `P(not eligible) = ${r.pLocationNo.toFixed(2)}`);
  chip("chipSeniority", SENIORITY[r.seniority], SENIORITY_TITLE[r.seniority]);
  chip("chipMust", r.gaps.length ? [plural(r.gaps.length, "gap"), "bad"] : r.mustMet ? ["All covered", "good"] : ["Some missing", "bad"]);
  list("gaps", r.gaps, "x");
  list("partial", r.partial, "minus");
  list("met", r.met);
  $("metCount").textContent = r.met.length;
  $("truncNote").hidden = !r.truncated;
  $("timing").textContent = timing;
}

async function remember(page, r, tier) {
  const hist = (await store.get("history")) || [];
  const row = { url: page.url.split("#")[0], title: page.title, company: page.company, location: page.location, fit: r.fit,
                verdict: r.verdict, loc: r.location, sen: r.seniority, nGaps: r.gaps.length, gaps: r.gaps.map((g) => g.text).slice(0, 5), tier, at: Date.now() };
  const next = [row, ...hist.filter((h) => h.url !== row.url)].slice(0, 2000);
  await store.set("history", next);
  $("savedCount").textContent = next.length;
}

// ---------- saved jobs ----------
let filter = "all";
const FILTERS = [["all", "All"], ["apply", "Apply"], ["stretch", "Stretch"], ["skip", "Skip"]];
const LOC_NOTE = { no: "Not eligible", "us-remote": "US-remote", unclear: "Location unclear" };

function note(h) {
  const n = h.nGaps ?? h.gaps.length;
  return [LOC_NOTE[h.loc], h.sen === "below" ? "More senior" : "", n ? plural(n, "gap") : "No gaps"].filter(Boolean).join(" · ");
}

async function renderRanked() {
  const hist = ((await store.get("history")) || []).sort((a, b) => b.fit - a.fit);
  $("savedCount").textContent = hist.length;
  $("rankEmpty").hidden = !!hist.length;
  $("filters").hidden = !hist.length;
  $("clearHist").hidden = !hist.length;
  $("filters").replaceChildren(...FILTERS.map(([id, label]) => {
    const n = id === "all" ? hist.length : hist.filter((h) => h.verdict === id).length;
    const b = el("button", { type: "button" }, label, el("span", { className: "count", textContent: n }));
    b.setAttribute("aria-pressed", filter === id);
    b.addEventListener("click", () => { filter = id; renderRanked(); });
    return b;
  }));
  $("rankList").replaceChildren(...hist.filter((h) => filter === "all" || h.verdict === filter).map((h) => el("li", {},
    el("a", { href: h.url, target: "_blank" },
      el("span", { className: `score v-${h.verdict}`, textContent: Math.round(h.fit * 100) }),
      el("span", { className: "body" },
        el("span", { className: "t", textContent: h.title || h.url }),
        el("span", { className: "sub", textContent: [h.company, h.location].filter(Boolean).join(" · ") }),
        el("span", { className: "tags2" },
          el("span", { className: `tag v-${h.verdict}`, textContent: VERDICT[h.verdict]?.[0] || h.verdict }),
          note(h)))))));
}
$("clearHist").addEventListener("click", async () => { await store.set("history", []); renderRanked(); });

// ---------- model choice (setup and settings) ----------
async function renderTiers() {
  const cached = {};
  for (const [k, t] of Object.entries(TIERS)) cached[k] = await isCached(repoFor(k), t.file);
  for (const box of [$("tiers"), $("tiersSetup")]) {
    box.replaceChildren(...Object.entries(TIERS).map(([k, t]) => {
      const input = el("input", { type: "radio", name: box.id, value: k, checked: settings.tier === k });
      input.addEventListener("change", async () => { settings.tier = k; await saveSettings(); renderTiers(); });
      const top = el("span", { className: "tierTop" }, el("strong", { textContent: t.label }));
      if (k === DEFAULTS.tier) top.append(el("span", { className: "badge", textContent: "Recommended" }));
      top.append(el("span", { className: "mono", textContent: cached[k] ? "downloaded" : `${t.mb} MB` }));
      return el("label", { className: "tier" }, input, el("span", { className: "tierBody" }, top, el("small", { textContent: t.note })));
    }));
  }
  $("repo").value = repoFor(settings.tier);
  const t = TIERS[settings.tier];
  $("downloadLabel").textContent = cached[settings.tier] ? "Continue" : `Download ${t.label} · ${t.mb} MB`;
}

$("downloadBtn").addEventListener("click", async () => {
  const t = TIERS[settings.tier];
  $("downloadBtn").disabled = true;
  $("dlBox").hidden = false;
  $("dlLabel").textContent = `Downloading ${t.label}…`;
  $("dlText").textContent = "";
  try {
    await load(repoFor(settings.tier), t.file, (p) => {
      if (p.phase) { $("dlLabel").textContent = "Starting the model…"; $("dlBar").style.width = "100%"; }
      else if (p.total) { $("dlText").textContent = `${mb(p.got)} / ${mb(p.total)} MB`; $("dlBar").style.width = `${(100 * p.got) / p.total}%`; }
    });
    $("dlBox").hidden = true;
    show("page");
    phase("ready");
  } catch (e) {
    console.error(e);
    $("dlLabel").textContent = `Download failed: ${e.message || e}`;
  } finally {
    $("downloadBtn").disabled = false;
  }
});

// ---------- CV ----------
let cvEdit = false;  // opened from Settings ("Replace") rather than the first-run setup
function openCv(edit) {
  cvEdit = edit;
  $("setupCv").classList.toggle("edit", edit);
  $("saveCv").textContent = edit ? "Save CV" : "Save and continue";
  $("cvSaved").textContent = "";
  show("setupCv");
}
$("editCv").addEventListener("click", () => openCv(true));
document.querySelector("#welcome [data-goto]").addEventListener("click", () => openCv(false));
$("cvBack").addEventListener("click", () => show(cvEdit ? "settings" : "welcome"));

async function loadCvFile(f) {
  const raw = await f.text();
  $("cvText").value = /\.html?$/i.test(f.name) ? htmlToText(new DOMParser().parseFromString(raw, "text/html")) : raw;
  $("dropTitle").textContent = f.name;
}
$("cvFile").addEventListener("change", (e) => e.target.files[0] && loadCvFile(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files[0]) loadCvFile(e.dataTransfer.files[0]);
});

$("saveCv").addEventListener("click", async () => {
  const cv = normalizeCv($("cvText").value);
  $("cvSaved").classList.toggle("error", cv.length < 200);
  if (cv.length < 200) { $("cvSaved").textContent = "That looks too short for a CV."; return; }
  await store.set("cv", cv);
  $("cvText").value = cv;
  if (cvEdit) show("settings");
  else show("setupModel");
});

$("deleteCv").addEventListener("click", async () => {
  await chrome.storage.local.remove("cv");
  $("cvText").value = "";
  $("dropTitle").textContent = "Drop your CV here";
  show("welcome");
});

// ---------- settings ----------
async function saveSettings() { await store.set("settings", settings); }

async function renderSettings() {
  const cv = await store.get("cv");
  $("cvStatus").textContent = cv ? "CV saved" : "No CV saved";
  $("cvInfo").textContent = cv ? `${cv.length.toLocaleString()} characters · in this browser only` : "Add one to start scoring.";
  $("editCv").textContent = cv ? "Replace" : "Add CV";
  $("deleteCv").disabled = !cv;
  const all = await chrome.permissions.contains({ origins: ["https://*/*"] });
  $("allowAll").disabled = all;
  $("allowAll").textContent = all ? "Allowed" : "Allow all";
  renderTiers();
}

$("repo").addEventListener("change", async () => {
  const v = $("repo").value.trim();
  if (v && v !== TIERS[settings.tier].repo) settings.repos[settings.tier] = v; else delete settings.repos[settings.tier];
  await saveSettings();
  renderTiers();
});
$("folder").addEventListener("change", async (e) => {
  const files = [...e.target.files].filter((f) => /\.(onnx|json)$/.test(f.name));
  if (!files.some((f) => f.name === TIERS[settings.tier].file)) { $("modelNote").textContent = `That folder has no ${TIERS[settings.tier].file}.`; return; }
  settings.repos[settings.tier] = await importFolder(files, settings.tier);
  await saveSettings();
  $("modelNote").textContent = "Model loaded from the folder.";
  renderTiers();
});
$("clearModels").addEventListener("click", async () => { await clearCache(); $("modelNote").textContent = "Downloaded models deleted."; renderTiers(); });
$("usRemote").addEventListener("change", async (e) => { settings.acceptUsRemote = e.target.checked; await saveSettings(); });
$("autoScore").addEventListener("change", async (e) => { settings.autoScore = e.target.checked; await saveSettings(); });

$("scoreBtn").addEventListener("click", score);
$("rescoreBtn").addEventListener("click", score);
document.querySelectorAll("[data-allow]").forEach((b) => b.addEventListener("click", async () => {
  if (!(await chrome.permissions.request({ origins: ["https://*/*"] }))) return;
  if (current === "settings") renderSettings(); else score();
}));

// re-score when the user lands on another page (only where Lacuna may read it)
async function maybeAuto() {
  if (!settings.autoScore || busy || current !== "page" || !(await store.get("cv"))) return;
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
  if (cv) $("cvText").value = cv;
  $("usRemote").checked = settings.acceptUsRemote;
  $("autoScore").checked = settings.autoScore;
  const gpu = !!navigator.gpu;
  for (const d of document.querySelectorAll(".device")) {
    d.classList.toggle("cpu", !gpu);
    d.querySelector(".deviceText").textContent = gpu ? "WebGPU found: scoring runs on your GPU." : "No WebGPU: scoring runs on the CPU (slower).";
  }
  $("version").textContent = `Lacuna ${chrome.runtime.getManifest().version} · MIT`;
  $("savedCount").textContent = ((await store.get("history")) || []).length;
  if (!cv) return show("welcome");
  show("page");
  phase("ready");
  maybeAuto();
})();
