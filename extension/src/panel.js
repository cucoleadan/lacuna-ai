import { encodePair, htmlToText, interpret, jobKey, normalizeCv } from "./core.js";
import { TIERS, clearCache, importFolder, isCached, load, run, tokenizer } from "./model.js";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const store = {
  get: (k) => chrome.storage.local.get(k).then((r) => r[k]),
  set: (k, v) => chrome.storage.local.set({ [k]: v }),
};
const DEFAULTS = { tier: "celer", repos: {}, acceptUsRemote: false, autoScore: true, sites: null };
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
  if (view === "page") checkTab();
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
  $("scoreLabel").textContent = p === "result" ? "Re-run on this page" : "Run on this page";
}

// ---------- scoring ----------
// the job sites Lacuna runs on by itself (subdomains included): the built-in ones (manifest host permissions minus the
// model hosts) until the user edits the list; settings.sites stays null till then, so new built-in sites reach them
const BUILT_IN = chrome.runtime.getManifest().host_permissions.filter((p) => !/huggingface|hf\.co/.test(p)).map((p) => new URL(p.replace("*.", "")).hostname);
const jobSites = () => settings.sites || BUILT_IN;
const listed = (host) => jobSites().find((d) => host === d || host.endsWith("." + d));
// "https://Jobs.Example.com/careers" -> "jobs.example.com"; "" if it isn't a domain
function domain(v) {
  try { const h = new URL(/^\w+:\/\//.test(v) ? v : `https://${v.trim()}`).hostname.replace(/^www\./, ""); return h.includes(".") ? h : ""; }
  catch { return ""; }
}

// call straight from a click: Chrome only shows the permission prompt for a user gesture
async function addSite(d) {
  if (!BUILT_IN.includes(d) && !(await chrome.permissions.request({ origins: [`https://*.${d}/*`] }).catch(() => false))) return false;
  settings.sites = [...new Set([...jobSites(), d])];
  await saveSettings();
  return true;
}
async function removeSite(d) {
  settings.sites = jobSites().filter((x) => x !== d);
  await saveSettings();
  if (!BUILT_IN.includes(d)) chrome.permissions.remove({ origins: [`https://*.${d}/*`] }).catch(() => {});
}
const repoFor = (tier) => settings.repos[tier] || TIERS[tier].repo;
let busy = false, scored = false, shownUrl = "";  // shownUrl: jobKey of the result on screen

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

async function score(given) {
  if (busy) return;
  const cv = await store.get("cv");
  if (!cv) return show("welcome");
  busy = true;
  $("scoreBtn").disabled = true;
  $("permNote").hidden = true;
  status("");
  try {
    let page = given;
    if (!page) {
      const tab = await activeTab();
      // tab.url is hidden on sites Lacuna has no access to; try anyway (activeTab may cover it)
      if (!tab || (tab.url && !tab.url.startsWith("http"))) { status("Open a job posting, then score it."); return; }
      try { page = await readPage(tab); } catch (e) {
        status("Lacuna can't read this page yet.", true);
        $("permNote").hidden = false;
        return;
      }
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
    const timing = `${enc.inputIds.length.toLocaleString()} tokens · ${s.toFixed(1)} s · ${m.device === "webgpu" ? "GPU" : "CPU"} · ${TIERS[tier].label}`;
    render(page, r, timing);
    if (!page.job) status("This page doesn't look like a job posting, so the score may not mean much.");
    await remember(page, r, tier, timing);
  } catch (e) {
    console.error(e);
    status(`Something went wrong: ${e.message || e}`, true);
  } finally {
    busy = false;
    $("scoreBtn").disabled = false;
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

// savedAt: the result comes from Saved (scored then), not from a run just now
function render(page, r, timing, savedAt) {
  scored = true;
  $("savedNote").hidden = !savedAt;
  if (savedAt) $("savedWhen").textContent = new Date(savedAt).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  shownUrl = jobKey(page.url);
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

async function remember(page, r, tier, timing) {
  const hist = (await store.get("history")) || [];
  const text = (x) => ({ text: x.text, pMust: +x.pMust.toFixed(2) });
  const row = { url: jobKey(page.url), title: page.title, company: page.company, location: page.location, fit: r.fit,
                verdict: r.verdict, loc: r.location, sen: r.seniority, nGaps: r.gaps.length, gaps: r.gaps.map((g) => g.text).slice(0, 5), tier, at: Date.now(),
                // what render() needs to show it again in full
                r: { fit: r.fit, verdict: r.verdict, location: r.location, pLocationNo: r.pLocationNo, seniority: r.seniority, mustMet: r.mustMet,
                     truncated: r.truncated, gaps: r.gaps.map(text), partial: r.partial.map(text), met: r.met.map(text) }, timing };
  const next = [row, ...hist.filter((h) => jobKey(h.url) !== row.url)].slice(0, 2000);  // rows saved before jobKey keep raw URLs
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
  $("clearHist").textContent = filter === "all" ? "Clear all" : `Clear ${VERDICT[filter][0]}`;
  $("rankList").replaceChildren(...hist.filter((h) => filter === "all" || h.verdict === filter).map((h) => el("li", {},
    el("button", { className: "iconBtn del", type: "button", title: "Remove", ariaLabel: `Remove ${h.title || h.url}`, innerHTML: ICON.x,
                   onclick: () => dropJobs((x) => x.url === h.url) }),
    el("a", { href: h.url, target: "_blank" },
      el("span", { className: `score v-${h.verdict}`, textContent: Math.round(h.fit * 100) }),
      el("span", { className: "body" },
        el("span", { className: "t", textContent: h.title || h.url }),
        el("span", { className: "sub", textContent: [h.company, h.location].filter(Boolean).join(" · ") }),
        el("span", { className: "tags2" },
          el("span", { className: `tag v-${h.verdict}`, textContent: VERDICT[h.verdict]?.[0] || h.verdict }),
          note(h)))))));
}
async function dropJobs(gone) {
  await store.set("history", ((await store.get("history")) || []).filter((h) => !gone(h)));
  renderRanked();
}
$("clearHist").addEventListener("click", () => dropJobs((h) => filter === "all" || h.verdict === filter));

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
  // the CV and the posting share the model's token budget; past it, the end of the longer one is cut
  if (cv) tokenizer(repoFor(settings.tier)).then(({ tok, cfg }) => {
    const n = tok.encode(cv, { add_special_tokens: false }).ids.length;
    $("cvInfo").textContent = `${cv.length.toLocaleString()} characters · ${n.toLocaleString()} of ${cfg.max_len.toLocaleString()} tokens · in this browser only`;
  }).catch(() => {});
  $("editCv").textContent = cv ? "Replace" : "Add CV";
  $("deleteCv").disabled = !cv;
  renderSites();
  renderTiers();
}

function renderSites() {
  $("siteList").replaceChildren(...jobSites().map((d) => el("li", {}, el("span", { className: "mono grow", textContent: d }),
    el("button", { type: "button", className: "iconBtn", title: "Remove", ariaLabel: `Remove ${d}`, innerHTML: ICON.x,
                   onclick: async () => { await removeSite(d); renderSites(); } }))));
  $("siteEmpty").hidden = !!jobSites().length;
  $("resetSites").hidden = !settings.sites;
}
$("addSite").addEventListener("submit", async (e) => {
  e.preventDefault();
  const d = domain($("newSite").value);
  if (!d) { $("siteNote").textContent = "Type a domain, like jobs.example.com."; return; }
  if (listed(d) === d) { $("siteNote").textContent = `${d} is already on the list.`; return; }
  const ok = await addSite(d);
  $("siteNote").textContent = ok ? `Added ${d}.` : `Chrome didn't allow ${d}, so it wasn't added.`;
  if (ok) $("newSite").value = "";
  renderSites();
});
$("resetSites").addEventListener("click", async () => {
  for (const d of jobSites()) if (!BUILT_IN.includes(d)) chrome.permissions.remove({ origins: [`https://*.${d}/*`] }).catch(() => {});
  settings.sites = null;
  await saveSettings();
  $("siteNote").textContent = "Back to the built-in list.";
  renderSites();
});

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

// "Run on this page": ask for this one site first (no prompt if Lacuna already has it), then score
$("scoreBtn").addEventListener("click", async () => {
  if (siteOrigin) await chrome.permissions.request({ origins: [siteOrigin] }).catch(() => false);
  score();
});
document.querySelectorAll("[data-allow]").forEach((b) => b.addEventListener("click", async () => {
  if (!(await chrome.permissions.request({ origins: ["https://*/*"] }))) return;
  if (current === "settings") renderSettings(); else score();
}));

// ---------- this site: what Lacuna sees on the active tab, and whether it scores it on its own ----------
let siteOrigin = "", siteEntry = "";  // the tab's origin pattern; the list entry that covers it ("" = none)
function site(host, line, state, auto) {
  $("siteHost").textContent = host;
  $("siteLine").textContent = line;
  $("siteDot").className = `dot ${state}`;
  $("siteAutoBox").hidden = auto === undefined;
  $("siteAuto").checked = !!auto;
}

// runs on every tab change: reads the page (locally, no model) and scores it only if it looks like a job posting
async function checkTab() {
  if (current !== "page" || busy) return;
  const tab = await activeTab(), url = tab?.url;
  const key = url?.startsWith("http") ? jobKey(url) : "";
  if (scored && key !== shownUrl) { scored = false; phase("ready"); status(""); }
  siteOrigin = url?.startsWith("https:") ? new URL(url).origin + "/*" : "";
  if (!url) return site("This site", "Not on your job-site list. Lacuna runs here only when you ask.", "");
  if (!url.startsWith("http")) return site("No web page", "Open a job posting to score it.", "");
  const hostname = new URL(url).hostname, host = hostname.replace(/^www\./, "");
  siteEntry = listed(hostname) || "";
  $("siteAutoBox").title = siteEntry ? `Lacuna runs by itself on ${siteEntry}. Turn off to take it off your job-site list.` : `Add ${host} to your job-site list`;
  if (!siteEntry) return site(host, "Not on your job-site list. Lacuna runs here only when you ask.", "", false);
  if (!(await chrome.permissions.contains({ origins: [new URL(url).origin + "/*"] })))
    return site(host, "On your list, but Chrome hasn't allowed it. Turn Auto off and on to allow it.", "noaccess", false);
  if (scored && key === shownUrl) return;  // the same job, maybe its application form: the result is already showing
  const saved = ((await store.get("history")) || []).find((h) => jobKey(h.url) === key);
  if (saved?.r) {
    site(host, "Already scored. Showing the saved result.", "job", true);
    return render({ url: saved.url, title: saved.title, company: saved.company, location: saved.location }, saved.r, saved.timing || "", saved.at);
  }
  // saved before results were kept in full
  if (saved) return site(host, `Already scored: ${Math.round(saved.fit * 100)} fit, ${VERDICT[saved.verdict]?.[0] || saved.verdict}. Press Run on this page to see it again.`, "job", true);
  let page;
  try { page = await readPage(tab); } catch { return site(host, "Lacuna can't read this page.", "noaccess", true); }
  if (!page?.job) return site(host, page?.chars < 200 ? "Almost no text on this page. Nothing to score." : "Doesn't look like a job posting, so Lacuna didn't run.", "", true);
  if (!settings.autoScore) return site(host, `Job posting found (${page.job}). Auto-scoring is off in Settings.`, "job", true);
  site(host, `Job posting found (${page.job}). Ran automatically.`, "auto", true);
  score(page);
}
$("siteAuto").addEventListener("change", async (e) => {
  if (!e.target.checked) await removeSite(siteEntry);
  else if (!(await addSite(siteEntry || $("siteHost").textContent))) e.target.checked = false;
  checkTab();
});
chrome.tabs.onActivated.addListener(checkTab);
chrome.tabs.onUpdated.addListener((id, info, tab) => { if (info.status === "complete" && tab.active) checkTab(); });

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
})();
