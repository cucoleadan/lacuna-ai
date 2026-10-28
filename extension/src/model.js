// Model files: downloaded once from Hugging Face (or loaded from a local folder), kept in the Cache API, run with
// ONNX Runtime Web on WebGPU (WASM fallback). Runs inside the side panel page.
import * as ort from "onnxruntime-web/webgpu";
import { Tokenizer } from "@huggingface/tokenizers";

ort.env.wasm.wasmPaths = chrome.runtime.getURL("ort/");
ort.env.wasm.numThreads = 1;  // extension pages are not cross-origin isolated: no WASM threads
ort.env.webgpu.powerPreference = "high-performance";  // laptops: the discrete GPU, not the integrated one

export const TIERS = {
  celer: { label: "Celer", repo: "lacuna-ai/lacuna-celer", file: "model_q4f16.onnx", mb: 76,
           note: "Fast and small. As accurate as Altus on our tests." },
  // fp16, not q4f16: 4-bit costs the large model 0.05 verdict F1, most of its edge over the small one
  altus: { label: "Altus", repo: "lacuna-ai/lacuna-altus", file: "model_fp16.onnx", mb: 790,
           note: "The large model. Needs about 2 GB of free GPU memory." },
};
const FILES = ["lacuna.json", "tokenizer.json", "tokenizer_config.json"];
const CACHE = "lacuna-models-v1";
const LOCAL = "https://lacuna.local/";  // cache keys for files picked from disk

// a Hugging Face repo id, or any base URL ending in "/" (self-hosted, or files imported from a folder)
const url = (repo, f) => (/^https?:\/\//.test(repo) ? repo + f : `https://huggingface.co/${repo}/resolve/main/${f}`);

async function fetchCached(u, onBytes) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(u);
  if (hit) return hit.arrayBuffer();
  const r = await fetch(u);
  if (!r.ok) throw new Error(`${r.status} for ${u}`);
  const total = +r.headers.get("content-length") || 0, reader = r.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onBytes?.(got, total);
  }
  const blob = new Blob(parts);
  await cache.put(u, new Response(blob));
  return blob.arrayBuffer();
}

/** Files chosen from a folder (model_q4f16.onnx, lacuna.json, tokenizer files) -> cached under a local key. */
export async function importFolder(files, name = "local") {
  const cache = await caches.open(CACHE), base = `${LOCAL}${name}/`;
  for (const f of files) await cache.put(base + f.name, new Response(f));
  return base;
}

export async function isCached(repo, file) {
  return !!(await (await caches.open(CACHE)).match(url(repo, file)));
}

export async function clearCache() { await caches.delete(CACHE); }

let loaded = null;  // {key, session, tok, cfg, device}
const tokenizers = {};  // repo -> Promise<{tok, cfg}>: the small files only, so the CV's token count needs no model

export function tokenizer(repo) {
  return tokenizers[repo] ||= Promise.all(FILES.map(async (f) => JSON.parse(new TextDecoder().decode(await fetchCached(url(repo, f))))))
    .then(([cfg, tj, tc]) => ({ cfg, tok: new Tokenizer(tj, tc) }))
    .catch((e) => { delete tokenizers[repo]; throw e; });
}

export async function load(repo, file, onProgress) {
  const key = repo + file;
  if (loaded?.key === key) return loaded;
  const { cfg, tok } = await tokenizer(repo);
  const bytes = await fetchCached(url(repo, file), (got, total) => onProgress?.({ got, total }));
  onProgress?.({ phase: "starting" });
  let session, device = "webgpu";
  try {
    if (!navigator.gpu) throw new Error("no WebGPU");
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" });
  } catch (e) {
    console.warn("WebGPU unavailable, using WASM:", e);
    device = "wasm";
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  }
  loaded = { key, session, tok, cfg, device };
  return loaded;
}

/** One forward pass -> plain arrays shaped like interpret() expects. */
export async function run(m, inputIds) {
  const n = inputIds.length;
  const ids = new ort.Tensor("int64", BigInt64Array.from(inputIds, BigInt), [1, n]);
  const mask = new ort.Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]);
  const o = await m.session.run({ input_ids: ids, attention_mask: mask });
  const arr = (t) => Array.from(t.data, Number);
  const met = arr(o.line_met);
  const out = Object.fromEntries(["verdict", "location", "seniority", "must_met"].map((h) => [h, arr(o[h])]));
  out.line_met = Array.from({ length: n }, (_, i) => met.slice(i * 4, i * 4 + 4));
  out.line_must = arr(o.line_must);
  for (const t of Object.values(o)) t.dispose?.();
  return out;
}
