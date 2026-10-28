// JS core vs the Python pipeline on the fixtures from test/make_fixtures.py (data/parity/, gitignored).
//   node test/parity.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-node";
import { encodePair, interpret, normalizeCv } from "../src/core.js";

const root = fileURLToPath(new URL("../../", import.meta.url));
const exp = root + "export/lacuna-celer/";
const read = (p) => JSON.parse(readFileSync(p, "utf8"));
const tok = new Tokenizer(read(exp + "tokenizer.json"), read(exp + "tokenizer_config.json"));
const cfg = read(exp + "lacuna.json");

const cvs = read(root + "data/parity/cv.json");
const cvBad = cvs.filter((c) => normalizeCv(c.raw) !== c.norm);
console.log(`normalizeCv: ${cvs.length - cvBad.length}/${cvs.length} identical`);
if (cvBad.length) {
  const a = normalizeCv(cvBad[0].raw).split("\n"), b = cvBad[0].norm.split("\n");
  const i = a.findIndex((l, k) => l !== b[k]);
  console.log("  first diff:\n  js: " + JSON.stringify(a[i]) + "\n  py: " + JSON.stringify(b[i]));
}

const pairs = read(root + "data/parity/pairs.json");
let idsSame = 0, linesSame = 0, linesNear = 0;
for (const p of pairs) {
  const e = encodePair(tok, p.cv, p.job, cfg.max_len);
  if (e.inputIds.length === p.ids.length && e.inputIds.every((x, i) => x === p.ids[i])) idsSame++;
  const js = e.lines.map((l) => [l.a, l.b]);
  if (JSON.stringify(js) === JSON.stringify(p.lines)) linesSame++;
  else if (js.length === p.lines.length && js.every((l, i) => Math.abs(l[0] - p.lines[i][0]) <= 1 && Math.abs(l[1] - p.lines[i][1]) <= 1)) linesNear++;
}
console.log(`encodePair: ids identical ${idsSame}/${pairs.length}; line ranges identical ${linesSame}, within 1 token ${linesNear}`);

const s = await ort.InferenceSession.create(exp + "model_q4f16.onnx");
let maxDiff = 0;
for (const p of pairs.filter((p) => p.logits)) {
  const e = encodePair(tok, p.cv, p.job, cfg.max_len);
  const ids = new ort.Tensor("int64", BigInt64Array.from(e.inputIds.map(BigInt)), [1, e.inputIds.length]);
  const mask = new ort.Tensor("int64", BigInt64Array.from(e.inputIds.map(() => 1n)), [1, e.inputIds.length]);
  const o = await s.run({ input_ids: ids, attention_mask: mask });
  const n = e.inputIds.length, arr = (t) => Array.from(t.data);
  const out = Object.fromEntries(["verdict", "location", "seniority", "must_met"].map((h) => [h, arr(o[h])]));
  const met = arr(o.line_met), must = arr(o.line_must);
  out.line_met = Array.from({ length: n }, (_, i) => met.slice(i * 4, i * 4 + 4));
  out.line_must = must;
  for (const h of ["verdict", "location", "seniority", "must_met"]) out[h].forEach((x, i) => { maxDiff = Math.max(maxDiff, Math.abs(x - p.logits[h][i])); });
  const r = interpret(out, e, p.job, cfg);
  console.log(`  fit ${r.fit.toFixed(2)} verdict ${r.verdict} location ${r.location} seniority ${r.seniority} gaps ${r.gaps.length} met ${r.met.length}`);
}
console.log(`ONNX logits, JS vs Python (same model): max abs diff ${maxDiff.toFixed(4)}`);
