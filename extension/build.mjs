// node build.mjs -> dist/ (load it in chrome://extensions with "Load unpacked")
import { build } from "esbuild";
import { cpSync, mkdirSync, rmSync } from "node:fs";

rmSync("dist", { recursive: true, force: true });
mkdirSync("dist/ort", { recursive: true });
await build({
  entryPoints: ["src/panel.js", "src/background.js"], outdir: "dist", bundle: true, format: "esm", target: "chrome121",
  minify: true, sourcemap: true, logLevel: "info",
  external: ["*.wasm"],
});
cpSync("static", "dist", { recursive: true });
cpSync("src/extract.js", "dist/extract.js");
for (const f of ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"]) {
  cpSync(`node_modules/onnxruntime-web/dist/${f}`, `dist/ort/${f}`);
}
