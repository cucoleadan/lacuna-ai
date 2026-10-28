// Brand assets from the vector logo sheet: cropped SVGs (mark, lockup dark/light text, app icon) and every PNG size.
//   node store/brand.mjs <logo sheet.svg> <out dir>
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { chromium } from "playwright";

const [src, out] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const sheet = readFileSync(src, "utf8");
const U = 2048 / 1254;  // sheet units per pixel of the 1254 px preview the boxes were measured on
const crop = (svg, [x0, y0, x1, y1], pad = 6) => {
  const [x, y, w, h] = [(x0 - pad) * U, (y0 - pad) * U, (x1 - x0 + 2 * pad) * U, (y1 - y0 + 2 * pad) * U].map((v) => +v.toFixed(2));
  return svg.replace(/<svg[^>]*>/, `<svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="${x} ${y} ${w} ${h}" width="${w}" height="${h}">`);
};
const light = (svg) => svg.replaceAll('fill="rgb(33,33,33)"', 'fill="rgb(246,247,251)"').replaceAll('fill="rgb(0,0,1)"', 'fill="rgb(246,247,251)"');
const BOX = { mark: [127, 366, 360, 621], lockup: [127, 366, 1136, 621], icon: [218, 924, 453, 1156] };
const svgs = {
  "mark.svg": crop(sheet, BOX.mark),
  "lockup-dark-text.svg": crop(sheet, BOX.lockup),
  "lockup-light-text.svg": light(crop(sheet, BOX.lockup)),
  "app-icon.svg": crop(sheet, BOX.icon, 1),
};
for (const [f, s] of Object.entries(svgs)) writeFileSync(`${out}/${f}`, s);

const b = await chromium.launch({ channel: "chromium" });
async function png(svg, w, h, path, inner = null) {
  const p = await b.newPage({ viewport: { width: w, height: h } });
  const [iw, ih] = inner || [w, h];
  await p.setContent(`<body style="margin:0;background:transparent;display:grid;place-items:center;width:${w}px;height:${h}px">
    <div style="width:${iw}px;height:${ih}px">${svg.replace(/width="[^"]*" height="[^"]*"/, `width="100%" height="100%"`)}</div></body>`);
  await p.screenshot({ path, omitBackground: true });
  await p.close();
}
const ratio = (s) => { const [, , w, h] = s.match(/viewBox="([^"]*)"/)[1].split(" ").map(Number); return w / h; };
const r = ratio(svgs["lockup-dark-text.svg"]);
await png(svgs["lockup-dark-text.svg"], 2000, Math.round(2000 / r), `${out}/lockup-dark-text.png`);
await png(svgs["lockup-light-text.svg"], 2000, Math.round(2000 / r), `${out}/lockup-light-text.png`);
const rm = ratio(svgs["mark.svg"]);
await png(svgs["mark.svg"], Math.round(1024 * rm), 1024, `${out}/mark.png`);
// toolbar sizes: the bare mark fills the square; larger sizes: the dark app icon
for (const n of [16, 32]) await png(svgs["mark.svg"], n, n, `${out}/icon-${n}.png`, rm < 1 ? [n * rm, n] : [n, n / rm]);
for (const n of [48, 128, 512, 1024]) await png(svgs["app-icon.svg"], n, n, `${out}/${n === 512 ? "avatar-512" : n === 1024 ? "app-icon" : "icon-" + n}.png`);
await png(svgs["app-icon.svg"], 128, 128, `${out}/store-icon-128.png`, [96, 96]);  // 96 px artwork in a 128 px square
await b.close();
