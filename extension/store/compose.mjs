// Chrome Web Store images from the raw captures (store/screens.mjs) and the logo:
// five 1280x800 screenshots, the 440x280 and 1400x560 promo tiles, the 128x128 store icon.
//   node store/compose.mjs <logo.png> <out dir>
import { mkdirSync, readFileSync } from "node:fs";
import { chromium } from "playwright";

const root = new URL("../../", import.meta.url).pathname;
const raw = root + "export/store-raw/";
const [logo, out] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const uri = (p) => `data:image/png;base64,${readFileSync(p).toString("base64")}`;
const logoUri = uri(logo);

const css = `
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#eef1ff;
background:radial-gradient(1100px 700px at 85% 15%,#1b1f5e 0%,#070a24 55%,#04061a 100%);overflow:hidden}
.cap{position:absolute;left:60px;top:96px;width:360px}
.cap .brand{display:flex;align-items:center;gap:12px;font-weight:600;letter-spacing:.02em;color:#a9b4ff;font-size:18px;margin-bottom:28px}
.cap .brand img{width:44px;height:44px;border-radius:10px}
.cap h1{font-size:42px;line-height:1.1;margin:0 0 18px;font-weight:750;letter-spacing:-.01em}
.cap h1 em{font-style:normal;background:linear-gradient(90deg,#3fe6ff,#a46bff);-webkit-background-clip:text;color:transparent}
.cap p{font-size:20px;line-height:1.45;color:#c4cae8;margin:0}
.win{position:absolute;right:52px;top:96px;width:770px;height:608px;border-radius:14px;overflow:hidden;background:#fff;
box-shadow:0 30px 80px rgba(0,0,0,.55),0 0 0 1px rgba(255,255,255,.08)}
.bar{height:34px;background:#e9ebf0;display:flex;align-items:center;gap:7px;padding:0 12px}
.bar i{width:11px;height:11px;border-radius:50%;background:#c9ccd4}
.bar .url{margin-left:14px;flex:1;height:20px;border-radius:10px;background:#fff}
.body{display:flex;height:574px}
/* job page: crop to its 720 px content column (of 900), panel at ~0.83x */
.page{flex:1;overflow:hidden;background:#f4f5f8}.page img{width:550px;margin-left:-43px;display:block}
.side{width:330px;border-left:1px solid #d9dce3;overflow:hidden;background:#fff}.side img{width:330px;display:block}
`;
const shot = (job, panel, h1, p) => `<style>${css}</style>
<div class="cap"><div class="brand"><img src="${logoUri}">Lacuna</div><h1>${h1}</h1><p>${p}</p></div>
<div class="win"><div class="bar"><i></i><i></i><i></i><div class="url"></div></div>
<div class="body"><div class="page"><img src="${uri(raw + "job-" + job + ".png")}"></div>
<div class="side"><img src="${uri(raw + "panel-" + panel + ".png")}"></div></div></div>`;

const SHOTS = [
  ["fullstack-ai", "fullstack-ai", "Mind the gap, <em>before you apply.</em>", "See how a job fits your CV, and what it asks for that your CV doesn't show, right next to the posting."],
  ["staff-ml", "staff-ml", "Know when a job <em>asks for more.</em>", "Seniority and missing requirements, read line by line from the posting."],
  ["backend-sf", "backend-sf", "Can you even <em>work there?</em>", "Lacuna reads where the job lets you work from the posting itself."],
  ["product-eng", "ranked", "Your best matches, <em>first.</em>", "Every job you score lands in one list, sorted by fit. Read the top five, skip the rest."],
  ["fullstack-ai", "settings", "Your CV never <em>leaves your browser.</em>", "There is no account and no server. The model runs on your own computer."],
];

const promo = (w, h, big) => `<style>${css}
.t{position:absolute;inset:0;display:flex;align-items:center;gap:${big ? 48 : 22}px;padding:0 ${big ? 110 : 34}px}
.t img{width:${big ? 300 : 150}px;height:${big ? 300 : 150}px;border-radius:${big ? 40 : 22}px}
.t h1{font-size:${big ? 92 : 44}px;margin:0;font-weight:780;letter-spacing:-.01em}
.t p{font-size:${big ? 34 : 17}px;margin:${big ? 14 : 8}px 0 0;color:#c4cae8}
.t em{font-style:normal;background:linear-gradient(90deg,#3fe6ff,#a46bff);-webkit-background-clip:text;color:transparent}
</style><div class="t"><img src="${logoUri}"><div><h1>Lacuna</h1><p>Mind the gap, <em>before you apply.</em></p>
${big ? '<p style="font-size:24px;color:#8f98c8;margin-top:22px">Job fit scored on your device. Your CV never leaves the browser.</p>' : ""}</div></div>`;

const b = await chromium.launch({ channel: "chromium" });
const render = async (html, w, h, path) => {
  const pg = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
  await pg.setContent(html, { waitUntil: "load" });
  await pg.screenshot({ path });
  await pg.close();
};
for (const [i, s] of SHOTS.entries()) await render(shot(...s), 1280, 800, `${out}/screenshot-${i + 1}.png`);
await render(promo(440, 280, false), 440, 280, `${out}/promo-small-440x280.png`);
await render(promo(1400, 560, true), 1400, 560, `${out}/promo-marquee-1400x560.png`);
// store icon: 96x96 artwork inside a 128x128 transparent square, as the store guidelines ask
await render(`<body style="margin:0;background:transparent"><img src="${logoUri}" style="width:96px;height:96px;margin:16px;border-radius:18px;display:block"></body>`,
  128, 128, `${out}/store-icon-128.png`);
await b.close();
