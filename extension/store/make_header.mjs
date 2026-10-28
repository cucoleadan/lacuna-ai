// README header images (light and dark) from the logo lockups, run from the repo root:
//   node extension/store/make_header.mjs <brand dir>   -> assets/header-{light,dark}.png
import { readFileSync } from "node:fs";
import { chromium } from "playwright";
const dir = process.argv[2];
const uri = (f) => `data:image/png;base64,${readFileSync(`${dir}/${f}`).toString("base64")}`;
const page = (dark) => `<body style="margin:0;width:1600px;height:440px;display:flex;align-items:center;justify-content:space-between;padding:0 110px;box-sizing:border-box;
  background:${dark ? "radial-gradient(900px 500px at 85% 20%,#1b1f5e 0%,#0b0d24 60%,#07081a 100%)" : "linear-gradient(135deg,#f6f7fa 0%,#eef1f6 100%)"};font-family:system-ui,-apple-system,'Segoe UI',sans-serif">
  <img src="${uri(dark ? "lockup-light-text.png" : "lockup-dark-text.png")}" style="width:720px">
  <div style="text-align:right;color:${dark ? "#e9ecff" : "#1c1f2b"}">
    <div style="font-size:46px;font-weight:750;letter-spacing:-.01em;line-height:1.15">Mind the gap,<br><span style="background:linear-gradient(90deg,#1e6bff,#9a2bff);-webkit-background-clip:text;color:transparent">before you apply.</span></div>
    <div style="font-size:22px;margin-top:16px;color:${dark ? "#aab2d8" : "#5b6275"}">Job fit for your CV, scored in your browser.</div>
  </div></body>`;
const b = await chromium.launch({ channel: "chromium" });
for (const dark of [false, true]) {
  const p = await b.newPage({ viewport: { width: 1600, height: 440 } });
  await p.setContent(page(dark), { waitUntil: "load" });
  await p.screenshot({ path: `assets/header-${dark ? "dark" : "light"}.png` });
}
await b.close();
