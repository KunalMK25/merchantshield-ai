#!/usr/bin/env node
// Captures the dashboard screenshots used in the README.
//
//   node scripts/capture_screenshots.mjs [baseUrl] [outDir]
//
// Needs the app running (API on :8000 and either the Vite dev server or the built frontend served by
// FastAPI) and Google Chrome installed. No npm dependencies: it drives headless Chrome directly over
// its remote-debugging protocol using Node's built-in WebSocket. Set CHROME_PATH to override Chrome.
//
// The shots show whatever the running app contains, so run a few retraining cycles first
// (python -m ml.retrain.job, once or twice with --drift amount_scale=2.5) for meaningful pages.

import { spawn } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const BASE = process.argv[2] || "http://localhost:5173";
const OUT = process.argv[3] || "screenshots";
const CHROME = process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9333;
const WIDTH = 1440;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const SHOTS = [
  { file: "evaluate.png", route: "evaluate", theme: "dark", wait: 1200, maxHeight: 1500,
    run: `[...document.querySelectorAll('button')].find(b => b.textContent.includes('Large purchase'))?.click()`, after: 3000 },
  { file: "evaluate-light.png", route: "evaluate", theme: "light", wait: 1200, maxHeight: 1500,
    run: `[...document.querySelectorAll('button')].find(b => b.textContent.includes('Large purchase'))?.click()`, after: 3000 },
  { file: "registry.png", route: "registry", theme: "dark", wait: 2000, maxHeight: 1300 },
  { file: "retraining.png", route: "retrain", theme: "dark", wait: 2000, maxHeight: 1100 },
  { file: "drift.png", route: "drift", theme: "dark", wait: 2500, maxHeight: 1700 },
  { file: "alerts.png", route: "alerts", theme: "dark", wait: 2500, maxHeight: 1300 },
];

async function main() {
  mkdirSync(OUT, { recursive: true });
  const profile = mkdtempSync(join(tmpdir(), "ms-shots-"));
  const chrome = spawn(CHROME, [
    "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
    "--hide-scrollbars", "--no-first-run", "--disable-gpu", "about:blank",
  ], { stdio: "ignore" });

  try {
    let target;
    for (let i = 0; i < 50 && !target; i++) {
      await sleep(200);
      try {
        const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
        target = list.find((t) => t.type === "page");
      } catch { /* Chrome still starting */ }
    }
    if (!target) throw new Error("Could not connect to Chrome. Is CHROME_PATH correct?");

    const ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
    let id = 0;
    const pending = new Map();
    ws.onmessage = (m) => {
      const msg = JSON.parse(m.data);
      if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
    };
    const send = (method, params = {}) => new Promise((res) => {
      const myId = ++id;
      pending.set(myId, res);
      ws.send(JSON.stringify({ id: myId, method, params }));
    });
    const evalJs = async (expression) =>
      (await send("Runtime.evaluate", { expression, returnByValue: true })).result?.result?.value;

    await send("Page.enable");
    for (const shot of SHOTS) {
      await send("Emulation.setDeviceMetricsOverride", { width: WIDTH, height: 900, deviceScaleFactor: 1.5, mobile: false });
      await send("Page.navigate", { url: `${BASE}/#/${shot.route}` });
      await sleep(600);
      await evalJs(`localStorage.setItem('ms-theme', '${shot.theme}')`);
      await send("Page.navigate", { url: `${BASE}/?t=${Date.now()}#/${shot.route}` });
      await sleep(shot.wait);
      if (shot.run) { await evalJs(shot.run); await sleep(shot.after || 1500); }
      const height = Math.min(await evalJs("document.documentElement.scrollHeight"), shot.maxHeight);
      await send("Emulation.setDeviceMetricsOverride", { width: WIDTH, height, deviceScaleFactor: 1.5, mobile: false });
      await sleep(400);
      const { result } = await send("Page.captureScreenshot", { format: "png" });
      writeFileSync(join(OUT, shot.file), Buffer.from(result.data, "base64"));
      console.log(`saved ${join(OUT, shot.file)}  (${WIDTH}x${height})`);
    }
    ws.close();
  } finally {
    chrome.kill();
  }
}

main().catch((e) => { console.error(e.message); process.exit(1); });
