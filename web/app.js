"use strict";
// Call Guard web: on-device deepfake voice check (ONNX Runtime Web, WASM, single thread).

const $ = (id) => document.getElementById(id);
const SR = 16000;
let cfg = { sr: SR, window: 64000, hop: 32000, threshold: 0.5, consecutive: 3 };
const MAX_WINDOWS = 30;
let session = null;
let big = null;       // {threshold} when the laptop server (serve.py) offers the XLS-R model
let useBig = false;

// ------------------------------------------------------------------ model

async function loadModel() {
  try {
    cfg = { ...cfg, ...(await (await fetch("model/config.json")).json()) };
    ort.env.wasm.numThreads = 1; // threads need cross-origin isolation (not on GitHub Pages)
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    session = await ort.InferenceSession.create("model/student.onnx", { executionProviders: ["wasm"] });
    $("model-status").textContent = "Model ready · works offline after this first load";
  } catch (e) {
    $("model-status").textContent = "Could not load the model: " + e.message;
  }
  try { // optional big model, only when served by serve.py on the laptop
    const info = await (await fetch("api/info")).json();
    if (info.big_model) {
      big = { threshold: info.threshold };
      useBig = true;
      $("model-pick").hidden = false;
      $("model-pick").value = "big";
      $("model-status").textContent = "Big model (XLS-R, on this laptop) + phone model ready";
    }
  } catch { /* plain static hosting: phone model only */ }
}

$("model-pick").onchange = (e) => { useBig = e.target.value === "big" && !!big; };

const threshold = () => (useBig ? big.threshold : cfg.threshold);
// Live listening hears the caller through a loudspeaker, which raises every score: the phone model
// has a separate threshold for it, calibrated on speakerphone audio (export_onnx.py).
const liveThreshold = () => (useBig ? big.threshold : cfg.live_threshold ?? cfg.threshold);

async function score(win) {
  const t0 = performance.now();
  if (useBig) { // audio goes only to the local server on this computer
    const r = await (await fetch("api/score", { method: "POST", body: win })).json();
    return { p: r.fake_prob, ms: performance.now() - t0 };
  }
  const out = await session.run({ audio: new ort.Tensor("float32", win, [1, win.length]) });
  return { p: out.prob.data[0], ms: performance.now() - t0 };
}

// 4 s windows every 2 s; long clips get MAX_WINDOWS spread evenly over the whole clip
// (same as dfad/data.py). Short clips are repeat-padded, like training. Returns [{start, w}].
function windows(x) {
  const n = cfg.window, hop = cfg.hop;
  if (x.length <= n) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = x.length ? x[i % x.length] : 0;
    return [{ start: 0, w }];
  }
  let starts = [];
  for (let s = 0; s + n <= x.length; s += hop) starts.push(s);
  if (starts.length > MAX_WINDOWS) {
    starts = Array.from({ length: MAX_WINDOWS }, (_, i) => Math.floor((i * (x.length - n)) / (MAX_WINDOWS - 1)));
  }
  return starts.map((s) => ({ start: s, w: x.slice(s, s + n) }));
}

function level(p, t = threshold()) { // on the risk scale, so it means the same for both models
  const r = risk(p, t);
  return r >= 0.5 ? "red" : r >= 0.35 ? "yellow" : "green";
}

// Display scale: the decision threshold maps to 50%, so ">50% fake" always means the FAKE verdict.
function risk(p, t = threshold()) {
  return p < t ? 0.5 * p / t : 0.5 + 0.5 * (p - t) / (1 - t);
}
const COLOR = { green: "var(--green)", yellow: "var(--yellow)", red: "var(--red)" };

// Simple energy VAD: enough 20 ms frames above -45 dBFS?
function hasSpeech(x) {
  const f = 320;
  let voiced = 0, frames = 0;
  for (let i = 0; i + f <= x.length; i += f, frames++) {
    let e = 0;
    for (let j = i; j < i + f; j++) e += x[j] * x[j];
    if (10 * Math.log10(e / f + 1e-12) > -45) voiced++;
  }
  return voiced / frames > 0.25;
}

// ------------------------------------------------------------------ tabs

for (const [tab, panel] of [["tab-clip", "panel-clip"], ["tab-live", "panel-live"]]) {
  $(tab).onclick = () => {
    for (const t of ["tab-clip", "tab-live"]) $(t).setAttribute("aria-selected", t === tab);
    $("panel-clip").hidden = panel !== "panel-clip";
    $("panel-live").hidden = panel !== "panel-live";
  };
}

// ------------------------------------------------------------------ check a clip

async function decodeTo16k(arrayBuf) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const buf = await ctx.decodeAudioData(arrayBuf);
  ctx.close();
  const off = new OfflineAudioContext(1, Math.ceil(buf.duration * SR), SR); // resample + downmix
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}

function wavBlob(x) { // 16-bit PCM WAV so recordings can be played back
  const b = new DataView(new ArrayBuffer(44 + x.length * 2));
  const s = (o, t) => [...t].forEach((c, i) => b.setUint8(o + i, c.charCodeAt(0)));
  s(0, "RIFF"); b.setUint32(4, 36 + x.length * 2, true); s(8, "WAVEfmt ");
  b.setUint32(16, 16, true); b.setUint16(20, 1, true); b.setUint16(22, 1, true);
  b.setUint32(24, SR, true); b.setUint32(28, SR * 2, true); b.setUint16(32, 2, true); b.setUint16(34, 16, true);
  s(36, "data"); b.setUint32(40, x.length * 2, true);
  for (let i = 0; i < x.length; i++) b.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x[i])) * 32767, true);
  return new Blob([b], { type: "audio/wav" });
}

async function checkClip(x, label, blob) {
  if (!session) return;
  const res = $("clip-result");
  res.hidden = false;
  // let the user hear exactly what was analysed (stays on the device: a local blob URL)
  const player = $("clip-audio");
  if (player.src) URL.revokeObjectURL(player.src);
  player.src = URL.createObjectURL(blob || wavBlob(x));
  player.hidden = false;
  $("verdict").className = "verdict";
  $("bars").innerHTML = "";
  const wins = windows(x);
  const ps = [];
  let ms = 0;
  for (const [i, { start, w }] of wins.entries()) {
    $("verdict").textContent = `Analysing ${(x.length / SR).toFixed(1)} s of audio… window ${i + 1}/${wins.length}`;
    const r = await score(w);
    ps.push(r.p);
    ms += r.ms;
    const bar = document.createElement("div");
    bar.style.height = Math.max(4, risk(r.p) * 100) + "%";
    bar.style.background = COLOR[level(r.p)];
    const t0 = start / SR;
    bar.title = `${t0.toFixed(0)}–${(t0 + cfg.window / SR).toFixed(0)} s: AI-voice risk ${(risk(r.p) * 100).toFixed(0)}%`;
    $("bars").appendChild(bar);
  }
  const p = ps.reduce((a, b) => a + b, 0) / ps.length;
  const fake = p >= threshold();
  const r = risk(p);
  $("verdict").textContent = fake
    ? `⚠ Likely AI-generated (AI-voice risk ${(r * 100).toFixed(0)}%)`
    : `✓ Likely real voice (AI-voice risk ${(r * 100).toFixed(0)}%)`;
  $("verdict").className = "verdict " + (fake ? "fake" : "real");
  $("clip-meta").textContent = `${label} · ${(x.length / SR).toFixed(1)} s · ${ps.length} window(s) · ` +
    `${(ms / ps.length).toFixed(0)} ms per window` + (x.length < SR * 3 ? " · short clip: less reliable" : "");
  x.fill(0); // audio is discarded after analysis
}

$("file").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    await checkClip(await decodeTo16k(await f.arrayBuffer()), f.name, f);
  } catch (err) {
    $("clip-result").hidden = false;
    $("verdict").textContent = "This file format can't be decoded on this device. Try WAV or MP3.";
  }
  e.target.value = "";
};

// ------------------------------------------------------------------ microphone

// Returns {stop} and calls onChunk(Float32Array at 16 kHz) every ~100 ms.
async function openMic(onChunk) {
  const stream = await navigator.mediaDevices.getUserMedia({
    // the loudspeaker is the signal here, so don't let the browser cancel or clean it
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
  });
  let ctx, src;
  try { // ask the browser to resample to 16 kHz for us
    ctx = new AudioContext({ sampleRate: SR });
    src = ctx.createMediaStreamSource(stream);
  } catch {
    ctx = new AudioContext();
    src = ctx.createMediaStreamSource(stream);
  }
  await ctx.audioWorklet.addModule("worklet.js");
  const node = new AudioWorkletNode(ctx, "capture");
  const ratio = ctx.sampleRate / SR;
  node.port.onmessage = (e) => onChunk(ratio === 1 ? e.data : downsample(e.data, ratio));
  src.connect(node);
  return {
    stop() {
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
    },
  };
}

function downsample(x, ratio) { // box-filter + linear pick, fine for speech band
  const out = new Float32Array(Math.floor(x.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const a = Math.floor(i * ratio), b = Math.min(x.length, Math.floor((i + 1) * ratio));
    let s = 0;
    for (let j = a; j < b; j++) s += x[j];
    out[i] = s / Math.max(1, b - a);
  }
  return out;
}

$("rec").onclick = async () => {
  const btn = $("rec");
  if (btn.disabled) return;
  const total = 8 * SR, x = new Float32Array(total);
  let n = 0, mic, timer;
  btn.disabled = true;
  btn.classList.add("recording");
  try {
    await new Promise(async (done, fail) => {
      try {
        mic = await openMic((c) => {
          const k = Math.min(c.length, total - n);
          x.set(c.subarray(0, k), n);
          n += k;
          btn.textContent = `● Recording ${Math.ceil((total - n) / SR)} s`;
          if (n >= total) done();
        });
        // a blocked mic (e.g. during a call) may never deliver audio: don't wait forever
        timer = setTimeout(() => { if (n === 0) fail(new Error("no audio from the microphone")); }, 3000);
      } catch (e) { fail(e); }
    }).finally(() => clearTimeout(timer));
    mic.stop();
    await checkClip(x, "Recording");
  } catch (e) {
    mic?.stop();
    alert("Microphone not available: " + e.message);
  }
  btn.disabled = false;
  btn.classList.remove("recording");
  btn.textContent = "● Record 8 s";
};

// ------------------------------------------------------------------ live Call Guard

let live = null;

async function startLive() {
  const ring = new Float32Array(cfg.window);
  let filled = 0, sinceHop = 0, streak = 0, busy = false, silentZeros = 0;
  const wake = await navigator.wakeLock?.request("screen").catch(() => null);
  const mic = await openMic(async (c) => {
    ring.copyWithin(0, c.length);           // slide window
    ring.set(c, ring.length - c.length);
    filled += c.length;
    sinceHop += c.length;
    if (filled < ring.length || sinceHop < cfg.hop || busy) return;
    sinceHop = 0;
    let peak = 0;
    for (let i = 0; i < ring.length; i++) peak = Math.max(peak, Math.abs(ring[i]));
    if (peak === 0) {                       // exact digital silence = the OS is blocking the mic
      if (++silentZeros >= 2) showLevel(null, "Mic is blocked (phone in a call?) — use a 2nd device");
      return;
    }
    silentZeros = 0;
    if (!hasSpeech(ring)) { showLevel(null, "Listening… (no speech)"); return; }
    busy = true;
    const { p, ms } = await score(ring.slice());
    busy = false;
    const t = liveThreshold();
    const lv = level(p, t);
    streak = lv === "red" ? streak + 1 : 0;
    showLevel(p, { green: "Low AI-voice risk", yellow: "Some AI-voice signs", red: "HIGH AI-voice risk" }[lv], t);
    $("live-meta").textContent = `AI-voice risk ${(risk(p, t) * 100).toFixed(0)}% · ${ms.toFixed(0)} ms · red streak ${streak}/${cfg.consecutive}`;
    if (streak === cfg.consecutive) raiseAlert();
  });
  live = { stop() { mic.stop(); wake?.release(); ring.fill(0); } };
}

function showLevel(p, text, t) {
  const fill = $("meter-fill");
  fill.style.width = p == null ? "0%" : Math.max(4, risk(p, t) * 100) + "%";
  fill.style.background = p == null ? "transparent" : COLOR[level(p, t)];
  $("meter-label").textContent = text;
}

function raiseAlert() {
  $("alert").hidden = false;
  $("alert").classList.add("flash");
  if (navigator.vibrate) navigator.vibrate([400, 150, 400, 150, 400]);
  else beep(); // iOS has no vibration API
}

function beep() {
  const ctx = new AudioContext(), o = ctx.createOscillator(), g = ctx.createGain();
  o.frequency.value = 880;
  g.gain.value = 0.2;
  o.connect(g).connect(ctx.destination);
  o.start();
  o.stop(ctx.currentTime + 0.6);
  o.onended = () => ctx.close();
}

$("alert-close").onclick = () => {
  $("alert").hidden = true;
  $("alert").classList.remove("flash");
};

$("live-toggle").onclick = async () => {
  const btn = $("live-toggle");
  if (live) {
    live.stop();
    live = null;
    btn.textContent = "Start Call Guard";
    showLevel(null, "Idle");
    $("live-meta").textContent = "";
    return;
  }
  if (!session) return;
  try {
    await startLive();
    btn.textContent = "Stop Call Guard";
    showLevel(null, "Listening…");
  } catch (e) {
    showLevel(null, "Microphone not available: " + e.message);
  }
};

// ------------------------------------------------------------------ boot

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
loadModel();
