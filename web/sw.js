// Offline cache: app shell precached; model + ONNX Runtime files cached on first use.
const CACHE = "callguard-v5";
const SHELL = ["./", "index.html", "style.css", "app.js", "worklet.js", "manifest.json",
  "model/config.json", "model/student.onnx", "icons/icon-192.png", "icons/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// Network first (so updates show up immediately), cache as the offline fallback.
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(fetch(e.request).then((res) => {
    if (res.ok || res.type === "opaque") {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(e.request, copy));
    }
    return res;
  }).catch(() => caches.match(e.request)));
});
