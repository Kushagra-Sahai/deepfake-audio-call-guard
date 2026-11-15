import puppeteer from "puppeteer-core";
const b = await puppeteer.launch({ executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
const p = await b.newPage();
p.on("console", (m) => console.log("console:", m.text()));
p.on("pageerror", (e) => console.log("pageerror:", e.message));
await p.goto(process.argv[2]);
await p.waitForFunction(() => document.title.startsWith("RESULT") || document.title.startsWith("ERR"), { timeout: 90000 });
console.log(await p.title());
console.log("meta:", await p.$eval("#clip-meta", (e) => e.textContent));
await b.close();
