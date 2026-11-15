"""Website server for the laptop: serves web/ and, if checkpoints/w2v.pt exists, the big XLS-R model.

  .venv\\Scripts\\python.exe serve.py          ->  http://localhost:8000

The page decodes audio to 16 kHz mono itself and POSTs the raw float32 samples to /api/score,
so audio never leaves this computer. Without w2v.pt the page simply uses the phone model.
"""
import json
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from dfad import ROOT, load_config

WEB = ROOT / "web"
_det, _lock = None, threading.Lock()


def detector():
    global _det
    if _det is None:
        ck = Path(load_config()["paths"]["checkpoints"]) / "w2v.pt"
        if not ck.exists():
            return None
        from dfad.infer import Detector
        _det = Detector(ck)
        print(f"big model loaded on {_det.device} (threshold {_det.threshold:.3f})", flush=True)
    return _det


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet: only API calls
        if "/api/" in self.path:
            super().log_message(fmt, *args)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/info":
            d = detector()
            return self._json({"big_model": d is not None, "name": "XLS-R (wav2vec 2.0)",
                               "threshold": d.threshold if d else None})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/score":
            return self._json({"error": "not found"}, 404)
        d = detector()
        if d is None:
            return self._json({"error": "no big model (checkpoints/w2v.pt)"}, 503)
        n = int(self.headers.get("Content-Length", 0))
        x = np.frombuffer(self.rfile.read(n), dtype=np.float32).copy()
        if len(x) < 1600:
            return self._json({"error": "audio too short"}, 400)
        with _lock:  # one GPU job at a time
            r = d.analyse(x)
        x[:] = 0  # nothing is kept
        return self._json({"window_probs": r["window_probs"].tolist(), "fake_prob": r["fake_prob"],
                           "threshold": r["threshold"], "hop": d.hop, "window": d.n})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    threading.Thread(target=detector, daemon=True).start()  # warm up in the background
    print(f"Call Guard website: http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("", port), partial(Handler, directory=str(WEB))).serve_forever()
