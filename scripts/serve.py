"""Resident pi-H3 generation service: models load once, requests run one at a time.

~/.venvs/pih3/bin/python scripts/serve.py [--port 8765] [--warmup]

curl -s localhost:8765/generate -d '{"prompt": "A fox in the snow", "height": 512, "width": 896, "frames": 121,
     "steps": 4, "image": "/abs/first.png", "last_image": "/abs/last.png", "out": "/abs/out.mp4"}'
-> {"out": "/abs/out.mp4", "timings": {...}}        GET /health -> {"busy": false, "loaded": [...]}
POST /unload frees the models (e.g. before running something else on the GPU).
Listens on 127.0.0.1 only; image/out paths are local files.
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from pih3_mlx.pipeline import PiH3Engine, save_mp4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--warmup", action="store_true", help="load the text encoder and 4-step transformer at start")
ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "outputs", "serve"))
a = ap.parse_args()

engine = PiH3Engine()
lock = threading.Lock()  # one generation at a time on the GPU


def _loaded():
    return [n for n, v in (("text_encoder", engine._enc), ("transformer", engine._tf_key)) if v is not None]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"busy": lock.locked(), "loaded": _loaded(), "transformer": engine._tf_key})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/unload":
            with lock:
                engine.unload()
            return self._send(200, {"loaded": _loaded()})
        if self.path != "/generate":
            return self._send(404, {"error": "not found"})
        if not body.get("prompt"):
            return self._send(400, {"error": "prompt required"})
        with lock:
            try:
                img, last = body.get("image"), body.get("last_image")
                res = engine.generate(body["prompt"], image=Image.open(img) if img else None,
                                      last_image=Image.open(last) if last else None, height=body.get("height"),
                                      width=body.get("width"), num_frames=body.get("frames", 121),
                                      steps=body.get("steps", 4), seed=body.get("seed", 42),
                                      sparse_attention=not body.get("dense_attention", False))
                out = body.get("out") or os.path.join(a.out_dir, time.strftime("%Y%m%d-%H%M%S") + ".mp4")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                save_mp4(res, out)
            except Exception as e:  # report and keep serving
                return self._send(500, {"error": repr(e)})
        self._send(200, {"out": out, "timings": {k: (round(v, 2) if isinstance(v, float) else v)
                                                 for k, v in res.timings.items()}})


if a.warmup:
    engine._text_encoder()
    engine._transformer(4)
print(f"pi-H3 service on http://127.0.0.1:{a.port} (loaded: {_loaded()})", flush=True)
ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
