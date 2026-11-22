"""pi-H3 on MLX: text/image-to-video with audio in 2-4 steps.

~/.venvs/pih3/bin/python scripts/generate.py "A red fox trotting through snow" --frames 121 --height 512 --width 896
~/.venvs/pih3/bin/python scripts/generate.py "The cat meows" --image cat.png
~/.venvs/pih3/bin/python scripts/generate.py --batch jobs.json     # models stay loaded across jobs

jobs.json: [{"prompt": "...", "image": "a.png", "last_image": "b.png", "height": 512, "width": 896, "frames": 121,
             "steps": 4, "seed": 42, "dense_attention": false, "out": "outputs/a.mp4"}, ...]
(every key but "prompt" optional; defaults from the command line)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from pih3_mlx.pipeline import PiH3Engine, save_mp4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?")
ap.add_argument("--batch", help="JSON list of jobs; models are loaded once")
ap.add_argument("--image", help="first frame")
ap.add_argument("--last-image", help="last frame")
ap.add_argument("--height", type=int)
ap.add_argument("--width", type=int)
ap.add_argument("--frames", type=int, default=121, help="5-15 s at 24 fps; rounded up to 17n+5")
ap.add_argument("--steps", type=int, default=4, choices=(2, 4),
                help="4 = final quality (use for people); 2 = fast draft only")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--transformer-bits", type=int, choices=(4, 8))
ap.add_argument("--text-encoder-bits", type=int, default=8, choices=(0, 4, 8))
ap.add_argument("--dense-attention", action="store_true", help="disable Sol sparse attention (slower, exact)")
ap.add_argument("--low-memory", action="store_true", help="load, use and free each model (slower, ~40 GB peak)")
ap.add_argument("--out", default="outputs/pih3.mp4")
a = ap.parse_args()
if not a.prompt and not a.batch:
    ap.error("give a prompt or --batch jobs.json")

jobs = json.load(open(a.batch)) if a.batch else [{"prompt": a.prompt}]
resident = not a.low_memory and len(jobs) > 1
engine = PiH3Engine(resident, resident, a.transformer_bits, a.text_encoder_bits or None)
for n, job in enumerate(jobs):
    get = lambda k, d: job.get(k, d)  # noqa: E731
    image, last = get("image", a.image), get("last_image", a.last_image)
    res = engine.generate(get("prompt", a.prompt), image=Image.open(image) if image else None,
                          last_image=Image.open(last) if last else None, height=get("height", a.height),
                          width=get("width", a.width), num_frames=get("frames", a.frames), steps=get("steps", a.steps),
                          seed=get("seed", a.seed), sparse_attention=not get("dense_attention", a.dense_attention))
    out = get("out", a.out if len(jobs) == 1 else os.path.splitext(a.out)[0] + f"_{n:03d}.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_mp4(res, out)
    print("wrote", out, flush=True)
