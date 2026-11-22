"""pi-H3 on MLX: text/image-to-video with audio in 2-4 steps.

~/.venvs/pih3/bin/python scripts/generate.py "A red fox trotting through snow" --frames 121 --height 512 --width 896
~/.venvs/pih3/bin/python scripts/generate.py "The cat meows" --image cat.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from pih3_mlx.pipeline import generate, save_mp4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("--image", help="first frame")
ap.add_argument("--last-image", help="last frame")
ap.add_argument("--height", type=int)
ap.add_argument("--width", type=int)
ap.add_argument("--frames", type=int, default=121, help="5-15 s at 24 fps; rounded up to 17n+5")
ap.add_argument("--steps", type=int, default=4, choices=(2, 4))
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--transformer-bits", type=int, choices=(4, 8))
ap.add_argument("--text-encoder-bits", type=int, default=8, choices=(0, 4, 8))
ap.add_argument("--dense-attention", action="store_true", help="disable Sol sparse attention (slower, exact)")
ap.add_argument("--out", default="outputs/pih3.mp4")
a = ap.parse_args()

res = generate(a.prompt, image=Image.open(a.image) if a.image else None,
               last_image=Image.open(a.last_image) if a.last_image else None, height=a.height, width=a.width,
               num_frames=a.frames, steps=a.steps, seed=a.seed, transformer_bits=a.transformer_bits,
               text_encoder_bits=a.text_encoder_bits or None,
               sparse_attention=not a.dense_attention)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
save_mp4(res, a.out)
print("wrote", a.out)
