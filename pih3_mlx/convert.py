"""Fuse the pruned H3 base transformer with a pi-H3 adapter and save MLX weights (optionally quantized).

python -m pih3_mlx.convert --steps 4 [--bits 8]
Writes `models_mlx/pi_h3_<N>step_dwd[_q<bits>]/{config.json, transformer-*.safetensors, dwd_head.safetensors}`.
"""

import argparse
import json
import os
import time

import mlx.core as mx
from mlx.utils import tree_flatten

from . import dwd_head
from .transformer import H3Transformer, build_transformer, quantize

# Weights root holding `models/` (original HF weights) and `models_mlx/` (converted). Defaults to the folder that
# contains this repository; set PIH3_ROOT to a `hf download vanch007/pi-H3-MLX --local-dir <dir>` folder instead.
ROOT = os.environ.get("PIH3_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "models")
SHARD_BYTES = 4 << 30


def mlx_model_dir(steps: int, bits: int | None = None, dwd: bool = True) -> str:
    name = f"pi_h3_{steps}step{'_dwd' if dwd else ''}" + (f"_q{bits}" if bits else "")
    return os.path.join(ROOT, "models_mlx", name)


def save_sharded(weights: dict, out_dir: str, prefix: str):
    shard, size, idx = {}, 0, 0
    items = list(weights.items())
    for i, (k, v) in enumerate(items):
        shard[k] = v
        size += v.nbytes
        if size >= SHARD_BYTES or i == len(items) - 1:
            mx.save_safetensors(os.path.join(out_dir, f"{prefix}-{idx:05d}.safetensors"), shard)
            shard, size, idx = {}, 0, idx + 1


def convert(steps: int, bits: int | None, dwd: bool = True):
    t0 = time.time()
    adapter = os.path.join(MODELS, "pi-H3", f"pi_h3_{steps}step{'_dwd' if dwd else ''}_adapter")
    model, cfg, adapter_w = build_transformer(os.path.join(MODELS, "MiniMax-H3-Pruned", "transformer"), adapter,
                                              quantize_bits=bits)
    out = mlx_model_dir(steps, bits, dwd)
    os.makedirs(out, exist_ok=True)
    cfg = dict(cfg, mlx_quantization=dict(bits=bits, group_size=64) if bits else None)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    save_sharded(dict(tree_flatten(model.parameters())), out, "transformer")
    if dwd:
        head = {k: v.astype(mx.bfloat16) for k, v in dwd_head.sanitize(adapter_w).items()}
        mx.save_safetensors(os.path.join(out, "dwd_head.safetensors"), head)
    print(f"saved {out} in {time.time() - t0:.0f}s")


def load_converted(model_dir: str):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    model = H3Transformer(cfg, with_dwd=cfg.get("dwd_head_config") is not None)
    q = cfg.get("mlx_quantization")
    if q:
        quantize(model, q["bits"], q["group_size"])
    files = sorted(f for f in os.listdir(model_dir) if f.startswith("transformer-") and f.endswith(".safetensors"))
    for f in files:
        model.load_weights(list(mx.load(os.path.join(model_dir, f)).items()), strict=False)
        mx.eval(model.parameters())
    head = None
    hp = os.path.join(model_dir, "dwd_head.safetensors")
    if os.path.exists(hp):
        head = dwd_head.SolH3DWDHead(cfg["hidden_size"])
        head.load_weights(list(mx.load(hp).items()), strict=True)
        mx.eval(head.parameters())
    return model, head, cfg


def text_encoder_dir(bits: int | None = 8) -> str:
    return os.path.join(ROOT, "models_mlx", f"text_encoder_q{bits}" if bits else "text_encoder_bf16")


def convert_text_encoder(bits: int | None = 8):
    """Pre-quantize the Qwen3-VL conditioner once, so generation loads ~34 GB of 8-bit weights instead of reading
    ~64 GB of bf16 and quantizing on every run."""
    from .text_encoder import load_text_encoder, save_text_encoder

    t0 = time.time()
    enc = load_text_encoder(os.path.join(MODELS, "MiniMax-H3-Pruned", "text_encoder"),
                            os.path.join(MODELS, "MiniMax-H3", "tokenizer"),
                            os.path.join(MODELS, "MiniMax-H3", "processor"), quantize_bits=bits)
    save_text_encoder(enc, text_encoder_dir(bits), bits)
    print(f"saved {text_encoder_dir(bits)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4, choices=(2, 4))
    ap.add_argument("--bits", type=int, default=None, choices=(4, 8))
    ap.add_argument("--text-encoder", action="store_true",
                    help="convert the text encoder instead (--bits defaults to 8 here; 0 keeps bf16)")
    a = ap.parse_args()
    if a.text_encoder:
        convert_text_encoder(8 if a.bits is None else (a.bits or None))
    else:
        convert(a.steps, a.bits)
