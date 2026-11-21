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

ROOT = "/Volumes/1T ssd/pi-h3-mlx"
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4, choices=(2, 4))
    ap.add_argument("--bits", type=int, default=None, choices=(4, 8))
    a = ap.parse_args()
    convert(a.steps, a.bits)
