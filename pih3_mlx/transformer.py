"""MiniMax H3 (pruned, adaln_rank=8) denoiser in MLX, with pi-H3 adapter fusion and DWD feature output.

References: `multimodalart/MiniMax-H3-Pruned/transformer/modeling_minimax_h3_pruned.py`,
`diffusers/models/transformers/transformer_minimax_h3.py`, `LakonLab/.../diffusers/minimax_h3.py`.

Batch size is always 1 (one packed request, guidance-distilled). The packed row order is
`[text | keyframe conditions | audio | target video]`, so every per-row quantity is applied on a handful of
contiguous runs instead of per-row gathers.
"""

import glob
import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .layout import MODALITY_NUM, Layout, adaln_runs

FP32_KEYS = ("time_table", "norm_out.", "dwd_norm_out.", "proj_in.", "audio_proj_in.", "proj_out.", "audio_proj_out.")
SEQ_CHUNK = 16384


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps):
        super().__init__()
        self.heads, self.head_dim, self.eps = heads, head_dim, eps
        inner = heads * head_dim
        self.to_q = nn.Linear(hidden, inner, bias=False)
        self.to_k = nn.Linear(hidden, inner, bias=False)
        self.to_v = nn.Linear(hidden, inner, bias=False)
        self.to_out = nn.Linear(inner, hidden, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)

    def __call__(self, x, rope=None):
        B, S, _ = x.shape
        shp = (B, S, self.heads, self.head_dim)
        q = mx.fast.rms_norm(self.to_q(x).reshape(shp), self.norm_q.weight, self.eps)
        k = mx.fast.rms_norm(self.to_k(x).reshape(shp), self.norm_k.weight, self.eps)
        v = self.to_v(x).reshape(shp)
        if rope is not None:
            q, k = _apply_rope(q, *rope), _apply_rope(k, *rope)
        o = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3), scale=self.head_dim ** -0.5)
        return self.to_out(o.transpose(0, 2, 1, 3).reshape(B, S, -1))


@mx.compile
def _apply_rope(x, cos, sin):
    r = cos.shape[-1]
    h = r // 2
    xr = x[..., :r]
    rot = mx.concatenate([-xr[..., h:], xr[..., :h]], axis=-1)
    return mx.concatenate([xr * cos + rot * sin, x[..., r:]], axis=-1)


@mx.compile
def _swiglu(h):
    f = h.shape[-1] // 2
    return h[..., :f] * nn.silu(h[..., f:])


class FeedForward(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.proj = nn.Linear(hidden, 2 * ffn, bias=False)
        self.out = nn.Linear(ffn, hidden, bias=False)

    def __call__(self, x):
        return self.out(_swiglu(self.proj(x)))


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps):
        super().__init__()
        self.eps = eps
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.ff = FeedForward(hidden, ffn)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class TokenRefiner(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, layers, eps, final_eps):
        super().__init__()
        self.refiner_blocks = [RefinerBlock(hidden, heads, head_dim, ffn, eps) for _ in range(layers)]
        self.final_norm = nn.RMSNorm(hidden, eps=final_eps)

    def __call__(self, x):
        for b in self.refiner_blocks:
            x = b(x)
        return self.final_norm(x)


class AdaLNProj(nn.Module):
    """Pruned AdaLN modulation: `linear(c) + folded_bias` over the rank-8 timestep coordinates (no silu)."""

    def __init__(self, rank, out):
        super().__init__()
        self.weight = mx.zeros((out, rank))
        self.bias = mx.zeros((out,))

    def __call__(self, temb):
        return temb @ self.weight.astype(mx.float32).T + self.bias.astype(mx.float32)


def _modulate(x, runs, scale, shift):
    """x (1, S, D); per run: x * (1 + scale[r]) + shift[r]."""
    parts = [x[:, s:e] * (1 + scale[r]) + shift[r] for s, e, r in runs]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)


def _gate(x, y, runs, gate):
    parts = [x[:, s:e] + gate[r] * y[:, s:e] for s, e, r in runs]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)


class TransformerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, rank, eps):
        super().__init__()
        self.hidden, self.eps = hidden, eps
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.ff = FeedForward(hidden, ffn)
        self.adaln_proj = AdaLNProj(rank, 6 * hidden * MODALITY_NUM)

    def __call__(self, x, temb, runs, rope):
        dtype = x.dtype
        mod = self.adaln_proj(temb).astype(dtype).reshape(-1, 6, self.hidden)  # rows: timestep * 3 + modality
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = (mod[:, i] for i in range(6))
        h = _modulate(mx.fast.rms_norm(x, self.norm1.weight, self.eps), runs, sc_a, sh_a)
        x = _gate(x, self.attn(h, rope), runs, g_a)
        mx.eval(x)
        out = []
        for i in range(0, x.shape[1], SEQ_CHUNK):  # the FFN is row-wise; chunk to bound the 2*ffn activation
            xi = x[:, i:i + SEQ_CHUNK]
            ri = [(max(s, i) - i, min(e, i + SEQ_CHUNK) - i, r) for s, e, r in runs if s < i + SEQ_CHUNK and e > i]
            hi = _modulate(mx.fast.rms_norm(xi, self.norm2.weight, self.eps), ri, sc_m, sh_m)
            out.append(_gate(xi, self.ff(hi), ri, g_m))
            mx.eval(out[-1])
        return out[0] if len(out) == 1 else mx.concatenate(out, axis=1)


class AdaLNOut(nn.Module):
    def __init__(self, hidden, rank, eps, silu_first=False, layer_norm=False, scale_first=False):
        super().__init__()
        self.eps, self.silu_first, self.layer_norm, self.scale_first = eps, silu_first, layer_norm, scale_first
        if not layer_norm:
            self.norm = nn.RMSNorm(hidden, eps=eps)
        self.weight = mx.zeros((2 * hidden, rank))
        self.bias = mx.zeros((2 * hidden,))

    def __call__(self, x, temb_row):
        c = nn.silu(temb_row) if self.silu_first else temb_row
        p = c @ self.weight.astype(mx.float32).T + self.bias.astype(mx.float32)
        a, b = mx.split(p, 2, axis=-1)
        scale, shift = (a, b) if self.scale_first else (b, a)
        xf = x.astype(mx.float32)
        n = mx.fast.layer_norm(xf, None, None, self.eps) if self.layer_norm else \
            mx.fast.rms_norm(xf, self.norm.weight.astype(mx.float32), self.eps)
        return n * (1 + scale) + shift


class H3Transformer(nn.Module):
    def __init__(self, cfg: dict, with_dwd: bool = True):
        super().__init__()
        self.cfg = cfg
        D, heads, hd, ffn = cfg["hidden_size"], cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["ffn_dim"]
        rank, eps = cfg["adaln_rank"], cfg["norm_eps"]
        pdim = cfg["in_channels"] * int(np.prod(cfg["patch_size"]))
        self.proj_in = nn.Linear(pdim, D)
        self.audio_proj_in = nn.Linear(cfg["audio_in_channels"], D)
        self.context_embedder = nn.Linear(cfg["text_dim"], D)
        self.time_table = mx.zeros((cfg["time_table_size"], rank))
        self.token_refiner = TokenRefiner(D, heads, hd, ffn, cfg["num_refiner_layers"], eps, cfg["final_norm_eps"])
        self.transformer_blocks = [TransformerBlock(D, heads, hd, ffn, rank, eps) for _ in range(cfg["num_layers"])]
        self.norm_out = AdaLNOut(D, rank, cfg["final_norm_eps"])
        self.proj_out = nn.Linear(D, pdim)
        self.audio_proj_out = nn.Linear(D, cfg["audio_in_channels"])
        if with_dwd:
            self.dwd_norm_out = AdaLNOut(D, rank, cfg["final_norm_eps"], silu_first=True, layer_norm=True,
                                         scale_first=True)
        inv = 1.0 / cfg["rope_theta"] ** (np.arange(0, 2 * cfg["rope_freq_dim"], 2) / (2 * cfg["rope_freq_dim"]))
        self._inv_freq = inv.astype(np.float32)

    def time_embed(self, t):
        """Linear interpolation in the (table_size, rank) coordinate table; t in [0, 1]."""
        table = self.time_table.astype(mx.float32)
        steps = table.shape[0] - 1
        pos = mx.clip(mx.array(t, mx.float32), 0, 1) * steps
        lo = mx.minimum(mx.floor(pos), steps - 1).astype(mx.int32)
        w = (pos - lo.astype(mx.float32))[:, None]
        return table[lo] + w * (table[lo + 1] - table[lo])

    def rope_tables(self, position_ids: np.ndarray, dtype):
        pos = position_ids.astype(np.float32)
        f = pos[:, :, None] * self._inv_freq[None, None]  # (S, 3, F)
        f = f.reshape(pos.shape[0], -1)
        f = np.concatenate([f, f], -1)
        return mx.array(np.cos(f))[None, :, None].astype(dtype), mx.array(np.sin(f))[None, :, None].astype(dtype)

    def encode_text(self, text_embeds):
        """context_embedder + token refiner; independent of the timestep, so computed once per request."""
        dtype = self.context_embedder.weight.dtype
        return self.token_refiner(self.context_embedder(text_embeds.astype(dtype)))

    def __call__(self, video_rows, audio_rows, text_hidden, timesteps, timestep_indices, layout: Layout, rope,
                 return_features=False):
        """
        video_rows: (cond + target video rows, 96), audio_rows: (audio rows, 32), text_hidden: (1, L, D) from
        `encode_text`; timesteps (K,) transformer t (= 1 - sigma); timestep_indices (S,) numpy.
        Returns raw head outputs (the pipeline's velocity is their negation) for the target video and audio rows,
        float32, plus the DWD features of the target video rows when requested.
        """
        dtype = text_hidden.dtype
        nc = layout.num_condition_video_rows
        v = self.proj_in(video_rows.astype(self.proj_in.weight.dtype)).astype(dtype)[None]
        a = self.audio_proj_in(audio_rows.astype(self.audio_proj_in.weight.dtype)).astype(dtype)[None]
        x = mx.concatenate([text_hidden, v[:, :nc], a, v[:, nc:]], axis=1)
        temb = self.time_embed(timesteps)
        runs = adaln_runs(timestep_indices * MODALITY_NUM + layout.token_tags)
        for blk in self.transformer_blocks:
            x = blk(x, temb, runs, rope)
        a0, v0 = layout.num_text + nc, layout.prefix_len
        xv, xa = x[0, v0:], x[0, a0:v0]
        tv, ta = temb[int(timestep_indices[v0])], temb[int(timestep_indices[a0])]
        hv = self.norm_out(xv, tv)
        out_v = hv @ self.proj_out.weight.astype(mx.float32).T + self.proj_out.bias.astype(mx.float32)
        ha = self.norm_out(xa, ta)
        out_a = ha @ self.audio_proj_out.weight.astype(mx.float32).T + self.audio_proj_out.bias.astype(mx.float32)
        feats = self.dwd_norm_out(xv, tv) if return_features else None
        return out_v, out_a, feats


# ---------------------------------------------------------------------------------------------- weights

def _rename(k: str) -> str | None:
    if k in ("adaln_basis", "adaln_mean") or k.startswith("dwd_pixel_refiner."):
        return None
    k = k.replace("time_embedder.table", "time_table")
    k = k.replace(".attn.to_out.0.", ".attn.to_out.").replace(".ff.net.0.proj.", ".ff.proj.").replace(
        ".ff.net.2.", ".ff.out.")
    for mod in ("adaln_proj", "norm_out", "dwd_norm_out"):
        k = k.replace(f"{mod}.folded_bias", f"{mod}.bias").replace(f"{mod}.linear.weight", f"{mod}.weight")
        k = k.replace(f"{mod}.linear.bias", f"{mod}.bias")
    return k


def sanitize(base: dict, adapter: dict | None = None, dtype=mx.bfloat16, lora_scale: float = 1.0) -> dict:
    """Merge the pruned base checkpoint with a pi-H3 adapter (full overrides + fused LoRA). Lazy arrays in, lazy
    arrays out; LoRA products are formed in float32."""
    out = {}
    for k, v in base.items():
        nk = _rename(k)
        if nk is not None:
            out[nk] = v
    lora = {}
    for k, v in (adapter or {}).items():
        if ".lora_A." in k or ".lora_B." in k:
            stem, ab = k.replace(".lora_A.weight", ""), "A" if ".lora_A." in k else "B"
            stem = stem.replace(".lora_B.weight", "")
            lora.setdefault(_rename(stem + ".weight"), {})[ab] = v
            continue
        nk = _rename(k)
        if nk is not None:
            out[nk] = v
    for nk, ab in lora.items():
        w = out[nk].astype(mx.float32) + lora_scale * (ab["B"].astype(mx.float32) @ ab["A"].astype(mx.float32))
        out[nk] = w
    for k in list(out):
        keep32 = k.startswith(FP32_KEYS) or k.endswith("adaln_proj.bias")
        out[k] = out[k].astype(mx.float32 if keep32 else dtype)
    return out


def load_safetensors_dir(path: str) -> dict:
    w = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        if not os.path.basename(f).startswith("._"):
            w.update(mx.load(f))
    return w


def build_transformer(base_dir: str, adapter_dir: str | None, dtype=mx.bfloat16, quantize_bits: int | None = None):
    """Build from the original HF checkpoints (slow path; `convert.py` saves the fused result)."""
    cfg = json.load(open(os.path.join(adapter_dir or base_dir, "config.json")))
    with_dwd = cfg.get("dwd_head_config") is not None
    model = H3Transformer(cfg, with_dwd=with_dwd)
    base = {k: v for k, v in load_safetensors_dir(base_dir).items()}
    adapter = load_safetensors_dir(adapter_dir) if adapter_dir else None
    weights = sanitize(base, adapter, dtype)
    _load_evaluated(model, weights)
    if quantize_bits:
        quantize(model, quantize_bits)
    return model, cfg, adapter


def _load_evaluated(model, weights: dict):
    """Load and evaluate block by block so the lazy graph (incl. LoRA products) never spans the whole model."""
    model.load_weights(list(weights.items()), strict=True)
    groups = {}
    for k in weights:
        parts = k.split(".")
        g = ".".join(parts[:2]) if parts[0] in ("transformer_blocks", "token_refiner") else parts[0]
        groups.setdefault(g, []).append(k)
    from mlx.utils import tree_flatten
    params = dict(tree_flatten(model.parameters()))
    for g, keys in groups.items():
        mx.eval([params[k] for k in keys])


def quantize(model, bits: int, group_size: int = 64):
    def pred(path, m):
        return isinstance(m, nn.Linear) and ("transformer_blocks" in path or "token_refiner" in path) \
            and m.weight.shape[-1] % group_size == 0
    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=pred)
