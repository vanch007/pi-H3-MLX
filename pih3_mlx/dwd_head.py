"""SolH3DWDHead in MLX: decodes pi-H3's final-layer features straight to pixels (DWD, no VAE decoder).

Port of `LakonLab/lakonlab/models/architectures/sol_h3/{model,sol_h3}.py`. The reference packs windows into varlen
FlashAttention rows; here the windows are padded dense tiles built with reshapes only, with a boolean key mask, which
is the same math (padded query rows are discarded) and maps onto `mx.fast.scaled_dot_product_attention`.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

NUM_HEADS, HEAD_DIM = 32, 64
DIM = NUM_HEADS * HEAD_DIM
FFN_HIDDEN = DIM * 4
NUM_LAYERS = 24
NORM_EPS = 1e-5
NUM_REGISTER_TOKENS = 4
ROPE_THETA = 100.0
ROPE_DIM = int(HEAD_DIM * 0.75)  # 48 rotated channels: 24 angle pairs
WINDOW, BLOCK, T_SHIFT, TIME_NORM = 16, 5, 2, 5
PATCH, PATCH_T, OUT_CH, FRAMES_TO_TRIM = 16, 4, 3, 3
S_GLOBAL = NUM_REGISTER_TOKENS + 1
FFN_CHUNK = 65536


def _rope_tables(T, H, W, s, st, dtype):
    """cos/sin (T, H, W, 1, 24) of window-local (t, y, x) ids for shifts (s, st)."""
    lt = ((np.arange(T) + st) % BLOCK + 0.5) / TIME_NORM * 2 - 1
    ly = ((np.arange(H) + s) % WINDOW + 0.5) / WINDOW * 2 - 1
    lx = ((np.arange(W) + s) % WINDOW + 0.5) / WINDOW * 2 - 1
    grid = np.stack(np.meshgrid(lt, ly, lx, indexing="ij"), -1).astype(np.float32)  # (T, H, W, 3)
    inv_freq = (1.0 / ROPE_THETA ** np.arange(0, 1, 6 / ROPE_DIM, dtype=np.float32)).astype(np.float32)
    ang = (2.0 * math.pi * grid[..., None] * inv_freq).reshape(T, H, W, 1, -1).astype(np.float32)
    return mx.array(np.cos(ang)).astype(dtype), mx.array(np.sin(ang)).astype(dtype)


def _rotate(x, cos, sin):
    """x (T, H, W, heads, 64): rotate channels [0:24] against [24:48]; [48:64] pass through."""
    half = cos.shape[-1]
    a, b, rest = x[..., :half], x[..., half:2 * half], x[..., 2 * half:]
    return mx.concatenate([a * cos - b * sin, b * cos + a * sin, rest], axis=-1)


class SolH3Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(DIM, 3 * DIM)
        self.to_out = nn.Linear(DIM, DIM)


class SolH3SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, 2 * FFN_HIDDEN)
        self.w2 = nn.Linear(FFN_HIDDEN, DIM)

    def __call__(self, x):
        h = self.w1(x)
        return self.w2(nn.silu(h[..., :FFN_HIDDEN]) * h[..., FFN_HIDDEN:])


class _Geometry:
    """Window / global attention plumbing shared by all layers with the same shift, built once per shape."""

    def __init__(self, T, H, W, shifted, dtype):
        s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
        self.s, self.st = s, st
        self.Tp = -(-(T + st) // BLOCK) * BLOCK
        self.Hp = -(-(H + s) // WINDOW) * WINDOW
        self.Wp = -(-(W + s) // WINDOW) * WINDOW
        self.nt, self.nh, self.nw = self.Tp // BLOCK, self.Hp // WINDOW, self.Wp // WINDOW
        self.cos, self.sin = _rope_tables(T, H, W, s, st, dtype)
        valid = np.zeros((self.Tp, self.Hp, self.Wp), bool)
        valid[st:st + T, s:s + H, s:s + W] = True
        vw = valid.reshape(self.nt, BLOCK, self.nh, WINDOW, self.nw, WINDOW).transpose(0, 2, 4, 1, 3, 5)
        vw = vw.reshape(self.nt, self.nh, self.nw, -1)
        # global tokens of the native blocks each temporal window overlaps (second one only if different)
        pairs = [(max(j * BLOCK - st, 0) // BLOCK, (min((j + 1) * BLOCK - st, T) - 1) // BLOCK) for j in range(self.nt)]
        self.g_blocks = mx.array(np.array([[b0, b1] for b0, b1 in pairs], np.int32))
        gv = np.array([[True] * S_GLOBAL + [b1 != b0] * S_GLOBAL for b0, b1 in pairs])
        gv = np.broadcast_to(gv[:, None, None], (self.nt, self.nh, self.nw, 2 * S_GLOBAL))
        mask = np.concatenate([vw, gv], -1).reshape(-1, 1, 1, vw.shape[-1] + 2 * S_GLOBAL)
        self.window_mask = mx.array(mask)
        nb = -(-T // BLOCK)
        tok_valid = np.arange(nb * BLOCK).reshape(nb, BLOCK) < T
        gmask = np.concatenate([np.repeat(tok_valid, H * W, axis=1), np.ones((nb, S_GLOBAL), bool)], 1)
        self.global_mask = mx.array(gmask[:, None, None, :])
        self.T, self.H, self.W, self.nb = T, H, W, nb

    def windows(self, x):
        """(T, H, W, heads, dh) -> (nW, heads, BLOCK*WINDOW*WINDOW, dh), zero padded."""
        T, H, W = self.T, self.H, self.W
        x = mx.pad(x, [(self.st, self.Tp - T - self.st), (self.s, self.Hp - H - self.s),
                       (self.s, self.Wp - W - self.s), (0, 0), (0, 0)])
        x = x.reshape(self.nt, BLOCK, self.nh, WINDOW, self.nw, WINDOW, NUM_HEADS, HEAD_DIM)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(self.nt * self.nh * self.nw, NUM_HEADS, BLOCK * WINDOW * WINDOW, HEAD_DIM)

    def unwindow(self, o):
        """(nW, heads, L, dh) -> (T*H*W, heads*dh) in token order."""
        o = o.reshape(self.nt, self.nh, self.nw, NUM_HEADS, BLOCK, WINDOW, WINDOW, HEAD_DIM)
        o = o.transpose(0, 4, 1, 5, 2, 6, 3, 7).reshape(self.Tp, self.Hp, self.Wp, DIM)
        o = o[self.st:self.st + self.T, self.s:self.s + self.H, self.s:self.s + self.W]
        return o.reshape(-1, DIM)

    def window_globals(self, gk):
        """(nb, S, heads, dh) -> (nW, heads, 2S, dh): the global keys/values every window appends."""
        g = mx.take(gk, self.g_blocks.reshape(-1), axis=0).reshape(self.nt, 1, 1, 2 * S_GLOBAL, NUM_HEADS, HEAD_DIM)
        g = mx.broadcast_to(g, (self.nt, self.nh, self.nw, 2 * S_GLOBAL, NUM_HEADS, HEAD_DIM))
        return g.reshape(-1, 2 * S_GLOBAL, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)


class SolH3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(DIM, eps=NORM_EPS)
        self.norm2 = nn.RMSNorm(DIM, eps=NORM_EPS)
        self.scale1 = mx.zeros((DIM,))
        self.scale2 = mx.zeros((DIM,))
        self.attn = SolH3Attention()
        self.ff = SolH3SwiGLU()

    def __call__(self, x, g, geo: _Geometry):
        """x (N, D) tokens in (T, H, W) order, g (nb*S, D) global tokens."""
        N, dtype = x.shape[0], x.dtype
        scale = HEAD_DIM ** -0.5
        h = mx.fast.rms_norm(mx.concatenate([x, g]).astype(mx.float32), self.norm1.weight.astype(mx.float32),
                             NORM_EPS).astype(dtype)
        qkv = self.attn.to_qkv(h).reshape(-1, NUM_HEADS, 3, HEAD_DIM)
        q = mx.fast.rms_norm(qkv[:, :, 0].astype(mx.float32), None, NORM_EPS).astype(dtype)
        k = mx.fast.rms_norm(qkv[:, :, 1].astype(mx.float32), None, NORM_EPS).astype(dtype)
        v = qkv[:, :, 2]
        shp = (geo.T, geo.H, geo.W, NUM_HEADS, HEAD_DIM)
        qx = _rotate(q[:N].reshape(shp), geo.cos, geo.sin)
        kx = _rotate(k[:N].reshape(shp), geo.cos, geo.sin)
        vx = v[:N].reshape(shp)
        gq = q[N:].reshape(geo.nb, S_GLOBAL, NUM_HEADS, HEAD_DIM)
        gk = k[N:].reshape(geo.nb, S_GLOBAL, NUM_HEADS, HEAD_DIM)
        gv = v[N:].reshape(geo.nb, S_GLOBAL, NUM_HEADS, HEAD_DIM)

        # global tokens attend over every token of their native block plus the block's own global tokens
        pad_t = geo.nb * BLOCK - geo.T
        kb = mx.pad(kx, [(0, pad_t), (0, 0), (0, 0), (0, 0), (0, 0)]).reshape(geo.nb, -1, NUM_HEADS, HEAD_DIM)
        vb = mx.pad(vx, [(0, pad_t), (0, 0), (0, 0), (0, 0), (0, 0)]).reshape(geo.nb, -1, NUM_HEADS, HEAD_DIM)
        og = mx.fast.scaled_dot_product_attention(
            gq.transpose(0, 2, 1, 3), mx.concatenate([kb, gk], 1).transpose(0, 2, 1, 3),
            mx.concatenate([vb, gv], 1).transpose(0, 2, 1, 3), scale=scale, mask=geo.global_mask)
        og = og.transpose(0, 2, 1, 3).reshape(-1, DIM)

        # tokens attend within their (5 latent x 16 x 16) window plus the overlapping blocks' global tokens
        o = mx.fast.scaled_dot_product_attention(
            geo.windows(qx), mx.concatenate([geo.windows(kx), geo.window_globals(gk)], 2),
            mx.concatenate([geo.windows(vx), geo.window_globals(gv)], 2), scale=scale, mask=geo.window_mask)
        o = geo.unwindow(o)
        x = x + self.attn.to_out(o) * self.scale1
        g = g + self.attn.to_out(og) * self.scale1
        return x, g

    def ffn(self, x):
        w = self.norm2.weight.astype(mx.float32)
        outs = []
        for i in range(0, x.shape[0], FFN_CHUNK):
            xi = x[i:i + FFN_CHUNK]
            hi = mx.fast.rms_norm(xi.astype(mx.float32), w, NORM_EPS).astype(xi.dtype)
            outs.append(xi + self.ff(hi) * self.scale2)
        return mx.concatenate(outs) if len(outs) > 1 else outs[0]


class SolH3DWDHead(nn.Module):
    def __init__(self, in_channels: int = 5376, src_patch: tuple = (32, 32), num_layers: int = 8):
        super().__init__()
        self.expansion = (src_patch[0] // PATCH, src_patch[1] // PATCH)
        self.feature_expand = nn.Linear(in_channels, DIM * self.expansion[0] * self.expansion[1])
        self.register_tokens = mx.zeros((1, NUM_REGISTER_TOKENS, DIM))
        self.blocks = [SolH3Block() for _ in range(num_layers)]
        self.norm_out = nn.LayerNorm(DIM, eps=NORM_EPS)
        self.proj_out = nn.Linear(DIM, OUT_CH * PATCH_T * PATCH * PATCH)
        self.first_frame_proj_out = nn.Linear(DIM, OUT_CH * (PATCH_T - FRAMES_TO_TRIM) * PATCH * PATCH)

    def __call__(self, features, token_grid):
        """features (t*h*w, in_channels) on a (t, h, w) token grid -> pixels (F, H*32... , 3) as (F, Hpix, Wpix, 3),
        in Sol's ImageNet-normalized RGB (apply `color_decode`)."""
        t, h, w = token_grid
        eh, ew = self.expansion
        dtype = self.register_tokens.dtype
        x = self.feature_expand(features.astype(dtype))
        x = x.reshape(t, h, w, eh, ew, DIM).transpose(0, 1, 3, 2, 4, 5).reshape(t, h * eh, w * ew, DIM)
        T, H, W = t, h * eh, w * ew
        nb = -(-T // BLOCK)
        g0 = mx.concatenate([self.register_tokens[0], mx.zeros((1, DIM), dtype)])
        g = mx.broadcast_to(g0[None], (nb, S_GLOBAL, DIM)).reshape(-1, DIM)
        x = x.reshape(-1, DIM)
        geos = {}
        first_index = NUM_LAYERS - len(self.blocks)
        for i, block in enumerate(self.blocks):
            shifted = (first_index + i) % 2 == 1
            if shifted not in geos:
                geos[shifted] = _Geometry(T, H, W, shifted, dtype)
            x, g = block(x, g, geos[shifted])
            x = block.ffn(x)
            g = block.ffn(g)
            mx.eval(x, g)
        return self.project(x.reshape(T, H, W, DIM))

    def project(self, x):
        """(T, H, W, D) -> (F, 16H, 16W, 3): one 1-frame + four 4-frame projections per 5-latent chunk."""
        T, H, W, _ = x.shape
        frames = []
        for c in range(0, T, BLOCK):
            xc = self.norm_out(x[c:c + BLOCK])
            first = self.first_frame_proj_out(xc[:1])  # (1, H, W, 3*1*16*16)
            first = first.reshape(H, W, OUT_CH, 1, PATCH, PATCH).transpose(3, 0, 4, 1, 5, 2)
            frames.append(first.reshape(1, H * PATCH, W * PATCH, OUT_CH))
            if xc.shape[0] > 1:
                n = xc.shape[0] - 1
                rest = self.proj_out(xc[1:]).reshape(n, H, W, OUT_CH, PATCH_T, PATCH, PATCH)
                rest = rest.transpose(0, 4, 1, 5, 2, 6, 3).reshape(n * PATCH_T, H * PATCH, W * PATCH, OUT_CH)
                frames.append(rest)
            mx.eval(frames[-1])
        return mx.concatenate(frames)


COLOR_MEAN = (-0.03, -0.088, -0.188)
COLOR_STD = (0.458, 0.448, 0.45)


def color_decode(pixels):
    """Sol ImageNet-normalized RGB (..., 3) -> [0, 1] (RGBColorEncoder affine decode, clamp, then (x+1)/2)."""
    x = pixels.astype(mx.float32) * mx.array(COLOR_STD) + mx.array(COLOR_MEAN)
    return (mx.clip(x, -1, 1) + 1) * 0.5


def sanitize(weights: dict, prefix: str = "dwd_pixel_refiner.") -> dict:
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
