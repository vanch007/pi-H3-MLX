"""MiniMax-H3 video VAE *encoder* for keyframes (i2v / first+last-frame conditioning) on Apple MLX.

Reference: `diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py` (`AutoencoderKLMiniMaxH3`) and
`diffusers/modular_pipelines/minimax_h3/{before_encoder,encoders,modular_pipeline}.py`.

A keyframe is a single frame. The reference encodes it with the tiled spatial encoder alone (`_encode` ->
`_encode_clip`), and every causal 3D conv sees exactly one real frame preceded by `kernel_t - 1` zero frames, so the
3x3x3 kernels collapse exactly to their last temporal slice: this module is a 2D (NHWC) port of that path, which is
bit-for-bit the same math. Temporal stride-2 downsamplers also produce exactly one frame from one, with the same
last-slice kernel.

Pipeline contract (what `MiniMaxH3ResizeStep` + `MiniMaxH3KeyframeVaeEncoderStep` do):
  1. `prepare_keyframes(image, last_image, height, width)` -> canvas + fitted PIL keyframes (also the images the
     Qwen3-VL text encoder sees).
  2. `encode_keyframes(encoder, keyframes, height, width)` -> one `(1, 24, 1, H/16, W/16)` float32 latent each.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------------------------------------------
# Constants of the released checkpoint (modular_pipeline.py)
# ----------------------------------------------------------------------------------------------------------------
PIXEL_MEAN = (0.485, 0.456, 0.406)  # ImageNet; `components.pixel_mean`
PIXEL_STD = (0.229, 0.224, 0.225)  # `components.pixel_std`
KEYFRAME_ENCODE_SEED = 42  # `components.keyframe_encode_seed`
CANVAS_SHORT_EDGE = 768  # `canvas_short_edge` config
CANVAS_MAX_PIXELS = 768 * 1344  # `canvas_max_pixels` config
CANVAS_MULTIPLE = 32  # vae spatial ratio (16) * transformer patch (2)
MIN_ASPECT_RATIO = 1 / 4
MAX_ASPECT_RATIO = 4

# Reference tile geometry (AutoencoderKLMiniMaxH3.__init__; tiling is on by default and part of the released output).
TILE_SAMPLE_MIN_SIZE = 256
TILE_SAMPLE_MIN_OVERLAP = 64


# ----------------------------------------------------------------------------------------------------------------
# Keyframe preprocessing
# ----------------------------------------------------------------------------------------------------------------
def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    canvas_multiple: int = CANVAS_MULTIPLE,
    short_edge: int = CANVAS_SHORT_EDGE,
    max_pixels: int = CANVAS_MAX_PIXELS,
    min_aspect_ratio: float = MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> tuple[int, int]:
    """Port of `modular_pipeline.resolve_canvas_size`. Returns `(height, width)`."""
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")
    ratio = aspect_width / aspect_height
    if not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:{1 / min_aspect_ratio:g} to {max_aspect_ratio:g}:1, got "
            f"{aspect_width}:{aspect_height} ({ratio:g})."
        )
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    m = canvas_multiple
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = (array * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def fit_keyframe(keyframe: Image.Image, height: int, width: int, stretch: bool) -> Image.Image:
    """
    Put one keyframe onto a `height x width` canvas like `MiniMaxH3ResizeStep`:
    - same size: untouched;
    - `stretch=True` (the geometry anchor, i.e. the first keyframe): PIL `resize((w, h), LANCZOS)`;
    - `stretch=False` (the follower): cover-resize with *rounded* sizes, LANCZOS, then crop centred at
      `(resized - canvas) // 2` (the released model's arithmetic, not `VaeImageProcessor`'s crop mode).
    """
    if keyframe.size == (width, height):
        return keyframe
    if stretch:
        return keyframe.resize((width, height), Image.Resampling.LANCZOS)
    scale = max(width / keyframe.size[0], height / keyframe.size[1])
    resized_size = (max(width, round(keyframe.size[0] * scale)), max(height, round(keyframe.size[1] * scale)))
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    resized = keyframe.resize(resized_size, Image.Resampling.LANCZOS)
    return resized.crop((left, top, left + width, top + height))


def prepare_keyframes(
    image=None,
    last_image=None,
    height: int | None = None,
    width: int | None = None,
    canvas_multiple: int = CANVAS_MULTIPLE,
    canvas_short_edge: int = CANVAS_SHORT_EDGE,
    canvas_max_pixels: int = CANVAS_MAX_PIXELS,
) -> tuple[list[Image.Image], tuple[str, ...], int, int]:
    """
    Port of `MiniMaxH3ResizeStep.__call__`.

    Returns `(keyframes, anchors, height, width)`: the fitted PIL keyframes in packed order (first, then last),
    which end each one anchors (`"first"` / `"last"`), and the canvas. The canvas comes from the first present
    keyframe's aspect ratio unless `height` / `width` are given. Feed the same `keyframes` to the Qwen3-VL text
    encoder and to `encode_keyframes`.
    """
    pairs = [(a, _to_pil(k)) for a, k in (("first", image), ("last", last_image)) if k is not None]
    if not pairs:
        raise ValueError("At least one of `image` / `last_image` is required.")
    if (height is None) != (width is None):
        raise ValueError("`height` and `width` have to be passed together, or neither of them.")
    if height is None:
        height, width = resolve_canvas_size(
            *pairs[0][1].size, canvas_multiple, canvas_short_edge, canvas_max_pixels
        )
    keyframes = [fit_keyframe(k, height, width, stretch=(i == 0)) for i, (_, k) in enumerate(pairs)]
    return keyframes, tuple(a for a, _ in pairs), height, width


def keyframe_pixels(keyframe, pixel_mean=PIXEL_MEAN, pixel_std=PIXEL_STD) -> mx.array:
    """uint8 HWC keyframe -> ImageNet-normalized `(1, H, W, 3)` float32 (the `encode_vae_condition` recipe)."""
    array = np.asarray(keyframe)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"A keyframe must be (H, W, 3) RGB, got {array.shape}; convert('RGB') it first.")
    x = mx.array(array).astype(mx.float32)[None] / 255.0
    return (x - mx.array(pixel_mean, dtype=mx.float32)) / mx.array(pixel_std, dtype=mx.float32)


# ----------------------------------------------------------------------------------------------------------------
# Encoder modules (single frame, NHWC)
# ----------------------------------------------------------------------------------------------------------------
def _reflect_pad(x: mx.array, top: int, bottom: int, left: int, right: int) -> mx.array:
    """torch `F.pad(mode="reflect")` on the H/W axes of an NHWC tensor (pads <= 1 in this model)."""
    parts = []
    if top:
        parts.append(x[:, 1 : top + 1][:, ::-1])
    parts.append(x)
    if bottom:
        parts.append(x[:, -bottom - 1 : -1][:, ::-1])
    if len(parts) > 1:
        x = mx.concatenate(parts, axis=1)
    parts = []
    if left:
        parts.append(x[:, :, 1 : left + 1][:, :, ::-1])
    parts.append(x)
    if right:
        parts.append(x[:, :, -right - 1 : -1][:, :, ::-1])
    if len(parts) > 1:
        x = mx.concatenate(parts, axis=2)
    return x


class CausalConv(nn.Module):
    """`MiniMaxH3VideoCausalConv3d` applied to one frame: the last temporal kernel slice as a 2D conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, spatial_padding: int = 0):
        super().__init__()
        self.weight = mx.zeros((out_ch, kernel_size, kernel_size, in_ch))
        self.bias = mx.zeros((out_ch,))
        self.stride = stride
        self.spatial_padding = spatial_padding

    def __call__(self, x: mx.array) -> mx.array:
        p = self.spatial_padding
        if p:
            x = _reflect_pad(x, p, p, p, p)
        if self.weight.shape[1] == 1 and self.stride == 1:
            y = x @ self.weight.reshape(self.weight.shape[0], -1).T
        else:
            y = mx.conv2d(x, self.weight, stride=self.stride)
        return y + self.bias


@mx.compile
def _group_norm_silu(x: mx.array, weight: mx.array, bias: mx.array, groups: mx.array, eps: mx.array) -> mx.array:
    n, h, w, c = x.shape
    g = groups.shape[0]
    xf = x.astype(mx.float32).reshape(n, h * w, g, c // g)
    mean = xf.mean(axis=(1, 3), keepdims=True)
    var = mx.square(xf - mean).mean(axis=(1, 3), keepdims=True)
    y = ((xf - mean) * mx.rsqrt(var + eps)).reshape(n, h, w, c)
    y = y * weight.astype(mx.float32) + bias.astype(mx.float32)
    return (y * mx.sigmoid(y)).astype(x.dtype)


class GroupNorm(nn.Module):
    """Per-frame GroupNorm (float32 statistics) fused with the SiLU that always follows it."""

    def __init__(self, num_groups: int, channels: int, eps: float):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))
        self._groups = mx.zeros((num_groups,))  # shape-only carrier so mx.compile keys on it
        self._eps = mx.array(eps, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:  # returns silu(norm(x))
        return _group_norm_silu(x, self.weight, self.bias, self._groups, self._eps)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int, eps: float):
        super().__init__()
        self.norm1 = GroupNorm(groups, in_ch, eps)
        self.conv1 = CausalConv(in_ch, out_ch, 3, spatial_padding=1)
        self.norm2 = GroupNorm(groups, out_ch, eps)
        self.conv2 = CausalConv(out_ch, out_ch, 3, spatial_padding=1)
        if in_ch != out_ch:
            self.conv_shortcut = CausalConv(in_ch, out_ch, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(self.norm1(x))
        h = self.conv2(self.norm2(h))
        if "conv_shortcut" in self:
            x = self.conv_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, ch: int, spatial_stride: int):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.conv = CausalConv(ch, ch, 3, stride=spatial_stride, spatial_padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        if self.spatial_stride == 2:
            x = _reflect_pad(x, 0, 1, 0, 1)
        # (A temporal-only downsampler has no spatial padding and shrinks H/W by 2, exactly like the reference.)
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_layers, t_factor, s_factor, groups, eps):
        super().__init__()
        self.resnets = [ResnetBlock(in_ch if i == 0 else out_ch, out_ch, groups, eps) for i in range(num_layers)]
        if t_factor * s_factor > 1:
            self.downsamplers = [Downsample(out_ch, s_factor)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if "downsamplers" in self:
            for d in self.downsamplers:
                x = d(x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        boc = list(cfg["block_out_channels"])
        groups, eps = cfg["norm_num_groups"], cfg["norm_eps"]
        if cfg.get("spatial_padding_mode", "reflect") != "reflect":
            raise NotImplementedError("Only `spatial_padding_mode='reflect'` (the released checkpoint) is ported.")
        self.conv_in = CausalConv(cfg["in_channels"], boc[0], 3, spatial_padding=1)
        block_in = [boc[0]] + boc[:-1]
        self.down_blocks = [
            DownBlock(
                block_in[i],
                boc[i],
                cfg["layers_per_block"],
                cfg["temporal_downsample_factors"][i],
                cfg["spatial_downsample_factors"][i],
                groups,
                eps,
            )
            for i in range(len(boc))
        ]
        self.norm_out = GroupNorm(groups, boc[-1], eps)
        self.conv_out = CausalConv(boc[-1], 2 * cfg["latent_channels"], 3, spatial_padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
            mx.eval(x)  # bound the graph / peak memory per level
        return self.conv_out(self.norm_out(x))


DEFAULT_CONFIG = {
    "in_channels": 3,
    "latent_channels": 24,
    "block_out_channels": [128, 256, 256, 512, 512, 1024],
    "layers_per_block": 2,
    "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
    "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
    "norm_num_groups": 32,
    "norm_eps": 1e-6,
    "spatial_padding_mode": "reflect",
    "latents_mean": [0.0] * 24,
    "latents_std": [1.0] * 24,
}


class VideoVAEEncoder(nn.Module):
    """`encoder` + `quant_conv` of `AutoencoderKLMiniMaxH3`, single-frame, with the reference spatial tiling."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.config = {k: cfg[k] for k in DEFAULT_CONFIG}
        self.encoder = Encoder(cfg)
        self.quant_conv = CausalConv(2 * cfg["latent_channels"], 2 * cfg["latent_channels"], 1)
        self.spatial_compression_ratio = math.prod(cfg["spatial_downsample_factors"])
        self.latent_channels = cfg["latent_channels"]
        self._latents_mean = np.asarray(cfg["latents_mean"], dtype=np.float32)
        self._latents_std = np.asarray(cfg["latents_std"], dtype=np.float32)
        self.use_tiling = True
        self.tile_sample_min_height = TILE_SAMPLE_MIN_SIZE
        self.tile_sample_min_width = TILE_SAMPLE_MIN_SIZE
        self.tile_sample_min_overlap_height = TILE_SAMPLE_MIN_OVERLAP
        self.tile_sample_min_overlap_width = TILE_SAMPLE_MIN_OVERLAP
        self.tile_batch_size = 8  # tiles run as one batch (GroupNorm is per sample, so batching is exact)

    @property
    def dtype(self) -> mx.Dtype:
        return self.encoder.conv_in.weight.dtype

    def encode_tile(self, x: mx.array) -> mx.array:
        return self.quant_conv(self.encoder(x.astype(self.dtype))).astype(mx.float32)

    # --- tiling: exact port of AutoencoderKLMiniMaxH3._split_tiles / _blend / _stitch_tiles / _encode_clip -------
    def _split_tiles(self, length: int, tile_size: int, min_overlap: int):
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio
        starts = [0]
        for i in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[i])
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a: mx.array, b: mx.array, blend_extent: int, axis: int) -> mx.array:
        blend_extent = min(a.shape[axis], b.shape[axis], blend_extent)
        shape = [1] * a.ndim
        shape[axis] = blend_extent
        pos = mx.arange(blend_extent, dtype=b.dtype)
        w_a = (1 - pos / blend_extent).reshape(shape)
        w_b = (pos / blend_extent).reshape(shape)
        sa = [slice(None)] * a.ndim
        sa[axis] = slice(-blend_extent, None)
        sb = [slice(None)] * b.ndim
        sb[axis] = slice(0, blend_extent)
        blended = a[tuple(sa)] * w_a + b[tuple(sb)] * w_b
        if blend_extent == b.shape[axis]:
            return blended
        sr = [slice(None)] * b.ndim
        sr[axis] = slice(blend_extent, None)
        return mx.concatenate([blended, b[tuple(sr)]], axis=axis)

    def _stitch(self, tiles, h_overlaps, w_overlaps) -> mx.array:  # NHWC: H axis 1, W axis 2
        rows = []
        for i, row in enumerate(tiles):
            out_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, h_overlaps[i - 1], axis=1)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, w_overlaps[j - 1], axis=2)
                if i < len(tiles) - 1:
                    tile = tile[:, : -h_overlaps[i]]
                if j < len(row) - 1:
                    tile = tile[:, :, : -w_overlaps[j]]
                out_row.append(tile)
            rows.append(mx.concatenate(out_row, axis=2))
        return mx.concatenate(rows, axis=1)

    def encode_moments(self, x: mx.array) -> mx.array:
        """ImageNet-normalized `(1, H, W, 3)` pixels -> posterior moments `(1, H/16, W/16, 2*latent_channels)`."""
        if not self.use_tiling:
            return self.encode_tile(x)
        height, width = x.shape[1], x.shape[2]
        ys, yl, yo = self._split_tiles(height, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        xs, xl, xo = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)
        crops = [x[:, i : i + ih, j : j + jw] for i, ih in zip(ys, yl) for j, jw in zip(xs, xl)]
        encoded: list[mx.array] = []
        k = 0
        while k < len(crops):  # batch equally-shaped consecutive tiles
            group = [crops[k]]
            while (
                k + len(group) < len(crops)
                and len(group) < self.tile_batch_size
                and crops[k + len(group)].shape == crops[k].shape
            ):
                group.append(crops[k + len(group)])
            out = self.encode_tile(mx.concatenate(group, axis=0) if len(group) > 1 else group[0])
            mx.eval(out)
            encoded.extend(out[n : n + 1] for n in range(len(group)))
            k += len(group)
        tiles = [encoded[r * len(xs) : (r + 1) * len(xs)] for r in range(len(ys))]
        r = self.spatial_compression_ratio
        return self._stitch(tiles, [o // r for o in yo], [o // r for o in xo])

    def __call__(self, x: mx.array) -> mx.array:
        return self.encode_moments(x)


# ----------------------------------------------------------------------------------------------------------------
# Posterior sampling (encode_vae_condition)
# ----------------------------------------------------------------------------------------------------------------
_warned_no_torch = False


def posterior_noise(shape_nc1hw: tuple[int, ...], seed: int = KEYFRAME_ENCODE_SEED) -> mx.array:
    """
    The exact noise `DiagonalGaussianDistribution.sample(generator=torch.Generator().manual_seed(seed))` draws:
    `torch.randn(shape, generator=cpu_gen, dtype=float32)` in `(1, C, 1, h, w)` layout, returned as NHWC `(1, h, w, C)`.
    Falls back to `mx.random` (not bit-identical to the reference) when torch is missing.
    """
    global _warned_no_torch
    try:
        import torch
    except ImportError:
        if not _warned_no_torch:
            warnings.warn("torch is not importable: keyframe posterior noise uses mx.random and will not match the "
                          "reference bit-for-bit.")
            _warned_no_torch = True
        noise = mx.random.normal(shape_nc1hw, key=mx.random.key(seed))
    else:
        noise = mx.array(torch.randn(shape_nc1hw, generator=torch.Generator().manual_seed(seed),
                                     dtype=torch.float32).numpy())
    return noise[:, :, 0].transpose(0, 2, 3, 1)


def encode_keyframe(encoder: VideoVAEEncoder, keyframe, seed: int = KEYFRAME_ENCODE_SEED) -> mx.array:
    """`encode_vae_condition` for one fitted keyframe -> normalized `(1, C, 1, h, w)` float32 latents."""
    moments = encoder.encode_moments(keyframe_pixels(keyframe))
    c = encoder.latent_channels
    mean, logvar = moments[..., :c], mx.clip(moments[..., c:], -30.0, 20.0)
    std = mx.exp(0.5 * logvar)
    _, h, w, _ = mean.shape
    latents = mean + std * posterior_noise((1, c, 1, h, w), seed)
    latents = latents.astype(mx.float16).astype(mx.float32)
    latents = (latents - mx.array(encoder._latents_mean)) / mx.array(encoder._latents_std)
    out = latents.transpose(0, 3, 1, 2)[:, :, None]  # (1, C, 1, h, w)
    mx.eval(out)
    return out


def encode_keyframes(
    encoder: VideoVAEEncoder, images: Sequence, height: int, width: int, seed: int = KEYFRAME_ENCODE_SEED
) -> list[mx.array]:
    """
    Port of `MiniMaxH3KeyframeVaeEncoderStep`: one `(1, latent_channels, 1, height // 16, width // 16)` float32
    latent per keyframe, in packed order. `images` should be `prepare_keyframes` output; any keyframe not already
    `height x width` is fitted with the same rule (first stretched, followers cover-cropped).
    """
    fitted = [fit_keyframe(_to_pil(im), height, width, stretch=(i == 0)) for i, im in enumerate(images)]
    return [encode_keyframe(encoder, im, seed) for im in fitted]


# ----------------------------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------------------------
ENCODER_PREFIXES = ("encoder.", "quant_conv.")


def sanitize(weights: dict) -> dict:
    """
    Original `AutoencoderKLMiniMaxH3` keys -> `VideoVAEEncoder` keys. Keeps only `encoder.*` / `quant_conv.*`; conv
    weights `(O, I, kt, kh, kw)` -> last temporal slice `(O, kh, kw, I)` (exact for a single causal frame).
    """
    out = {}
    for key, value in weights.items():
        if not key.startswith(ENCODER_PREFIXES):
            continue
        if key.endswith(".weight") and value.ndim == 5:
            value = value[:, :, -1].transpose(0, 2, 3, 1)
        out[key] = value
    return out


def _shards_for_encoder(model_dir: Path) -> list[Path]:
    index = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        names = sorted({f for k, f in weight_map.items() if k.startswith(ENCODER_PREFIXES)})
        return [model_dir / n for n in names]
    single = model_dir / "diffusion_pytorch_model.safetensors"
    if single.exists():
        return [single]
    return sorted(p for p in model_dir.glob("*.safetensors") if not p.name.startswith("._"))


def load_video_vae_encoder(model_dir: str | Path, dtype: mx.Dtype = mx.float32) -> VideoVAEEncoder:
    """
    Build the encoder from `<model_dir>/config.json` (the HF `vae/` folder) and lazily load only the encoder +
    quant_conv tensors from the shards that hold them. The reference pins the encoder to float32
    (`_keep_in_fp32_modules`), so float32 is the parity default; bfloat16 is faster at ~1e-2 relative error.
    """
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    encoder = VideoVAEEncoder(config)
    weights: dict = {}
    for shard in _shards_for_encoder(model_dir):
        loaded = mx.load(str(shard))
        weights.update({k: v for k, v in loaded.items() if k.startswith(ENCODER_PREFIXES)})
        del loaded
    weights = {k: v.astype(dtype) for k, v in sanitize(weights).items()}
    encoder.load_weights(list(weights.items()), strict=True)
    mx.eval(encoder.parameters())
    return encoder
