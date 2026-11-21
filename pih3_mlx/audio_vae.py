"""MLX port of the MiniMax-H3 audio VAE **decoder** (`AutoencoderKLMiniMaxH3Audio.decode`) and the pipeline glue
around it (`MiniMaxH3AfterDenoiseStep` audio unpacking + `MiniMaxH3AudioDecodeStep` denormalization).

Reference: `diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py`,
`diffusers/modular_pipelines/minimax_h3/decoders.py`.

Architecture (decoder path only; the DAC encoder / attention `pre_block` / posterior heads are not ported):

    latents [B, 32, T] (pipeline layout, normalized)
      -> denormalize with per-channel latents_mean / latents_std            (float32, as the pipeline)
      -> dec_in_proj  Conv1d(32 -> latent_dim, k=1)
      -> BigVGAN: conv_pre k7 -> for each rate: ConvTranspose1d upsampler -> mean of AMP blocks
         -> alias-free SnakeBeta -> conv_post k7 (no bias) -> clamp[-1, 1]
    waveform [B, T * 800] at 32 kHz; B = 2 is the stereo pair (the VAE is mono, stereo rides the batch axis).

MLX layout is channels-last: activations are `(B, L, C)`, conv weights `(O, K, I)`. `weight_norm` (`weight_g` /
`weight_v`) is folded into plain weights in `sanitize`.

The alias-free activation (2x Kaiser-sinc upsample -> SnakeBeta -> 2x Kaiser-sinc downsample, all with replicate
padding) is a single fused Metal kernel: one read of the input, one write of the output, float32 math inside, no
materialized 2x-rate intermediate.

Precision: the reference keeps the whole audio VAE in float32 (`_keep_in_fp32_modules`, "bf16 is ~20 dB quieter"),
so `load_audio_vae` defaults to float32; bfloat16 is supported for speed, with the activation math still in float32.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

AUDIO_CHANNELS = 2  # MiniMax-H3 soundtrack is stereo, packed channel-major
AUDIO_LATENTS_PER_SECOND = 40

_FILTER_TAPS = 12  # Activation1d(ratio=2, kernel_size=12) is hard-wired in the reference


@dataclass
class AudioVAEConfig:
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    latent_dim: int = 2048
    latent_channels: int = 32
    decoder_dim: int = 1024
    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    sampling_rate: int = 32000
    latents_mean: list[float] | None = None
    latents_std: list[float] | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict) -> "AudioVAEConfig":
        known = {k for k in cls.__dataclass_fields__ if k != "extra"}
        kw = {k: v for k, v in cfg.items() if k in known}
        for k in ("encoder_rates", "decoder_rates", "decoder_kernel_sizes", "resblock_kernel_sizes"):
            if k in kw:
                kw[k] = tuple(int(v) for v in kw[k])
        if "resblock_dilation_sizes" in kw:
            kw["resblock_dilation_sizes"] = tuple(tuple(int(d) for d in ds) for ds in kw["resblock_dilation_sizes"])
        return cls(**kw, extra={k: v for k, v in cfg.items() if k not in known})

    @property
    def hop_length(self) -> int:
        return math.prod(self.decoder_rates)


# --------------------------------------------------------------------------------------------------------------------
# Fused alias-free SnakeBeta activation (UpSample1d x2 -> SnakeBeta -> DownSample1d x2)
# --------------------------------------------------------------------------------------------------------------------
#
# Derivation (ratio 2, K = 12, filters fu / fd):
#   upsample : u[m] = 2 * sum_i x[clamp(i)] * fu[m + 5 - 2i],  i in [floor(m/2) - 3 + (m&1), +6)   (m in [0, 2T))
#   snake    : z[m] = u + inv_beta * sin(alpha * u)^2
#   downsample (replicate pad 5 | 6, stride 2):  y[t] = sum_k fd[k] * z[clamp(2t + k - 5, 0, 2T - 1)]
_ACT_SOURCE = """
    uint gid = thread_position_in_grid.x;
    int T = x_shape[1];
    int C = x_shape[2];
    int c = gid % C;
    int bt = gid / C;
    int t = bt % T;
    int b = bt / T;
    size_t base = (size_t)b * (size_t)T * (size_t)C + (size_t)c;
    float a = alpha[c];
    float ib = inv_beta[c];
    int T2 = 2 * T;
    float acc = 0.0f;
    for (int k = 0; k < 12; ++k) {
        int m = clamp(2 * t + k - 5, 0, T2 - 1);
        int lo = (m >> 1) - 3 + (m & 1);
        float u = 0.0f;
        for (int j = 0; j < 6; ++j) {
            int i = lo + j;
            int ic = clamp(i, 0, T - 1);
            u += float(x[base + (size_t)ic * (size_t)C]) * fu[m + 5 - 2 * i];
        }
        u *= 2.0f;
        float s = metal::precise::sin(a * u);
        acc += fd[k] * (u + ib * s * s);
    }
    out[gid] = static_cast<OutT>(acc);
"""

_act_kernel = mx.fast.metal_kernel(
    name="pih3_alias_free_snakebeta",
    input_names=["x", "alpha", "inv_beta", "fu", "fd"],
    output_names=["out"],
    source=_ACT_SOURCE,
)


def alias_free_snakebeta(x: mx.array, alpha_exp: mx.array, inv_beta: mx.array, fu: mx.array, fd: mx.array):
    """x: (B, T, C) contiguous; alpha_exp / inv_beta: (C,) float32; fu / fd: (12,) float32 -> (B, T, C) x.dtype."""
    n = x.size
    return _act_kernel(
        inputs=[x, alpha_exp, inv_beta, fu, fd],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        template=[("OutT", x.dtype)],
    )[0]


def alias_free_snakebeta_ref(x, alpha_exp, inv_beta, fu, fd):
    """Pure-op reference of `alias_free_snakebeta` (float32), used only for testing the kernel."""
    x = x.astype(mx.float32)
    T = x.shape[1]
    xp = mx.concatenate([mx.repeat(x[:, :1], 5, axis=1), x, mx.repeat(x[:, -1:], 5, axis=1)], axis=1)
    C = x.shape[2]
    w = mx.broadcast_to(fu.reshape(1, -1, 1), (C, _FILTER_TAPS, 1))
    u = 2.0 * mx.conv_transpose1d(xp, w, stride=2, groups=C)[:, 15 : 15 + 2 * T]
    z = u + inv_beta * mx.sin(alpha_exp * u) ** 2
    zp = mx.concatenate([mx.repeat(z[:, :1], 5, axis=1), z, mx.repeat(z[:, -1:], 6, axis=1)], axis=1)
    wd = mx.broadcast_to(fd.reshape(1, -1, 1), (C, _FILTER_TAPS, 1))
    return mx.conv1d(zp, wd, stride=2, groups=C)


class Activation1d(nn.Module):
    """`MiniMaxH3AudioActivation1d(MiniMaxH3AudioSnakeBeta(C))`; parameters stored flat (see `sanitize`)."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))
        self.up_filter = mx.zeros((_FILTER_TAPS,))
        self.down_filter = mx.zeros((_FILTER_TAPS,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha.astype(mx.float32))
        inv_beta = 1.0 / (mx.exp(self.beta.astype(mx.float32)) + 1e-9)
        return alias_free_snakebeta(
            x, alpha, inv_beta, self.up_filter.astype(mx.float32), self.down_filter.astype(mx.float32)
        )


class AMPBlock(nn.Module):
    """BigVGAN `AMPBlock1`: per dilation, `x += conv2(act2(conv1(act1(x))))`."""

    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]):
        super().__init__()
        self.convs1 = [
            nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size * d - d) // 2, dilation=d)
            for d in dilation
        ]
        self.convs2 = [
            nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2) for _ in dilation
        ]
        self.activations = [Activation1d(channels) for _ in range(2 * len(dilation))]

    def __call__(self, x: mx.array) -> mx.array:
        for i, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):
            r = conv1(self.activations[2 * i](x))
            r = conv2(self.activations[2 * i + 1](r))
            x = x + r
        return x


class BigVGANDecoder(nn.Module):
    """`(B, T, latent_dim) -> (B, T * prod(rates), 1)`, clamped to [-1, 1]."""

    def __init__(self, cfg: AudioVAEConfig):
        super().__init__()
        c0 = cfg.decoder_dim
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(cfg.latent_dim, c0, 7, padding=3)
        # `ups.<i>.0` nesting kept so parameter names match the checkpoint.
        self.ups = [
            [nn.ConvTranspose1d(c0 // 2**i, c0 // 2 ** (i + 1), k, stride=r, padding=(k - r) // 2)]
            for i, (r, k) in enumerate(zip(cfg.decoder_rates, cfg.decoder_kernel_sizes))
        ]
        self.resblocks = []
        for i in range(len(cfg.decoder_rates)):
            ch = c0 // 2 ** (i + 1)
            for k, d in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(ch, k, tuple(d)))
        self.activation_post = Activation1d(ch)
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_pre(x)
        nk = self.num_kernels
        for i, up in enumerate(self.ups):
            x = up[0](x)
            acc = self.resblocks[i * nk](x)
            for j in range(1, nk):
                acc = acc + self.resblocks[i * nk + j](x)
            x = acc * (1.0 / nk)
        x = self.conv_post(self.activation_post(x))
        return mx.clip(x, -1.0, 1.0)


class AudioVAEDecoder(nn.Module):
    """Decoder half of `AutoencoderKLMiniMaxH3Audio` (`dec_in_proj` + BigVGAN `decoder`)."""

    def __init__(self, cfg: AudioVAEConfig):
        super().__init__()
        self.config = cfg
        self.dec_in_proj = nn.Conv1d(cfg.latent_channels, cfg.latent_dim, 1)
        self.decoder = BigVGANDecoder(cfg)
        c = cfg.latent_channels
        mean = cfg.latents_mean if cfg.latents_mean is not None else [0.0] * c
        std = cfg.latents_std if cfg.latents_std is not None else [1.0] * c
        # Kept out of the parameter tree (leading underscore) and always float32.
        self._latents_mean = mx.array(mean, dtype=mx.float32)
        self._latents_std = mx.array(std, dtype=mx.float32)

    @property
    def sampling_rate(self) -> int:
        return self.config.sampling_rate

    @property
    def hop_length(self) -> int:
        return self.config.hop_length

    @property
    def dtype(self) -> mx.Dtype:
        return self.dec_in_proj.weight.dtype

    def _decode_impl(self, latents_btc: mx.array) -> mx.array:
        return self.decoder(self.dec_in_proj(latents_btc.astype(self.dtype)))

    def __call__(self, latents_btc: mx.array) -> mx.array:
        """Raw decode (no denormalization): `(B, T, latent_channels)` -> `(B, T * hop, 1)` in the module dtype."""
        # mx.compile measured at ~2% here (conv-bound) and would freeze weights as constants, so it is not used.
        return self._decode_impl(latents_btc)

    def denormalize(self, latents_bct: mx.array) -> mx.array:
        """Pipeline denormalization in float32: `latents * std + mean`, per channel, on `(B, C, T)`."""
        x = latents_bct.astype(mx.float32)
        return x * self._latents_std.reshape(1, -1, 1) + self._latents_mean.reshape(1, -1, 1)


# --------------------------------------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------------------------------------

_DROP_PREFIXES = ("encoder.", "pre_block.", "mean_proj.", "logs_proj.")


def _fold_weight_norm(g: mx.array, v: mx.array) -> mx.array:
    """torch `weight_norm(dim=0)`: `v * g / ||v||` with the norm over every axis but 0 (float32)."""
    g = g.astype(mx.float32)
    v = v.astype(mx.float32)
    norm = mx.sqrt(mx.sum(v * v, axis=tuple(range(1, v.ndim)), keepdims=True))
    return v * (g / norm)


def sanitize(weights: dict) -> dict:
    """Map HF `AutoencoderKLMiniMaxH3Audio` state-dict keys/layouts to `AudioVAEDecoder` parameters.

    - drops the encoder side (`encoder.*`, `pre_block.*`, `mean_proj.*`, `logs_proj.*`);
    - folds `weight_g` / `weight_v` (or `parametrizations.weight.original0/1`) into `weight`;
    - Conv1d `(O, I, K)` -> `(O, K, I)`; ConvTranspose1d `(I, O, K)` -> `(O, K, I)`;
    - `activations.N.act.{alpha,beta}` -> `activations.N.{alpha,beta}`, resampler `filter` buffers ->
      `up_filter` / `down_filter` of shape (12,).
    Accepts `mx.array`, numpy arrays or torch tensors as values.
    """
    w = {}
    for k, v in weights.items():
        if k.startswith(_DROP_PREFIXES):
            continue
        if not isinstance(v, mx.array):
            if hasattr(v, "detach"):  # torch tensor
                v = v.detach().cpu().float().numpy()
            v = mx.array(np.asarray(v))
        w[k] = v

    # Fold weight norm.
    for k in [k for k in w if k.endswith(("weight_g", "parametrizations.weight.original0"))]:
        if k.endswith("weight_g"):
            stem, kv = k[: -len("weight_g")], k[: -len("weight_g")] + "weight_v"
        else:
            stem = k[: -len("parametrizations.weight.original0")]
            kv = stem + "parametrizations.weight.original1"
        w[stem + "weight"] = _fold_weight_norm(w.pop(k), w.pop(kv))

    out = {}
    for k, v in w.items():
        if k.endswith(".act.alpha") or k.endswith(".act.beta"):
            out[k.replace(".act.", ".")] = v
        elif k.endswith(".upsample.filter"):
            out[k.replace(".upsample.filter", ".up_filter")] = v.reshape(-1)
        elif k.endswith(".downsample.lowpass.filter"):
            out[k.replace(".downsample.lowpass.filter", ".down_filter")] = v.reshape(-1)
        elif k.endswith(".weight") and v.ndim == 3:
            if k.startswith("decoder.ups."):
                out[k] = v.transpose(1, 2, 0)  # (I, O, K) -> (O, K, I)
            else:
                out[k] = v.transpose(0, 2, 1)  # (O, I, K) -> (O, K, I)
        else:
            out[k] = v
    return out


def _keep_fp32(name: str) -> bool:
    return name.endswith(("alpha", "beta", "up_filter", "down_filter"))


def build_audio_vae(cfg: AudioVAEConfig | dict, weights: dict, dtype: mx.Dtype = mx.float32) -> AudioVAEDecoder:
    """Build an `AudioVAEDecoder` from a config and an (unsanitized, HF-layout) state dict."""
    if isinstance(cfg, dict):
        cfg = AudioVAEConfig.from_dict(cfg)
    model = AudioVAEDecoder(cfg)
    params = sanitize(weights)
    params = {k: (v.astype(mx.float32) if _keep_fp32(k) else v.astype(dtype)) for k, v in params.items()}
    model.load_weights(list(params.items()), strict=True)
    mx.eval(model.parameters())
    return model


def load_audio_vae(model_dir: str | Path, dtype: mx.Dtype = mx.float32) -> AudioVAEDecoder:
    """Load the decoder from an HF `audio_vae/` folder (`config.json` + `diffusion_pytorch_model.safetensors`).

    Defaults to float32 because the reference pins the whole audio VAE to fp32; pass `mx.bfloat16` for speed.
    """
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    files = sorted(p for p in model_dir.glob("*.safetensors") if not p.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {model_dir}")
    weights = {}
    for f in files:
        weights.update(mx.load(str(f)))  # lazy
    weights = {k: v for k, v in weights.items() if not k.startswith(_DROP_PREFIXES)}
    return build_audio_vae(cfg, weights, dtype)


# --------------------------------------------------------------------------------------------------------------------
# Pipeline glue
# --------------------------------------------------------------------------------------------------------------------


def audio_rows_to_latents(
    audio_rows, num_audio_latents: int, num_condition_audio_rows: int = 0, audio_channels: int = AUDIO_CHANNELS
) -> mx.array:
    """`MiniMaxH3AfterDenoiseStep` audio part: packed channel-major rows `(num_cond + ch * T, C)` -> `(ch, C, T)`."""
    rows = mx.array(audio_rows) if not isinstance(audio_rows, mx.array) else audio_rows
    rows = rows[num_condition_audio_rows:]
    rows = rows.reshape(audio_channels, num_audio_latents, rows.shape[-1])
    return rows.transpose(0, 2, 1)


def decode_audio(audio_vae: AudioVAEDecoder, audio_latents) -> np.ndarray:
    """`MiniMaxH3AudioDecodeStep`: normalized latents `(2, 32, T)` (pipeline layout) -> float32 waveform `(2, T*800)`.

    Denormalizes in float32 (`x * latents_std + mean`), decodes each stereo channel as a batch item, clamps to
    [-1, 1]. The reference returns `(1, 2, samples)`; this returns the squeezed `(channels, samples)` numpy array.
    Sampling rate: `audio_vae.sampling_rate` (32000).
    """
    x = audio_latents if isinstance(audio_latents, mx.array) else mx.array(np.asarray(audio_latents, np.float32))
    if x.ndim != 3:
        raise ValueError(f"audio_latents must be (channels, latent_channels, T), got {x.shape}")
    x = audio_vae.denormalize(x).transpose(0, 2, 1)  # (B, T, C)
    wav = audio_vae(x)[..., 0].astype(mx.float32)
    mx.eval(wav)
    return np.array(wav)
