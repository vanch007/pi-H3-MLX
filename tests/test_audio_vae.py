"""Parity tests for `pih3_mlx.audio_vae` against the diffusers `AutoencoderKLMiniMaxH3Audio` decoder."""

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch

from pih3_mlx import audio_vae as av

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "MiniMax-H3" / "audio_vae"
WEIGHTS = MODEL_DIR / "diffusion_pytorch_model.safetensors"
WEIGHTS_BYTES = 605_000_000  # ~0.605 GB when complete


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max()), float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def _torch_model(**cfg):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio

    return AutoencoderKLMiniMaxH3Audio(**cfg).eval()


def _torch_decode(model, latents_bct: np.ndarray) -> np.ndarray:
    """MiniMaxH3AudioDecodeStep, verbatim math."""
    with torch.no_grad():
        mean = torch.tensor(model.config.latents_mean).view(1, -1, 1)
        std = torch.tensor(model.config.latents_std).view(1, -1, 1)
        x = torch.from_numpy(latents_bct) * std + mean
        audio = model.decode(x, return_dict=False)[0]
        return audio.float().permute(1, 0, 2)[0].numpy()  # (1, 2, S) -> (2, S)


SMALL_CFG = dict(
    encoder_dim=4,
    encoder_rates=(2, 5, 2),
    latent_dim=64,
    latent_channels=8,
    num_attention_heads=2,
    decoder_dim=64,
    decoder_rates=(5, 2, 2),
    decoder_kernel_sizes=(9, 4, 4),
    resblock_kernel_sizes=(3, 7, 11),
    resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    sampling_rate=32000,
)


def test_activation_kernel_matches_ops():
    rng = np.random.default_rng(0)
    for T, C in [(1, 3), (7, 5), (64, 16)]:
        x = mx.array(rng.standard_normal((2, T, C)).astype(np.float32))
        a = mx.array(np.exp(rng.standard_normal(C) * 0.3).astype(np.float32))
        ib = mx.array((1.0 / np.exp(rng.standard_normal(C) * 0.3)).astype(np.float32))
        fu = mx.array(rng.standard_normal(12).astype(np.float32) * 0.2)
        fd = mx.array(rng.standard_normal(12).astype(np.float32) * 0.2)
        y = av.alias_free_snakebeta(x, a, ib, fu, fd)
        r = av.alias_free_snakebeta_ref(x, a, ib, fu, fd)
        assert y.shape == r.shape
        _, rel = _rel(y, r)
        assert rel < 1e-5, (T, C, rel)


def test_audio_rows_to_latents_layout():
    T, C = 5, 4
    rows = np.random.default_rng(1).standard_normal((3 + 2 * T, C)).astype(np.float32)
    got = np.array(av.audio_rows_to_latents(rows, T, num_condition_audio_rows=3))
    ref = torch.from_numpy(rows)[3:].reshape(2, T, C).permute(0, 2, 1).numpy()
    np.testing.assert_array_equal(got, ref)


def test_decoder_parity_small_random():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    c = SMALL_CFG["latent_channels"]
    cfg = dict(
        SMALL_CFG,
        latents_mean=rng.standard_normal(c).tolist(),
        latents_std=(1.0 + rng.random(c)).tolist(),
    )
    tm = _torch_model(**cfg)
    # Perturb every parameter so weight_g != ||weight_v|| and alpha / beta are non-trivial.
    with torch.no_grad():
        for name, p in tm.named_parameters():
            if name.endswith(("alpha", "beta")):
                p.copy_(torch.randn_like(p) * 0.3)
            elif name.endswith("weight_g"):
                p.mul_(0.5 + torch.rand_like(p))
            elif name.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.05)
    state = {k: v for k, v in tm.state_dict().items()}
    mm = av.build_audio_vae(json.loads(json.dumps(dict(tm.config))), state, dtype=mx.float32)

    T = 13
    lat = rng.standard_normal((2, c, T)).astype(np.float32) * 0.5
    ref = _torch_decode(tm, lat)
    got = av.decode_audio(mm, lat)
    assert got.shape == ref.shape == (2, T * 20)
    assert got.dtype == np.float32
    mae, rel = _rel(got, ref)
    print(f"\n[small random fp32] max abs {mae:.3e}  rel {rel:.3e}  (ref max {np.abs(ref).max():.3f})")
    assert rel < 1e-4


def _weights_complete() -> bool:
    if not WEIGHTS.exists() or (MODEL_DIR / "diffusion_pytorch_model.safetensors.aria2").exists():
        return False
    return WEIGHTS.stat().st_size >= 0.99 * WEIGHTS_BYTES


@pytest.mark.skipif(not _weights_complete(), reason="audio_vae weights not fully downloaded")
def test_real_weights_parity_and_speed():
    from safetensors.torch import load_file

    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    tm = _torch_model(**{k: v for k, v in cfg.items() if not k.startswith("_")})
    sd = load_file(str(WEIGHTS))
    missing, unexpected = tm.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    assert not [k for k in missing if not k.endswith("zero_k_bias")], missing[:5]

    rng = np.random.default_rng(0)
    T = 20  # 0.5 s
    lat = rng.standard_normal((2, 32, T)).astype(np.float32)
    ref = _torch_decode(tm, lat)

    m32 = av.load_audio_vae(MODEL_DIR, dtype=mx.float32)
    got = av.decode_audio(m32, lat)
    mae, rel = _rel(got, ref)
    print(f"\n[real fp32, T={T}] max abs {mae:.3e}  rel {rel:.3e}  (ref max {np.abs(ref).max():.3f})")
    assert rel < 1e-3

    m16 = av.load_audio_vae(MODEL_DIR, dtype=mx.bfloat16)
    g16 = av.decode_audio(m16, lat)
    snr = 10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / np.sum((g16 - ref).astype(np.float64) ** 2))
    print(f"[real bf16, T={T}] max abs {np.abs(g16 - ref).max():.3e}  SNR vs fp32 ref {snr:.1f} dB")

    T10 = 10 * av.AUDIO_LATENTS_PER_SECOND
    lat10 = rng.standard_normal((2, 32, T10)).astype(np.float32)
    for name, m in (("bf16", m16), ("fp32", m32)):
        av.decode_audio(m, lat10)  # warmup / compile
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            wav = av.decode_audio(m, lat10)
            ts.append(time.perf_counter() - t0)
        assert wav.shape == (2, T10 * 800)
        print(f"[real {name}] 10 s stereo clip: {min(ts) * 1e3:.0f} ms (best of 3)")
