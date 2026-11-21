"""Parity tests for pih3_mlx.video_vae against the diffusers MiniMax-H3 reference."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from PIL import Image

from diffusers import AutoencoderKLMiniMaxH3
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines.minimax_h3 import encoders as ref_encoders
from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as ref_mp
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3ResizeStep
from diffusers.modular_pipelines.modular_pipeline import PipelineState

from pih3_mlx import video_vae as vv

VAE_DIR = Path(__file__).resolve().parents[2] / "models" / "MiniMax-H3" / "vae"

SMALL_CONFIG = dict(
    in_channels=3,
    out_channels=3,
    latent_channels=8,
    block_out_channels=(8, 16, 16, 32),
    layers_per_block=2,
    spatial_downsample_factors=(2, 2, 2, 2),
    temporal_downsample_factors=(1, 2, 2, 1),
    norm_num_groups=4,
    norm_eps=1e-6,
    decoder_num_layers=1,
    decoder_num_attention_heads=2,
    decoder_attention_head_dim=24,
    decoder_num_register_tokens=1,
    latents_mean=tuple(np.linspace(-0.5, 0.5, 8).tolist()),
    latents_std=tuple(np.linspace(0.5, 2.0, 8).tolist()),
)


def _rand_image(w, h, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (max(h // 8, 1), max(w // 8, 1), 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((w, h), Image.Resampling.BILINEAR)
    noise = rng.integers(-20, 21, (h, w, 3))
    return Image.fromarray(np.clip(np.asarray(img).astype(int) + noise, 0, 255).astype(np.uint8))


@pytest.fixture(scope="module")
def small_pair():
    torch.manual_seed(0)
    ref = AutoencoderKLMiniMaxH3(**SMALL_CONFIG).eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if name.startswith(("encoder.", "quant_conv.")):
                if "norm" in name and name.endswith("weight"):
                    p.copy_(1.0 + 0.2 * torch.randn_like(p))
                elif name.endswith("bias"):
                    p.copy_(0.1 * torch.randn_like(p))
    cfg = {k: list(v) if isinstance(v, tuple) else v for k, v in SMALL_CONFIG.items()}
    mlx_enc = vv.VideoVAEEncoder(cfg)
    weights = {k: mx.array(v.detach().numpy()) for k, v in ref.state_dict().items()}
    sanitized = vv.sanitize(weights)
    assert all(k.startswith(("encoder.", "quant_conv.")) for k in sanitized)
    mlx_enc.load_weights(list(sanitized.items()), strict=True)
    return ref, mlx_enc


def _err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    abs_err = np.abs(a - b).max()
    return abs_err, abs_err / max(np.abs(b).max(), 1e-12)


@pytest.mark.parametrize("hw", [(128, 192), (320, 576), (512, 896)])
def test_moments_parity_fp32(small_pair, hw):
    ref, enc = small_pair
    h, w = hw
    img = _rand_image(w, h, seed=h)
    x = vv.keyframe_pixels(img)
    moments = np.asarray(enc.encode_moments(x)).transpose(0, 3, 1, 2)
    xt = torch.from_numpy(np.asarray(x)).permute(0, 3, 1, 2)[:, :, None]
    with torch.no_grad():
        ref_moments = ref.encode(xt).latent_dist.parameters[:, :, 0].numpy()
    assert moments.shape == ref_moments.shape
    abs_err, rel_err = _err(moments, ref_moments)
    print(f"\nmoments {h}x{w}: max abs {abs_err:.3e}, rel {rel_err:.3e}")
    assert rel_err < 1e-4


@pytest.mark.parametrize("hw", [(128, 192), (512, 896)])
def test_encode_vae_condition_parity(small_pair, hw):
    ref, enc = small_pair
    h, w = hw
    img = _rand_image(w, h, seed=7)
    ours = np.asarray(vv.encode_keyframes(enc, [img], h, w)[0])
    pixels = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None, :, None]
    with torch.no_grad():
        theirs = ref_encoders.encode_vae_condition(ref, pixels, vv.PIXEL_MEAN, vv.PIXEL_STD, 42).numpy()
    assert ours.shape == theirs.shape == (1, 8, 1, h // 16, w // 16)
    assert ours.dtype == np.float32
    abs_err, rel_err = _err(ours, theirs)
    # float16 rounding of the sampled latent can flip by one fp16 ulp where fp32 results differ in the last bits.
    exact = np.mean(ours == theirs)
    print(f"\ncondition {h}x{w}: max abs {abs_err:.3e}, rel {rel_err:.3e}, bit-exact fraction {exact:.4f}")
    assert rel_err < 2e-3
    assert exact > 0.99


@pytest.mark.parametrize(
    "size",
    [(1920, 1080), (1080, 1920), (1024, 1024), (800, 533), (640, 1280), (4000, 1000), (333, 1000), (1344, 768)],
)
def test_resolve_canvas_size(size):
    assert vv.resolve_canvas_size(*size) == ref_mp.resolve_canvas_size(*size, 32, 768, 768 * 1344)


def _ref_resize(image, last_image, height=None, width=None):
    components = SimpleNamespace(
        canvas_multiple=32,
        config=SimpleNamespace(canvas_short_edge=768, canvas_max_pixels=768 * 1344),
        image_processor=VaeImageProcessor(vae_scale_factor=16),
    )
    state = PipelineState()
    for k, v in dict(image=image, last_image=last_image, height=height, width=width).items():
        state.set(k, v)
    MiniMaxH3ResizeStep()(components, state)
    return state.get("keyframes"), state.get("keyframe_anchors"), state.get("height"), state.get("width")


@pytest.mark.parametrize(
    "first,last",
    [((1280, 720), (1000, 1000)), ((720, 1280), (1920, 1080)), ((900, 601), (603, 905)), ((768, 1344), None),
     (None, (1600, 900))],
)
def test_prepare_keyframes_matches_reference(first, last):
    image = _rand_image(*first, seed=1) if first else None
    last_image = _rand_image(*last, seed=2) if last else None
    ours = vv.prepare_keyframes(image, last_image)
    theirs = _ref_resize(image, last_image)
    assert ours[1:] == tuple(theirs[1:])
    assert len(ours[0]) == len(theirs[0])
    for a, b in zip(ours[0], theirs[0]):
        assert a.size == b.size == (ours[3], ours[2])
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_prepare_keyframes_explicit_canvas():
    image, last_image = _rand_image(1280, 720, 3), _rand_image(700, 1000, 4)
    ours = vv.prepare_keyframes(image, last_image, height=512, width=896)
    theirs = _ref_resize(image, last_image, 512, 896)
    for a, b in zip(ours[0], theirs[0]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _real_weights_ready():
    index = VAE_DIR / "diffusion_pytorch_model.safetensors.index.json"
    if not index.exists():
        return False
    shards = set(json.loads(index.read_text())["weight_map"].values())
    expected = {5.061e9, 4.956e9, 0.399e9}
    sizes = [(VAE_DIR / s).stat().st_size if (VAE_DIR / s).exists() else 0 for s in shards]
    return all(s > 0 for s in sizes) and all(any(abs(s - e) / e < 0.01 for e in expected) for s in sizes)


@pytest.mark.skipif(not _real_weights_ready(), reason="MiniMax-H3 VAE shards not fully downloaded")
def test_real_weights_parity():
    h, w = 512, 896
    img = _rand_image(w, h, seed=11)
    t0 = time.perf_counter()
    enc = vv.load_video_vae_encoder(VAE_DIR, mx.float32)
    t_load = time.perf_counter() - t0
    vv.encode_keyframes(enc, [img], h, w)  # warm-up (compile)
    t0 = time.perf_counter()
    ours = np.asarray(vv.encode_keyframes(enc, [img], h, w)[0])
    t_mlx = time.perf_counter() - t0
    ref = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).eval()
    pixels = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None, :, None]
    t0 = time.perf_counter()
    with torch.no_grad():
        theirs = ref_encoders.encode_vae_condition(ref, pixels, vv.PIXEL_MEAN, vv.PIXEL_STD, 42).numpy()
    t_ref = time.perf_counter() - t0
    abs_err, rel_err = _err(ours, theirs)
    print(f"\nreal {h}x{w}: load {t_load:.1f}s, mlx fp32 {t_mlx:.2f}s, torch cpu {t_ref:.2f}s, "
          f"max abs {abs_err:.3e}, rel {rel_err:.3e}, bit-exact {np.mean(ours == theirs):.4f}")
    assert ours.shape == (1, 24, 1, h // 16, w // 16)
    assert rel_err < 5e-3
