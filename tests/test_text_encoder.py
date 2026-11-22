"""Parity tests for pih3_mlx.text_encoder against transformers' Qwen3VLForConditionalGeneration (tiny random config)."""

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from PIL import Image

from pih3_mlx import text_encoder as te

from pih3_mlx.convert import ROOT as _ROOT  # noqa: E402
ROOT = Path(_ROOT)
TOKENIZER_DIR = ROOT / "models/MiniMax-H3/tokenizer"
PROCESSOR_DIR = ROOT / "models/MiniMax-H3/processor"
TEXT_ENCODER_DIR = ROOT / "models/MiniMax-H3-Pruned/text_encoder"
PROMPT = "A red fox trots across a snowy meadow at dawn, its breath fogging in the cold air. Soft wind, distant crows."

TINY_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "model_type": "qwen3_vl",
    "tie_word_embeddings": True,
    "text_config": {
        "attention_bias": False,
        "head_dim": 32,
        "hidden_act": "silu",
        "hidden_size": 64,
        "intermediate_size": 128,
        "max_position_embeddings": 4096,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 4,
        "num_hidden_layers": 7,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [6, 5, 5], "rope_type": "default"},
        "rope_theta": 5000000,
        "vocab_size": 151936,
    },
    "vision_config": {
        "deepstack_visual_indexes": [1, 2, 3],
        "depth": 4,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 32,
        "in_channels": 3,
        "intermediate_size": 64,
        "model_type": "qwen3_vl",
        "num_heads": 2,
        "num_position_embeddings": 64,
        "out_hidden_size": 64,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}
LAYER = 6  # tiny analogue of hidden_states[50] of a 51-layer stack


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


@pytest.fixture(scope="module")
def image_processor():
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(str(PROCESSOR_DIR))


@pytest.fixture(scope="module")
def tiny(tokenizer, image_processor):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(0)
    cfg = json.loads(json.dumps(TINY_CONFIG))
    hf = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**cfg)).float().eval()
    with torch.no_grad():  # random init leaves some params ~0 / norms at 1: perturb so every weight matters
        for name, p in hf.named_parameters():
            if "norm" in name:
                p.add_(0.1 * torch.randn_like(p))
            elif p.dim() == 1:
                p.copy_(0.05 * torch.randn_like(p))
    weights = {k: mx.array(v.detach().numpy()) for k, v in hf.state_dict().items()}
    enc = te.make_encoder(TINY_CONFIG, weights, tokenizer, image_processor, dtype=mx.float32, layer=LAYER)
    return hf, enc


def _image(h=200, w=300, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def _hf_hidden_states(hf, image_processor, token_ids, keyframes=None):
    """Mirror diffusers' get_qwen3vl_prompt_embeds on the tiny model; also return HF mrope position ids."""
    from transformers import Qwen3VLProcessor  # noqa: F401  (create_mm_token_type_ids semantics below)

    ids = torch.tensor([token_ids])
    mm = torch.tensor([[1 if t == 151655 else 2 if t == 151656 else 0 for t in token_ids]])
    kwargs = {}
    if keyframes:
        v = image_processor(images=keyframes, return_tensors="pt")
        kwargs = {"pixel_values": v["pixel_values"].float(), "image_grid_thw": v["image_grid_thw"]}
    with torch.no_grad():
        out = hf.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=mm,
            use_cache=False,
            output_hidden_states=True,
            **kwargs,
        )
        pos = None
        if keyframes:
            pos, _ = hf.model.get_rope_index(ids, mm, image_grid_thw=kwargs["image_grid_thw"])
    return [h[0].numpy() for h in out.hidden_states], pos


def _err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    abs_err = np.abs(a - b).max()
    return abs_err, abs_err / max(np.abs(b).max(), 1e-12)


def test_rope_index_text_only():
    pos = te.get_rope_index(list(range(17)))
    assert pos.shape == (3, 17) and (pos == np.arange(17)).all()


def test_text_only_parity(tiny, tokenizer, image_processor):
    hf, enc = tiny
    ids, tags, pv, grid = te.build_presentation(tokenizer, image_processor, PROMPT)
    assert pv is None and (tags == te.TEXT_TAG).all()
    ref, _ = _hf_hidden_states(hf, image_processor, ids)
    assert len(ref) == TINY_CONFIG["text_config"]["num_hidden_layers"] + 1
    for k in range(1, LAYER + 1):
        out = np.array(te.encode_tokens(enc, ids, layer=k)[0])
        abs_err, rel_err = _err(out, ref[k])
        print(f"text-only hidden_states[{k}]: max abs {abs_err:.3e}  rel {rel_err:.3e}")
        assert rel_err < 1e-4, (k, abs_err, rel_err)
    embeds, tags2 = te.encode_prompt(enc, PROMPT)
    assert embeds.shape == (1, len(ids), 64) and embeds.dtype == mx.float32
    np.testing.assert_array_equal(tags2, tags)


def test_image_presentation_parity(tiny, tokenizer, image_processor):
    hf, enc = tiny
    frames = [_image()]
    ids, tags, pv, grid = te.build_presentation(tokenizer, image_processor, PROMPT, frames)
    ref, hf_pos = _hf_hidden_states(hf, image_processor, ids, frames)
    pos = te.get_rope_index(ids, grid, 151655, 2)
    np.testing.assert_array_equal(pos, hf_pos[:, 0].numpy())
    n_ds = len(TINY_CONFIG["vision_config"]["deepstack_visual_indexes"])
    # transformers records hidden_states[k] at the decoder-layer output, i.e. before deepstack injection for k <= n_ds;
    # from k = n_ds + 1 on (incl. the H3 read-out layer) both conventions coincide.
    for k in range(n_ds + 1, LAYER + 1):
        out = np.array(te.encode_tokens(enc, ids, pv, grid, layer=k)[0])
        abs_err, rel_err = _err(out, ref[k])
        print(f"image hidden_states[{k}]: max abs {abs_err:.3e}  rel {rel_err:.3e}")
        assert rel_err < 1e-4, (k, abs_err, rel_err)


def test_two_keyframes_rope_index(tiny, tokenizer, image_processor):
    hf, enc = tiny
    frames = [_image(200, 300, 1), _image(256, 180, 2)]
    ids, tags, pv, grid = te.build_presentation(tokenizer, image_processor, "go", frames)
    _, hf_pos = _hf_hidden_states(hf, image_processor, ids, frames)
    np.testing.assert_array_equal(te.get_rope_index(ids, grid), hf_pos[:, 0].numpy())


def _reference_tokens(tokenizer, processor_image, prompt, keyframes):
    """Run diffusers' MiniMaxH3FL2VATextEncoderStep presentation with the conditioner call stubbed out."""
    import diffusers.modular_pipelines.minimax_h3.encoders as enc_mod
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG

    captured = {}

    def fake_embeds(text_encoder, processor, token_ids, vision_inputs=None, **kw):
        captured["ids"] = list(token_ids)
        return torch.zeros(1, len(token_ids), 1)

    class Proc:
        image_processor = processor_image

    class Comp:
        pass

    comp = Comp()
    comp.tokenizer, comp.processor = tokenizer, Proc()
    comp.text_encoder = type("TE", (), {"dtype": torch.float32})()
    comp.text_tag, comp.video_tag, comp.text_encoder_layer = MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG, 50
    comp._execution_device = "cpu"

    class BS:
        pass

    bs = BS()
    bs.prompt, bs.keyframes = prompt, keyframes
    step = enc_mod.MiniMaxH3FL2VATextEncoderStep()
    step.get_block_state = lambda state: bs
    step.set_block_state = lambda state, b: None
    orig = enc_mod.get_qwen3vl_prompt_embeds
    enc_mod.get_qwen3vl_prompt_embeds = fake_embeds
    try:
        step(comp, None)
    finally:
        enc_mod.get_qwen3vl_prompt_embeds = orig
    return captured["ids"], bs.text_token_tags.numpy()


@pytest.mark.parametrize("n_frames", [0, 1, 2])
def test_presentation_matches_reference(tokenizer, image_processor, n_frames):
    frames = [_image(720, 1280, i) for i in range(n_frames)] or None
    ids, tags, _, _ = te.build_presentation(tokenizer, image_processor, PROMPT, frames)
    ref_ids, ref_tags = _reference_tokens(tokenizer, image_processor, PROMPT, frames)
    assert ids == ref_ids
    np.testing.assert_array_equal(tags, ref_tags)
    if n_frames == 0:
        assert ids == tokenizer(PROMPT, add_special_tokens=False)["input_ids"]


def _real_weights_ready():
    index = TEXT_ENCODER_DIR / "model.safetensors.index.json"
    if not index.exists():
        return False
    shards = set(json.loads(index.read_text())["weight_map"].values())
    return all((TEXT_ENCODER_DIR / s).exists() and not (TEXT_ENCODER_DIR / (s + ".aria2")).exists() for s in shards)


@pytest.mark.skipif(not _real_weights_ready() or os.environ.get("PIH3_SKIP_REAL"), reason="real shards not ready")
def test_real_weights_smoke():
    t0 = time.perf_counter()
    enc = te.load_text_encoder(TEXT_ENCODER_DIR, TOKENIZER_DIR, PROCESSOR_DIR, quantize_bits=None)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    embeds, tags = te.encode_prompt(enc, PROMPT)
    mx.eval(embeds)
    t_text = time.perf_counter() - t0
    t0 = time.perf_counter()
    embeds_i, tags_i = te.encode_prompt(enc, PROMPT, [_image(720, 1280)])
    mx.eval(embeds_i)
    t_img = time.perf_counter() - t0
    print(
        f"real: load {t_load:.1f}s, text {embeds.shape} {t_text:.2f}s, image {embeds_i.shape} {t_img:.2f}s, "
        f"peak {mx.get_peak_memory() / 2**30:.1f} GiB, std {float(embeds.astype(mx.float32).std()):.3f}"
    )
    assert embeds.shape[-1] == 5120 and embeds.dtype == mx.bfloat16
    assert bool(mx.isfinite(embeds).all()) and bool(mx.isfinite(embeds_i).all())
    assert (tags_i == te.VIDEO_TAG).sum() > 0


@pytest.mark.parametrize("bits,dtype,tol", [(None, mx.bfloat16, 5e-2), (8, mx.float32, 2e-2), (4, mx.float32, 2e-1)])
def test_bf16_and_quantized(tiny, tokenizer, image_processor, bits, dtype, tol):
    hf, enc32 = tiny
    weights = {k: mx.array(v.detach().numpy()) for k, v in hf.state_dict().items()}
    enc = te.make_encoder(TINY_CONFIG, weights, tokenizer, image_processor, dtype=dtype, quantize_bits=bits, layer=LAYER)
    frames = [_image()]
    ref, tags_ref = te.encode_prompt(enc32, PROMPT, frames)
    out, tags = te.encode_prompt(enc, PROMPT, frames)
    assert out.dtype == dtype
    abs_err, rel_err = _err(np.array(out.astype(mx.float32)), np.array(ref))
    print(f"bits={bits} dtype={dtype}: max abs {abs_err:.3e} rel {rel_err:.3e}")
    assert rel_err < tol
    np.testing.assert_array_equal(tags, tags_ref)
