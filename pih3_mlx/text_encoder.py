"""MiniMax H3 text conditioner (Qwen3-VL) in MLX.

MiniMax H3 conditions its transformer on ``hidden_states[50]`` of a 51-layer Qwen3-VL language model: the residual
stream right after the 50th decoder layer, *before* the final RMSNorm. Layer 51, the final norm and the LM head are
never needed, so they are neither built nor loaded.

The presentation (reference: ``diffusers/.../minimax_h3/encoders.py``) is the prompt verbatim with no chat template and
no special tokens; for ``fl2va`` every keyframe prepends a ``"<Picture i>: "`` label and a vision block
``<|vision_start|> <|image_pad|>*n <|vision_end|>``. Per-row MiniMax tags: text rows = 1, vision-block rows = 0 (video).

Model classes (decoder layers with interleaved MRoPE, vision tower with deepstack mergers) are reused from
``mlx_vlm.models.qwen3_vl``; this module drives them directly: embeddings -> vision scatter -> 50 decoder layers with
deepstack injection after layers 0..2 -> pre-norm hidden state. 3D MRoPE position ids follow transformers'
``Qwen3VLModel.get_rope_index`` and are computed in numpy.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.qwen3_vl import Model as Qwen3VLModel
from mlx_vlm.models.qwen3_vl import ModelConfig

VIDEO_TAG = 0  # MINIMAX_H3_VIDEO_TAG — vision-block rows
TEXT_TAG = 1  # MINIMAX_H3_TEXT_TAG
TEXT_ENCODER_LAYER = 50

__all__ = [
    "TEXT_TAG",
    "VIDEO_TAG",
    "TEXT_ENCODER_LAYER",
    "H3TextEncoder",
    "build_model",
    "sanitize",
    "load_text_encoder",
    "save_text_encoder",
    "load_converted_text_encoder",
    "build_presentation",
    "get_rope_index",
    "encode_tokens",
    "encode_prompt",
]


# --------------------------------------------------------------------------------------------------------------------
# Model construction / weights
# --------------------------------------------------------------------------------------------------------------------


def build_model(config: dict, num_layers: int = TEXT_ENCODER_LAYER) -> Qwen3VLModel:
    """Build the mlx-vlm Qwen3-VL model truncated to ``num_layers`` decoder layers (no LM head).

    Parameters are lazy random arrays until weights are loaded, so construction allocates nothing.
    """
    config = json.loads(json.dumps(config))  # deep copy
    text = config["text_config"]
    total = text["num_hidden_layers"]
    if total <= num_layers:
        raise ValueError(
            f"hidden_states[{num_layers}] needs more than {num_layers} decoder layers (pre-norm), got {total}."
        )
    text["num_hidden_layers"] = num_layers
    text["tie_word_embeddings"] = True  # never build an lm_head
    # transformers 5 serializes `rope_parameters` (incl. rope_theta); 4.x / the H3 checkpoint uses `rope_scaling`.
    rope = dict(text.get("rope_scaling") or text.get("rope_parameters") or {})
    text.setdefault("rope_theta", rope.pop("rope_theta", 5000000))
    rope.pop("rope_theta", None)
    text["rope_scaling"] = rope
    config["vision_config"]["model_type"] = "qwen3_vl"
    cfg = ModelConfig.from_dict(config)
    model = Qwen3VLModel(cfg)
    model.eval()
    return model


def _wanted(key: str, num_layers: int) -> bool:
    if key.startswith("lm_head") or key == "model.language_model.norm.weight":
        return False
    prefix = "model.language_model.layers."
    if key.startswith(prefix):
        return int(key[len(prefix) :].split(".", 1)[0]) < num_layers
    return key.startswith(("model.language_model.", "model.visual."))


def sanitize(weights: dict, num_layers: int = TEXT_ENCODER_LAYER) -> dict:
    """Map HF ``Qwen3VLForConditionalGeneration`` keys/layouts to the mlx-vlm module; drop unused weights."""
    out = {}
    for key, value in weights.items():
        if not _wanted(key, num_layers):
            continue
        if key.startswith("model.language_model."):
            new_key = "language_model.model." + key[len("model.language_model.") :]
        else:
            new_key = "vision_tower." + key[len("model.visual.") :]
            if new_key == "vision_tower.patch_embed.proj.weight" and value.ndim == 5:
                value = value.transpose(0, 2, 3, 4, 1)  # (O, C, T, P, P) -> (O, T, P, P, C)
        out[new_key] = value
    return out


def _quantize_predicate(path: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and module.weight.shape[-1] % 64 == 0


def _load_into(model: Qwen3VLModel, weights: dict, dtype, quantize_bits: int | None, group_size: int = 64):
    """Load sanitized weights; evaluate the LM one decoder layer at a time (quantizing it if asked) to cap peak memory.

    Vision-tower weights stay lazy (mx.load is lazy), so text-only prompts never read them from disk.
    """
    weights = {k: (v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in weights.items()}
    lm = model.language_model.model
    groups: dict[str, list] = {"rest": []}
    for k, v in weights.items():
        if k.startswith("language_model.model.layers."):
            idx = k.split(".")[3]
            groups.setdefault(idx, []).append((k, v))
        else:
            groups["rest"].append((k, v))
    missing = set(dict(_flatten_params(model))) - set(weights) - {"language_model.model.norm.weight"}
    if missing:
        raise ValueError(f"missing text-encoder weights: {sorted(missing)[:8]} ...")

    model.load_weights(groups.pop("rest"), strict=False)
    mx.eval(lm.embed_tokens.parameters())
    for i, layer in enumerate(lm.layers):
        model.load_weights(groups.pop(str(i)), strict=False)
        if quantize_bits:
            nn.quantize(layer, group_size=group_size, bits=quantize_bits, class_predicate=_quantize_predicate)
        mx.eval(layer.parameters())
    del weights, groups
    gc.collect()


def _flatten_params(module: nn.Module):
    from mlx.utils import tree_flatten

    return tree_flatten(module.parameters())


@dataclass
class H3TextEncoder:
    model: Qwen3VLModel
    tokenizer: object
    image_processor: object | None
    config: dict
    layer: int = TEXT_ENCODER_LAYER
    dtype: mx.Dtype = mx.bfloat16
    token_ids: dict = field(default_factory=dict)

    @property
    def image_token_id(self) -> int:
        return self.config.get("image_token_id", 151655)

    @property
    def merge_size(self) -> int:
        return self.config["vision_config"].get("spatial_merge_size", 2)


def make_encoder(
    config: dict,
    weights: dict,
    tokenizer,
    image_processor=None,
    dtype=mx.bfloat16,
    quantize_bits: int | None = None,
    layer: int = TEXT_ENCODER_LAYER,
) -> H3TextEncoder:
    """Build an encoder from a config dict and an (unsanitized, HF-keyed) weight dict. Used by the loader and tests."""
    model = build_model(config, layer)
    _load_into(model, sanitize(weights, layer), dtype, quantize_bits)
    ids = {}
    if tokenizer is not None:
        for name in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
            ids[name] = tokenizer.convert_tokens_to_ids(name)
    return H3TextEncoder(model, tokenizer, image_processor, config, layer, dtype, ids)


def load_text_encoder(
    model_dir,
    tokenizer_dir,
    processor_dir,
    dtype=mx.bfloat16,
    quantize_bits: int | None = None,
    layer: int = TEXT_ENCODER_LAYER,
) -> H3TextEncoder:
    """Load the H3 Qwen3-VL conditioner from an HF ``text_encoder`` directory.

    Only shards/tensors that are used are touched (no lm_head, no layers >= ``layer``, no final norm). Language-model
    linears are optionally quantized (``quantize_bits`` in {4, 8}, group size 64) layer by layer while loading.
    Embeddings and the vision tower stay in ``dtype``.
    """
    from transformers import AutoImageProcessor, AutoTokenizer

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            if _wanted(key, layer):
                shards.setdefault(shard, []).append(key)
        weights = {}
        for shard, keys in sorted(shards.items()):
            data = mx.load(str(model_dir / shard))  # lazy
            for k in keys:
                weights[k] = data[k]
            del data
    else:
        weights = {}
        for f in sorted(model_dir.glob("*.safetensors")):
            weights.update({k: v for k, v in mx.load(str(f)).items() if _wanted(k, layer)})

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    image_processor = AutoImageProcessor.from_pretrained(str(processor_dir)) if processor_dir else None
    return make_encoder(config, weights, tokenizer, image_processor, dtype, quantize_bits, layer)


SHARD_BYTES = 4 << 30


def save_text_encoder(encoder: H3TextEncoder, out_dir, quantize_bits: int | None):
    """Save the built (optionally quantized) encoder as MLX safetensors shards + config, for `load_converted_*`."""
    from mlx.utils import tree_flatten

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(encoder.config, h3_layer=encoder.layer,
               mlx_quantization=dict(bits=quantize_bits, group_size=64) if quantize_bits else None)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    shard, size, idx = {}, 0, 0
    items = tree_flatten(encoder.model.parameters())
    for i, (k, v) in enumerate(items):
        shard[k] = v
        size += v.nbytes
        if size >= SHARD_BYTES or i == len(items) - 1:
            mx.save_safetensors(str(out_dir / f"model-{idx:05d}.safetensors"), shard)
            shard, size, idx = {}, 0, idx + 1


def load_converted_text_encoder(model_dir, tokenizer_dir, processor_dir, dtype=mx.bfloat16) -> H3TextEncoder:
    """Load an encoder written by `save_text_encoder` (already quantized: no bf16 detour, ~half the peak memory)."""
    from transformers import AutoImageProcessor, AutoTokenizer

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    layer, q = config.pop("h3_layer"), config.pop("mlx_quantization")
    model = build_model(config, layer)
    if q:
        for blk in model.language_model.model.layers:
            nn.quantize(blk, group_size=q["group_size"], bits=q["bits"], class_predicate=_quantize_predicate)
    for f in sorted(model_dir.glob("model-*.safetensors")):
        model.load_weights(list(mx.load(str(f)).items()), strict=False)
        mx.eval(model.language_model.parameters())  # the vision tower stays lazy until an image is encoded
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    image_processor = AutoImageProcessor.from_pretrained(str(processor_dir)) if processor_dir else None
    ids = {name: tokenizer.convert_tokens_to_ids(name) for name in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")}
    return H3TextEncoder(model, tokenizer, image_processor, config, layer, dtype, ids)


# --------------------------------------------------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------------------------------------------------


def build_presentation(tokenizer, image_processor, prompt: str, keyframes=None):
    """Tokenize the H3 presentation exactly as ``MiniMaxH3{,FL2VA}TextEncoderStep``.

    Returns ``(token_ids list[int], token_tags np.int64 (L,), pixel_values np.float32 | None,
    image_grid_thw np.int64 (N, 3) | None)``.
    """
    if not isinstance(prompt, str):
        raise ValueError(f"MiniMax-H3 packs one request into one sequence; prompt must be a str, got {type(prompt)}.")
    token_ids: list[int] = []
    token_tags: list[int] = []
    pixel_values = grid = None
    if keyframes:
        vision = image_processor(images=list(keyframes), return_tensors="np")
        pixel_values = np.asarray(vision["pixel_values"], dtype=np.float32)
        grid = np.asarray(vision["image_grid_thw"], dtype=np.int64)
        merge = image_processor.merge_size**2
        vs, pad, ve = tokenizer.convert_tokens_to_ids(["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"])
        for index in range(len(keyframes)):
            n = int(np.prod(grid[index])) // merge
            label = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            vision_ids = [vs] + [pad] * n + [ve]
            token_ids += label + vision_ids
            token_tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(vision_ids)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    return token_ids, np.asarray(token_tags, dtype=np.int64), pixel_values, grid


def get_rope_index(
    token_ids, image_grid_thw=None, image_token_id: int = 151655, spatial_merge_size: int = 2
) -> np.ndarray:
    """3D MRoPE positions ``(3, L)`` for one image/text sequence, as transformers' ``Qwen3VLModel.get_rope_index``.

    Text runs advance all three axes together; an image run of grid (t, h, w) gets (start + t_i, start + h_i,
    start + w_i) over the merged grid and advances the cursor by ``max(h, w) // merge``. Text-only == arange.
    """
    ids = np.asarray(token_ids)
    L = ids.shape[0]
    if image_grid_thw is None or len(image_grid_thw) == 0:
        return np.broadcast_to(np.arange(L, dtype=np.int64), (3, L)).copy()
    is_img = ids == image_token_id
    out = np.zeros((3, L), dtype=np.int64)
    grids = iter(np.asarray(image_grid_thw).tolist())
    pos, i = 0, 0
    while i < L:
        j = i
        while j < L and is_img[j] == is_img[i]:
            j += 1
        if not is_img[i]:
            out[:, i:j] = np.arange(j - i) + pos
            pos += j - i
        else:
            t, h, w = next(grids)
            gh, gw = h // spatial_merge_size, w // spatial_merge_size
            tt, hh, ww = np.meshgrid(np.arange(t), np.arange(gh), np.arange(gw), indexing="ij")
            block = np.stack([tt.ravel(), hh.ravel(), ww.ravel()]) + pos
            if block.shape[1] != j - i:
                raise ValueError(f"image run of {j - i} tokens does not match grid {t}x{h}x{w}")
            out[:, i:j] = block
            pos += max(h, w) // spatial_merge_size
        i = j
    return out


# --------------------------------------------------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------------------------------------------------


def encode_tokens(
    encoder: H3TextEncoder,
    token_ids,
    pixel_values=None,
    image_grid_thw=None,
    layer: int | None = None,
) -> mx.array:
    """Hidden state after decoder layer ``layer`` (default: ``encoder.layer``) before the final norm: (1, L, D)."""
    model = encoder.model
    lm = model.language_model.model
    layer = encoder.layer if layer is None else layer
    if not 0 <= layer <= len(lm.layers):
        raise ValueError(f"layer must be in [0, {len(lm.layers)}], got {layer}")
    ids_np = np.asarray(token_ids, dtype=np.int64)
    ids = mx.array(ids_np[None].astype(np.int32))
    h = lm.embed_tokens(ids)

    deepstack, vis_idx = [], None
    if pixel_values is not None:
        vis_idx_np = np.flatnonzero(ids_np == encoder.image_token_id)
        vis_idx = mx.array(vis_idx_np.astype(np.uint32))
        vt = model.vision_tower
        pv = mx.array(np.asarray(pixel_values)).astype(vt.patch_embed.proj.weight.dtype)
        feats, deepstack = vt(pv, mx.array(np.asarray(image_grid_thw, dtype=np.int32)))
        if feats.shape[0] != vis_idx_np.size:
            raise ValueError(f"{vis_idx_np.size} image tokens but {feats.shape[0]} vision features")
        h[:, vis_idx] = feats.astype(h.dtype)[None]
        mx.eval(h, deepstack)

    pos = get_rope_index(ids_np, image_grid_thw, encoder.image_token_id, encoder.merge_size)
    position_ids = mx.array(pos[:, None, :].astype(np.int32))  # (3, B=1, L)
    rotary = lm.layers[0].self_attn.rotary_emb if lm.layers else None
    position_embeddings = None
    if rotary is not None and not rotary.fused_apply:
        position_embeddings = rotary(h, position_ids)

    mask = "causal" if h.shape[1] > 1 else None
    for i in range(layer):
        h = lm.layers[i](h, mask, None, position_ids, position_embeddings)
        if i < len(deepstack):
            h[:, vis_idx] = h[:, vis_idx] + deepstack[i].astype(h.dtype)[None]
        mx.eval(h)
    return h


def encode_prompt(encoder: H3TextEncoder, prompt: str, keyframes=None):
    """Encode a MiniMax H3 presentation.

    Returns ``(prompt_embeds mx.array (1, L, hidden) in encoder.dtype, text_token_tags np.int64 (L,))`` where tags are
    1 for text rows and 0 for vision-block rows (``<|vision_start|>``, image pads, ``<|vision_end|>``).
    """
    token_ids, tags, pixel_values, grid = build_presentation(
        encoder.tokenizer, encoder.image_processor, prompt, keyframes
    )
    embeds = encode_tokens(encoder, token_ids, pixel_values, grid)
    return embeds.astype(encoder.dtype), tags
