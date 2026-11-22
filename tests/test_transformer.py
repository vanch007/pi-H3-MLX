"""fp32 parity of the MLX H3 transformer against the pruned diffusers reference on a tiny random config."""

import importlib.util
import os
import sys

import mlx.core as mx
import numpy as np
import pytest
import torch

from pih3_mlx import layout as L
from pih3_mlx.convert import ROOT
from pih3_mlx.transformer import H3Transformer, sanitize

REF = os.path.join(ROOT, "models", "MiniMax-H3-Pruned", "transformer", "modeling_minimax_h3_pruned.py")
CFG = dict(num_attention_heads=4, attention_head_dim=32, hidden_size=64, num_layers=3, num_refiner_layers=2,
           ffn_dim=96, in_channels=4, audio_in_channels=8, patch_size=[1, 2, 2], text_dim=48, rope_freq_dim=4,
           rope_theta=10000.0, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5, adaln_rank=8,
           time_table_size=17, freq_dim=16, time_embed_hidden_dim=64, time_embed_dim=32)


def _ref_cls():
    spec = importlib.util.spec_from_file_location("h3_pruned_ref", REF)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3_pruned_ref"] = mod
    spec.loader.exec_module(mod)
    return mod.MiniMaxH3PrunedTransformer3DModel


def _randomize(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if name.endswith("inv_freq"):
                continue
            if "norm" in name and name.endswith("weight"):
                p.copy_(1 + 0.2 * torch.randn(p.shape, generator=g))
            elif "folded_bias" in name or "table" in name:
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))


@pytest.mark.parametrize("anchors", [(), ("first",), ("first", "last")])
def test_transformer_parity(anchors):
    torch.manual_seed(0)
    ref = _ref_cls()(**{k: (tuple(v) if k == "patch_size" else v) for k, v in CFG.items()}).float().eval()
    _randomize(ref, 1)
    n_text = 7
    tags = np.ones(n_text, np.int64)
    tags[2:4] = L.VIDEO_TAG
    lay = L.build_layout(tags, 124, 64, 96, anchors)
    sig_v, sig_a = 0.83, 0.61
    t_uniq, t_idx = L.row_timesteps(lay, sig_v, sig_a)
    g = torch.Generator().manual_seed(3)
    nv, na = lay.video_indices.shape[0], lay.audio_indices.shape[0]
    video = torch.randn(1, nv, 16, generator=g)
    audio = torch.randn(1, na, 8, generator=g)
    text = torch.randn(1, n_text, 48, generator=g)
    with torch.no_grad():
        out = ref(hidden_states=video, audio_hidden_states=audio, encoder_hidden_states=text,
                  timestep=torch.tensor(t_uniq), timestep_indices=torch.tensor(t_idx),
                  token_tags=torch.tensor(lay.token_tags), position_ids=torch.tensor(lay.position_ids).float(),
                  video_indices=torch.tensor(lay.video_indices), audio_indices=torch.tensor(lay.audio_indices),
                  text_indices=torch.tensor(lay.text_indices), return_dict=False)
    want_v = out[0][0, lay.num_condition_video_rows:].numpy()
    want_a = out[1][0].numpy()

    model = H3Transformer(CFG, with_dwd=False)
    model.sparse_attention = False  # parity vs the dense reference; Sol is covered by test_sol_attn.py
    w = sanitize({k: mx.array(v.detach().numpy()) for k, v in ref.state_dict().items()}, dtype=mx.float32)
    model.load_weights(list(w.items()), strict=True)
    rope = model.rope_tables(lay.position_ids, mx.float32)
    th = model.encode_text(mx.array(text.numpy()))
    ov, oa, _ = model(mx.array(video[0].numpy()), mx.array(audio[0].numpy()), th, t_uniq, t_idx, lay, rope)
    for got, want, name in ((np.array(ov), want_v, "video"), (np.array(oa), want_a, "audio")):
        err = np.abs(got - want).max() / np.abs(want).max()
        print(f"{anchors} {name}: rel err {err:.2e}")
        assert got.shape == want.shape and err < 1e-4
