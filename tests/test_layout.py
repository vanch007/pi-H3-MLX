import mlx.core as mx
import numpy as np
import pytest
import torch
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep

from pih3_mlx import layout as L
from pih3_mlx.transformer import AdaLNOut


@pytest.mark.parametrize("frames,h,w,anchors", [(243, 768, 1344, ()), (125, 512, 896, ("first",)),
                                                (344, 1344, 768, ("first", "last")), (130, 416, 736, ("last",))])
def test_layout_matches_reference(frames, h, w, anchors):
    tags = np.ones(37, np.int64)
    tags[5:20] = 0
    lay = L.build_layout(tags, frames, h, w, anchors)
    ref = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.tensor(tags), lay.num_latent_frames, lay.latent_height, lay.latent_width, lay.num_audio_latents,
        (1, 2, 2), 2, 2, 0, anchors)
    for got, want in zip((lay.position_ids, lay.token_tags, lay.video_indices, lay.audio_indices, lay.text_indices),
                         ref[:5]):
        np.testing.assert_array_equal(got, want.numpy())
    assert lay.num_condition_video_rows == ref[5]
    for sv, sa in ((0.99, 0.9), (0.3, 0.1)):
        uniq, idx = L.row_timesteps(lay, sv, sa)
        r_uniq, r_idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            torch.tensor(lay.video_indices), torch.tensor(lay.audio_indices), lay.num_condition_video_rows, 0,
            lay.num_text, sv, sa, min(sv, 1 - 0.999), 0.0)
        np.testing.assert_allclose(uniq, 1 - r_uniq.numpy(), rtol=0, atol=0)
        np.testing.assert_array_equal(idx, r_idx.numpy())


def test_dwd_norm_out():
    torch.manual_seed(0)
    ref = AdaLayerNormContinuous(64, 8, elementwise_affine=False, eps=1e-5).float()
    with torch.no_grad():
        ref.linear.weight.normal_(0, 0.3)
        ref.linear.bias.normal_(0, 0.3)
    x, c = torch.randn(10, 64), torch.randn(1, 8)
    scale, shift = ref.linear(ref.silu(c)).chunk(2, dim=-1)
    want = ref.norm(x) * (1 + scale) + shift  # MiniMaxH3DWDAdaLayerNormOut.forward
    m = AdaLNOut(64, 8, 1e-5, silu_first=True, layer_norm=True, scale_first=True)
    m.weight, m.bias = mx.array(ref.linear.weight.detach().numpy()), mx.array(ref.linear.bias.detach().numpy())
    got = np.array(m(mx.array(x.numpy()), mx.array(c[0].numpy())))
    np.testing.assert_allclose(got, want.detach().numpy(), atol=2e-5)
