import mlx.core as mx
import numpy as np
import pytest
import torch

from pih3_mlx import dwd_head as D
from tests._lakonlab_stubs import randomize_, sol_h3_head_cls


def _to_mlx(ref):
    return {k: mx.array(v.detach().float().numpy()) for k, v in ref.state_dict().items()}


@pytest.mark.parametrize("grid,layers", [((7, 10, 9), 2), ((12, 3, 4), 3)])
def test_dwd_head_parity(grid, layers):
    torch.manual_seed(0)
    Ref = sol_h3_head_cls()
    ref = Ref(in_channels=64, src_patch_size=(4, 32, 32), first_frame_src_patch_size=(1, 32, 32),
              first_frame_period=5, num_layers=layers).float()
    randomize_(ref, seed=1)
    t, h, w = grid
    feats = torch.randn(1, t * h * w, 64, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        want = ref(None, feats, grid)[0]  # (F, 3, H, W)
    head = D.SolH3DWDHead(in_channels=64, num_layers=layers)
    head.load_weights(list(_to_mlx(ref).items()), strict=True)
    got = head(mx.array(feats[0].numpy()), grid)  # (F, H, W, 3)
    got = np.array(got).transpose(0, 3, 1, 2)
    want = want.numpy()
    assert got.shape == want.shape
    err = np.abs(got - want).max() / np.abs(want).max()
    print(f"grid {grid}: max rel err {err:.2e}")
    assert err < 2e-4
