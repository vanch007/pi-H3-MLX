"""Sol sparse attention Metal kernel vs a float32 reference with identical routing, and vs dense attention when
every block is routed exactly."""

import mlx.core as mx
import numpy as np
import pytest

from pih3_mlx.sol_attn import BLOCK, LOG2E, sol_attention, sol_reference, sol_route


def _qkv(T, H, seed=0):
    mx.random.seed(seed)
    return [mx.random.normal((1, T, H, 128)).astype(mx.bfloat16) for _ in range(3)]


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.linalg.norm(b)).item()


@pytest.mark.parametrize("T,H,prefix", [(1000, 2, 0), (777, 3, 150), (64, 1, 0), (2113, 2, 1200)])
def test_kernel_matches_reference(T, H, prefix):
    q, k, v = _qkv(T, H)
    out = sol_attention(q, k, v, prefix)
    assert _rel(out, sol_reference(q, k, v, prefix)) < 4e-3  # bf16 output rounding


@pytest.mark.parametrize("T,prefix", [(1000, 0), (777, 150)])
def test_all_exact_is_dense(T, prefix):
    q, k, v = _qkv(T, 2, seed=1)
    dense = mx.fast.scaled_dot_product_attention(*(x.transpose(0, 2, 1, 3) for x in (q, k, v)), scale=128 ** -0.5)
    assert _rel(sol_attention(q, k, v, prefix, tau=-1e30), dense.transpose(0, 2, 1, 3)) < 4e-3


def test_routing_matches_reference_formulas():
    """Independent NumPy port of the Triton reference preprocessing (`triton_ref/preprocess.py`, exact threshold) and
    routing (`triton_ref/fwd.py`): routes may differ only for scores within float rounding of the threshold."""
    T, H, prefix, tau = 1500, 2, 200, 1.0
    q, k, v = _qkv(T, H, seed=2)
    kc, vm, lens, exact = sol_route(q[0], k[0], v[0], prefix, tau)
    qn, kn = (np.array(x[0].astype(mx.float32)) for x in (q, k))
    N = -(-T // BLOCK)
    sl2 = 128 ** -0.5 * LOG2E
    pad = lambda x: np.concatenate([x, np.zeros((N * BLOCK - T,) + x.shape[1:], x.dtype)]).reshape(N, BLOCK, H, 128)
    ln = np.minimum(T - np.arange(N) * BLOCK, BLOCK)[:, None, None]
    qbar, kcn = pad(qn).sum(1) / ln, pad(kn).sum(1) / ln
    kcn = np.array(mx.array(kcn).astype(mx.bfloat16).astype(mx.float32))
    assert np.allclose(np.array(kc.astype(mx.float32)), kcn, rtol=8e-3, atol=1e-6)  # one bf16 ulp
    margin, route = [], np.zeros((H, N, N), bool)
    for h in range(H):
        kh = kcn[:, h]
        mean, second = kh.mean(0), kh.T @ kh / N
        qb = qbar[:, h]
        raw_mean, raw_sm = qb @ mean, np.einsum("nd,de,ne->n", qb, second, qb)
        thr = raw_mean * sl2 + tau * np.sqrt(np.maximum(raw_sm - raw_mean ** 2, 0) * sl2 ** 2 + 1e-6)
        scores = qb @ kh.T * sl2
        i = np.arange(N)
        route[h] = (scores > thr[:, None]) | (np.abs(i[:, None] - i[None]) <= 1) | (i[None] < -(-prefix // BLOCK))
        margin.append(np.abs(scores - thr[:, None]))
    diff = route != np.array(exact)
    assert (np.stack(margin)[diff] < 2e-2).all()  # bf16 rounding of q_bar / second moment in the reference
    assert diff.mean() < 5e-3
