"""Sol sparse attention on Apple silicon (custom Metal kernel), forward only.

Port of LakonLab `lakonlab/ops/sol_attn` (Triton reference `triton_ref/fwd.py`, routing `bsa_metadata.py`) with
`thresh_type='exact'` and the H3 prefix sink (`lakonlab/ops/attention.py::sol_attention`):

* tokens are split into 64-token blocks; each KV block j gets a summary key kc_j = mean(k) and value sum vc_j;
* query block i attends *exactly* to KV block j when the routing score
  mean_{q in i}(q . kc_j) * scale * log2(e) exceeds a per-(i, head) threshold
  (mean + tau * std of q_bar . kc over all j, from the first and second moments of kc), when |i - j| <= 1, or when
  block j overlaps the prefix sink `[0, prefix)`;
* every other block contributes its summary as len_j identical keys kc_j with mean value vc_j / len_j
  (implemented as one key with additive logit bias ln(len_j));
* query rows inside the prefix (text, keyframe conditions, audio) get dense attention over all keys.

The kernel is MLX's steel flash-attention kernel (32-row query tile per threadgroup, 4 simdgroups x 8 rows,
16-key tiles, 8x8 simdgroup MMAs) with its KV loop replaced by the route: exact blocks, then block summaries.
"""

import os
import re

import mlx.core as mx

BLOCK = 64
LOG2E = 1.4426950408889634

def _steel_header() -> str:
    """MLX's own steel attention building blocks (MIT, Apple), inlined from the installed package's headers."""
    import mlx

    root = os.path.join(list(mlx.__path__)[0], "include")
    seen = {"mlx/backend/metal/kernels/utils.h", "mlx/backend/metal/kernels/bf16.h",
            "mlx/backend/metal/kernels/bf16_math.h"}  # already in the custom-kernel preamble

    def inline(path):
        if path in seen:
            return ""
        seen.add(path)
        out = []
        for line in open(os.path.join(root, path)):
            m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
            if m:
                out.append(inline(m.group(1)))
            elif line.strip() != "#pragma once":
                out.append(line)
        return "".join(out)

    return inline("mlx/backend/metal/kernels/steel/attn/attn.h")


_HEADER = _steel_header() + r"""
struct SolMaxOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return metal::max(x, y); } };
struct SolSumOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return x + y; } };
struct SolMulOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return x * y; } };
struct SolDivOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return x / y; } };
struct SolExpSubOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return fast::exp2(x - y); } };
"""

# Adapted from MLX `steel/attn/kernels/steel_attention.h` (WM=4, WN=1, TQ=1): the KV loop walks the exact 64-token
# blocks of this query block's route, then the block summaries with a log2(len) logit bias.
_SOURCE = r"""
  using namespace mlx::steel;
  typedef bfloat16_t T;
  typedef float AccumType;
  constexpr int BD = 128, BV = 128, WM = 4;
  constexpr short padQ = 8, padK = 8, padV = 8;
  constexpr short LDQ_tgp = BD + padQ, LDK_tgp = BK + padK, LDV_tgp = BV + padV;
  constexpr short tgp_mem_0 = (BK + padK) * BD;
  constexpr short tgp_mem_1 = BK * (BV + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;
  threadgroup T Qs[BQ * LDQ_tgp];
  threadgroup T KV_smem[tgp_mem_s];
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  const int TT = dims[0], N = dims[1], H = dims[2];
  const int HD = H * BD;
  const AccumType scale = scale_log2[0];
  const ushort simd_group_id = simdgroup_index_in_threadgroup;
  const ushort simd_lane_id = thread_index_in_simdgroup;
  const int qt = threadgroup_position_in_grid.x, h = threadgroup_position_in_grid.y;
  const int q0 = qt * BQ;
  const int qb = q0 / 64;
  auto route = exact + ((long)h * N + qb) * N;  // device or constant (tiny inputs)

  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, WM * 32>;
  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, WM * 32>;
  using VBlockLoader = BlockLoaderT<T, BK, BV, LDV_tgp, 1, 0, WM * 32>;

  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  constexpr int TK = BK / kFragSize, TD = BD / kFragSize, TV = BV / kFragSize;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, 1, TV, MMAFrag_acc_t> Otile;
  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y, sn = simd_coord.x;
  const short tm = kFragSize * simd_group_id;
  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  {
    QBlockLoader loader_q(q + (long)q0 * HD + h * BD, HD, Qs, simd_group_id, simd_lane_id);
    if (q0 + BQ > TT) loader_q.load_safe(short2(BD, TT - q0)); else loader_q.load_unsafe();
  }

  AccumType max_score = Limits<AccumType>::finite_min;
  AccumType sum_score = 0;
  constexpr AccumType neg_inf = Limits<AccumType>::finite_min;

  int j = 0;
  while (j < N && !route[j]) ++j;
  int sub = 0, sstart = 0;
  while (true) {
    const device T* srcK; const device T* srcV;
    int start, limit;
    bool is_exact;
    if (j < N) {
      is_exact = true; srcK = k; srcV = v; limit = TT;
      start = j * 64 + sub * BK;
      ++sub;
      if (sub == 64 / BK || start + BK >= TT) { sub = 0; ++j; while (j < N && !route[j]) ++j; }
    } else {
      if (sstart >= N) break;
      is_exact = false; srcK = kc; srcV = vm; limit = N;
      start = sstart; sstart += BK;
    }
    const int rem = limit - start;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      KBlockLoader loader_k(srcK + (long)start * HD + h * BD, HD, Ks, simd_group_id, simd_lane_id);
      if (rem < BK) loader_k.load_safe(short2(BD, rem)); else loader_k.load_unsafe();
    }
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * kFragSize]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(&Ks[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }
    STEEL_PRAGMA_UNROLL
    for (short jt = 0; jt < TK; jt++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < 2; jj++) {
        const int col = start + sn + jt * kFragSize + jj;
        AccumType s = Stile.frag_at(0, jt)[jj] * scale;
        if (is_exact) {
          if (col >= TT) s = neg_inf;
        } else {
          s = (col >= N || route[col]) ? neg_inf : s + log2len[col];
        }
        Stile.frag_at(0, jt)[jj] = s;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      VBlockLoader loader_v(srcV + (long)start * HD + h * BD, HD, Vs, simd_group_id, simd_lane_id);
      if (rem < BK) loader_v.load_safe(short2(BV, rem)); else loader_v.load_unsafe();
    }

    AccumType new_max = max_score;
    Stile.template row_reduce<SolMaxOp>(&new_max);
    Stile.template row_bin_op<SolExpSubOp>(&new_max);
    // P enters the MMA rounded to bf16, as in the reference kernels
    STEEL_PRAGMA_UNROLL
    for (short jt = 0; jt < TK; jt++) {
      Stile.frag_at(0, jt)[0] = AccumType(T(Stile.frag_at(0, jt)[0]));
      Stile.frag_at(0, jt)[1] = AccumType(T(Stile.frag_at(0, jt)[1]));
    }
    AccumType factor = fast::exp2(max_score - new_max);
    max_score = new_max;
    AccumType sum_tmp = 0;
    Stile.template row_reduce<SolSumOp>(&sum_tmp);
    sum_score = sum_score * factor + sum_tmp;
    Otile.template row_bin_op<SolMulOp>(&factor);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TV; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        simdgroup_barrier(mem_flags::mem_none);
        Vtile.template load<T, 1, 1, LDV_tgp, 1>(&Vs[Vs_offset + ik * kFragSize * LDV_tgp + id * kFragSize]);
        simdgroup_barrier(mem_flags::mem_none);
        MMAFrag_acc_t::mma(Otile.frag_at(0, id), Stile.frag_at(0, ik), Vtile.frag_at(0, 0), Otile.frag_at(0, id));
      }
    }
  }

  Otile.template row_bin_op<SolDivOp>(&sum_score);
  threadgroup_barrier(mem_flags::mem_none);
  device T* O = out + (long)(q0 + tm + sm) * HD + h * BD + sn;
  if (q0 + BQ > TT) {
    auto dims2 = short2(BV - sn, TT - (q0 + tm + sm));
    if (dims2.x > 0 && dims2.y > 0) Otile.template store_safe<T, 1, 1>(O, HD, dims2);
  } else {
    Otile.template store<T, 1, 1>(O, HD);
  }
"""

_kernel = mx.fast.metal_kernel(
    name="pih3_sol_attention",
    input_names=["q", "k", "v", "kc", "vm", "log2len", "exact", "dims", "scale_log2"],
    output_names=["out"],
    source=_SOURCE,
    header=_HEADER,
)
BQ, BK = 32, 16
DENSITY_LOG = None  # set to a list to collect each call's fraction of exact blocks (mx scalars)


def _block_sums(x, T, N):
    """(T, H, D) -> float32 per-block sums (N, H, D) over the valid rows."""
    pad = N * BLOCK - T
    xf = x.astype(mx.float32)
    if pad:
        xf = mx.concatenate([xf, mx.zeros((pad,) + x.shape[1:], mx.float32)])
    return xf.reshape(N, BLOCK, *x.shape[1:]).sum(1)


def sol_route(q, k, v, prefix: int, tau: float = 1.0, scale: float | None = None):
    """Block summaries and routing for (T, H, 128) bf16 q/k/v.

    Returns kc (N, H, D) bf16 mean keys, vm (N, H, D) bf16 mean values, block lengths (N,) float32 and the exact-block mask
    (H, N, N) bool, matching the reference's preprocessing (bf16 summaries, `exact` thresholds)."""
    T, H, D = q.shape
    scale = D ** -0.5 if scale is None else scale
    sl2 = scale * LOG2E
    N = -(-T // BLOCK)
    lens = mx.minimum(T - mx.arange(N) * BLOCK, BLOCK).astype(mx.float32)
    inv = (1.0 / lens)[:, None, None]
    qbar = _block_sums(q, T, N) * inv  # float32 pooled queries
    kc = (_block_sums(k, T, N) * inv).astype(mx.bfloat16)
    vc = _block_sums(v, T, N).astype(mx.bfloat16)  # the reference stores bf16 sums
    vm = (vc.astype(mx.float32) * inv).astype(mx.bfloat16)

    kch = kc.astype(mx.float32).transpose(1, 0, 2)  # (H, N, D)
    mean = kch.mean(1)  # (H, D)
    second = (kch.transpose(0, 2, 1) @ kch / N).astype(mx.bfloat16).astype(mx.float32)  # (H, D, D)
    qb = qbar.astype(mx.bfloat16).astype(mx.float32).transpose(1, 0, 2)  # (H, N, D)
    raw_mean = (qb * mean[:, None]).sum(-1)
    raw_sm = ((qb @ second) * qb).sum(-1)
    var = mx.maximum(raw_sm - raw_mean * raw_mean, 0.0) * (sl2 * sl2)
    thr = raw_mean * sl2 + tau * mx.sqrt(var + 1e-6)  # (H, N)

    scores = (qbar.transpose(1, 0, 2) @ kch.transpose(0, 2, 1)) * sl2  # (H, N, N)
    idx = mx.arange(N)
    near = mx.abs(idx[:, None] - idx[None, :]) <= 1
    sink = idx[None, :] < -(-prefix // BLOCK)
    exact = (scores > thr[:, :, None]) | near[None] | sink[None]
    return kc, vm, lens, exact


def sol_attention(q, k, v, prefix: int = 0, tau: float = 1.0, scale: float | None = None, return_density=False):
    """Sol attention for (1, T, H, 128) bf16 q/k/v (BTHD, as produced by the H3 attention) -> (1, T, H, 128)."""
    if q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError("sol_attention expects (1, T, H, 128)")
    q, k, v = (x[0].astype(mx.bfloat16) for x in (q, k, v))
    T, H, D = q.shape
    scale = D ** -0.5 if scale is None else scale
    kc, vm, lens, exact = sol_route(q, k, v, prefix, tau, scale)
    N = kc.shape[0]
    out = _kernel(  # the kernel works in the log2 domain: summary bias log2(len)
        # one padding element each: size-1 inputs would be bound as `constant` scalars instead of device arrays
        inputs=[q, k, v, kc, vm, mx.concatenate([mx.log2(lens), mx.zeros((1,))]),
                mx.concatenate([exact.astype(mx.uint8).reshape(-1), mx.zeros((1,), mx.uint8)]),
                mx.array([T, N, H], mx.int32),
                mx.array([scale * LOG2E], mx.float32)],
        grid=(128 * (-(-T // BQ)), H, 1),
        threadgroup=(128, 1, 1),
        template=[("BQ", BQ), ("BK", BK)],
        output_shapes=[(T, H, D)],
        output_dtypes=[mx.bfloat16],
    )[0]
    if prefix:
        qt, kt, vt = (x.transpose(1, 0, 2)[None] for x in (q[:prefix], k, v))
        dense = mx.fast.scaled_dot_product_attention(qt, kt, vt, scale=scale)[0].transpose(1, 0, 2)
        out = mx.concatenate([dense, out[prefix:]])
    out = out[None]
    if DENSITY_LOG is not None:
        DENSITY_LOG.append(exact.astype(mx.float32).mean())
    if return_density:
        return out, exact.astype(mx.float32).mean()
    return out


def sol_reference(q, k, v, prefix: int = 0, tau: float = 1.0, scale: float | None = None):
    """Slow float32 reference of `sol_attention` (same routing), one query block at a time. Testing only."""
    q, k, v = (x[0] for x in (q, k, v))
    T, H, D = q.shape
    scale = D ** -0.5 if scale is None else scale
    kc, vm, lens, exact = sol_route(q, k, v, prefix, tau, scale)
    N = kc.shape[0]
    qf, kf, vf = (x.astype(mx.float32).transpose(1, 0, 2) for x in (q, k, v))  # (H, T, D)
    kcf, vmf = (x.astype(mx.float32).transpose(1, 0, 2) for x in (kc, vm))  # (H, N, D)
    tok_block = mx.arange(T) // BLOCK
    outs = []
    for i in range(N):
        qi = qf[:, i * BLOCK:(i + 1) * BLOCK]
        ex = exact[:, i]  # (H, N)
        tok_mask = mx.take(ex, tok_block, axis=1)  # (H, T)
        s_tok = (qi @ kf.transpose(0, 2, 1)) * scale + mx.where(tok_mask, 0.0, -mx.inf)[:, None]
        s_sum = (qi @ kcf.transpose(0, 2, 1)) * scale + mx.where(ex, -mx.inf, mx.log(lens)[None])[:, None]
        s = mx.concatenate([s_tok, s_sum], -1)
        p = mx.softmax(s, axis=-1)
        outs.append(p[..., :T] @ vf + p[..., T:] @ vmf)
    out = mx.concatenate(outs, 1).transpose(1, 0, 2)
    if prefix:
        qt, kt, vt = (x.astype(mx.float32).transpose(1, 0, 2)[None] for x in (q[:prefix], k, v))
        dense = mx.fast.scaled_dot_product_attention(qt, kt, vt, scale=scale)[0].transpose(1, 0, 2)
        out = mx.concatenate([dense, out[prefix:]])
    return out[None]
