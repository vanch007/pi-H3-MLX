"""Time one full-size H3 transformer block (random weights) at real sequence lengths."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx, mlx.nn as nn, numpy as np
from pih3_mlx.transformer import TransformerBlock, quantize

def bench(S, bits=None, reps=3):
    blk = TransformerBlock(5376, 56, 128, 14336, 8, 1e-5)
    blk.update(nn.utils.tree_map(lambda p: (mx.random.normal(p.shape) * 0.02).astype(mx.bfloat16), blk.parameters()))
    if bits:
        nn.quantize(blk, 64, bits)
    mx.eval(blk.parameters())
    x = mx.random.normal((1, S, 5376)).astype(mx.bfloat16)
    temb = mx.random.normal((3, 8))
    runs = [(0, 1200, 4), (1200, S, 0)]
    cos = mx.random.normal((1, S, 1, 96)).astype(mx.bfloat16); sin = cos
    mx.eval(x, temb, cos)
    blk(x, temb, runs, (cos, sin)); mx.eval(x)
    ts = []
    for _ in range(reps):
        t = time.time(); y = blk(x, temb, runs, (cos, sin)); mx.eval(y); ts.append(time.time() - t)
    flops = 2 * S * 5376 * (4 * 7168 + 3 * 14336) + 4 * S * S * 7168
    dt = min(ts)
    print(f"S={S:6d} bits={bits}: {dt*1000:8.1f} ms/block  {flops/dt/1e12:5.1f} TFLOPS  -> 50 blocks/step {dt*50:6.1f}s  peak {mx.get_peak_memory()/1e9:.1f}GB", flush=True)
    mx.reset_peak_memory()

for S in (17000, 40000, 73500):
    bench(S)
bench(17000, 8)
