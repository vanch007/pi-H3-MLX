# pi-H3-MLX

Native Apple-silicon (MLX + Metal) port of [pi-H3](https://huggingface.co/Lakonik/pi-H3): text-to-video,
image-to-video and first+last-frame-to-video **with synchronized audio** in 2–4 steps, built on
[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and the [LakonLab](https://github.com/Lakonik/LakonLab)
reference code. No PyTorch or CUDA at inference time.

- Weights: [vanch007/pi-H3-MLX](https://huggingface.co/vanch007/pi-H3-MLX) (4-step DWD transformer, 8-bit text
  encoder, VAEs, tokenizer)
- Highlights
  - **Sol sparse attention as a custom Metal kernel** built on MLX's steel attention templates: only ~17–24 % of
    64-token key blocks are computed exactly, 3.3× faster per block than MLX dense SDPA at 73k tokens.
  - DWD pixel head (no video VAE decoder), audio VAE decoder, video VAE keyframe encoder, Qwen3-VL text encoder.
  - Pre-quantized 8-bit text encoder (load 79 s → 31 s, peak 76 GB → 30 GB, identical output).
  - `PiH3Engine` keeps models resident across requests; batch CLI and a local HTTP service.

## Requirements

- Apple-silicon Mac with **≥ 96 GB unified memory** (128 GB recommended). Resident mode peaks at 80–105 GB;
  `--low-memory` loads one model at a time (~40 GB peak at 512×896, slower).
- Python 3.12, ~80 GB of disk for the weights.

## Install

```bash
git clone https://github.com/vanch007/pi-H3-MLX && cd pi-H3-MLX
python3.12 -m venv ~/.venvs/pih3 && ~/.venvs/pih3/bin/pip install -r requirements.txt
hf download vanch007/pi-H3-MLX --local-dir /path/to/pi-h3-weights
export PIH3_ROOT=/path/to/pi-h3-weights
```

`PIH3_ROOT` holds `models_mlx/` (converted weights) and `models/MiniMax-H3/` (VAEs, tokenizer, processor). Without
it, the folder that contains this repository is used.

## Generate

```bash
# text-to-video, 512x896, 5 s (121 frames -> 124), 4 steps
~/.venvs/pih3/bin/python scripts/generate.py "A chef tosses vegetables in a flaming wok, sizzling sounds" \
    --height 512 --width 896 --frames 121 --out outputs/chef.mp4

# image-to-video (canvas follows the image unless --height/--width are given)
~/.venvs/pih3/bin/python scripts/generate.py "The woman smiles and says hello" --image first.png --height 512 --width 896

# first + last frame
~/.venvs/pih3/bin/python scripts/generate.py "He walks to the whiteboard and draws a chart" \
    --image first.png --last-image last.png --height 512 --width 896

# many jobs, models loaded once
~/.venvs/pih3/bin/python scripts/generate.py --batch jobs.json

# resident local HTTP service (127.0.0.1 only)
~/.venvs/pih3/bin/python scripts/serve.py --warmup
curl -s localhost:8765/generate -d '{"prompt": "A fox in the snow", "height": 512, "width": 896}'
```

Frames are rounded up to 17n+5 and must be 5–15 s at 24 fps (121, 243, 345). `--steps 4` is the default and the
one to use for people; `--steps 2` is for fast drafts (it needs the 2-step weights, made with
`python -m pih3_mlx.convert --steps 2` from the original checkpoints). `--dense-attention` turns Sol off.

## Performance (MacBook Pro 16", M3 Max 40-core GPU, 128 GB, 4 steps, Sol on)

| Case | Tokens | Step (cool) | Denoise | Sol exact blocks |
|---|---|---|---|---|
| T2V 512×896, 5 s | 17k | 75 s | 5.0 min | 17–18 % |
| T2V 512×896, 5 s, dense attention | 17k | 104 s | 8.3 min | — |
| T2V 512×896, 10 s | 33k | 191 s | 16 min | 17 % |
| T2V 512×896, 14 s | 47k | 358 s | 26 min | 17 % |
| I2V 512×896, 5 s | 17k | 93 s | 8–10 min | 21 % |
| First+last frame 512×896, 5 s | 19k | 108 s | 8 min | 24 % |
| I2V 768×1344, 5 s | 40k | 376 s | 27 min | 20 % |

Laptops throttle under sustained load: step times rose 1.5–3.7× after several minutes on this machine, so the
denoise column includes some throttling. Text encoding, model loading and decoding add ~1–3 min on a cold start and
seconds on a resident engine.

## Layout

| Path | Contents |
|---|---|
| `pih3_mlx/layout.py` | packed `[text, cond, audio, video]` sequence, RoPE positions, per-row timesteps |
| `pih3_mlx/sol_attn.py` | Sol routing + Metal kernel |
| `pih3_mlx/transformer.py` | pruned H3 transformer, LoRA fusion, DWD features |
| `pih3_mlx/dwd_head.py` | DWD pixel decoder |
| `pih3_mlx/text_encoder.py` | Qwen3-VL conditioner (`hidden_states[50]`), 8-bit save/load |
| `pih3_mlx/video_vae.py`, `audio_vae.py` | keyframe encoder, audio decoder |
| `pih3_mlx/pipeline.py` | `PiH3Engine`, `generate`, `save_mp4` |
| `pih3_mlx/convert.py` | original HF checkpoints → MLX weights |
| `scripts/` | `generate.py`, `serve.py`, `real_matrix.py` (test matrix), `bench_block.py` |
| `tests/` | fp32 parity tests against the PyTorch reference (need torch, diffusers and LakonLab) |

## Converting from the original checkpoints

Download `MiniMaxAI/MiniMax-H3` (vae, audio_vae, tokenizer, processor), the pruned transformer and text encoder,
and `Lakonik/pi-H3` into `$PIH3_ROOT/models/{MiniMax-H3,MiniMax-H3-Pruned,pi-H3}`, then:

```bash
python -m pih3_mlx.convert --steps 4          # fused 4-step DWD transformer (bf16, 39 GB)
python -m pih3_mlx.convert --steps 2          # optional 2-step drafts
python -m pih3_mlx.convert --text-encoder     # 8-bit text encoder (27 GB)
```

## License

Code: Apache 2.0 (see `LICENSE` and `NOTICE`); ported from LakonLab and diffusers (Apache 2.0).

The weights are Model Derivatives of MiniMax H3 and are governed by the **MiniMax H3 Community License Agreement**,
including its Acceptable Use Policy and its territorial scope (it does not cover the EU, the UK, the Republic of
Korea or the USA). MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026
MiniMax. All Rights Reserved. Powered by MiniMax H3.

Credits: pi-H3 by Hansheng Chen et al.; MiniMax H3 by MiniMax; Qwen3-VL by the Qwen team; MLX by Apple.

---

## 中文简介

pi-H3 的 Apple 芯片原生 MLX 移植版，支持文生视频、图生视频和首尾帧生视频，并同步生成音频，只需 2–4 步。
Sol 稀疏注意力改写成了自定义 Metal 内核，文本编码器预先做了 8-bit 量化，引擎可以常驻内存。需要 96 GB 以上统一内存
（推荐 128 GB）。用法见上文：先 `hf download vanch007/pi-H3-MLX`，设置 `PIH3_ROOT`，再运行 `scripts/generate.py`。
权重受 MiniMax H3 社区许可证约束。
