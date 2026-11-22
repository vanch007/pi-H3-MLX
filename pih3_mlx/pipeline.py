"""End-to-end pi-H3 text/image/first-last-frame -> video + audio on MLX (DWD adapters: no video VAE decoder).

Mirrors `LakonLab/lakonlab/pipelines/pipeline_minimax_h3.py` (`MiniMaxH3DWDBlocks`) with the Euler scheduler of
`demo/example_pih3_pipeline.py`. `PiH3Engine` keeps components resident across requests (or loads,
uses and frees them one at a time for the lowest peak memory); `generate` is the one-shot wrapper.
"""

import gc
import os
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from . import layout as L
from .convert import MODELS, load_converted, mlx_model_dir
from . import sol_attn
from .dwd_head import color_decode
from .scheduler import pih3_schedules


def _log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def _free():
    gc.collect()
    mx.clear_cache()


class _Noise:
    """Gaussian noise in the reference's draw order. Uses torch's CPU generator when available so a seed gives the
    same noise as the PyTorch pipeline; otherwise MLX's."""

    def __init__(self, seed):
        try:
            import torch
            self.g, self.torch = torch.Generator().manual_seed(seed), torch
        except ImportError:
            self.torch, self.key = None, mx.random.key(seed)

    def __call__(self, shape):
        if self.torch is not None:
            return self.torch.randn(shape, generator=self.g, dtype=self.torch.float32).numpy()
        self.key, k = mx.random.split(self.key)
        return np.array(mx.random.normal(shape, key=k))


@dataclass
class Result:
    video: np.ndarray  # (F, H, W, 3) uint8
    audio: np.ndarray  # (2, samples) float32
    sampling_rate: int
    timings: dict


class PiH3Engine:
    """pi-H3 components held across requests.

    keep_text_encoder / keep_transformer = True keep the 8-bit text encoder (~27 GB) and the transformer of the last
    used step count (~39 GB + 1.2 GB DWD head) resident, so only the first request pays the ~100 s of loading. Both
    False reproduce the load-use-free flow with the lowest peak memory (~40 GB at 512x896).
    """

    def __init__(self, keep_text_encoder: bool = True, keep_transformer: bool = True,
                 transformer_bits: int | None = None, text_encoder_bits: int | None = 8, models_dir: str = MODELS):
        self.keep_text_encoder, self.keep_transformer = keep_text_encoder, keep_transformer
        self.transformer_bits, self.text_encoder_bits, self.models_dir = transformer_bits, text_encoder_bits, models_dir
        self._enc = self._tf = self._tf_key = self._vae = self._avae = None

    # ---- components
    def _text_encoder(self):
        if self._enc is None:
            from .convert import text_encoder_dir
            from .text_encoder import load_converted_text_encoder, load_text_encoder
            m = self.models_dir
            tok, proc = os.path.join(m, "MiniMax-H3", "tokenizer"), os.path.join(m, "MiniMax-H3", "processor")
            conv = text_encoder_dir(self.text_encoder_bits)
            if os.path.exists(os.path.join(conv, "config.json")):  # pre-quantized: `convert --text-encoder`
                self._enc = load_converted_text_encoder(conv, tok, proc)
            else:
                self._enc = load_text_encoder(os.path.join(m, "MiniMax-H3-Pruned", "text_encoder"), tok, proc,
                                              quantize_bits=self.text_encoder_bits)
        return self._enc

    def _transformer(self, steps):
        key = (steps, self.transformer_bits)
        if self._tf_key != key:
            self._tf = self._tf_key = None
            _free()
            self._tf, self._tf_key = load_converted(mlx_model_dir(steps, self.transformer_bits)), key
        return self._tf

    def _video_vae(self):
        if self._vae is None:
            from .video_vae import load_video_vae_encoder
            self._vae = load_video_vae_encoder(os.path.join(self.models_dir, "MiniMax-H3", "vae"))
        return self._vae

    def _audio_vae(self):
        if self._avae is None:
            from .audio_vae import load_audio_vae
            self._avae = load_audio_vae(os.path.join(self.models_dir, "MiniMax-H3", "audio_vae"))
        return self._avae

    def unload(self):
        self._enc = self._tf = self._tf_key = self._vae = self._avae = None
        _free()

    # ---- one request
    def generate(self, prompt: str, image=None, last_image=None, height=None, width=None, num_frames: int = 121,
                 steps: int = 4, seed: int = 42, sparse_attention: bool = True) -> Result:
        timings, t_all = {}, time.time()

        # 1. keyframes and canvas
        from .video_vae import prepare_keyframes, resolve_canvas_size
        if image is not None or last_image is not None:
            keyframes, anchors, height, width = prepare_keyframes(
                image=None if image is None else image.convert("RGB"),
                last_image=None if last_image is None else last_image.convert("RGB"), height=height, width=width)
        else:
            keyframes, anchors = [], ()
            if height is None:
                height, width = resolve_canvas_size(16, 9)
        _log(f"canvas {height}x{width}, {num_frames} frames, anchors {anchors}")

        # 2. text conditioning (Qwen3-VL hidden_states[50])
        t = time.time()
        from .text_encoder import encode_prompt
        prompt_embeds, text_tags = encode_prompt(self._text_encoder(), prompt, keyframes or None)
        mx.eval(prompt_embeds)
        if not self.keep_text_encoder:
            self._enc = None
        _free()
        timings["text_encoder"] = time.time() - t
        _log(f"text encoded: {prompt_embeds.shape[1]} tokens")

        # 3. keyframe latents
        cond_latents = []
        if keyframes:
            t = time.time()
            from .video_vae import encode_keyframes
            cond_latents = [np.array(z) for z in encode_keyframes(self._video_vae(), keyframes, height, width)]
            if not self.keep_text_encoder:
                self._vae = None
            _free()
            timings["vae_encoder"] = time.time() - t

        # 4. layout, noise, schedules
        lay = L.build_layout(text_tags, num_frames, height, width, anchors)
        noise = _Noise(seed)
        cond_rows = [L.patchify(L.KEYFRAME_NOISE_AUG * z + (1 - L.KEYFRAME_NOISE_AUG) * noise(z.shape))
                     for z in cond_latents]
        video = L.patchify(noise((1, L.VAE_LATENT_CHANNELS, lay.num_latent_frames, lay.latent_height,
                                  lay.latent_width)))
        audio = mx.array(noise((lay.num_audio_rows, L.AUDIO_LATENT_CHANNELS)))
        cond = mx.array(np.concatenate(cond_rows)) if cond_rows else None
        video = mx.array(video)
        sched_v, sched_a = pih3_schedules(steps)
        _log(f"sequence length {lay.seq_len} (video {video.shape[0]}, audio {lay.num_audio_rows}, "
             f"text {lay.num_text})")

        # 5. denoise
        t = time.time()
        model, head, cfg = self._transformer(steps)
        timings["load_transformer"] = time.time() - t
        model.sparse_attention = sparse_attention
        dtype = model.context_embedder.weight.dtype  # never quantized
        rope = model.rope_tables(lay.position_ids, dtype)
        text_hidden = model.encode_text(prompt_embeds.astype(dtype))
        mx.eval(text_hidden, *rope)
        feats = None
        sol_attn.DENSITY_LOG = densities = [] if sparse_attention else None
        t = time.time()
        for i in range(steps):
            ts = time.time()
            t_uniq, t_idx = L.row_timesteps(lay, float(sched_v.sigmas[i]), float(sched_a.sigmas[i]))
            rows = video if cond is None else mx.concatenate([cond, video])
            out_v, out_a, f = model(rows, audio, text_hidden, t_uniq, t_idx, lay, rope,
                                    return_features=i == steps - 1)
            video = sched_v.step(video, -out_v, i)
            audio = sched_a.step(audio, -out_a, i)
            feats = f if f is not None else feats
            mx.eval(video, audio, *([feats] if feats is not None else []))
            dens = ""
            if densities:
                timings.setdefault("sol_density", []).append(round(mx.mean(mx.stack(densities)).item(), 4))
                dens, densities[:] = f", Sol exact blocks {timings['sol_density'][-1]:.1%}", []
            _log(f"step {i + 1}/{steps}: {time.time() - ts:.1f}s{dens}")
        timings["denoise"] = time.time() - t
        sol_attn.DENSITY_LOG = None
        del model, text_hidden, rope
        _free()

        # 6. DWD pixels and audio
        t = time.time()
        pixels = head(feats, lay.token_grid)
        frames = (color_decode(pixels) * 255 + 0.5).astype(mx.uint8)
        frames = np.array(frames)[:L.align_num_frames(num_frames)]
        del pixels, feats, head
        if not self.keep_transformer:
            self._tf = self._tf_key = None
        _free()
        timings["dwd_decode"] = time.time() - t
        t = time.time()
        from .audio_vae import audio_rows_to_latents, decode_audio
        avae = self._audio_vae()
        wav = decode_audio(avae, audio_rows_to_latents(audio, lay.num_audio_latents))
        if not self.keep_transformer:
            self._avae = None
        timings["audio_decode"] = time.time() - t
        timings["total"] = time.time() - t_all
        _log(f"done: {frames.shape} video, {wav.shape} audio; {timings}")
        return Result(frames, wav, avae.sampling_rate, timings)


def generate(prompt: str, image=None, last_image=None, height=None, width=None, num_frames: int = 121,
             steps: int = 4, seed: int = 42, transformer_bits: int | None = None, text_encoder_bits: int | None = 8,
             sparse_attention: bool = True, models_dir: str = MODELS) -> Result:
    """One-shot generation (loads, uses and frees each component). Use `PiH3Engine` for repeated requests."""
    engine = PiH3Engine(False, False, transformer_bits, text_encoder_bits, models_dir)
    return engine.generate(prompt, image, last_image, height, width, num_frames, steps, seed, sparse_attention)


def save_mp4(result: Result, path: str, fps: int = L.FPS, crf: int = 18):
    """H.264 (yuv420p) + AAC via PyAV."""
    import av
    with av.open(path, "w") as c:
        vs = c.add_stream("libx264", rate=fps)
        F, H, W, _ = result.video.shape
        vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
        vs.options = {"crf": str(crf), "preset": "medium"}
        as_ = c.add_stream("aac", rate=result.sampling_rate)
        as_.layout = "stereo"
        for fr in result.video:
            for p in vs.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                c.mux(p)
        for p in vs.encode():
            c.mux(p)
        wav = np.ascontiguousarray(result.audio.astype(np.float32))
        n = 1024
        for s in range(0, wav.shape[1], n):
            af = av.AudioFrame.from_ndarray(wav[:, s:s + n], format="fltp", layout="stereo")
            af.sample_rate = result.sampling_rate
            af.pts = s
            for p in as_.encode(af):
                c.mux(p)
        for p in as_.encode():
            c.mux(p)
