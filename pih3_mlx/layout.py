"""Packed-sequence layout of MiniMax H3: `[text | keyframe conditions | audio | video]`.

numpy port of `diffusers/modular_pipelines/minimax_h3/{before_denoise,modular_pipeline}.py` (t2va / fl2va only).
Everything here is static per request and computed once on the CPU.
"""

from dataclasses import dataclass

import numpy as np

VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
MODALITY_NUM = 3
FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
AUDIO_LATENT_CHANNELS = 32
VAE_LATENT_CHANNELS = 24
VAE_SPATIAL_RATIO = 16
FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 17, 5
PATCH_SIZE = (1, 2, 2)
CANVAS_MULTIPLE = VAE_SPATIAL_RATIO * PATCH_SIZE[2]
MIN_DURATION, MAX_DURATION = 5.0, 15.0
KEYFRAME_NOISE_AUG = 0.999

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


def align_num_frames(num_frames: int) -> int:
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    assert num_frames % FRAMES_PER_CHUNK == LATENTS_PER_CHUNK
    return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def _spatial_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE


def _temporal_grid(num_latent_frames: int, origin: float) -> np.ndarray:
    spans = np.array([_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % 5] for i in range(num_latent_frames)])
    # torch cumsum of float64 is sequential, like np.cumsum
    return origin + np.concatenate([[0.0], np.cumsum(spans[:-1])])


@dataclass
class Layout:
    position_ids: np.ndarray  # (S, 3) float64
    token_tags: np.ndarray  # (S,) int
    video_indices: np.ndarray  # (cond + target video rows,)
    audio_indices: np.ndarray
    text_indices: np.ndarray
    num_text: int
    num_condition_video_rows: int
    num_audio_rows: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int

    @property
    def seq_len(self) -> int:
        return self.position_ids.shape[0]

    @property
    def token_grid(self) -> tuple[int, int, int]:
        return self.num_latent_frames, self.latent_height // PATCH_SIZE[1], self.latent_width // PATCH_SIZE[2]

    @property
    def prefix_len(self) -> int:
        """Rows before the target video: text, conditions, audio (the reference's `sol_prefix`)."""
        return self.num_text + self.num_condition_video_rows + self.num_audio_rows


def build_layout(text_token_tags, num_frames: int, height: int, width: int, keyframe_anchors=()) -> Layout:
    if height % CANVAS_MULTIPLE or width % CANVAS_MULTIPLE:
        raise ValueError(f"height/width must be multiples of {CANVAS_MULTIPLE}, got {height}x{width}")
    num_frames = align_num_frames(num_frames)
    if not MIN_DURATION <= num_frames / FPS <= MAX_DURATION:
        raise ValueError(f"duration must be within [{MIN_DURATION}, {MAX_DURATION}] s, got {num_frames} frames")
    T = video_latent_num_frames(num_frames)
    lh, lw = height // VAE_SPATIAL_RATIO, width // VAE_SPATIAL_RATIO
    A = audio_latent_num_frames(num_frames)
    _, ph, pw = PATCH_SIZE
    text_token_tags = np.asarray(text_token_tags, dtype=np.int64)
    n_text = text_token_tags.shape[0]
    rows_per_frame = (lh // ph) * (lw // pw)
    n_cond = len(keyframe_anchors) * rows_per_frame
    n_audio = A * AUDIO_CHANNELS
    n_video = T * rows_per_frame
    S = n_text + n_cond + n_audio + n_video
    cond_start, audio_start = n_text, n_text + n_cond
    video_start = audio_start + n_audio

    pos = np.zeros((S, 3), dtype=np.float64)
    pos[:n_text, 0] = np.arange(n_text, dtype=np.float64)
    sqrt_area = np.sqrt(lh * lw)
    hg, wg = _spatial_grid(lh, ph, sqrt_area), _spatial_grid(lw, pw, sqrt_area)
    gh, gw = np.meshgrid(hg, wg, indexing="ij")
    frame_grid = np.stack([gh.reshape(-1), gw.reshape(-1)], -1)

    for i, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            t0 = float(n_text)
        elif anchor == "last":
            spans = np.ones(T, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for off in range(5):
                spans[off::5] *= _ROPE_FRAMES_PER_LATENT[off]
            t0 = float(n_text) + float(spans.sum()) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"bad keyframe anchor {anchor!r}")
        rows = slice(cond_start + i * rows_per_frame, cond_start + (i + 1) * rows_per_frame)
        pos[rows, 0] = t0
        pos[rows, 1:] = frame_grid

    audio_time = float(n_text) + np.arange(A, dtype=np.float64)
    pos[audio_start:video_start, 0] = np.tile(audio_time, AUDIO_CHANNELS)
    pos[audio_start:video_start, 2] = np.concatenate([np.full(A, wg[0]), np.full(n_audio - A, wg[-1])])

    vp = np.empty((T, rows_per_frame, 3), dtype=np.float64)
    vp[:, :, 0] = _temporal_grid(T, float(n_text))[:, None]
    vp[:, :, 1:] = frame_grid[None]
    pos[video_start:] = vp.reshape(-1, 3)

    video_indices = np.concatenate([np.arange(cond_start, audio_start), np.arange(video_start, S)])
    audio_indices = np.arange(audio_start, video_start)
    text_indices = np.arange(n_text)
    tags = np.empty(S, dtype=np.int64)
    tags[text_indices] = text_token_tags
    tags[audio_indices] = AUDIO_TAG
    tags[video_indices] = VIDEO_TAG
    return Layout(pos, tags, video_indices, audio_indices, text_indices, n_text, n_cond, n_audio, T, lh, lw, A)


def row_timesteps(layout: Layout, video_sigma: float, audio_sigma: float):
    """Per-row noise level in the scheduler's sigma convention (pipeline_minimax_h3 SetTimestepsStepMod), reduced to
    `(unique transformer timesteps t = 1 - sigma, index of every row)`."""
    rt = np.full(layout.seq_len, video_sigma, dtype=np.float32)
    rt[layout.video_indices[: layout.num_condition_video_rows]] = min(video_sigma, 1 - KEYFRAME_NOISE_AUG)
    rt[layout.audio_indices] = audio_sigma
    uniq, inv = np.unique(rt, return_inverse=True)
    return (1 - uniq).astype(np.float32), inv.astype(np.int64)


def adaln_runs(adaln_indices: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous `(start, end, table_row)` runs of a per-row AdaLN index; H3 layouts have only a handful."""
    change = np.flatnonzero(np.diff(adaln_indices)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [adaln_indices.shape[0]]])
    return [(int(s), int(e), int(adaln_indices[s])) for s, e in zip(starts, ends)]


def patchify(latents: np.ndarray) -> np.ndarray:
    """(1, C, T, H, W) -> (T*H/2*W/2, C*4) rows, frame-major then row-major (patchify_video_latents)."""
    _, c, t, h, w = latents.shape
    x = latents.reshape(1, c, t, 1, h // 2, 2, w // 2, 2).transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * 4)


def unpatchify(rows: np.ndarray, t: int, h: int, w: int, c: int = VAE_LATENT_CHANNELS) -> np.ndarray:
    x = rows.reshape(1, t, h // 2, w // 2, c, 1, 2, 2).transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(1, c, t, h, w)
