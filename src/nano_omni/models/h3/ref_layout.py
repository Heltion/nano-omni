"""Place reference timelines before the generated audio and video streams."""

import dataclasses
import math
from typing import Literal

import numpy
from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceBlock:
    """Latent dimensions; video packs 2x2 patches and audio has two channels."""

    kind: Literal["image", "audio", "video", "video_audio"]
    frames: int = 0
    height: int = 0
    width: int = 0
    audio_frames: int = 0

    @property
    def video_tokens(self) -> int:
        return self.frames * self.height * self.width // 4

    @property
    def tokens(self) -> int:
        return self.video_tokens + 2 * self.audio_frames


def spatial_grid(
    height: int, width: int
) -> tuple[NDArray[numpy.float64], tuple[numpy.float64, numpy.float64]]:
    """Return packed-patch coordinates and the two audio channel positions."""
    area = math.sqrt(height * width)
    axes = [
        (
            numpy.arange(size // 2, dtype=numpy.float64) / (size // 2) * (size / area)
            + (1 - size / area) / 2
        )
        * 32
        for size in (height, width)
    ]
    h, w = numpy.meshgrid(*axes, indexing="ij")
    return numpy.stack((h.ravel(), w.ravel()), axis=1), (axes[1][0], axes[1][-1])


def video_spans(frames: int) -> NDArray[numpy.float64]:
    """Each five-latent group represents 1, 4, 4, 4, 4 source frames."""
    return numpy.where(numpy.arange(frames) % 5 == 0, 1.0, 4.0) * (5.0 / 3.0)


def positions(
    text_tokens: int,
    video_frames: int,
    height: int,
    width: int,
    audio_frames: int,
    blocks: tuple[ReferenceBlock, ...],
) -> tuple[NDArray[numpy.float32], tuple[tuple[int, int, int, int], ...]]:
    """Return [token, time/height/width] and (start, length, modality, timestep)."""
    target_grid, target_edges = spatial_grid(height, width)
    text = numpy.zeros((text_tokens, 3), dtype=numpy.float64)
    text[:, 0] = numpy.arange(text_tokens)
    parts = [text]
    segments: list[tuple[int, int, int, int]] = []
    cursor, row = float(text_tokens), text_tokens
    target = ReferenceBlock("video_audio", video_frames, height, width, audio_frames)
    for index, block in enumerate((*blocks, target)):
        reference = index < len(blocks)
        grid, edges = (
            spatial_grid(block.height, block.width)
            if reference and block.kind != "audio"
            else (target_grid, target_edges)
        )
        # A paired reference shares one start time, with channel-major audio first.
        if block.kind == "audio" or block.audio_frames or not reference:
            frames = block.audio_frames
            audio = numpy.zeros((2 * frames, 3), dtype=numpy.float64)
            audio[:, 0] = numpy.tile(cursor + numpy.arange(frames), 2)
            audio[:frames, 2], audio[frames:, 2] = edges
            parts.append(audio)
            segments.append((row, len(audio), 2, 3 if reference else 1))
            row += len(audio)
            if block.kind == "audio":
                cursor += frames
                continue
        video = numpy.empty((block.frames, len(grid), 3), dtype=numpy.float64)
        spans = video_spans(block.frames)
        times = cursor + numpy.concatenate(([0.0], numpy.cumsum(spans[:-1])))
        video[:, :, 0] = times[:, None]
        video[:, :, 1:] = grid[None]
        video = video.reshape(-1, 3)
        parts.append(video)
        segments.append((row, len(video), 0, 2 if reference else 0))
        row += len(video)
        cursor += (
            1.0
            if block.kind == "image"
            else max(block.audio_frames, float(spans.sum()))
        )
    # Accumulate fractional frame times in FP64 and round once at the upload boundary.
    return numpy.concatenate(parts).astype(numpy.float32), tuple(segments)
