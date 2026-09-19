"""Encode reference pixels in 17-frame clips and overlapping spatial tiles."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, cast

import numpy
from numpy.typing import NDArray

from nano_omni.models.h3.video_vae_encoder import (
    H3VideoVaeEncoder,
    H3VideoVaeEncoderArgs,
)
from nano_omni.models.h3.video_vae_tiles import spatial_tiles

if TYPE_CHECKING:
    from nano_omni.core.runtime.buffers import PipelineMemory

REFERENCE_TILE_BATCH = 32


def reference_tile_count(
    frames: int, height: int, width: int, tile_size: int
) -> int:
    """Count temporal and spatial encoder volumes."""
    return (
        math.ceil(frames / 17)
        * len(spatial_tiles(height, tile_size)[0])
        * len(spatial_tiles(width, tile_size)[0])
    )


def reference_batch_counts(
    frames: int, height: int, width: int, tile_size: int
) -> tuple[int, ...]:
    """Split reference tiles into bounded GPU input batches."""
    count = reference_tile_count(frames, height, width, tile_size)
    full, remainder = divmod(count, REFERENCE_TILE_BATCH)
    return (REFERENCE_TILE_BATCH,) * full + ((remainder,) if remainder else ())


def blend(
    previous: NDArray[numpy.float32],
    current: NDArray[numpy.float32],
    overlap: int,
    axis: Literal[2, 3],
) -> NDArray[numpy.float32]:
    """Blend the leading spatial overlap without changing either raw latent tile."""
    a, b = [slice(None)] * 4, [slice(None)] * 4
    a[axis], b[axis] = slice(-overlap, None), slice(0, overlap)
    shape = [1] * 4
    shape[axis] = overlap
    weight = numpy.arange(overlap, dtype=numpy.float32).reshape(shape) / overlap
    result = current.copy()
    result[tuple(b)] = previous[tuple(a)] * (1 - weight) + current[tuple(b)] * weight
    return result


class ReferenceVideoEncoder:
    def __init__(
        self,
        models: dict[int, H3VideoVaeEncoder],
        memory: PipelineMemory,
        tile_size: int,
    ) -> None:
        self.models = models
        self.memory = memory
        self.tile_size = tile_size

    def encode_batch(
        self, inputs: list[NDArray[numpy.float32]]
    ) -> list[NDArray[numpy.float32]]:
        """Encode one bounded batch and return its equally shaped latent tiles."""
        model = self.models[len(inputs)]
        packed_input = numpy.concatenate([value.reshape(-1) for value in inputs])
        source = self.memory.upload(packed_input)
        frames, height, width = model.spec.shape
        shape = (24, (frames + 3) // 4, height // 16, width // 16)
        size = math.prod(shape)
        output = self.memory.empty(
            (len(inputs) * size * numpy.dtype(numpy.float32).itemsize,), numpy.uint8
        )
        try:
            model.run(H3VideoVaeEncoderArgs(source, output, self.memory.runtime))
            packed = cast(NDArray[numpy.float32], self.memory.download(output)).view(
                numpy.float32
            )
        finally:
            self.memory.release(source, output)
        return [
            packed[start : start + size].reshape(shape)
            for start in range(0, len(packed), size)
        ]

    def encode(self, frames: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
        """Encode every tile under one weight plan, then stitch the downloaded results."""
        tile_size = self.tile_size
        y, y_overlap = spatial_tiles(frames.shape[1], tile_size)
        x, x_overlap = spatial_tiles(frames.shape[2], tile_size)
        # Starts are latent coordinates; overlaps from the shared helper are pixels.
        y_overlap = tuple(overlap // 16 for overlap in y_overlap)
        x_overlap = tuple(overlap // 16 for overlap in x_overlap)
        counts = reference_batch_counts(
            len(frames), frames.shape[1], frames.shape[2], tile_size
        )
        assert set(self.models) == set(counts)
        inputs: list[NDArray[numpy.float32]] = []
        encoded: list[NDArray[numpy.float32]] = []
        batch = iter(counts)
        limit = next(batch)
        for start in range(0, len(frames), 17):
            clip = frames[start : start + 17]
            if len(clip) < 17:
                clip = numpy.concatenate(
                    (clip, numpy.repeat(clip[-1:], 17 - len(clip), axis=0))
                )
            for top in y:
                for left in x:
                    tile = clip[
                        :,
                        top * 16 : top * 16 + min(tile_size, frames.shape[1]),
                        left * 16 : left * 16 + min(tile_size, frames.shape[2]),
                    ]
                    value = numpy.ascontiguousarray(tile.transpose(3, 0, 1, 2) * 2 - 1)
                    inputs.append(value)
                    if len(inputs) == limit:
                        encoded.extend(self.encode_batch(inputs))
                        inputs.clear()
                        limit = next(batch, 0)
        assert not inputs and not limit

        chunks: list[NDArray[numpy.float32]] = []
        cursor = 0
        for _ in range(0, len(frames), 17):
            previous: list[NDArray[numpy.float32]] = []
            stitched: list[NDArray[numpy.float32]] = []
            for i in range(len(y)):
                row = encoded[cursor : cursor + len(x)]
                cursor += len(x)
                tiles = []
                for j, raw in enumerate(row):
                    tile = raw
                    # Both neighbors stay unblended: vertical first, horizontal second.
                    if i:
                        tile = blend(previous[j], tile, y_overlap[i - 1], 2)
                    if j:
                        tile = blend(row[j - 1], tile, x_overlap[j - 1], 3)
                    if i + 1 < len(y):
                        tile = tile[:, :, : -y_overlap[i]]
                    if j + 1 < len(x):
                        tile = tile[:, :, :, : -x_overlap[j]]
                    tiles.append(tile)
                stitched.append(numpy.concatenate(tiles, axis=3))
                previous = row
            chunks.append(numpy.concatenate(stitched, axis=2))
        return numpy.ascontiguousarray(numpy.concatenate(chunks, axis=1)[:, :-3])
