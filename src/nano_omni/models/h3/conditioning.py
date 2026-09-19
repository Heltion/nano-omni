import dataclasses
from pathlib import Path

import numpy
from PIL import Image
from tokenizers import Tokenizer

VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655


@dataclasses.dataclass(frozen=True, slots=True)
class H3VisionInput:
    image: numpy.ndarray
    patches: numpy.ndarray
    position_indices: numpy.ndarray
    position_weights: numpy.ndarray
    rope_positions: numpy.ndarray
    inverse_frequencies: numpy.ndarray
    grid_height: int
    grid_width: int

    @property
    def merged_tokens(self) -> int:
        return self.grid_height * self.grid_width // 4


@dataclasses.dataclass(frozen=True, slots=True)
class H3Presentation:
    tokens: numpy.ndarray
    scatter_spans: tuple[tuple[int, int], ...]
    modality_spans: tuple[tuple[int, int], ...]
    mrope_positions: numpy.ndarray
    inverse_frequencies: numpy.ndarray


def load_keyframe(path: Path, width: int, height: int, *, cover: bool) -> numpy.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if cover:
            scale = max(width / image.width, height / image.height)
            resized = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
            left = (resized.width - width) // 2
            top = (resized.height - height) // 2
            image = resized.crop((left, top, left + width, top + height))
        else:
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        storage = numpy.frombuffer(image.tobytes(), dtype=numpy.uint8)
        return storage.reshape(1, height, width, 3).astype(numpy.float32) / 255.0


def float32_to_bf16(value: numpy.ndarray) -> numpy.ndarray:
    return (value.view(numpy.uint32) >> 16).astype(numpy.uint16)


def prepare_vision_input(image: numpy.ndarray) -> H3VisionInput:
    assert image.ndim == 4 and image.shape[0] in (1, 2) and image.shape[3] >= 3
    height, width = image.shape[1:3]
    factor = 32
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    assert (resized_height, resized_width) == (height, width)
    pixels = image[..., :3].transpose(0, 3, 1, 2) * 2.0 - 1.0
    grid_height = resized_height // 16
    grid_width = resized_width // 16
    patches = (
        (numpy.repeat(pixels, 2, axis=0) if image.shape[0] == 1 else pixels)
        .reshape(1, 2, 3, grid_height // 2, 2, 16, grid_width // 2, 2, 16)
        .transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        .reshape(grid_height * grid_width, 3 * 2 * 16 * 16)
    )
    patches = float32_to_bf16(numpy.ascontiguousarray(patches))

    side = 48
    rows = numpy.linspace(0, side - 1, grid_height, dtype=numpy.float32)
    columns = numpy.linspace(0, side - 1, grid_width, dtype=numpy.float32)
    row_floor = rows.astype(numpy.int32)
    column_floor = columns.astype(numpy.int32)
    row_ceil = numpy.minimum(row_floor + 1, side - 1)
    column_ceil = numpy.minimum(column_floor + 1, side - 1)
    row_fraction = rows - row_floor
    column_fraction = columns - column_floor
    indices = numpy.stack(
        (
            row_floor[:, None] * side + column_floor[None, :],
            row_floor[:, None] * side + column_ceil[None, :],
            row_ceil[:, None] * side + column_floor[None, :],
            row_ceil[:, None] * side + column_ceil[None, :],
        )
    )
    weights = numpy.stack(
        (
            (1 - row_fraction)[:, None] * (1 - column_fraction)[None, :],
            (1 - row_fraction)[:, None] * column_fraction[None, :],
            row_fraction[:, None] * (1 - column_fraction)[None, :],
            row_fraction[:, None] * column_fraction[None, :],
        )
    )

    def merge_order(value: numpy.ndarray) -> numpy.ndarray:
        return (
            value.reshape(4, grid_height // 2, 2, grid_width // 2, 2)
            .transpose(0, 1, 3, 2, 4)
            .reshape(4, -1)
        )

    block_rows = numpy.arange(grid_height // 2)
    block_columns = numpy.arange(grid_width // 2)
    intra = numpy.arange(2)
    rope_rows = block_rows[:, None, None, None] * 2 + intra[None, None, :, None]
    rope_rows = numpy.broadcast_to(rope_rows, (grid_height // 2, grid_width // 2, 2, 2))
    rope_columns = block_columns[None, :, None, None] * 2 + intra[None, None, None, :]
    rope_columns = numpy.broadcast_to(
        rope_columns, (grid_height // 2, grid_width // 2, 2, 2)
    )
    rope_positions = numpy.stack((rope_rows.reshape(-1), rope_columns.reshape(-1)), 1)
    inverse_frequencies = 1.0 / (
        10000.0 ** (numpy.arange(0, 36, 2, dtype=numpy.float32) / 36)
    )
    return H3VisionInput(
        image,
        patches,
        numpy.ascontiguousarray(merge_order(indices), dtype=numpy.uint32),
        numpy.ascontiguousarray(merge_order(weights), dtype=numpy.float32),
        numpy.ascontiguousarray(rope_positions, dtype=numpy.float32),
        numpy.ascontiguousarray(inverse_frequencies, dtype=numpy.float32),
        grid_height,
        grid_width,
    )


def prepare_video_vae_input(image: numpy.ndarray) -> numpy.ndarray:
    assert image.ndim == 4 and image.shape[0] == 1 and image.shape[3] == 3
    return numpy.ascontiguousarray(image[0].transpose(2, 0, 1)[:, None] * 2.0 - 1.0)


def build_presentation(
    tokenizer: Tokenizer,
    prompt: str,
    vision_inputs: tuple[H3VisionInput, ...],
) -> H3Presentation:
    token_ids: list[int] = []
    scatter_spans = []
    modality_spans = []
    grids = []
    for index, vision in enumerate(vision_inputs, 1):
        token_ids.extend(
            tokenizer.encode(f"<Picture {index}>: ", add_special_tokens=False).ids
        )
        block_start = len(token_ids)
        token_ids.append(VISION_START)
        scatter_start = len(token_ids)
        token_ids.extend([IMAGE_PAD] * vision.merged_tokens)
        scatter_spans.append((scatter_start, vision.merged_tokens))
        token_ids.append(VISION_END)
        modality_spans.append((block_start, vision.merged_tokens + 2))
        grids.append((1, vision.grid_height, vision.grid_width))
    token_ids.extend(tokenizer.encode(prompt, add_special_tokens=False).ids)
    if not token_ids:
        token_ids.append(151643)
    positions = _mrope_positions(len(token_ids), tuple(scatter_spans), tuple(grids))
    inverse_frequencies = 1.0 / (
        5_000_000.0 ** (numpy.arange(0, 128, 2, dtype=numpy.float32) / 128)
    )
    return H3Presentation(
        numpy.asarray(token_ids, dtype=numpy.uint32),
        tuple(scatter_spans),
        tuple(modality_spans),
        positions,
        inverse_frequencies,
    )


def _mrope_positions(
    sequence: int,
    spans: tuple[tuple[int, int], ...],
    grids: tuple[tuple[int, int, int], ...],
) -> numpy.ndarray:
    positions = numpy.zeros((sequence, 3), dtype=numpy.float32)
    offset = 0
    cursor = 0
    for (start, length), (_, height, width) in zip(spans, grids, strict=True):
        positions[cursor:start] = numpy.arange(cursor + offset, start + offset)[:, None]
        positions[start : start + length, 0] = start + offset
        positions[start : start + length, 1] = (
            numpy.broadcast_to(
                numpy.arange(height // 2)[:, None], (height // 2, width // 2)
            ).reshape(-1)[:length]
            + start
            + offset
        )
        positions[start : start + length, 2] = (
            numpy.broadcast_to(
                numpy.arange(width // 2)[None, :], (height // 2, width // 2)
            ).reshape(-1)[:length]
            + start
            + offset
        )
        end = start + length
        offset += max(height, width) // 2 - length
        cursor = end
    positions[cursor:] = numpy.arange(cursor + offset, sequence + offset)[:, None]
    return positions
