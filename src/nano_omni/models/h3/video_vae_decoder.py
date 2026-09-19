"""Plan tiled H3 decoding directly into its final pitched NV12 allocation."""

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path

from nano_omni.core.layout import ActivationLayout
from nano_omni.core.model import (
    FileMetadata,
    ModelMetadata,
    PlannedModel,
)
from nano_omni.core.planning import operators
from nano_omni.core.planning.scheduling import Action, Copy
from nano_omni.core.runtime.execution import CudaRuntime
from nano_omni.core.tensor import DType, TensorDesc, TensorKind
from nano_omni.core.weights import WeightRegistry
from nano_omni.models.h3.video_vae_tiles import spatial_tiles
from nano_omni.ops.minimax_h3.video_vae import (
    VideoVaeBlend,
    VideoVaeDecodeTile,
    VideoVaeSetup,
    VideoVaeTemporalStore,
)

VIDEO_VAE_BLOCKS = 36
VIDEO_VAE_HEADS = 32
VIDEO_VAE_DIM = 64
VIDEO_VAE_ROPE_DIM = 48
VIDEO_VAE_MEMORY_GRANULARITY = 64 << 20
VIDEO_VAE_MEMORY_RESERVE = 1 << 30


@dataclasses.dataclass(frozen=True, slots=True)
class H3VideoVaeDecoderSpec:
    latent_frames: int = 17
    latent_height: int = 22
    latent_width: int = 38


@dataclasses.dataclass(frozen=True, slots=True)
class H3VideoVaeDecoderArgs:
    latent: TensorDesc
    runtime: CudaRuntime


@dataclasses.dataclass(frozen=True, slots=True)
class H3VideoVaeDecoderConfig:
    nv12_offset: int
    nv12_nbytes: int
    pitch: int
    frames: int
    height: int
    width: int


def read_h3_video_vae_metadata(path: Path) -> ModelMetadata:
    metadata = ModelMetadata((FileMetadata.read(0, path),))
    validate_h3_video_vae_metadata(metadata)
    return metadata


def validate_h3_video_vae_metadata(metadata: ModelMetadata) -> None:
    weights = metadata.files[0].weights
    expected = {
        "latents_mean": (24,),
        "latents_std": (24,),
        "post_quant_conv.weight": (24, 24, 1, 1, 1),
        "post_quant_conv.bias": (24,),
        "decoder.x_embedder.weight": (2048, 24),
        "decoder.x_embedder.bias": (2048,),
        "decoder.register_tokens": (1, 4, 2048),
        "decoder.norm_out.weight": (2048,),
        "decoder.norm_out.bias": (2048,),
        "decoder.proj_out.weight": (3072, 2048),
        "decoder.proj_out.bias": (3072,),
    }
    for name, shape in expected.items():
        assert weights[name].dtype == DType.F16
        assert weights[name].shape == shape
    for index in range(VIDEO_VAE_BLOCKS):
        prefix = f"decoder.transformer_blocks.{index}"
        block_expected = {
            f"{prefix}.norm1.weight": (2048,),
            f"{prefix}.attn.to_qkv.weight": (6144, 2048),
            f"{prefix}.attn.to_qkv.bias": (6144,),
            f"{prefix}.attn.to_out.weight": (2048, 2048),
            f"{prefix}.attn.to_out.bias": (2048,),
            f"{prefix}.scale1": (2048,),
            f"{prefix}.norm2.weight": (2048,),
            f"{prefix}.ff.w1.weight": (16384, 2048),
            f"{prefix}.ff.w1.bias": (16384,),
            f"{prefix}.ff.w2.weight": (2048, 8192),
            f"{prefix}.ff.w2.bias": (2048,),
            f"{prefix}.scale2": (2048,),
        }
        for name, shape in block_expected.items():
            assert weights[name].dtype == DType.F16
            assert weights[name].shape == shape


@dataclasses.dataclass(slots=True)
class H3VideoVaeDecoder(
    PlannedModel[
        H3VideoVaeDecoderSpec,
        H3VideoVaeDecoderArgs,
        H3VideoVaeDecoderConfig,
    ]
):
    @classmethod
    def read_metadata(
        cls, path: Path, additional_weights: Sequence[Path] = ()
    ) -> ModelMetadata:
        assert not additional_weights
        metadata = read_h3_video_vae_metadata(path)
        return metadata

    @classmethod
    def plan(
        cls,
        metadata: ModelMetadata,
        spec: H3VideoVaeDecoderSpec,
        total_memory: int,
        workspace_limit: int | None = None,
    ) -> tuple[H3VideoVaeDecoderConfig, list[Action], int]:
        available = cls.workspace_capacity(
            total_memory - VIDEO_VAE_MEMORY_RESERVE, workspace_limit
        )
        if available <= 0:
            raise MemoryError("H3 Video VAE requires a 1 GiB VRAM reserve")
        high = available // VIDEO_VAE_MEMORY_GRANULARITY
        if high == 0:
            raise MemoryError("H3 Video VAE has no allocatable VRAM")
        workspace_nbytes = high * VIDEO_VAE_MEMORY_GRANULARITY
        try:
            commands, nv12_offset, nv12_nbytes, frames, surface_height, pitch = (
                build_decoder_plan(metadata, spec, workspace_nbytes)
            )
        except MemoryError as error:
            raise MemoryError(
                "H3 Video VAE weights and workspace do not fit with a 1 GiB reserve"
            ) from error
        height = surface_height * 2 // 3
        config = H3VideoVaeDecoderConfig(
            nv12_offset,
            nv12_nbytes,
            pitch,
            frames,
            height,
            spec.latent_width * 16,
        )
        return config, commands, workspace_nbytes

    def run(self, args: H3VideoVaeDecoderArgs) -> TensorDesc:
        input = TensorDesc.from_pointer(args.latent.data_ptr(), args.latent.num_bytes)
        self.execute(args.runtime, (input,), ())
        workspace = args.runtime.workspace
        assert workspace is not None
        return TensorDesc.from_pointer(
            workspace.data_ptr() + self.config.nv12_offset,
            self.config.nv12_nbytes,
        )


def build_decoder_plan(
    metadata: ModelMetadata, spec: H3VideoVaeDecoderSpec, capacity: int
) -> tuple[list[Action], int, int, int, int, int]:
    """Keep decoded chunk buffers live while emitting disjoint final NV12 segments."""
    registry = WeightRegistry(metadata)
    weights = decoder_weight_refs(registry)
    storage = ActivationLayout()
    latent, normalized, rope = 0, 1, 2
    latent_shape = (24, spec.latent_frames, spec.latent_height, spec.latent_width)
    storage.require(latent, math.prod(latent_shape) * DType.F32.itemsize)
    storage.require(normalized, math.prod(latent_shape) * DType.F16.itemsize)
    storage.require(rope, 2 * 1797 * 24 * DType.F32.itemsize)
    ops: list[
        VideoVaeSetup | VideoVaeDecodeTile | VideoVaeBlend | VideoVaeTemporalStore
    ] = [
        VideoVaeSetup(latent, shape=latent_shape)
        .with_weights((weights["latents_mean"], weights["latents_std"]))
        .to(normalized, rope)
    ]
    y_starts, y_overlaps = spatial_tiles(spec.latent_height * 16)
    x_starts, x_overlaps = spatial_tiles(spec.latent_width * 16)
    # A five-frame clip repeats its final latent to fill one seven-latent tile.
    temporal_starts = tuple(range(0, max(1, spec.latent_frames - 6), 5))
    tile_ids, row_ids, vertical_ids = (3,), (5, 6), (7, 8)
    tile_shape = (3, 28, 256, 256)
    row_widths = [256, 256]
    row_width = 256
    # The last tile writes the vertical result; only intermediate row prefixes live here.
    for x_index, overlap in enumerate(x_overlaps[:-1]):
        row_width += 256 - overlap
        slot = x_index % 2
        row_widths[slot] = max(row_widths[slot], row_width)
    vertical_capacity = (
        3,
        28,
        spec.latent_height * 16,
        spec.latent_width * 16,
    )
    for identity in tile_ids:
        storage.require(identity, math.prod(tile_shape) * DType.F32.itemsize)
    for identity, row_width in zip(row_ids, row_widths, strict=True):
        storage.require(
            identity, 3 * 28 * 256 * row_width * DType.F32.itemsize
        )
    for identity in vertical_ids:
        storage.require(identity, math.prod(vertical_capacity) * DType.F32.itemsize)
    chunk_ids = (9, 10)
    for identity in chunk_ids:
        storage.require(identity, math.prod(vertical_capacity) * DType.F32.itemsize)
    output_frames = 17 * ((spec.latent_frames - 2) // 5) + 5
    width = spec.latent_width * 16
    pitch = (width + 255) // 256 * 256
    # Each temporal store owns its final segment; no full-video FP32 RGB allocation.
    nv12 = 11
    nv12_shape = (output_frames, spec.latent_height * 16 * 3 // 2, pitch)
    storage.require(nv12, math.prod(nv12_shape) * DType.U8.itemsize)
    nv12_frame_nbytes = math.prod(nv12_shape[1:]) * DType.U8.itemsize
    for temporal_index, temporal_start in enumerate(temporal_starts):
        chunk_id = chunk_ids[temporal_index % 2]
        chunk_shape = vertical_capacity
        vertical = None
        vertical_shape = None
        for y_index, y_start in enumerate(y_starts):
            finish_vertical = y_index > 0 and len(x_starts) > 1
            row = None
            row_shape = None
            for x_index, x_start in enumerate(x_starts):
                is_last = x_index == len(x_starts) - 1
                # Preserve the first complete row in the vertical accumulator. The
                # two horizontal workspaces can then be reused by subsequent rows.
                if is_last and finish_vertical:
                    target = (
                        chunk_id
                        if y_index == len(y_starts) - 1
                        else vertical_ids[y_index % 2]
                    )
                elif is_last and len(y_starts) == 1:
                    target = chunk_id
                elif is_last and y_index == 0:
                    target = vertical_ids[0]
                elif x_index == 0:
                    target = tile_ids[0] if len(x_starts) > 1 else row_ids[0]
                else:
                    target = row_ids[(x_index - 1) % 2]
                decode = VideoVaeDecodeTile(
                    (
                        normalized,
                        (
                            (temporal_start * latent_shape[2] + y_start)
                            * latent_shape[3]
                            + x_start
                        )
                        * DType.F16.itemsize,
                    ),
                    rope,
                    latent_shape=latent_shape,
                    previous=row,
                    previous_shape=row_shape,
                    overlap=0 if row is None else x_overlaps[x_index - 1],
                    vertical=vertical if is_last and finish_vertical else None,
                    vertical_shape=vertical_shape
                    if is_last and finish_vertical
                    else None,
                    vertical_overlap=y_overlaps[y_index - 1] if finish_vertical else 0,
                )
                ops.append(decode.with_weights(weights).to(target))
                row, row_shape = target, decode.output_shape
            assert row is not None and row_shape is not None
            if vertical is None or finish_vertical:
                # A multi-tile row finishes directly in the vertical accumulator.
                vertical, vertical_shape = row, row_shape
                continue
            assert vertical_shape is not None
            target = (
                chunk_id if y_index == len(y_starts) - 1 else vertical_ids[y_index % 2]
            )
            blend = VideoVaeBlend(
                vertical,
                row,
                first_shape=vertical_shape,
                second_shape=row_shape,
                axis=2,
                overlap=y_overlaps[y_index - 1],
            )
            storage.require(
                target, math.prod(blend.output_shape) * DType.F32.itemsize
            )
            ops.append(blend.to(target))
            vertical, vertical_shape = target, blend.output_shape
        previous = chunk_ids[max(0, temporal_index - 1) % 2]
        ops.append(
            VideoVaeTemporalStore(
                previous if temporal_index > 0 else None,
                chunk_id,
                chunk_shape=chunk_shape,
                output_frames=min(
                    output_frames - temporal_index * 17,
                    17 + (5 if temporal_index == len(temporal_starts) - 1 else 0),
                ),
                pitch=pitch,
            ).to((nv12, temporal_index * 17 * nv12_frame_nbytes))
        )
    commands, stats = operators.commands(ops, storage, registry, capacity)
    offsets = stats["activation_offsets"]
    commands.insert(
        0,
        Copy(
            TensorDesc.bytes(
                TensorKind.INPUT, 0, 0, storage.requirements[latent][0]
            ),
            TensorDesc.bytes(
                TensorKind.WORKSPACE,
                0,
                offsets[latent],
                storage.requirements[latent][0],
            ),
            "compute",
        ),
    )
    return (
        commands,
        offsets[nv12],
        storage.requirements[nv12][0],
        *nv12_shape,
    )


def decoder_weight_refs(registry: WeightRegistry) -> dict[str, TensorDesc]:
    names = {
        "latents_mean",
        "latents_std",
        "post_quant_conv.weight",
        "post_quant_conv.bias",
        "decoder.x_embedder.weight",
        "decoder.x_embedder.bias",
        "decoder.register_tokens",
        "decoder.norm_out.weight",
        "decoder.norm_out.bias",
        "decoder.proj_out.weight",
        "decoder.proj_out.bias",
    }
    for index in range(VIDEO_VAE_BLOCKS):
        prefix = f"decoder.transformer_blocks.{index}"
        names.update(
            f"{prefix}.{suffix}"
            for suffix in (
                "norm1.weight",
                "attn.to_qkv.weight",
                "attn.to_qkv.bias",
                "attn.to_out.weight",
                "attn.to_out.bias",
                "scale1",
                "norm2.weight",
                "ff.w1.weight",
                "ff.w1.bias",
                "ff.w2.weight",
                "ff.w2.bias",
                "scale2",
            )
        )
    return {name: registry.reference(name) for name in names}
