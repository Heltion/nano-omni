"""Extract an FP16/BF16 CFHW region as contiguous FHW-by-C tokens."""

import dataclasses

import tilelang

from nano_omni.core.kernel import (
    Arguments,
    Config,
    Kernel,
    Workload,
)
from nano_omni.core.tensor import DType, TensorDesc


class VolumeToTokensWorkload(Workload):
    """Full-volume CFHW dimensions and the extracted region's FHW dimensions."""

    channels: int
    input_frames: int
    input_height: int
    input_width: int
    tile_frames: int
    tile_height: int
    tile_width: int
    input_strides: tuple[int, int, int, int]
    dtype: DType = DType.BF16


class VolumeToTokensConfig(Config):
    """Output tokens per block and CUDA threads per block."""

    tile_tokens: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class VolumeToTokensArguments(Arguments):
    """Input[C, F, H, W] and output[tile_frames * tile_height * tile_width, C].

    The input address is shifted to the region origin. Its one-dimensional span
    covers every element reached through ``input_strides``.
    Output token order is frame, y, x; channels form each contiguous output row.
    """

    input: TensorDesc
    output: TensorDesc
    tile_frames: int
    tile_height: int
    tile_width: int
    channels: int
    input_frames: int
    input_height: int
    input_width: int
    input_strides: tuple[int, int, int, int]


def strided_span(dimensions: tuple[int, ...], strides: tuple[int, ...]) -> int:
    """Return the smallest element span containing a positive strided view."""
    assert len(dimensions) == len(strides)
    assert all(dimension > 0 for dimension in dimensions)
    assert all(stride > 0 for stride in strides)
    return 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(dimensions, strides, strict=True)
    )


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def volume_to_tokens(
    channels,
    input_frames,
    input_height,
    input_width,
    tile_frames,
    tile_height,
    tile_width,
    input_strides,
    dtype,
    tile_tokens=128,
    threads=256,
):
    import tilelang.language as T

    tokens = tile_frames * tile_height * tile_width
    available_frames = min(input_frames, tile_frames)
    input_span = strided_span(
        (channels, available_frames, tile_height, tile_width),
        input_strides,
    )
    storage_type = T.float16 if dtype == DType.F16 else T.bfloat16

    @T.prim_func
    def main(
        input: T.Tensor([input_span], storage_type),
        output: T.Tensor([tokens, channels], storage_type),
    ):
        with T.Kernel(T.ceildiv(tokens, tile_tokens), threads=threads) as block:
            for local, channel in T.Parallel(tile_tokens, channels):
                token = block * tile_tokens + local
                if token < tokens:
                    spatial = token % (tile_height * tile_width)
                    frame = token // (tile_height * tile_width)
                    y = spatial // tile_width
                    x = spatial % tile_width
                    if input_frames < tile_frames:
                        # H3 short clips repeat the last latent, matching temporal padding.
                        source_frame = T.min(frame, input_frames - 1)
                    else:
                        source_frame = frame
                    source = (
                        channel * input_strides[0]
                        + source_frame * input_strides[1]
                        + y * input_strides[2]
                        + x * input_strides[3]
                    )
                    output[token, channel] = input[source]

    return main.with_attr(
        "global_symbol",
        "volume_to_tokens_"
        f"{channels}_{input_frames}_{input_height}_{input_width}_"
        f"{tile_frames}_{tile_height}_{tile_width}_"
        f"{'_'.join(map(str, input_strides))}_{dtype.value}_"
        f"{tile_tokens}_{threads}",
    )

class VolumeToTokensKernel(
    Kernel[VolumeToTokensArguments, VolumeToTokensWorkload, VolumeToTokensConfig]
):
    """Shift the input origin while retaining the original full-volume strides."""

    name = "volume_to_tokens"
    program = volume_to_tokens

    @classmethod
    def make_arguments(
        cls, workload: VolumeToTokensWorkload
    ) -> VolumeToTokensArguments:
        """Describe a full-stride input rooted at the selected tile and its tokens."""
        return VolumeToTokensArguments(
            input=TensorDesc.empty(workload.dtype, (_input_span(workload),)),
            output=TensorDesc.empty(
                workload.dtype,
                (
                    workload.tile_frames * workload.tile_height * workload.tile_width,
                    workload.channels,
                ),
            ),
            tile_frames=workload.tile_frames,
            tile_height=workload.tile_height,
            tile_width=workload.tile_width,
            channels=workload.channels,
            input_frames=workload.input_frames,
            input_height=workload.input_height,
            input_width=workload.input_width,
            input_strides=workload.input_strides,
        )

    @classmethod
    def make_workload(
        cls, arguments: VolumeToTokensArguments
    ) -> VolumeToTokensWorkload:
        """Specialize on full/tile shapes; region starts only shift the runtime pointer."""
        workload = VolumeToTokensWorkload(
            channels=arguments.channels,
            input_frames=arguments.input_frames,
            input_height=arguments.input_height,
            input_width=arguments.input_width,
            tile_frames=arguments.tile_frames,
            tile_height=arguments.tile_height,
            tile_width=arguments.tile_width,
            input_strides=arguments.input_strides,
            dtype=arguments.input.dtype,
        )
        assert arguments.input.dtype == arguments.output.dtype
        assert arguments.input.dtype in (DType.F16, DType.BF16)
        assert arguments.input.shape == (_input_span(workload),)
        assert workload.input_frames > 0
        assert workload.tile_height <= workload.input_height
        assert workload.tile_width <= workload.input_width
        assert arguments.output.shape == (
            workload.tile_frames * workload.tile_height * workload.tile_width,
            workload.channels,
        )
        return workload

    @classmethod
    def ref_program(cls, arguments: VolumeToTokensArguments) -> None:
        """Extract the tile at the addressed origin and repeat a short final frame."""
        import torch

        workload = cls.make_workload(arguments)
        available_frames = min(workload.input_frames, workload.tile_frames)
        tile = arguments.input.as_torch().as_strided(
            (
                workload.channels,
                available_frames,
                workload.tile_height,
                workload.tile_width,
            ),
            workload.input_strides,
        )
        if workload.input_frames < workload.tile_frames:
            tile = torch.cat(
                (
                    tile,
                    tile[:, -1:].repeat(
                        1, workload.tile_frames - workload.input_frames, 1, 1
                    ),
                ),
                dim=1,
            )
        arguments.output.as_torch().copy_(
            tile.permute(1, 2, 3, 0).reshape(-1, workload.channels)
        )

    @classmethod
    def make_config(
        cls, workload: VolumeToTokensWorkload
    ) -> VolumeToTokensConfig:
        """Select the measured launch configuration."""
        del workload
        return VolumeToTokensConfig(threads=128, tile_tokens=64)


def _input_span(workload: VolumeToTokensWorkload) -> int:
    available_frames = min(workload.input_frames, workload.tile_frames)
    dimensions = (
        workload.channels,
        available_frames,
        workload.tile_height,
        workload.tile_width,
    )
    return strided_span(dimensions, workload.input_strides)
