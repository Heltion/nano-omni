"""Fuse channel affine conversion with an audio layout transpose."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class ChannelAffineTransposeWorkload(Workload):
    """Direction and channel/stereo/frame dimensions."""

    inverse: bool = False
    channels: int
    stereo: int
    frames: int


@dataclasses.dataclass(frozen=True, slots=True)
class ChannelAffineTransposeArguments(Arguments):
    """Input, per-channel mean/std, and transposed output."""

    input: TensorDesc
    mean: TensorDesc
    std: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def channel_affine_transpose(
    inverse, channels, stereo, frames, tile_elements=256, threads=128
):
    import tilelang.language as T

    rows = stereo * frames
    elements = rows * channels
    input_shape = (rows, channels) if inverse else (channels, stereo, frames)
    output_shape = (channels, stereo, frames) if inverse else (rows, channels)

    @T.prim_func
    def main(
        input: T.Tensor(input_shape, T.float32),
        mean: T.Tensor((channels,), T.float32),
        std: T.Tensor((channels,), T.float32),
        output: T.Tensor(output_shape, T.float32),
    ):
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < elements:
                    row = index // channels
                    channel = index % channels
                    stereo_index = row // frames
                    frame = row % frames
                    if inverse:
                        output[channel, stereo_index, frame] = (
                            input[row, channel] - mean[channel]
                        ) / std[channel]
                    else:
                        output[row, channel] = (
                            input[channel, stereo_index, frame] * std[channel]
                            + mean[channel]
                        )

    return main.with_attr(
        "global_symbol",
        f"channel_affine_transpose_{int(inverse)}_{channels}_{stereo}_{frames}_{tile_elements}_{threads}",
    )


class ChannelAffineTransposeKernel(
    ElementwiseKernel[
        ChannelAffineTransposeArguments,
        ChannelAffineTransposeWorkload,
    ]
):
    name = "channel_affine_transpose"
    program = channel_affine_transpose
    launch = (256, 128)

    @classmethod
    def make_arguments(
        cls, workload: ChannelAffineTransposeWorkload
    ) -> ChannelAffineTransposeArguments:
        """Describe channel-first and flattened layouts in either direction."""
        channel_shape = (workload.channels, workload.stereo, workload.frames)
        flat_shape = (workload.stereo * workload.frames, workload.channels)
        input_shape, output_shape = (
            (flat_shape, channel_shape)
            if workload.inverse
            else (channel_shape, flat_shape)
        )
        return ChannelAffineTransposeArguments(
            input=TensorDesc.empty(DType.F32, input_shape),
            mean=TensorDesc.empty(DType.F32, (workload.channels,)),
            std=TensorDesc.empty(DType.F32, (workload.channels,)),
            output=TensorDesc.empty(DType.F32, output_shape),
        )

    @classmethod
    def make_workload(
        cls, arguments: ChannelAffineTransposeArguments
    ) -> ChannelAffineTransposeWorkload:
        inverse = len(arguments.input.shape) == 2
        assert len(arguments.output.shape) == (3 if inverse else 2)
        channel_shape = arguments.output.shape if inverse else arguments.input.shape
        assert len(channel_shape) == 3
        channels, stereo, frames = channel_shape
        flat_shape = (stereo * frames, channels)
        assert (
            arguments.input.shape if inverse else arguments.output.shape
        ) == flat_shape
        assert arguments.mean.shape == arguments.std.shape == (channels,)
        assert all(
            tensor.dtype == DType.F32
            for tensor in (
                arguments.input,
                arguments.mean,
                arguments.std,
                arguments.output,
            )
        )
        return ChannelAffineTransposeWorkload(
            inverse=inverse,
            channels=channels,
            stereo=stereo,
            frames=frames,
        )

    @classmethod
    def ref_program(cls, arguments: ChannelAffineTransposeArguments) -> None:
        """Apply channel affine conversion while transposing the audio layout."""
        workload = cls.make_workload(arguments)
        input = arguments.input.as_torch()
        mean = arguments.mean.as_torch()
        std = arguments.std.as_torch()
        if workload.inverse:
            result = (
                ((input - mean) / std)
                .view(workload.stereo, workload.frames, workload.channels)
                .permute(2, 0, 1)
            )
        else:
            result = (
                (input * std[:, None, None] + mean[:, None, None])
                .permute(1, 2, 0)
                .reshape(arguments.output.shape)
            )
        arguments.output.as_torch().copy_(result)
