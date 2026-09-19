"""Transpose contiguous F32 audio between channel-first and stereo-major rows."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class H3AudioPackWorkload(Workload):
    """Axis lengths in elements; reverse restores the channel-first arrangement."""

    channels: int
    stereo: int
    frames: int
    reverse: bool


@dataclasses.dataclass(frozen=True, slots=True)
class H3AudioPackArguments(Arguments):
    """F32 [channels, stereo, frames] <-> [stereo * frames, channels].

    Forward rows use stereo_index * frames + frame; reverse swaps the layouts.
    Each contiguous view spans 4 * channels * stereo * frames bytes. The caller
    supplies any slice base address; these scalar dimensions do not offset it.
    """

    input: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def h3_audio_pack(channels, stereo, frames, reverse, tile_elements=256, threads=128):
    import tilelang.language as T

    rows = stereo * frames
    elements = channels * rows
    audio_shape = [channels, stereo, frames]
    row_shape = [rows, channels]
    input_shape = row_shape if reverse else audio_shape
    output_shape = audio_shape if reverse else row_shape

    @T.prim_func
    def main(
        input: T.Tensor(input_shape, T.float32),
        output: T.Tensor(output_shape, T.float32),
    ):
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for offset in T.Parallel(tile_elements):
                linear = block * tile_elements + offset
                if linear < elements:
                    if reverse:
                        channel = linear // rows
                        remaining = linear % rows
                        stereo_index = remaining // frames
                        frame = remaining % frames
                        row = stereo_index * frames + frame
                        output[channel, stereo_index, frame] = input[row, channel]
                    else:
                        row = linear // channels
                        channel = linear % channels
                        stereo_index = row // frames
                        frame = row % frames
                        output[row, channel] = input[channel, stereo_index, frame]

    return main.with_attr(
        "global_symbol",
        f"h3_audio_pack_{channels}_{stereo}_{frames}_{int(reverse)}_{tile_elements}_{threads}",
    )


class H3AudioPackKernel(ElementwiseKernel[H3AudioPackArguments, H3AudioPackWorkload]):
    name = "h3_audio_pack"
    program = h3_audio_pack
    launch = (256, 256)

    @classmethod
    def make_arguments(cls, workload: H3AudioPackWorkload) -> H3AudioPackArguments:
        """Describe the channel-first and stereo-major audio layouts."""
        channel_shape = (workload.channels, workload.stereo, workload.frames)
        row_shape = (workload.stereo * workload.frames, workload.channels)
        return H3AudioPackArguments(
            input=TensorDesc.empty(
                DType.F32, row_shape if workload.reverse else channel_shape
            ),
            output=TensorDesc.empty(
                DType.F32, channel_shape if workload.reverse else row_shape
            ),
        )

    @classmethod
    def make_workload(cls, arguments: H3AudioPackArguments) -> H3AudioPackWorkload:
        reverse = len(arguments.input.shape) == 2
        channel_shape = arguments.output.shape if reverse else arguments.input.shape
        row_shape = arguments.input.shape if reverse else arguments.output.shape
        assert len(channel_shape) == 3 and len(row_shape) == 2
        channels, stereo, frames = channel_shape
        assert row_shape == (stereo * frames, channels)
        assert arguments.input.dtype == arguments.output.dtype == DType.F32
        return H3AudioPackWorkload(
            channels=channels,
            stereo=stereo,
            frames=frames,
            reverse=reverse,
        )

    @classmethod
    def ref_program(cls, arguments: H3AudioPackArguments) -> None:
        """Transpose between channel-first audio and contiguous stereo rows."""
        workload = cls.make_workload(arguments)
        source = arguments.input.as_torch()
        output = arguments.output.as_torch()
        if workload.reverse:
            output.copy_(
                source.view(workload.stereo, workload.frames, workload.channels)
                .permute(2, 0, 1)
                .contiguous()
            )
        else:
            output.copy_(source.permute(1, 2, 0).reshape(output.shape))
