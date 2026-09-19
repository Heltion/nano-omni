"""FP32 one-dimensional transposed convolution with optional bias."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.convolution.base import ConvolutionConfig


class ConvTranspose1dWorkload(Workload):
    # Each stride phase uses M = stereo * ceil(output_frames / stride),
    # N = output_channels, K = input_channels * ceil(kernel_size / stride).
    # Invalid phase taps and input frames contribute zero.
    input_channels: int
    frames: int
    stereo: int
    output_channels: int
    kernel_size: int
    stride: int
    padding: int  # Crop both ends of the full transposed-convolution output.
    output_padding: int  # Extend only the output tail; no input values are added.
    use_bias: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ConvTranspose1dArguments(Arguments):
    input: TensorDesc
    weight: TensorDesc
    bias: TensorDesc | None
    output: TensorDesc
    stride: int
    padding: int


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def conv_transpose1d(
    input_channels,
    frames,
    stereo,
    output_channels,
    kernel_size,
    stride,
    padding,
    output_padding,
    use_bias,
    tile_m=64,
    tile_n=64,
    tile_k=32,
    threads=128,
    stages=2,
):
    import tilelang.language as T

    output_frames = (
        (frames - 1) * stride - 2 * padding + kernel_size + output_padding
    )
    phase_frames = T.ceildiv(output_frames, stride)
    phase_rows = stereo * phase_frames
    taps = T.ceildiv(kernel_size, stride)
    inner = input_channels * taps

    @T.prim_func
    def main(
        input: T.Tensor([stereo, frames, input_channels], T.float32),
        weight: T.Tensor(
            [input_channels, output_channels, kernel_size],
            T.float32,
        ),
        bias: T.Tensor([output_channels], T.float32),
        output: T.Tensor([stereo, output_frames, output_channels], T.float32),
    ):
        with T.Kernel(
            T.ceildiv(output_channels, tile_n),
            T.ceildiv(phase_rows, tile_m),
            stride,
            threads=threads,
        ) as (column, row, phase):
            input_shared = T.alloc_shared([tile_m, tile_k], T.float32)
            weight_shared = T.alloc_shared([tile_n, tile_k], T.float32)
            accumulator = T.alloc_fragment([tile_m, tile_n], T.float32)
            T.clear(accumulator)
            for inner_tile in T.Pipelined(
                T.ceildiv(inner, tile_k), num_stages=stages
            ):
                for i, j in T.Parallel(tile_m, tile_k):
                    phase_row = row * tile_m + i
                    reduction = inner_tile * tile_k + j
                    stream = phase_row // phase_frames
                    output_frame = (
                        phase + phase_row % phase_frames * stride
                    )
                    input_channel = reduction // taps
                    tap = reduction % taps
                    first_kernel_frame = (
                        output_frame + padding
                    ) % stride
                    kernel_frame = first_kernel_frame + tap * stride
                    input_frame = (
                        output_frame + padding - kernel_frame
                    ) // stride
                    input_shared[i, j] = T.if_then_else(
                        (phase_row < phase_rows)
                        & (output_frame < output_frames)
                        & (reduction < inner)
                        & (kernel_frame < kernel_size)
                        & (input_frame >= 0)
                        & (input_frame < frames),
                        input[stream, input_frame, input_channel],
                        0.0,
                    )
                for i, j in T.Parallel(tile_n, tile_k):
                    output_channel = column * tile_n + i
                    reduction = inner_tile * tile_k + j
                    input_channel = reduction // taps
                    tap = reduction % taps
                    first_kernel_frame = (
                        phase + padding
                    ) % stride
                    kernel_frame = first_kernel_frame + tap * stride
                    weight_shared[i, j] = T.if_then_else(
                        (output_channel < output_channels)
                        & (reduction < inner)
                        & (kernel_frame < kernel_size),
                        weight[input_channel, output_channel, kernel_frame],
                        0.0,
                    )
                T.gemm(
                    input_shared,
                    weight_shared,
                    accumulator,
                    transpose_B=True,
                )
            for i, j in T.Parallel(tile_m, tile_n):
                phase_row = row * tile_m + i
                stream = phase_row // phase_frames
                output_frame = phase + phase_row % phase_frames * stride
                output_channel = column * tile_n + j
                if use_bias and output_channel < output_channels:
                    accumulator[i, j] += bias[output_channel]
                if (
                    phase_row < phase_rows
                    and output_frame < output_frames
                    and output_channel < output_channels
                ):
                    output[stream, output_frame, output_channel] = accumulator[i, j]

    return main.with_attr(
        "global_symbol",
        "conv_transpose1d_"
        f"{input_channels}_{frames}_{stereo}_{output_channels}_{kernel_size}_{stride}_"
        f"{padding}_{output_padding}_{int(use_bias)}_"
        f"{tile_m}_{tile_n}_{tile_k}_{threads}_{stages}",
    )

class ConvTranspose1dKernel(
    Kernel[ConvTranspose1dArguments, ConvTranspose1dWorkload, ConvolutionConfig]
):
    name = "conv_transpose1d"
    program = conv_transpose1d
    @classmethod
    def make_arguments(
        cls, workload: ConvTranspose1dWorkload
    ) -> ConvTranspose1dArguments:
        """Describe time-major audio, transposed-convolution weights, and output."""
        output_frames = (
            (workload.frames - 1) * workload.stride
            - 2 * workload.padding
            + workload.kernel_size
            + workload.output_padding
        )
        return ConvTranspose1dArguments(
            input=TensorDesc.empty(
                DType.F32,
                (workload.stereo, workload.frames, workload.input_channels),
            ),
            weight=TensorDesc.empty(
                DType.F32,
                (
                    workload.input_channels,
                    workload.output_channels,
                    workload.kernel_size,
                ),
            ),
            bias=(
                TensorDesc.empty(DType.F32, (workload.output_channels,))
                if workload.use_bias
                else None
            ),
            output=TensorDesc.empty(
                DType.F32,
                (workload.stereo, output_frames, workload.output_channels),
            ),
            stride=workload.stride,
            padding=workload.padding,
        )

    @classmethod
    def make_workload(
        cls, arguments: ConvTranspose1dArguments
    ) -> ConvTranspose1dWorkload:
        assert len(arguments.input.shape) == len(arguments.weight.shape) == 3
        assert all(tensor.dtype == DType.F32 for tensor in arguments.values())
        assert arguments.weight.shape[0] == arguments.input.shape[2]
        base_frames = (
            (arguments.input.shape[1] - 1) * arguments.stride
            - 2 * arguments.padding
            + arguments.weight.shape[2]
        )
        output_padding = arguments.output.shape[1] - base_frames
        assert 0 <= output_padding < arguments.stride
        if arguments.bias is not None:
            assert arguments.bias.shape == (arguments.weight.shape[1],)
        workload = ConvTranspose1dWorkload(
            input_channels=arguments.input.shape[2],
            frames=arguments.input.shape[1],
            stereo=arguments.input.shape[0],
            output_channels=arguments.weight.shape[1],
            kernel_size=arguments.weight.shape[2],
            stride=arguments.stride,
            padding=arguments.padding,
            output_padding=output_padding,
            use_bias=arguments.bias is not None,
        )
        assert arguments.output.shape == (
            workload.stereo,
            base_frames + output_padding,
            workload.output_channels,
        )
        return workload

    @classmethod
    def ref_program(cls, arguments: ConvTranspose1dArguments) -> None:
        """Evaluate deterministic FP32 transposed convolution."""
        import torch

        workload = cls.make_workload(arguments)
        bias = arguments.bias.as_torch() if arguments.bias is not None else None
        with torch.backends.cudnn.flags(
            benchmark=False, deterministic=True, allow_tf32=False
        ):
            result = torch.nn.functional.conv_transpose1d(
                arguments.input.as_torch().transpose(1, 2),
                arguments.weight.as_torch(),
                bias,
                workload.stride,
                workload.padding,
                workload.output_padding,
            ).transpose(1, 2)
        arguments.output.as_torch().copy_(result)

    @classmethod
    def make_config(
        cls, workload: ConvTranspose1dWorkload
    ) -> ConvolutionConfig:
        """Return the current production launch configuration."""
        del workload
        return ConvolutionConfig(
            tile_m=128, tile_n=64, tile_k=16, threads=128, stages=2
        )
