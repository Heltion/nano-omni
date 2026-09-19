"""FP32 one-dimensional convolution with optional fused epilogue."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.convolution.base import ConvolutionConfig


class Conv1dWorkload(Workload):
    # GEMM M = stereo * output_frames, N = output_channels, K = channels * taps.
    # Streams are concatenated along rows; padding never reads another stream.
    input_channels: int
    frames: int
    stereo: int
    output_channels: int
    kernel_size: int
    stride: int
    dilation: int
    padding: int  # Zero padding on both ends of each stream.
    use_bias: bool
    use_residual: bool
    clamp: bool


@dataclasses.dataclass(frozen=True, slots=True)
class Conv1dArguments(Arguments):
    input: TensorDesc
    weight: TensorDesc
    bias: TensorDesc | None
    residual: TensorDesc | None
    output: TensorDesc
    stride: int
    dilation: int
    padding: int
    clamp: bool


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def conv1d(
    input_channels,
    frames,
    stereo,
    output_channels,
    kernel_size,
    stride,
    dilation,
    padding,
    use_bias,
    use_residual,
    clamp,
    tile_m=64,
    tile_n=64,
    tile_k=32,
    threads=128,
    stages=2,
):
    import tilelang.language as T

    output_frames = (
        frames
        + 2 * padding
        - dilation * (kernel_size - 1)
        - 1
    ) // stride + 1
    rows = stereo * output_frames
    inner = input_channels * kernel_size

    @T.prim_func
    def main(
        input: T.Tensor([stereo, frames, input_channels], T.float32),
        weight: T.Tensor(
            [output_channels, input_channels, kernel_size],
            T.float32,
        ),
        bias: T.Tensor([output_channels], T.float32),
        residual: T.Tensor([stereo, output_frames, output_channels], T.float32),
        output: T.Tensor([stereo, output_frames, output_channels], T.float32),
    ):
        with T.Kernel(
            T.ceildiv(output_channels, tile_n),
            T.ceildiv(rows, tile_m),
            threads=threads,
        ) as (column, row):
            input_shared = T.alloc_shared([tile_m, tile_k], T.float32)
            weight_shared = T.alloc_shared([tile_n, tile_k], T.float32)
            accumulator = T.alloc_fragment([tile_m, tile_n], T.float32)
            T.clear(accumulator)
            for inner_tile in T.Pipelined(
                T.ceildiv(inner, tile_k), num_stages=stages
            ):
                for i, j in T.Parallel(tile_m, tile_k):
                    output_row = row * tile_m + i
                    reduction = inner_tile * tile_k + j
                    stream = output_row // output_frames
                    output_frame = output_row % output_frames
                    input_channel = reduction // kernel_size
                    kernel_frame = reduction % kernel_size
                    input_frame = (
                        output_frame * stride + kernel_frame * dilation - padding
                    )
                    input_shared[i, j] = T.if_then_else(
                        (output_row < rows)
                        & (reduction < inner)
                        & (input_frame >= 0)
                        & (input_frame < frames),
                        input[stream, input_frame, input_channel],
                        0.0,
                    )
                for i, j in T.Parallel(tile_n, tile_k):
                    output_channel = column * tile_n + i
                    reduction = inner_tile * tile_k + j
                    input_channel = reduction // kernel_size
                    kernel_frame = reduction % kernel_size
                    weight_shared[i, j] = T.if_then_else(
                        (output_channel < output_channels) & (reduction < inner),
                        weight[output_channel, input_channel, kernel_frame],
                        0.0,
                    )
                T.gemm(
                    input_shared,
                    weight_shared,
                    accumulator,
                    transpose_B=True,
                )
            for i, j in T.Parallel(tile_m, tile_n):
                output_row = row * tile_m + i
                output_channel = column * tile_n + j
                if use_bias and output_channel < output_channels:
                    accumulator[i, j] += bias[output_channel]
                if (
                    use_residual
                    and output_row < rows
                    and output_channel < output_channels
                ):
                    accumulator[i, j] += residual[
                        output_row // output_frames,
                        output_row % output_frames,
                        output_channel,
                    ]
                if output_row < rows and output_channel < output_channels:
                    if clamp:
                        output[
                            output_row // output_frames,
                            output_row % output_frames,
                            output_channel,
                        ] = T.max(-1.0, T.min(1.0, accumulator[i, j]))
                    else:
                        output[
                            output_row // output_frames,
                            output_row % output_frames,
                            output_channel,
                        ] = accumulator[i, j]

    return main



class Conv1dKernel(Kernel[Conv1dArguments, Conv1dWorkload, ConvolutionConfig]):
    name = "conv1d"
    program = conv1d
    @classmethod
    def make_arguments(cls, workload: Conv1dWorkload) -> Conv1dArguments:
        """Describe time-major audio, convolution parameters, and output."""
        output_frames = (
            workload.frames
            + 2 * workload.padding
            - workload.dilation * (workload.kernel_size - 1)
            - 1
        ) // workload.stride + 1
        output_shape = (workload.stereo, output_frames, workload.output_channels)
        return Conv1dArguments(
            input=TensorDesc.empty(
                DType.F32,
                (workload.stereo, workload.frames, workload.input_channels),
            ),
            weight=TensorDesc.empty(
                DType.F32,
                (
                    workload.output_channels,
                    workload.input_channels,
                    workload.kernel_size,
                ),
            ),
            bias=(
                TensorDesc.empty(DType.F32, (workload.output_channels,))
                if workload.use_bias
                else None
            ),
            residual=(
                TensorDesc.empty(DType.F32, output_shape)
                if workload.use_residual
                else None
            ),
            output=TensorDesc.empty(DType.F32, output_shape),
            stride=workload.stride,
            dilation=workload.dilation,
            padding=workload.padding,
            clamp=workload.clamp,
        )

    @classmethod
    def make_workload(cls, arguments: Conv1dArguments) -> Conv1dWorkload:
        assert len(arguments.input.shape) == len(arguments.weight.shape) == 3
        assert all(tensor.dtype == DType.F32 for tensor in arguments.values())
        assert arguments.weight.shape[1] == arguments.input.shape[2]
        workload = Conv1dWorkload(
            input_channels=arguments.input.shape[2],
            frames=arguments.input.shape[1],
            stereo=arguments.input.shape[0],
            output_channels=arguments.weight.shape[0],
            kernel_size=arguments.weight.shape[2],
            stride=arguments.stride,
            dilation=arguments.dilation,
            padding=arguments.padding,
            use_bias=arguments.bias is not None,
            use_residual=arguments.residual is not None,
            clamp=arguments.clamp,
        )
        expected_frames = (
            workload.frames
            + 2 * workload.padding
            - workload.dilation * (workload.kernel_size - 1)
            - 1
        ) // workload.stride + 1
        if arguments.bias is not None:
            assert arguments.bias.shape == (workload.output_channels,)
        expected_output = (
            workload.stereo,
            expected_frames,
            workload.output_channels,
        )
        if arguments.residual is not None:
            assert arguments.residual.shape == expected_output
        assert arguments.output.shape == expected_output
        return workload

    @classmethod
    def ref_program(cls, arguments: Conv1dArguments) -> None:
        """Evaluate deterministic FP32 convolution and its fused epilogue."""
        import torch

        workload = cls.make_workload(arguments)
        input = arguments.input.as_torch().transpose(1, 2)
        weight = arguments.weight.as_torch()
        bias = arguments.bias.as_torch() if arguments.bias is not None else None
        residual = (
            arguments.residual.as_torch() if arguments.residual is not None else None
        )
        output = arguments.output.as_torch()
        with torch.backends.cudnn.flags(
            benchmark=False, deterministic=True, allow_tf32=False
        ):
            result = torch.nn.functional.conv1d(
                input,
                weight,
                bias,
                workload.stride,
                workload.padding,
                workload.dilation,
            ).transpose(1, 2)
        if residual is not None:
            result += residual
        if workload.clamp:
            result.clamp_(-1, 1)
        output.copy_(result)

    @classmethod
    def make_config(cls, workload: Conv1dWorkload) -> ConvolutionConfig:
        """Choose an implicit-GEMM tile from the output-channel width.

        A 128-column tile is worthwhile for the middle widths.  At 512
        channels it halves the column-block count too far, while 32 channels do
        not fill it.  The selected pipeline depth follows the shared-memory
        footprint of that tile.
        """
        if workload.input_channels <= 8 and workload.output_channels <= 8:
            return ConvolutionConfig(
                tile_m=64,
                tile_n=16,
                tile_k=16,
                threads=32,
                stages=2,
            )
        if workload.output_channels <= 16:
            return ConvolutionConfig(
                tile_m=128,
                tile_n=16,
                tile_k=32,
                threads=128,
                stages=2,
            )
        if workload.output_channels <= 32:
            tile_n, stages = 32, 2
        elif workload.output_channels >= 512:
            tile_n, stages = 64, 1
        else:
            tile_n, stages = 128, 3
        return ConvolutionConfig(
            tile_m=64,
            tile_n=tile_n,
            tile_k=32,
            threads=128,
            stages=stages,
        )
