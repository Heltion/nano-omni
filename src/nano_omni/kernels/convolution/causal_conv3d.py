import dataclasses
import math

import tilelang

from nano_omni.core.kernel import Arguments, Kernel, MmaType, Tops, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.convolution.base import ConvolutionConfig


class CausalConv3dWorkload(Workload):
    input_channels: int
    frames: int
    height: int
    width: int
    output_channels: int
    kernel_frames: int
    kernel_height: int
    kernel_width: int
    stride_frames: int
    stride_height: int
    stride_width: int
    padding_frames: int  # All 2 * padding_frames zeros precede the first frame.
    padding_height: int
    padding_width: int
    pad_end_height: int
    pad_end_width: int
    input_dtype: DType
    output_dtype: DType = DType.BF16
    normalize_pixels: bool
    use_residual: bool

    @property
    def output_shape(self) -> tuple[int, int, int]:
        # Time is left-padded; space reflects on both sides, including pad_end.
        # The DSL uses one reflection, so spatial pads must be smaller than the axis.
        return (
            (self.frames + 2 * self.padding_frames - self.kernel_frames)
            // self.stride_frames
            + 1,
            (
                self.height
                + 2 * self.padding_height
                + self.pad_end_height
                - self.kernel_height
            )
            // self.stride_height
            + 1,
            (
                self.width
                + 2 * self.padding_width
                + self.pad_end_width
                - self.kernel_width
            )
            // self.stride_width
            + 1,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class CausalConv3dArguments(Arguments):
    input: TensorDesc
    weight: TensorDesc
    bias: TensorDesc
    residual: TensorDesc | None
    output: TensorDesc
    normalize_pixels: bool
    stride_frames: int
    stride_height: int
    stride_width: int
    padding_frames: int
    padding_height: int
    padding_width: int


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def causal_conv3d(
    input_channels,
    frames,
    height,
    width,
    output_channels,
    kernel_frames,
    kernel_height,
    kernel_width,
    stride_frames,
    stride_height,
    stride_width,
    padding_frames,
    padding_height,
    padding_width,
    pad_end_height,
    pad_end_width,
    input_dtype,
    output_dtype,
    normalize_pixels,
    use_residual,
    tile_m=64,
    tile_n=64,
    tile_k=32,
    threads=256,
    stages=2,
):
    import tilelang.language as T

    tl_input_dtype = {
        DType.F32: T.float32,
        DType.F16: T.float16,
        DType.BF16: T.bfloat16,
    }[input_dtype]
    tl_output_dtype = T.float16 if output_dtype == DType.F16 else T.bfloat16
    output_frames = (frames + 2 * padding_frames - kernel_frames) // stride_frames + 1
    output_height = (
        height + 2 * padding_height + pad_end_height - kernel_height
    ) // stride_height + 1
    output_width = (
        width + 2 * padding_width + pad_end_width - kernel_width
    ) // stride_width + 1
    rows = output_frames * output_height * output_width
    kernel_volume = kernel_frames * kernel_height * kernel_width
    inner = input_channels * kernel_volume
    temporal_padding = padding_frames * 2

    @T.prim_func
    def main(
        input: T.Tensor(
            [
                input_channels,
                frames,
                height,
                width,
            ],
            tl_input_dtype,
        ),
        weight: T.Tensor(
            [
                output_channels,
                input_channels,
                kernel_frames,
                kernel_height,
                kernel_width,
            ],
            T.float16,
        ),
        bias: T.Tensor([output_channels], T.float16),
        residual: T.Tensor(
            [output_channels, output_frames, output_height, output_width],
            tl_output_dtype,
        ),
        output: T.Tensor(
            [output_channels, output_frames, output_height, output_width],
            tl_output_dtype,
        ),
    ):
        with T.Kernel(
            T.ceildiv(output_channels, tile_n),
            T.ceildiv(rows, tile_m),
            threads=threads,
        ) as (column, row):
            input_shared = T.alloc_shared([tile_m, tile_k], T.float16)
            weight_shared = T.alloc_shared([tile_n, tile_k], T.float16)
            accumulator = T.alloc_fragment([tile_m, tile_n], T.float32)
            T.clear(accumulator)
            for block in T.Pipelined(T.ceildiv(inner, tile_k), num_stages=stages):
                for i, j in T.Parallel(tile_m, tile_k):
                    output_position = row * tile_m + i
                    reduction = block * tile_k + j
                    output_t = output_position // (output_height * output_width)
                    output_y = (output_position // output_width) % output_height
                    output_x = output_position % output_width
                    input_channel = reduction // kernel_volume
                    kernel_position = reduction % kernel_volume
                    kernel_t = kernel_position // (kernel_height * kernel_width)
                    kernel_y = (kernel_position // kernel_width) % kernel_height
                    kernel_x = kernel_position % kernel_width
                    input_t = output_t * stride_frames + kernel_t - temporal_padding
                    input_y = output_y * stride_height + kernel_y - padding_height
                    input_x = output_x * stride_width + kernel_x - padding_width
                    reflected_y = T.if_then_else(
                        input_y < 0,
                        -input_y,
                        T.if_then_else(
                            input_y >= height,
                            2 * height - 2 - input_y,
                            input_y,
                        ),
                    )
                    reflected_x = T.if_then_else(
                        input_x < 0,
                        -input_x,
                        T.if_then_else(
                            input_x >= width,
                            2 * width - 2 - input_x,
                            input_x,
                        ),
                    )
                    valid = (
                        (output_position < rows)
                        & (reduction < inner)
                        & (input_t >= 0)
                        & (input_t < frames)
                    )
                    raw = T.if_then_else(
                        valid,
                        input[input_channel, input_t, reflected_y, reflected_x],
                        0.0,
                    )
                    if normalize_pixels:
                        mean = T.if_then_else(
                            input_channel == 0,
                            0.485,
                            T.if_then_else(input_channel == 1, 0.456, 0.406),
                        )
                        inverse_std = T.if_then_else(
                            input_channel == 0,
                            1.0 / 0.229,
                            T.if_then_else(
                                input_channel == 1, 1.0 / 0.224, 1.0 / 0.225
                            ),
                        )
                        value = T.if_then_else(
                            valid,
                            ((raw + 1.0) * 0.5 - mean) * inverse_std,
                            0.0,
                        )
                    else:
                        value = raw
                    input_shared[i, j] = value
                for i, j in T.Parallel(tile_n, tile_k):
                    output_channel = column * tile_n + i
                    reduction = block * tile_k + j
                    input_channel = reduction // kernel_volume
                    kernel_position = reduction % kernel_volume
                    kernel_t = kernel_position // (kernel_height * kernel_width)
                    kernel_y = (kernel_position // kernel_width) % kernel_height
                    kernel_x = kernel_position % kernel_width
                    weight_shared[i, j] = T.if_then_else(
                        (output_channel < output_channels) & (reduction < inner),
                        weight[
                            output_channel,
                            input_channel,
                            kernel_t,
                            kernel_y,
                            kernel_x,
                        ],
                        0.0,
                    )
                T.gemm(
                    input_shared,
                    weight_shared,
                    accumulator,
                    transpose_B=True,
                )
            for i, j in T.Parallel(tile_m, tile_n):
                output_position = row * tile_m + i
                output_channel = column * tile_n + j
                if output_position < rows and output_channel < output_channels:
                    output_t = output_position // (output_height * output_width)
                    output_y = (output_position // output_width) % output_height
                    output_x = output_position % output_width
                    if use_residual:
                        output[output_channel, output_t, output_y, output_x] = (
                            accumulator[i, j]
                            + bias[output_channel]
                            + residual[output_channel, output_t, output_y, output_x]
                        )
                    else:
                        output[output_channel, output_t, output_y, output_x] = (
                            accumulator[i, j] + bias[output_channel]
                        )

    return main


class CausalConv3dKernel(
    Kernel[CausalConv3dArguments, CausalConv3dWorkload, ConvolutionConfig]
):
    name = "causal_conv3d"
    program = causal_conv3d
    @classmethod
    def make_arguments(cls, workload: CausalConv3dWorkload) -> CausalConv3dArguments:
        """Describe causal convolution operands and its optional residual."""
        output_shape = (workload.output_channels, *workload.output_shape)
        return CausalConv3dArguments(
            input=TensorDesc.empty(
                workload.input_dtype,
                (
                    workload.input_channels,
                    workload.frames,
                    workload.height,
                    workload.width,
                ),
            ),
            weight=TensorDesc.empty(
                DType.F16,
                (
                    workload.output_channels,
                    workload.input_channels,
                    workload.kernel_frames,
                    workload.kernel_height,
                    workload.kernel_width,
                ),
            ),
            bias=TensorDesc.empty(DType.F16, (workload.output_channels,)),
            residual=(
                TensorDesc.empty(workload.output_dtype, output_shape)
                if workload.use_residual
                else None
            ),
            output=TensorDesc.empty(workload.output_dtype, output_shape),
            normalize_pixels=workload.normalize_pixels,
            stride_frames=workload.stride_frames,
            stride_height=workload.stride_height,
            stride_width=workload.stride_width,
            padding_frames=workload.padding_frames,
            padding_height=workload.padding_height,
            padding_width=workload.padding_width,
        )

    @classmethod
    def tops(cls, arguments: CausalConv3dArguments) -> Tops:
        workload = cls.make_workload(arguments)
        # Implicit GEMM: M = output positions, N = channels, K = input channels * taps.
        # Both shared operands use the FP16 tensor-core path.
        m = math.prod(workload.output_shape)
        n = workload.output_channels
        k = (
            workload.input_channels
            * workload.kernel_frames
            * workload.kernel_height
            * workload.kernel_width
        )
        return {MmaType.F16F16F32: 2 * m * n * k}

    @classmethod
    def ref_program(cls, arguments: CausalConv3dArguments) -> None:
        """Evaluate causal time padding and reflected spatial convolution."""
        import torch

        workload = cls.make_workload(arguments)
        source = arguments.input.as_torch().float()
        if workload.normalize_pixels:
            mean = torch.tensor([0.485, 0.456, 0.406], device=source.device)[
                :, None, None, None
            ]
            std = torch.tensor([0.229, 0.224, 0.225], device=source.device)[
                :, None, None, None
            ]
            source = ((source + 1) * 0.5 - mean) / std
        source = source.half().float().unsqueeze(0)
        source = torch.nn.functional.pad(
            source,
            (
                workload.padding_width,
                workload.padding_width + workload.pad_end_width,
                workload.padding_height,
                workload.padding_height + workload.pad_end_height,
                0,
                0,
            ),
            mode="reflect",
        )
        source = torch.nn.functional.pad(
            source, (0, 0, 0, 0, 2 * workload.padding_frames, 0)
        )
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            result = torch.nn.functional.conv3d(
                source,
                arguments.weight.as_torch().half().float(),
                arguments.bias.as_torch().float(),
                stride=(
                    workload.stride_frames,
                    workload.stride_height,
                    workload.stride_width,
                ),
            )[0]
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        if arguments.residual is not None:
            result += arguments.residual.as_torch().float()
        arguments.output.as_torch().copy_(result)

    @classmethod
    def make_config(cls, workload: CausalConv3dWorkload) -> ConvolutionConfig:
        """Select the current production launch from convolution shape rules."""
        output_rows = math.prod(workload.output_shape)
        pointwise = (
            workload.kernel_frames
            == workload.kernel_height
            == workload.kernel_width
            == 1
        )
        large_spatial_tile = (
            workload.input_channels <= 256
            and workload.output_channels == 256
            and workload.stride_height == 1
            and not workload.use_residual
        )
        dense_spatiotemporal = (
            not pointwise
            and not workload.normalize_pixels
            and workload.frames > 1
            and workload.stride_height == workload.stride_width == 1
            and workload.output_channels >= 128
            and output_rows > 1024
        )
        if workload.normalize_pixels:
            tile_m = 128
        elif pointwise:
            tile_m = 128 if output_rows >= 4096 else 64
        elif workload.input_channels >= 512 or output_rows <= 1024:
            tile_m = 32
        elif large_spatial_tile and not dense_spatiotemporal:
            tile_m = 128
        else:
            tile_m = 64
        tile_n = (
            128
            if dense_spatiotemporal or large_spatial_tile
            else 32
            if workload.output_channels < 64
            else 64
        )
        threads = (
            256
            if workload.normalize_pixels
            or large_spatial_tile
            or (workload.input_channels < 512 and output_rows > 1024)
            else 128
        )
        tile_k = 16 if pointwise and workload.input_channels >= 256 else 32
        if (
            not pointwise
            and workload.frames > 1
            and workload.stride_height == workload.stride_width == 1
            and tile_m == 64
            and threads == 256
        ):
            tile_k = 64
        stages = 3 if (tile_m, tile_n, tile_k, threads) == (64, 64, 16, 128) else 1
        if (tile_m, tile_n, tile_k, threads) == (128, 64, 16, 256):
            stages = 2
        return ConvolutionConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            threads=threads,
            stages=stages,
        )

    @classmethod
    def make_workload(cls, arguments: CausalConv3dArguments) -> CausalConv3dWorkload:
        input = arguments.input
        weight = arguments.weight
        output = arguments.output
        pad_end_height = (
            (output.shape[2] - 1) * arguments.stride_height
            - input.shape[2]
            - 2 * arguments.padding_height
            + weight.shape[3]
        )
        pad_end_width = (
            (output.shape[3] - 1) * arguments.stride_width
            - input.shape[3]
            - 2 * arguments.padding_width
            + weight.shape[4]
        )
        assert 0 <= pad_end_height < arguments.stride_height
        assert 0 <= pad_end_width < arguments.stride_width
        workload = CausalConv3dWorkload(
            input_channels=input.shape[0],
            frames=input.shape[1],
            height=input.shape[2],
            width=input.shape[3],
            output_channels=weight.shape[0],
            kernel_frames=weight.shape[2],
            kernel_height=weight.shape[3],
            kernel_width=weight.shape[4],
            stride_frames=arguments.stride_frames,
            stride_height=arguments.stride_height,
            stride_width=arguments.stride_width,
            padding_frames=arguments.padding_frames,
            padding_height=arguments.padding_height,
            padding_width=arguments.padding_width,
            pad_end_height=pad_end_height,
            pad_end_width=pad_end_width,
            input_dtype=arguments.input.dtype,
            output_dtype=arguments.output.dtype,
            normalize_pixels=arguments.normalize_pixels,
            use_residual=arguments.residual is not None,
        )
        assert workload.output_shape == output.shape[1:]
        assert input.dtype in (DType.F16, DType.BF16, DType.F32)
        assert output.dtype in (DType.F16, DType.BF16)
        if arguments.residual is not None:
            assert arguments.residual.dtype == output.dtype
        assert output.shape[0] == workload.output_channels
        return workload
