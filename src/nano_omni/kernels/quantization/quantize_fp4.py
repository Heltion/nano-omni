"""Quantize BF16 rows into packed FP4 values and blocked UE4M3 scales."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class QuantizeFp4Workload(Workload):
    """Logical matrix dimensions and optional scaling inputs."""

    num_tokens: int
    columns: int
    use_pre_scale: bool
    dynamic_input_scale: bool = False


class QuantizeFp4Config(Config):
    """Row and column tile sizes plus CUDA thread count."""

    tile_m: int
    tile_n: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class QuantizeFp4Arguments(Arguments):
    """Input, optional scales, packed values, and blocked output scales."""

    input: TensorDesc
    pre_scale: TensorDesc | None
    input_scale: TensorDesc | None
    quantized: TensorDesc
    scales: TensorDesc
    static_input_scale: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def quantize_fp4(
    num_tokens,
    columns,
    use_pre_scale,
    dynamic_input_scale=False,
    tile_m=32,
    tile_n=128,
    threads=128,
):
    """Build packed FP4 quantization with tensor-core scale layout."""
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    scale_columns = columns // 16
    num_padded_tokens = T.ceildiv(dynamic_num_tokens, 128) * 128

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        input: T.Tensor([dynamic_num_tokens, columns], T.bfloat16),
        pre_scale: T.Tensor([columns], T.bfloat16),
        input_scale: T.Tensor([1], T.float32),
        packed_output: T.Tensor([dynamic_num_tokens, columns // 2], T.uint8),
        scales: T.Tensor([num_padded_tokens, scale_columns], T.float8_e4m3fn),
        static_input_scale: T.Tensor([1], T.float32),
    ):
        output = T.view(
            packed_output,
            shape=[dynamic_num_tokens, columns],
            dtype=T.float4_e2m1fn,
        )
        with T.Kernel(
            T.ceildiv(columns, tile_n),
            T.ceildiv(num_padded_tokens, tile_m),
            threads=threads,
        ) as (column, row):
            values = T.alloc_fragment([tile_m, tile_n // 16, 16], T.float32)
            block_scales = T.alloc_fragment([tile_m, tile_n // 16], T.float32)
            tensor_scale = T.alloc_local([1], T.float32)
            tensor_scale[0] = (
                input_scale[0] if dynamic_input_scale else static_input_scale[0]
            )
            for i, group, j in T.Parallel(tile_m, tile_n // 16, 16):
                logical_row = row * tile_m + i
                logical_column = column * tile_n + group * 16 + j
                value = T.if_then_else(
                    logical_row < dynamic_num_tokens and logical_column < columns,
                    input[logical_row, logical_column],
                    0.0,
                )
                if use_pre_scale:
                    values[i, group, j] = T.if_then_else(
                        logical_row < dynamic_num_tokens and logical_column < columns,
                        T.cast(
                            T.cast(value * pre_scale[logical_column], T.bfloat16),
                            T.float32,
                        ),
                        0.0,
                    )
                else:
                    values[i, group, j] = value
            T.reduce_absmax(values, block_scales, dim=2)
            for i, group in T.Parallel(tile_m, tile_n // 16):
                block_scales[i, group] /= tensor_scale[0] * 6.0
            for i, group in T.Parallel(tile_m, tile_n // 16):
                logical_row = row * tile_m + i
                logical_column = column * (tile_n // 16) + group
                if logical_row < num_padded_tokens and logical_column < scale_columns:
                    linear = (
                        (
                            (logical_row // 128) * (scale_columns // 4)
                            + logical_column // 4
                        )
                        * 128
                        + logical_row % 128
                    ) * 4 + logical_column % 4
                    scale_group = linear // 512
                    offset = linear % 512
                    blocked_linear = (
                        (scale_group * 32 + (offset % 128) // 4) * 4 + offset // 128
                    ) * 4 + offset % 4
                    scales[
                        blocked_linear // scale_columns,
                        blocked_linear % scale_columns,
                    ] = block_scales[i, group] * (
                        tensor_scale[0] if dynamic_input_scale else 1.0
                    )
            for i, j in T.Parallel(tile_m, tile_n):
                logical_row = row * tile_m + i
                logical_column = column * tile_n + j
                if logical_row < dynamic_num_tokens and logical_column < columns:
                    scale = block_scales[i, j // 16]
                    safe_scale = T.if_then_else(scale == 0.0, 1.0, scale)
                    output[logical_row, logical_column] = values[i, j // 16, j % 16] / (
                        safe_scale * tensor_scale[0]
                    )

    return main.with_attr(
        "global_symbol",
        "quantize_fp4_"
        f"{columns}_{int(use_pre_scale)}_{int(dynamic_input_scale)}_"
        f"{tile_m}_{tile_n}_{threads}",
    )


class QuantizeFp4Kernel(
    Kernel[QuantizeFp4Arguments, QuantizeFp4Workload, QuantizeFp4Config]
):
    """Produce packed FP4 data and scales consumed by block-scaled MMA."""

    name = "quantize_fp4"
    program = quantize_fp4

    @classmethod
    def make_arguments(cls, workload: QuantizeFp4Workload) -> QuantizeFp4Arguments:
        """Describe the complete fixed-position TileLang ABI."""
        assert workload.columns % 64 == 0
        num_padded_tokens = -(-workload.num_tokens // 128) * 128
        return QuantizeFp4Arguments(
            input=TensorDesc.empty(
                DType.BF16, (workload.num_tokens, workload.columns)
            ),
            pre_scale=(
                TensorDesc.empty(DType.BF16, (workload.columns,))
                if workload.use_pre_scale
                else None
            ),
            input_scale=(
                TensorDesc.empty(DType.F32, (1,))
                if workload.dynamic_input_scale
                else None
            ),
            quantized=TensorDesc.empty(
                DType.U8, (workload.num_tokens, workload.columns // 2)
            ),
            scales=TensorDesc.empty(
                DType.FP8_UE4M3, (num_padded_tokens, workload.columns // 16)
            ),
            static_input_scale=TensorDesc.empty(DType.F32, (1,)),
        )

    @classmethod
    def make_workload(cls, arguments: QuantizeFp4Arguments) -> QuantizeFp4Workload:
        """Validate the packed ABI and recover its specialization."""
        assert arguments.input.dtype == DType.BF16
        assert len(arguments.input.shape) == 2
        num_tokens, columns = arguments.input.shape
        assert columns % 64 == 0
        if arguments.pre_scale is not None:
            assert arguments.pre_scale.dtype == DType.BF16
            assert arguments.pre_scale.shape == (columns,)
        if arguments.input_scale is not None:
            assert arguments.input_scale.dtype == DType.F32
            assert arguments.input_scale.shape == (1,)
        assert arguments.quantized.dtype == DType.U8
        assert arguments.quantized.shape == (num_tokens, columns // 2)
        assert arguments.scales.dtype == DType.FP8_UE4M3
        assert arguments.scales.shape == (
            -(-num_tokens // 128) * 128,
            columns // 16,
        )
        assert arguments.static_input_scale.dtype == DType.F32
        assert arguments.static_input_scale.shape == (1,)
        return QuantizeFp4Workload(
            num_tokens=num_tokens,
            columns=columns,
            use_pre_scale=arguments.pre_scale is not None,
            dynamic_input_scale=arguments.input_scale is not None,
        )

    @classmethod
    def make_config(cls, workload: QuantizeFp4Workload) -> QuantizeFp4Config:
        """Return the current production launch configuration."""
        del workload
        return QuantizeFp4Config(threads=128, tile_m=32, tile_n=128)

    @classmethod
    def ref_program(cls, arguments: QuantizeFp4Arguments) -> None:
        """Evaluate packing and blocked-scale layout with vectorized Torch."""
        import torch

        value = arguments.input.as_torch()
        if arguments.pre_scale is not None:
            value = (value * arguments.pre_scale.as_torch()).to(torch.bfloat16)
        value = value.float()
        factor = (
            arguments.input_scale.as_torch()
            if arguments.input_scale is not None
            else arguments.static_input_scale.as_torch()
        )
        rows, columns = value.shape
        groups = value.reshape(rows, columns // 16, 16)
        block_scale = groups.abs().amax(dim=2) / (factor * 6)
        safe_scale = torch.where(block_scale == 0, 1, block_scale)
        normalized = groups / (safe_scale[..., None] * factor)
        boundaries = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=value.device
        )
        magnitude = normalized.abs()
        codes = torch.bucketize(magnitude, boundaries)
        ties = (
            (codes < 7)
            & (magnitude == boundaries[codes.clamp(max=6)])
            & (codes % 2 == 1)
        )
        codes = (
            (codes + ties | normalized.signbit().long() * 8)
            .byte()
            .reshape(rows, columns)
        )
        arguments.quantized.as_torch().copy_(codes[:, ::2] | codes[:, 1::2] << 4)

        scale_rows = -(-rows // 128) * 128
        logical_scales = torch.zeros(
            (scale_rows, columns // 16), dtype=torch.float32, device=value.device
        )
        logical_scales[:rows] = block_scale * (
            factor if arguments.input_scale is not None else 1
        )
        arguments.scales.as_torch().copy_(
            logical_scales.reshape(scale_rows // 128, 4, 32, columns // 64, 4)
            .permute(0, 3, 2, 1, 4)
            .contiguous()
            .reshape(arguments.scales.shape)
        )
