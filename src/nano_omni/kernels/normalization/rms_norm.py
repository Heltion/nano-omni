"""Apply weighted RMS normalization to independent FP16/BF16 rows."""

import dataclasses
from typing import Literal

import tilelang

from nano_omni.core.kernel import Arguments, Kernel
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.normalization.base import (
    RowNormalizationConfig,
    RowNormalizationWorkload,
)


class RmsNormWorkload(RowNormalizationWorkload):
    """Row shape, epsilon, weight dtype, and output storage dtype."""

    weight_dtype: Literal[DType.F16, DType.BF16] = DType.BF16
    input_dtype: Literal[DType.F16, DType.BF16] = DType.BF16
    output_dtype: Literal[DType.F16, DType.BF16] = DType.BF16


@dataclasses.dataclass(frozen=True, slots=True)
class RmsNormArguments(Arguments):
    """Input rows, column weights, output rows, and runtime epsilon."""

    input: TensorDesc
    weight: TensorDesc
    output: TensorDesc
    epsilon: float

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def rms_norm(
    num_tokens,
    columns,
    epsilon,
    weight_dtype,
    input_dtype,
    output_dtype,
    tile_tokens=1,
    threads=128,
):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    reduction_columns = T.ceildiv(columns, threads) * threads
    weight_type = T.float16 if weight_dtype == DType.F16 else T.bfloat16
    input_type = T.float16 if input_dtype == DType.F16 else T.bfloat16
    output_type = T.float16 if output_dtype == DType.F16 else T.bfloat16

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        input: T.Tensor((dynamic_num_tokens, columns), input_type),
        weight: T.Tensor((columns,), weight_type),
        output: T.Tensor((dynamic_num_tokens, columns), output_type),
    ):
        with T.Kernel(
            T.ceildiv(dynamic_num_tokens, tile_tokens), threads=threads
        ) as block:
            values = T.alloc_fragment((tile_tokens, reduction_columns), T.float32)
            squares = T.alloc_fragment((tile_tokens, reduction_columns), T.float32)
            totals = T.alloc_fragment((tile_tokens,), T.float32)
            inverse = T.alloc_fragment((tile_tokens,), T.float32)
            for local_row, column in T.Parallel(tile_tokens, reduction_columns):
                row = block * tile_tokens + local_row
                if reduction_columns == columns:
                    values[local_row, column] = input[row, column]
                else:
                    values[local_row, column] = T.if_then_else(
                        (row < dynamic_num_tokens) & (column < columns),
                        input[row, column],
                        0.0,
                    )
                squares[local_row, column] = (
                    values[local_row, column] * values[local_row, column]
                )
            T.reduce_sum(squares, totals, dim=1)
            for local_row in T.Parallel(tile_tokens):
                inverse[local_row] = T.rsqrt(totals[local_row] / columns + epsilon)
            for local_row, column in T.Parallel(tile_tokens, reduction_columns):
                row = block * tile_tokens + local_row
                if row < dynamic_num_tokens and column < columns:
                    value = (
                        values[local_row, column] * inverse[local_row] * weight[column]
                    )
                    if output_dtype == DType.F16:
                        output[row, column] = T.cast(value, T.float16)
                    else:
                        output[row, column] = value

    epsilon_tag = str(epsilon).replace("-", "m").replace(".", "p")
    return main.with_attr(
        "global_symbol",
        f"rms_norm_{columns}_{tile_tokens}_{threads}_{epsilon_tag}_{weight_dtype.value}_{input_dtype.value}_{output_dtype.value}",
    )


class RmsNormKernel(Kernel[RmsNormArguments, RmsNormWorkload, RowNormalizationConfig]):
    name = "rms_norm"
    program = rms_norm

    @classmethod
    def make_arguments(cls, workload: RmsNormWorkload) -> RmsNormArguments:
        """Describe input rows, per-column weight, and normalized output."""
        shape = (workload.num_tokens, workload.columns)
        return RmsNormArguments(
            input=TensorDesc.empty(workload.input_dtype, shape),
            weight=TensorDesc.empty(workload.weight_dtype, (workload.columns,)),
            output=TensorDesc.empty(workload.output_dtype, shape),
            epsilon=workload.epsilon,
        )

    @classmethod
    def make_workload(cls, arguments: RmsNormArguments) -> RmsNormWorkload:
        assert len(arguments.input.shape) == 2
        assert arguments.output.shape == arguments.input.shape
        assert arguments.weight.shape == (arguments.input.shape[1],)
        assert arguments.input.dtype in (DType.F16, DType.BF16)
        assert arguments.output.dtype in (DType.F16, DType.BF16)
        assert arguments.weight.dtype in (DType.F16, DType.BF16)
        return RmsNormWorkload(
            num_tokens=arguments.input.shape[0],
            columns=arguments.input.shape[1],
            epsilon=arguments.epsilon,
            input_dtype=arguments.input.dtype,
            weight_dtype=arguments.weight.dtype,
            output_dtype=arguments.output.dtype,
        )

    @classmethod
    def make_config(cls, workload: RmsNormWorkload) -> RowNormalizationConfig:
        """Return the current production launch configuration."""
        threads = 64 if (workload.num_tokens, workload.columns) == (4096, 5376) else 128
        return RowNormalizationConfig(threads=threads, tile_tokens=1)

    @classmethod
    def ref_program(cls, arguments: RmsNormArguments) -> None:
        """Normalize complete rows in FP32 and apply the column weights."""
        import torch

        value = arguments.input.as_torch().float()
        inverse = torch.rsqrt(
            value.square().mean(dim=1, keepdim=True) + arguments.epsilon
        )
        result = value * inverse * arguments.weight.as_torch().float()
        arguments.output.as_torch().copy_(
            result.half() if arguments.output.dtype == DType.F16 else result.bfloat16()
        )
