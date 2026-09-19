"""Row-wise layer normalization with optional affine parameters."""

import dataclasses
from typing import Literal

import tilelang

from nano_omni.core.kernel import Arguments, Kernel
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.normalization.base import (
    RowNormalizationConfig,
    RowNormalizationWorkload,
)


class LayerNormWorkload(RowNormalizationWorkload):
    # Reduce columns in FP32; input and output share the selected F32/BF16 dtype.
    dtype: Literal[DType.F16, DType.BF16, DType.F32] = DType.BF16
    use_affine: bool
    affine_dtype: Literal[DType.F16, DType.BF16, DType.F32]


@dataclasses.dataclass(frozen=True, slots=True)
class LayerNormArguments(Arguments):
    input: TensorDesc
    weight: TensorDesc | None
    bias: TensorDesc | None
    output: TensorDesc
    epsilon: float


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def layer_norm(
    num_tokens,
    columns,
    epsilon,
    dtype,
    use_affine,
    affine_dtype,
    tile_tokens=1,
    threads=128,
):
    import tilelang.language as T

    storage_type = {
        DType.F16: T.float16,
        DType.BF16: T.bfloat16,
        DType.F32: T.float32,
    }[dtype]
    affine_type = {
        DType.F16: T.float16,
        DType.BF16: T.bfloat16,
        DType.F32: T.float32,
    }[affine_dtype]
    reduction_columns = -(-columns // threads) * threads
    epsilon_tag = str(epsilon).replace("-", "m").replace(".", "p")

    @T.prim_func
    def main(
        input: T.Tensor([num_tokens, columns], storage_type),
        weight: T.Tensor([columns], affine_type),
        bias: T.Tensor([columns], affine_type),
        output: T.Tensor([num_tokens, columns], storage_type),
    ):
        T.annotate_pass_configs({"tl.disable_vectorize_256": dtype == DType.F32})
        with T.Kernel(T.ceildiv(num_tokens, tile_tokens), threads=threads) as block:
            values = T.alloc_fragment([tile_tokens, reduction_columns], T.float32)
            totals = T.alloc_fragment([tile_tokens], T.float32)
            squares = T.alloc_fragment([tile_tokens, reduction_columns], T.float32)
            inverse = T.alloc_fragment([tile_tokens], T.float32)
            for i, column in T.Parallel(tile_tokens, reduction_columns):
                row = block * tile_tokens + i
                values[i, column] = T.if_then_else(
                    (row < num_tokens) & (column < columns),
                    input[row, column],
                    0.0,
                )
            T.reduce_sum(values, totals, dim=1)
            for i, column in T.Parallel(tile_tokens, reduction_columns):
                # Padding must not contribute mean-subtracted values to variance.
                values[i, column] = T.if_then_else(
                    column < columns,
                    values[i, column] - totals[i] / columns,
                    0.0,
                )
                squares[i, column] = values[i, column] * values[i, column]
            T.reduce_sum(squares, totals, dim=1)
            for i in T.Parallel(tile_tokens):
                inverse[i] = T.rsqrt(totals[i] / columns + epsilon)
            for i, column in T.Parallel(tile_tokens, reduction_columns):
                row = block * tile_tokens + i
                if row < num_tokens and column < columns:
                    normalized = values[i, column] * inverse[i]
                    output[row, column] = T.if_then_else(
                        use_affine,
                        normalized * weight[column] + bias[column],
                        normalized,
                    )

    return main.with_attr(
        "global_symbol",
        "layer_norm_"
        f"{num_tokens}_{columns}_{tile_tokens}_{threads}_{epsilon_tag}_{int(use_affine)}_"
        f"{affine_dtype.value}_{dtype.value}",
    )


class LayerNormKernel(
    Kernel[LayerNormArguments, LayerNormWorkload, RowNormalizationConfig]
):
    name = "layer_norm"
    program = layer_norm

    @classmethod
    def make_arguments(cls, workload: LayerNormWorkload) -> LayerNormArguments:
        """Describe normalized rows and optional affine parameters."""
        shape = (workload.num_tokens, workload.columns)
        affine_shape = (workload.columns,)
        return LayerNormArguments(
            input=TensorDesc.empty(workload.dtype, shape),
            weight=(
                TensorDesc.empty(workload.affine_dtype, affine_shape)
                if workload.use_affine
                else None
            ),
            bias=(
                TensorDesc.empty(workload.affine_dtype, affine_shape)
                if workload.use_affine
                else None
            ),
            output=TensorDesc.empty(workload.dtype, shape),
            epsilon=workload.epsilon,
        )

    @classmethod
    def make_config(cls, workload: LayerNormWorkload) -> RowNormalizationConfig:
        """Select the production configuration for this workload."""
        if workload.dtype == DType.F16:
            return RowNormalizationConfig(threads=128, tile_tokens=1)
        if workload.columns in (1152, 4608):
            return RowNormalizationConfig(threads=64, tile_tokens=1)
        return RowNormalizationConfig(threads=256, tile_tokens=4)

    @classmethod
    def make_workload(cls, arguments: LayerNormArguments) -> LayerNormWorkload:
        assert len(arguments.input.shape) == 2
        assert arguments.input.dtype in (DType.F16, DType.BF16, DType.F32)
        assert arguments.output.dtype == arguments.input.dtype
        assert arguments.output.shape == arguments.input.shape
        assert (arguments.weight is None) == (arguments.bias is None), (
            "layer norm weight and bias must be provided together"
        )
        if arguments.weight is not None and arguments.bias is not None:
            assert arguments.weight.dtype in (DType.F16, DType.BF16, DType.F32)
            assert arguments.bias.dtype == arguments.weight.dtype
            assert (
                arguments.weight.shape
                == arguments.bias.shape
                == (arguments.input.shape[1],)
            )
        return LayerNormWorkload(
            dtype=arguments.input.dtype,
            num_tokens=arguments.input.shape[0],
            columns=arguments.input.shape[1],
            epsilon=arguments.epsilon,
            use_affine=arguments.weight is not None,
            affine_dtype=(
                arguments.weight.dtype
                if arguments.weight is not None
                else arguments.input.dtype
            ),
        )

    @classmethod
    def ref_program(cls, arguments: LayerNormArguments) -> None:
        """Normalize rows in FP32 and apply optional affine parameters."""
        import torch

        value = arguments.input.as_torch().float()
        centered = value - value.mean(dim=1, keepdim=True)
        result = centered * torch.rsqrt(
            centered.square().mean(dim=1, keepdim=True) + arguments.epsilon
        )
        if arguments.weight is not None:
            assert arguments.bias is not None
            result *= arguments.weight.as_torch().float()
            result += arguments.bias.as_torch().float()
        arguments.output.as_torch().copy_(result)
