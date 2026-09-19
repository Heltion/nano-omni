"""RMS normalization followed by per-column affine modulation."""

import dataclasses
from typing import TYPE_CHECKING

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc

if TYPE_CHECKING:
    from tilelang.language.eager import PrimFunc


class AdaptiveRmsNormWorkload(Workload):
    num_tokens: int
    columns: int
    output_dtype: DType


class AdaptiveRmsNormConfig(Config):
    tile_tokens: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class AdaptiveRmsNormArguments(Arguments):
    """BF16 input/weight/shift/scale and a BF16 or F32 output."""

    input: TensorDesc
    weight: TensorDesc
    shift: TensorDesc
    scale: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def adaptive_rms_norm(
    num_tokens: int,
    columns: int,
    output_dtype: DType,
    tile_tokens: int = 1,
    threads: int = 64,
) -> "PrimFunc[..., None]":
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    output_type = T.float32 if output_dtype == DType.F32 else T.bfloat16

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        input: T.Tensor((dynamic_num_tokens, columns), T.bfloat16),
        weight: T.Tensor((columns,), T.bfloat16),
        shift: T.Tensor((columns,), T.bfloat16),
        scale: T.Tensor((columns,), T.bfloat16),
        output: T.Tensor((dynamic_num_tokens, columns), output_type),
    ):
        with T.Kernel(
            T.ceildiv(dynamic_num_tokens, tile_tokens), threads=threads
        ) as block:
            values = T.alloc_fragment((tile_tokens, columns), T.float32)
            squares = T.alloc_fragment((tile_tokens, columns), T.float32)
            totals = T.alloc_fragment((tile_tokens,), T.float32)
            for local_row, column in T.Parallel(tile_tokens, columns):
                row = block * tile_tokens + local_row
                value = T.if_then_else(
                    row < dynamic_num_tokens, input[row, column], 0.0
                )
                values[local_row, column] = value
                squares[local_row, column] = value * value
            T.reduce_sum(squares, totals, dim=1)
            for local_row, column in T.Parallel(tile_tokens, columns):
                row = block * tile_tokens + local_row
                if row < dynamic_num_tokens:
                    normalized = (
                        values[local_row, column]
                        * T.rsqrt(totals[local_row] / columns + 1e-5)
                        * weight[column]
                    )
                    output[row, column] = (
                        normalized * (1.0 + scale[column]) + shift[column]
                    )

    return main.with_attr(
        "global_symbol",
        f"adaptive_rms_norm_{columns}_{output_dtype}_{tile_tokens}_{threads}",
    )


class AdaptiveRmsNormKernel(
    Kernel[AdaptiveRmsNormArguments, AdaptiveRmsNormWorkload, AdaptiveRmsNormConfig]
):
    name = "adaptive_rms_norm"
    program = adaptive_rms_norm

    @classmethod
    def make_arguments(
        cls, workload: AdaptiveRmsNormWorkload
    ) -> AdaptiveRmsNormArguments:
        """Describe BF16 normalization inputs and the selected output dtype."""
        shape = (workload.num_tokens, workload.columns)
        return AdaptiveRmsNormArguments(
            input=TensorDesc.empty(DType.BF16, shape),
            weight=TensorDesc.empty(DType.BF16, (workload.columns,)),
            shift=TensorDesc.empty(DType.BF16, (workload.columns,)),
            scale=TensorDesc.empty(DType.BF16, (workload.columns,)),
            output=TensorDesc.empty(workload.output_dtype, shape),
        )

    @classmethod
    def make_config(cls, workload: AdaptiveRmsNormWorkload) -> AdaptiveRmsNormConfig:
        """Select the current output-dtype-specific production launch."""
        # Small F32 outputs are launch-bound; one extra warp reduces that cost.
        if workload.output_dtype == DType.F32 and workload.num_tokens <= 768:
            return AdaptiveRmsNormConfig(tile_tokens=1, threads=64)
        return AdaptiveRmsNormConfig(
            tile_tokens=1,
            threads=32 if workload.output_dtype == DType.F32 else 64,
        )

    @classmethod
    def make_workload(
        cls, arguments: AdaptiveRmsNormArguments
    ) -> AdaptiveRmsNormWorkload:
        num_tokens, columns = arguments.input.shape
        assert all(
            tensor.dtype == DType.BF16
            for tensor in (
                arguments.input,
                arguments.weight,
                arguments.shift,
                arguments.scale,
            )
        )
        assert arguments.weight.shape == (columns,)
        assert arguments.shift.shape == (columns,)
        assert arguments.scale.shape == (columns,)
        assert arguments.output.shape == (num_tokens, columns)
        assert arguments.output.dtype in (DType.BF16, DType.F32)
        return AdaptiveRmsNormWorkload(
            num_tokens=num_tokens,
            columns=columns,
            output_dtype=arguments.output.dtype,
        )

    @classmethod
    def ref_program(cls, arguments: AdaptiveRmsNormArguments) -> None:
        """Apply RMS normalization and per-column affine modulation."""
        import torch

        value = arguments.input.as_torch().float()
        inverse = torch.rsqrt(value.square().mean(dim=1, keepdim=True) + 1e-5)
        normalized = value * inverse * arguments.weight.as_torch().float()
        result = normalized * (1.0 + arguments.scale.as_torch().float())
        result += arguments.shift.as_torch().float()
        arguments.output.as_torch().copy_(result)
