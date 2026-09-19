"""Concatenate contiguous BF16 or FP32 matrices along axis zero."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class ConcatRowsWorkload(Workload):
    """Shared BF16/F32 dtype, input row counts and common feature width."""

    dtype: DType = DType.BF16
    num_first_tokens: int
    num_second_tokens: int
    columns: int


@dataclasses.dataclass(frozen=True, slots=True)
class ConcatRowsArguments(Arguments):
    """Matching row-major BF16/F32 matrices with one common column count.

    Output has first_rows + second_rows rows: all first rows, then all second
    rows. Row stride is columns elements; arbitrary strides are not supported.
    """

    first: TensorDesc
    second: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_first_tokens", self.first.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def concat_rows(
    dtype,
    num_first_tokens,
    num_second_tokens,
    columns,
    tile_elements=1024,
    threads=256,
):
    import tilelang.language as T

    dynamic_num_first_tokens = T.dynamic("num_first_tokens")
    storage_type = T.float32 if dtype == DType.F32 else T.bfloat16
    num_tokens = dynamic_num_first_tokens + num_second_tokens
    num_elements = num_tokens * columns

    @T.prim_func
    def main(
        num_first_tokens: dynamic_num_first_tokens,
        first: T.Tensor([dynamic_num_first_tokens, columns], storage_type),
        second: T.Tensor([num_second_tokens, columns], storage_type),
        output: T.Tensor([num_tokens, columns], storage_type),
    ):
        with T.Kernel(T.ceildiv(num_elements, tile_elements), threads=threads) as block:
            for offset in T.Parallel(tile_elements):
                linear = block * tile_elements + offset
                if linear < num_elements:
                    row = linear // columns
                    column = linear % columns
                    output[row, column] = T.if_then_else(
                        row < dynamic_num_first_tokens,
                        first[T.min(row, dynamic_num_first_tokens - 1), column],
                        second[T.max(row - dynamic_num_first_tokens, 0), column],
                    )

    return main.with_attr(
        "global_symbol",
        f"concat_rows_{dtype}_{num_second_tokens}_{columns}_{tile_elements}_{threads}",
    )


class ConcatRowsKernel(ElementwiseKernel[ConcatRowsArguments, ConcatRowsWorkload]):
    """Copy rows without interleaving or changing their shared element dtype."""

    name = "concat_rows"
    program = concat_rows
    launch = (256, 128)

    @classmethod
    def make_arguments(cls, workload: ConcatRowsWorkload) -> ConcatRowsArguments:
        """Describe both input matrices and their concatenated output."""
        return ConcatRowsArguments(
            TensorDesc.empty(
                workload.dtype, (workload.num_first_tokens, workload.columns)
            ),
            TensorDesc.empty(
                workload.dtype, (workload.num_second_tokens, workload.columns)
            ),
            TensorDesc.empty(
                workload.dtype,
                (
                    workload.num_first_tokens + workload.num_second_tokens,
                    workload.columns,
                ),
            ),
        )

    @classmethod
    def make_workload(cls, arguments: ConcatRowsArguments) -> ConcatRowsWorkload:
        """Recover the workload and validate both matrices and their output."""
        assert len(arguments.first.shape) == len(arguments.second.shape) == 2
        num_first_tokens, columns = arguments.first.shape
        num_second_tokens, second_columns = arguments.second.shape
        assert second_columns == columns
        assert arguments.first.dtype in (DType.BF16, DType.F32)
        assert arguments.first.dtype == arguments.second.dtype == arguments.output.dtype
        assert arguments.output.shape == (
            num_first_tokens + num_second_tokens,
            columns,
        )
        return ConcatRowsWorkload(
            dtype=arguments.first.dtype,
            num_first_tokens=num_first_tokens,
            num_second_tokens=num_second_tokens,
            columns=columns,
        )

    @classmethod
    def ref_program(cls, arguments: ConcatRowsArguments) -> None:
        """Concatenate the two addressed inputs along their row axis."""
        import torch

        torch.cat(
            (arguments.first.as_torch(), arguments.second.as_torch()),
            dim=0,
            out=arguments.output.as_torch(),
        )
