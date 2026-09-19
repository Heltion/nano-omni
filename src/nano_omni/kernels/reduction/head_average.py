"""Average attention heads and contiguous feature groups."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class HeadAverageWorkload(Workload):
    """Input head layout and output column count."""

    rows: int
    heads: int
    dim: int
    columns: int


class HeadAverageConfig(Config):
    """CUDA threads assigned to each row."""

    threads: int = 128


@dataclasses.dataclass(frozen=True, slots=True)
class HeadAverageArguments(Arguments):
    """F32 input `[rows, heads, dim]` and output `[rows, columns]`."""

    input: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def head_average(rows, heads, dim, columns, threads=128):
    """Build the grouped mean reduction."""
    import tilelang.language as T

    group = dim // columns
    assert dim % columns == 0

    @T.prim_func
    def main(
        input: T.Tensor([rows, heads, dim], T.float32),
        output: T.Tensor([rows, columns], T.float32),
    ):
        with T.Kernel(rows, threads=threads) as row:
            values = T.alloc_fragment([columns, heads * group], T.float32)
            means = T.alloc_fragment([columns], T.float32)
            for column, item in T.Parallel(columns, heads * group):
                values[column, item] = input[
                    row, item // group, column * group + item % group
                ]
            T.reduce_sum(values, means, dim=1)
            for column in T.Parallel(columns):
                output[row, column] = means[column] / (heads * group)

    return main


class HeadAverageKernel(
    Kernel[HeadAverageArguments, HeadAverageWorkload, HeadAverageConfig]
):
    """Reduce the head and within-column group dimensions."""

    name = "head_average"
    program = head_average

    @classmethod
    def make_arguments(cls, workload: HeadAverageWorkload) -> HeadAverageArguments:
        """Describe the input and output tensors."""
        return HeadAverageArguments(
            input=TensorDesc.empty(
                DType.F32, (workload.rows, workload.heads, workload.dim)
            ),
            output=TensorDesc.empty(DType.F32, (workload.rows, workload.columns)),
        )

    @classmethod
    def make_workload(cls, arguments: HeadAverageArguments) -> HeadAverageWorkload:
        """Validate the tensor contract and recover reduction dimensions."""
        assert arguments.input.dtype == arguments.output.dtype == DType.F32
        assert len(arguments.input.shape) == 3
        assert len(arguments.output.shape) == 2
        rows, heads, dim = arguments.input.shape
        assert arguments.output.shape[0] == rows
        columns = arguments.output.shape[1]
        assert heads > 0 and columns > 0 and dim % columns == 0
        return HeadAverageWorkload(
            rows=rows,
            heads=heads,
            dim=dim,
            columns=columns,
        )

    @classmethod
    def make_config(
        cls, workload: HeadAverageWorkload
    ) -> HeadAverageConfig:
        """Select the measured production configuration."""
        del workload
        return HeadAverageConfig(threads=128)

    @classmethod
    def ref_program(cls, arguments: HeadAverageArguments) -> None:
        """Evaluate the grouped mean directly with Torch."""
        workload = cls.make_workload(arguments)
        group = workload.dim // workload.columns
        value = arguments.input.as_torch().reshape(
            workload.rows, workload.heads, workload.columns, group
        )
        arguments.output.as_torch().copy_(value.mean(dim=(1, 3)))
