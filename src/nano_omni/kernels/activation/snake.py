"""Apply a per-column Snake activation to an FP32 matrix."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class SnakeWorkload(Workload):
    """Matrix shape; one alpha value is shared by each column."""

    rows: int
    columns: int


@dataclasses.dataclass(frozen=True, slots=True)
class SnakeArguments(Arguments):
    """Activation matrix, per-column alpha, and output matrix."""

    input: TensorDesc
    alpha: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def snake(rows, columns, tile_elements=1024, threads=256):
    import tilelang.language as T

    elements = rows * columns

    @T.prim_func
    def main(
        input: T.Tensor((rows, columns), T.float32),
        alpha: T.Tensor((columns,), T.float32),
        output: T.Tensor((rows, columns), T.float32),
    ):
        T.annotate_pass_configs({"tl.disable_vectorize_256": True})
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for offset in T.Parallel(tile_elements):
                linear = block * tile_elements + offset
                if linear < elements:
                    row = linear // columns
                    column = linear % columns
                    value = input[row, column]
                    scale = alpha[column]
                    wave = T.sin(scale * value)
                    output[row, column] = value + wave * wave / (scale + 1e-9)

    return main.with_attr(
        "global_symbol", f"snake_{rows}_{columns}_{tile_elements}_{threads}"
    )


class SnakeKernel(ElementwiseKernel[SnakeArguments, SnakeWorkload]):
    name = "snake"
    program = snake
    launch = (256, 128)

    @classmethod
    def make_arguments(cls, workload: SnakeWorkload) -> SnakeArguments:
        """Describe separate activation, alpha, and output tensors."""
        shape = (workload.rows, workload.columns)
        return SnakeArguments(
            input=TensorDesc.empty(DType.F32, shape),
            alpha=TensorDesc.empty(DType.F32, (workload.columns,)),
            output=TensorDesc.empty(DType.F32, shape),
        )

    @classmethod
    def make_workload(cls, arguments: SnakeArguments) -> SnakeWorkload:
        assert len(arguments.input.shape) == 2
        rows, columns = arguments.input.shape
        assert arguments.alpha.shape == (columns,)
        assert arguments.output.shape == arguments.input.shape
        assert (
            arguments.input.dtype
            == arguments.alpha.dtype
            == arguments.output.dtype
            == DType.F32
        )
        return SnakeWorkload(rows=rows, columns=columns)

    @classmethod
    def ref_program(cls, arguments: SnakeArguments) -> None:
        """Apply the Snake transform into the output tensor."""
        value = arguments.input.as_torch()
        alpha = arguments.alpha.as_torch()
        wave = (alpha * value).sin()
        arguments.output.as_torch().copy_(value + wave.square() / (alpha + 1e-9))
