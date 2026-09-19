"""Add matching BF16 or F32 residual tensors without broadcasting."""

import dataclasses
import math

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class ResidualWorkload(Workload):
    dtype: DType
    num_elements: int


@dataclasses.dataclass(frozen=True, slots=True)
class ResidualArguments(Arguments):
    """Add input and skip with identical shape/dtype into matching output storage.

    Leading dimensions flatten into rows; the final axis supplies columns.
    Output may exactly alias either input, since each element is read before its
    sum is stored. No broadcasting or shifted overlapping views are supported.
    """

    input: TensorDesc
    skip: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_elements", math.prod(self.input.shape)),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def residual(num_elements, dtype, tile_elements=256, threads=128):
    import tilelang.language as T

    dynamic_num_elements = T.dynamic("num_elements")
    tilelang_dtype = T.bfloat16 if dtype == "BF16" else T.float32

    @T.prim_func
    def main(
        num_elements: dynamic_num_elements,
        input: T.Tensor([dynamic_num_elements], tilelang_dtype),
        skip: T.Tensor([dynamic_num_elements], tilelang_dtype),
        output: T.Tensor([dynamic_num_elements], tilelang_dtype),
    ):
        with T.Kernel(
            T.ceildiv(dynamic_num_elements, tile_elements), threads=threads
        ) as bid:
            for i in T.Parallel(tile_elements):
                offset = bid * tile_elements + i
                if offset < dynamic_num_elements:
                    output[offset] = input[offset] + skip[offset]

    return main.with_attr(
        "global_symbol", f"residual_{dtype}_{tile_elements}_{threads}"
    )


class ResidualKernel(ElementwiseKernel[ResidualArguments, ResidualWorkload]):
    name = "residual"
    program = residual
    launch = (256, 128)

    @classmethod
    def make_arguments(cls, workload: ResidualWorkload) -> ResidualArguments:
        """Describe three flat tensors with the workload's shared dtype."""
        shape = (workload.num_elements,)
        return ResidualArguments(
            TensorDesc.empty(workload.dtype, shape),
            TensorDesc.empty(workload.dtype, shape),
            TensorDesc.empty(workload.dtype, shape),
        )

    @classmethod
    def make_workload(cls, arguments: ResidualArguments) -> ResidualWorkload:
        """Recover the flat size and validate all three tensor contracts."""
        input = arguments.input
        assert input.dtype in (DType.BF16, DType.F32)
        assert arguments.skip.dtype == arguments.output.dtype == input.dtype
        assert arguments.skip.shape == arguments.output.shape == input.shape
        return ResidualWorkload(dtype=input.dtype, num_elements=math.prod(input.shape))

    @classmethod
    def ref_program(cls, arguments: ResidualArguments) -> None:
        """Add the addressed input and skip tensors into output."""
        arguments.output.as_torch().copy_(
            arguments.input.as_torch() + arguments.skip.as_torch()
        )
