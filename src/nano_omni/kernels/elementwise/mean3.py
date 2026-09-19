"""Average three matching FP32 tensors element by element."""

import dataclasses
import math

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class Mean3Workload(Workload):
    """Number of independent FP32 elements."""

    elements: int


@dataclasses.dataclass(frozen=True, slots=True)
class Mean3Arguments(Arguments):
    """Three matching inputs and one output."""

    first: TensorDesc
    second: TensorDesc
    third: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def mean3(elements, tile_elements=1024, threads=256):
    import tilelang.language as T

    @T.prim_func
    def main(
        first: T.Tensor((elements,), T.float32),
        second: T.Tensor((elements,), T.float32),
        third: T.Tensor((elements,), T.float32),
        output: T.Tensor((elements,), T.float32),
    ):
        T.annotate_pass_configs({"tl.disable_vectorize_256": True})
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for offset in T.Parallel(tile_elements):
                element = block * tile_elements + offset
                if element < elements:
                    output[element] = (
                        first[element] + second[element] + third[element]
                    ) / 3.0

    return main.with_attr(
        "global_symbol", f"mean3_{elements}_{tile_elements}_{threads}"
    )


class Mean3Kernel(ElementwiseKernel[Mean3Arguments, Mean3Workload]):
    name = "mean3"
    program = mean3
    launch = (256, 128)

    @classmethod
    def make_arguments(cls, workload: Mean3Workload) -> Mean3Arguments:
        """Describe three independent flat inputs and one matching output."""
        shape = (workload.elements,)
        return Mean3Arguments(
            first=TensorDesc.empty(DType.F32, shape),
            second=TensorDesc.empty(DType.F32, shape),
            third=TensorDesc.empty(DType.F32, shape),
            output=TensorDesc.empty(DType.F32, shape),
        )

    @classmethod
    def make_workload(cls, arguments: Mean3Arguments) -> Mean3Workload:
        shape = arguments.first.shape
        assert (
            arguments.second.shape
            == arguments.third.shape
            == arguments.output.shape
            == shape
        )
        assert all(
            tensor.dtype == DType.F32
            for tensor in (
                arguments.first,
                arguments.second,
                arguments.third,
                arguments.output,
            )
        )
        return Mean3Workload(elements=math.prod(shape))

    @classmethod
    def ref_program(cls, arguments: Mean3Arguments) -> None:
        """Average the three addressed inputs into output."""
        arguments.output.as_torch().copy_(
            (
                arguments.first.as_torch()
                + arguments.second.as_torch()
                + arguments.third.as_torch()
            )
            / 3.0
        )
