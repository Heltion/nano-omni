"""Copy one contiguous tensor into separate storage."""

import dataclasses
import math

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class CopyWorkload(Workload):
    """Element type and count of a contiguous tensor."""

    dtype: DType
    num_elements: int


@dataclasses.dataclass(frozen=True, slots=True)
class CopyArguments(Arguments):
    """Separate source and destination tensors."""

    input: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_elements", math.prod(self.input.shape)),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def copy(dtype, num_elements, tile_elements=1024, threads=256):
    import tilelang.language as T

    dynamic_num_elements = T.dynamic("num_elements")
    element_type = T.bfloat16 if dtype == DType.BF16 else T.float32

    @T.prim_func
    def main(
        num_elements: dynamic_num_elements,
        input: T.Tensor((dynamic_num_elements,), element_type),
        output: T.Tensor((dynamic_num_elements,), element_type),
    ):
        with T.Kernel(
            T.ceildiv(dynamic_num_elements, tile_elements), threads=threads
        ) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < dynamic_num_elements:
                    output[index] = input[index]

    return main.with_attr(
        "global_symbol", f"copy_{dtype.value}_{tile_elements}_{threads}"
    )


class CopyKernel(ElementwiseKernel[CopyArguments, CopyWorkload]):
    name = "copy"
    program = copy
    launch = (1024, 256)

    @classmethod
    def make_arguments(cls, workload: CopyWorkload) -> CopyArguments:
        """Describe a flat source and destination for this copy."""
        shape = (workload.num_elements,)
        return CopyArguments(
            input=TensorDesc.empty(workload.dtype, shape),
            output=TensorDesc.empty(workload.dtype, shape),
        )

    @classmethod
    def make_workload(cls, arguments: CopyArguments) -> CopyWorkload:
        assert arguments.output.dtype == arguments.input.dtype
        assert arguments.output.shape == arguments.input.shape
        return CopyWorkload(
            dtype=arguments.input.dtype,
            num_elements=math.prod(arguments.input.shape),
        )

    @classmethod
    def ref_program(cls, arguments: CopyArguments) -> None:
        """Copy the addressed input into the addressed output."""
        arguments.output.as_torch().copy_(arguments.input.as_torch())
