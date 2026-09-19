"""Apply the tanh GELU approximation to contiguous BF16 values."""

import dataclasses
import math

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class GeluWorkload(Workload):
    """Number of contiguous elements."""

    elements: int


@dataclasses.dataclass(frozen=True, slots=True)
class GeluArguments(Arguments):
    """Matching BF16 input and output tensors."""

    input: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def gelu(elements, tile_elements=1024, threads=256):
    import tilelang.language as T

    @T.prim_func
    def main(
        input: T.Tensor((elements,), T.bfloat16),
        output: T.Tensor((elements,), T.bfloat16),
    ):
        T.annotate_pass_configs({"tl.enable_fast_math": True})
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < elements:
                    value = input[index]
                    cubic = value * value * value
                    output[index] = (
                        0.5
                        * value
                        * (
                            1.0
                            + T.tanh(0.7978845608028654 * (value + 0.044715 * cubic))
                        )
                    )

    return main.with_attr("global_symbol", f"gelu_{elements}_{tile_elements}_{threads}")


class GeluKernel(ElementwiseKernel[GeluArguments, GeluWorkload]):
    name = "gelu"
    program = gelu
    launch = (1024, 256)

    @classmethod
    def make_arguments(cls, workload: GeluWorkload) -> GeluArguments:
        """Describe separate BF16 input and output tensors."""
        shape = (workload.elements,)
        return GeluArguments(
            input=TensorDesc.empty(DType.BF16, shape),
            output=TensorDesc.empty(DType.BF16, shape),
        )

    @classmethod
    def make_workload(cls, arguments: GeluArguments) -> GeluWorkload:
        assert arguments.input.shape == arguments.output.shape
        assert arguments.input.dtype == arguments.output.dtype == DType.BF16
        return GeluWorkload(elements=math.prod(arguments.input.shape))

    @classmethod
    def ref_program(cls, arguments: GeluArguments) -> None:
        """Apply the same tanh GELU approximation as the TileLang kernel."""
        import torch

        value = arguments.input.as_torch().float()
        arguments.output.as_torch().copy_(
            0.5
            * value
            * (1.0 + torch.tanh(0.7978845608028654 * (value + 0.044715 * value**3)))
        )
