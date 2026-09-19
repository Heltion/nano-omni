"""Multiply an FP32 gate by the tanh GELU approximation of an FP32 input."""

import dataclasses
import math

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class GeluMultiplyWorkload(Workload):
    """Number of contiguous elements."""

    elements: int


@dataclasses.dataclass(frozen=True, slots=True)
class GeluMultiplyArguments(Arguments):
    """Independent input, gate, and output tensors."""

    input: TensorDesc
    gate: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def gelu_multiply(elements, tile_elements=256, threads=128):
    import tilelang.language as T

    @T.prim_func
    def main(
        input: T.Tensor((elements,), T.float32),
        gate: T.Tensor((elements,), T.float32),
        output: T.Tensor((elements,), T.float32),
    ):
        T.annotate_pass_configs(
            {"tl.enable_fast_math": True, "tl.disable_vectorize_256": True}
        )
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < elements:
                    value = input[index]
                    cubic = value * value * value
                    output[index] = (
                        gate[index]
                        * 0.5
                        * value
                        * (
                            1.0
                            + T.tanh(0.7978845608028654 * (value + 0.044715 * cubic))
                        )
                    )

    return main.with_attr(
        "global_symbol", f"gelu_multiply_{elements}_{tile_elements}_{threads}"
    )


class GeluMultiplyKernel(
    ElementwiseKernel[GeluMultiplyArguments, GeluMultiplyWorkload]
):
    name = "gelu_multiply"
    program = gelu_multiply
    launch = (256, 128)

    @classmethod
    def make_arguments(cls, workload: GeluMultiplyWorkload) -> GeluMultiplyArguments:
        """Describe independent flat input, gate, and output tensors."""
        shape = (workload.elements,)
        return GeluMultiplyArguments(
            input=TensorDesc.empty(DType.F32, shape),
            gate=TensorDesc.empty(DType.F32, shape),
            output=TensorDesc.empty(DType.F32, shape),
        )

    @classmethod
    def make_workload(cls, arguments: GeluMultiplyArguments) -> GeluMultiplyWorkload:
        shape = arguments.input.shape
        assert arguments.gate.shape == arguments.output.shape == shape
        assert (
            arguments.input.dtype
            == arguments.gate.dtype
            == arguments.output.dtype
            == DType.F32
        )
        return GeluMultiplyWorkload(elements=math.prod(shape))

    @classmethod
    def ref_program(cls, arguments: GeluMultiplyArguments) -> None:
        """Multiply the gate by the same tanh GELU approximation."""
        import torch

        value = arguments.input.as_torch()
        arguments.output.as_torch().copy_(
            arguments.gate.as_torch()
            * 0.5
            * value
            * (1.0 + torch.tanh(0.7978845608028654 * (value + 0.044715 * value**3)))
        )
