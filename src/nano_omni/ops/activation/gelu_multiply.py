"""Emit tanh-approximate GELU(input) multiplied by a separate F32 gate."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.activation.gelu_multiply import (
    GeluMultiplyArguments,
    GeluMultiplyKernel,
)


class GeluMultiplyOp(Op[None]):
    def __init__(
        self,
        input: int | tuple[int, int],
        gate: int | tuple[int, int],
        *,
        shape: tuple[int, int],
    ) -> None:
        inputs = tuple(
            TensorDesc.activation(value, DType.F32, shape) for value in (input, gate)
        )
        super().__init__(*inputs, outputs=((DType.F32, shape),))

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        input, gate = self.inputs
        (output,) = self.bound_outputs
        return (
            GeluMultiplyKernel(
                GeluMultiplyArguments(
                    input,
                    gate,
                    output,
                )
            ),
        )
