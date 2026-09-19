"""Average three F32 audio branches using the existing fused kernel."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.mean3 import Mean3Arguments, Mean3Kernel


class Mean3Op(Op[None]):
    def __init__(
        self,
        first: int | tuple[int, int],
        second: int | tuple[int, int],
        third: int | tuple[int, int],
        *,
        shape: tuple[int, ...],
    ) -> None:
        inputs = tuple(
            TensorDesc.activation(position, DType.F32, shape)
            for position in (first, second, third)
        )
        super().__init__(*inputs, outputs=((DType.F32, shape),))

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        first, second, third = self.inputs
        return (Mean3Kernel(Mean3Arguments(first, second, third, output)),)
