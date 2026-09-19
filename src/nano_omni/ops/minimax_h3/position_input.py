"""Apply H3 rotary frequencies to the caller's ordered [rows, 3] positions."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.specialized.h3_rope import H3RopeArguments, H3RopeKernel


class PositionInput(Op[TensorDesc]):
    def __init__(self, input: int | tuple[int, int], *, rows: int) -> None:
        super().__init__(
            TensorDesc.activation(input, DType.F32, (rows, 3)),
            outputs=((DType.F32, (2, rows, 48)),),
        )

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (positions,) = self.inputs
        (output,) = self.bound_outputs
        return (
            H3RopeKernel(
                H3RopeArguments(
                    positions,
                    self.bound_weights,
                    output,
                )
            ),
        )
