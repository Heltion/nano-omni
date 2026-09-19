"""Per-channel Snake activation for the DAC audio encoder."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.activation.snake import SnakeArguments, SnakeKernel


class SnakeOp(Op[TensorDesc]):
    """Apply Snake to an explicitly bound activation."""

    def __init__(
        self, input: int | tuple[int, int], *, rows: int, columns: int
    ) -> None:
        shape = (rows, columns)
        super().__init__(
            TensorDesc.activation(input, DType.F32, shape),
            outputs=((DType.F32, shape),),
        )

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        return (
            SnakeKernel(
                SnakeArguments(
                    self.inputs[0],
                    self.bound_weights,
                    output,
                )
            ),
        )
