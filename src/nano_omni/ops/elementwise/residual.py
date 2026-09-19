"""Bind residual addition without changing its dtype or broadcast-free shape."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.residual import ResidualArguments, ResidualKernel


class ResidualOp(Op[None]):
    def __init__(
        self,
        input: int | tuple[int, int],
        skip: int | tuple[int, int],
        *,
        dtype: DType,
        shape: tuple[int, ...],
    ) -> None:
        inputs = tuple(
            TensorDesc.activation(position, dtype, shape) for position in (input, skip)
        )
        super().__init__(*inputs, outputs=((dtype, shape),))

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        input, skip = self.inputs
        return (ResidualKernel(ResidualArguments(input, skip, output)),)
