"""Bind a contiguous tensor copy between model activation slices."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.copy import CopyArguments, CopyKernel


class CopyOp(Op[None]):
    def __init__(
        self,
        input: int | tuple[int, int],
        *,
        dtype: DType,
        shape: tuple[int, ...],
    ) -> None:
        super().__init__(
            TensorDesc.activation(input, dtype, shape), outputs=((dtype, shape),)
        )

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        return (CopyKernel(CopyArguments(self.inputs[0], self.bound_outputs[0])),)
