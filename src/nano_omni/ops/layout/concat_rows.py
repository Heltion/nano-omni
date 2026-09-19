"""Concatenate two contiguous matrices along their row dimension."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.layout.concat_rows import ConcatRowsArguments, ConcatRowsKernel


class ConcatRowsOp(Op[None]):
    def __init__(
        self,
        first: int | tuple[int, int],
        second: int | tuple[int, int],
        *,
        dtype: DType,
        first_shape: tuple[int, ...],
        second_shape: tuple[int, ...],
    ) -> None:
        inputs = (
            TensorDesc.activation(first, dtype, first_shape),
            TensorDesc.activation(second, dtype, second_shape),
        )
        output_shape = (first_shape[0] + second_shape[0], first_shape[1])
        super().__init__(*inputs, outputs=((dtype, output_shape),))

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        return (
            ConcatRowsKernel(
                ConcatRowsArguments(
                    self.inputs[0],
                    self.inputs[1],
                    output,
                )
            ),
        )
