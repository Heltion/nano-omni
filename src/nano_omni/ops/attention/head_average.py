"""Bind F32 head averaging without allocating intermediate activations."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.reduction.head_average import (
    HeadAverageArguments,
    HeadAverageKernel,
)


class HeadAverageOp(Op[None]):
    """Average heads and contiguous within-head groups into output columns."""

    def __init__(
        self,
        input: int | tuple[int, int],
        *,
        rows: int,
        input_columns: int,
        heads: int,
        output_columns: int,
    ) -> None:
        assert input_columns % heads == 0
        input_shape = (rows, heads, input_columns // heads)
        output_shape = (rows, output_columns)
        super().__init__(
            TensorDesc.activation(input, DType.F32, input_shape),
            outputs=((DType.F32, output_shape),),
        )

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        return (
            HeadAverageKernel(
                HeadAverageArguments(
                    self.inputs[0],
                    output,
                )
            ),
        )
